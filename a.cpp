#include <cstdlib>
#include <iostream>
#include <complex>
#define MKL_Complex8 std::complex<float>
#include "mkl.h"
#include <cilk/cilk.h>
#include <omp.h>

int const N_PLAYERS = 16; // have hacked the partitioning to depend on this 16 so be careful
long const N_GRID = 1 << 12;
long const N_GRID_2 = 1 << 11;
long const N_GRID_4 = 1 << 10;
double const M_PI = 3.14159265358979323846;
double const M_SQRT1_2 = 1.0 / std::sqrt(2.0);
double const SQRTN_2PI = std::sqrt(0.5 / M_PI * N_GRID);

// udag basically the wrong way round here, see notes there was a change of notation :(
class FourierTransform1D {
  double* const s;
  MKL_Complex8* const post_factor;
  double* const h_neg_theta;
  double* const h_theta;
  double* const h_neg_theta_to_alpha_less_one;
  double* const h_theta_to_alpha_less_one;
  MKL_Complex8* const darray;
  DFTI_DESCRIPTOR_HANDLE hand;
public:
  FourierTransform1D() :
    s((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    post_factor((MKL_Complex8*)mkl_malloc(N_GRID * sizeof(MKL_Complex8), 64)),
    h_neg_theta((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    h_theta((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    h_neg_theta_to_alpha_less_one((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    h_theta_to_alpha_less_one((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    darray((MKL_Complex8*)mkl_malloc(3 * N_GRID * sizeof(MKL_Complex8), 64))
  {
    s[0: N_GRID] = 0.5 * std::sqrt(2.0 * M_PI / N_GRID) * (2 * __sec_implicit_index(0) - N_GRID + 1);
    float const a = M_PI * (N_GRID - 1) / (2 * N_GRID);
    for (int j = 0; j != N_GRID; ++j) {
      post_factor[j] = MKL_Complex8(0.0f, a * (N_GRID - 1 - 2 * j));
    }
    vcExp(N_GRID, post_factor, post_factor);
    float const b = -2.0f / std::sqrt(2.0f * M_PI * N_GRID);
    for (int j = 0; j != N_GRID; ++j) {
      post_factor[j] = b * post_factor[j];
    }
    MKL_LONG const length = N_GRID;
    DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_COMPLEX, 1, length);
    DftiSetValue(hand, DFTI_NUMBER_OF_TRANSFORMS, 3);
    DftiSetValue(hand, DFTI_INPUT_DISTANCE, N_GRID);
    DftiCommitDescriptor(hand);
  }
  ~FourierTransform1D() {
    mkl_free(s);
    mkl_free(post_factor);
    mkl_free(h_neg_theta);
    mkl_free(h_theta);
    mkl_free(h_neg_theta_to_alpha_less_one);
    mkl_free(h_theta_to_alpha_less_one);
    mkl_free(darray);
    DftiFreeDescriptor(&hand);
  }
  void run(double const alpha, double const theta, float* const buffer) {
    MKL_Complex8* const chi_U_s = darray;
    MKL_Complex8* const d_theta_chi_U_s = darray + N_GRID;
    MKL_Complex8* const d_theta_theta_chi_U_s = darray + 2 * N_GRID;
    double const tan_theta = std::tan(theta);
    double const sec_squared_theta = std::pow(std::cos(theta), -2);
    h_neg_theta[0: N_GRID] = 0.5 * M_SQRT1_2 * (1.0 - tan_theta) * s[0: N_GRID];
    h_theta[0: N_GRID] = 0.5 * M_SQRT1_2 * (1.0 + tan_theta) * s[0: N_GRID];
    vdAbs(N_GRID, h_neg_theta, h_neg_theta_to_alpha_less_one);
    vdAbs(N_GRID, h_theta, h_theta_to_alpha_less_one);
    vdPowx(N_GRID, h_neg_theta_to_alpha_less_one, alpha - 1.0, h_neg_theta_to_alpha_less_one);
    vdPowx(N_GRID, h_theta_to_alpha_less_one, alpha - 1.0, h_theta_to_alpha_less_one);
    if (h_neg_theta[0: N_GRID] < 0.0) {
      h_neg_theta_to_alpha_less_one[0: N_GRID] = -h_neg_theta_to_alpha_less_one[0: N_GRID];
    }
    if (h_theta[0: N_GRID] < 0.0) {
      h_theta_to_alpha_less_one[0: N_GRID] = -h_theta_to_alpha_less_one[0: N_GRID];
    }
    for (int j = 0; j != N_GRID; ++j) {
      double const h_neg_theta_to_alpha_j = h_neg_theta_to_alpha_less_one[j] * h_neg_theta[j];
      double const h_theta_to_alpha_j = h_theta_to_alpha_less_one[j] * h_theta[j];
      chi_U_s[j] = MKL_Complex8(-(h_neg_theta_to_alpha_j + h_theta_to_alpha_j), -M_PI * (N_GRID - 1) * j / N_GRID);
    }
    vcExp(N_GRID, chi_U_s, chi_U_s);
    d_theta_chi_U_s[0: N_GRID] = 0.5 * M_SQRT1_2 * alpha * sec_squared_theta
      * (h_neg_theta_to_alpha_less_one[0: N_GRID] - h_theta_to_alpha_less_one[0: N_GRID]);
    vcMul(N_GRID, d_theta_chi_U_s, chi_U_s, d_theta_chi_U_s);
    d_theta_theta_chi_U_s[0: N_GRID] = 0.125 * alpha * alpha * sec_squared_theta * sec_squared_theta * s[0: N_GRID]
      * std::pow(h_neg_theta_to_alpha_less_one[0: N_GRID] - h_theta_to_alpha_less_one[0: N_GRID], 2)
      + M_SQRT1_2 * alpha * tan_theta * sec_squared_theta * (h_neg_theta_to_alpha_less_one[0: N_GRID] - h_theta_to_alpha_less_one[0: N_GRID])
      - 0.125 * alpha * (alpha - 1.0) * s[0: N_GRID] * sec_squared_theta * sec_squared_theta
      * (h_neg_theta_to_alpha_less_one[0: N_GRID] / h_neg_theta[0: N_GRID] + h_theta_to_alpha_less_one[0: N_GRID] / h_theta[0: N_GRID]);
    vcMul(N_GRID, d_theta_theta_chi_U_s, chi_U_s, d_theta_theta_chi_U_s);
    for (int j = 0; j != N_GRID; ++j) {
      float const s_j = s[j];
      chi_U_s[j] = chi_U_s[j] / s_j;
    }
    DftiComputeBackward(hand, chi_U_s);
    vcMul(N_GRID, chi_U_s, post_factor, chi_U_s);
    vcMul(N_GRID, d_theta_chi_U_s, post_factor, d_theta_chi_U_s);
    vcMul(N_GRID, d_theta_theta_chi_U_s, post_factor, d_theta_theta_chi_U_s);
    buffer[0: N_GRID_2] = chi_U_s[N_GRID_4: N_GRID_2].imag();
    buffer[N_GRID_2: N_GRID_2] = d_theta_chi_U_s[N_GRID_4: N_GRID_2].imag();
    buffer[2 * N_GRID_2: N_GRID_2] = d_theta_theta_chi_U_s[N_GRID_4: N_GRID_2].imag();
  }
};

class FourierTransform2D {
  double* const s;
  MKL_Complex8* const post_factor;
  double* const h_neg_theta;
  double* const h_theta;
  double* const h_neg_theta_to_alpha_less_one;
  double* const h_theta_to_alpha_less_one;
  double* const h_neg_theta_to_alpha;
  double* const h_theta_to_alpha;
  double* const h_neg_theta_to_alpha_less_two;
  double* const h_theta_to_alpha_less_two;
  double* const d_theta_1_chi_U_s_;
  double* const d_theta_2_chi_U_s_;
  MKL_Complex8* const darray;
  DFTI_DESCRIPTOR_HANDLE hand;
public:
  FourierTransform2D() :
    s((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    post_factor((MKL_Complex8*)mkl_malloc(N_GRID * N_GRID * sizeof(MKL_Complex8), 64)),
    h_neg_theta((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    h_theta((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    h_neg_theta_to_alpha_less_one((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    h_theta_to_alpha_less_one((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    h_neg_theta_to_alpha((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    h_theta_to_alpha((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    h_neg_theta_to_alpha_less_two((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    h_theta_to_alpha_less_two((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    d_theta_1_chi_U_s_((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    d_theta_2_chi_U_s_((double*)mkl_malloc(N_GRID * sizeof(double), 64)),
    darray((MKL_Complex8*)mkl_malloc(6 * N_GRID * N_GRID * sizeof(MKL_Complex8), 64))
  {
    s[0: N_GRID] = 0.5 * std::sqrt(2.0 * M_PI / N_GRID) * (2 * __sec_implicit_index(0) - N_GRID + 1);
    float const a = M_PI * (N_GRID - 1) / N_GRID;
    for (int k1 = 0; k1 != N_GRID; ++k1) {
      for (int k2 = 0; k2 != N_GRID; ++k2) {
	post_factor[N_GRID * k1 + k2] = MKL_Complex8(0.0f, a * (N_GRID - 1 - k1 - k2));
      }
    }
    vcExp(N_GRID * N_GRID, post_factor, post_factor);
    float const b = 4.0f / (2.0f * M_PI * N_GRID);
    for (int k = 0; k != N_GRID * N_GRID; ++k) {
      post_factor[k] = b * post_factor[k];
    }
    MKL_LONG const length[2] = {N_GRID, N_GRID};
    DftiCreateDescriptor(&hand, DFTI_SINGLE, DFTI_COMPLEX, 2, length);
    DftiSetValue(hand, DFTI_NUMBER_OF_TRANSFORMS, 6);
    DftiSetValue(hand, DFTI_INPUT_DISTANCE, N_GRID * N_GRID);
    DftiCommitDescriptor(hand);
  }
  ~FourierTransform2D() {
    mkl_free(s);
    mkl_free(post_factor);
    mkl_free(h_neg_theta);
    mkl_free(h_theta);
    mkl_free(h_neg_theta_to_alpha_less_one);
    mkl_free(h_theta_to_alpha_less_one);
    mkl_free(h_neg_theta_to_alpha);
    mkl_free(h_theta_to_alpha);
    mkl_free(h_neg_theta_to_alpha_less_two);
    mkl_free(h_theta_to_alpha_less_two);
    mkl_free(d_theta_1_chi_U_s_);
    mkl_free(d_theta_2_chi_U_s_);
    mkl_free(darray);
    DftiFreeDescriptor(&hand);
  }
  void run(double const alpha, double const theta_1, double const theta_2, float* const buffer) {
    MKL_Complex8* const chi_U_s = darray;
    MKL_Complex8* const d_theta_1_chi_U_s = darray + N_GRID * N_GRID;
    MKL_Complex8* const d_theta_2_chi_U_s = darray + 2 * N_GRID * N_GRID;
    MKL_Complex8* const d_theta_1_theta_1_chi_U_s = darray + 3 * N_GRID * N_GRID;
    MKL_Complex8* const d_theta_1_theta_2_chi_U_s = darray + 4 * N_GRID * N_GRID;
    MKL_Complex8* const d_theta_2_theta_2_chi_U_s = darray + 5 * N_GRID * N_GRID;
    double const tan_theta_1 = std::tan(theta_1);
    double const tan_theta_2 = std::tan(theta_2);
    double const sec_squared_theta_1 = std::pow(std::cos(theta_1), -2);
    double const sec_squared_theta_2 = std::pow(std::cos(theta_2), -2);
    for (int j1 = 0; j1 != N_GRID; ++j1) {
      double const s_j1 = s[j1];
      h_neg_theta[0: N_GRID] = 0.5 * M_SQRT1_2 * (1.0 - tan_theta_1) * s_j1 + 0.5 * M_SQRT1_2 * (1.0 - tan_theta_2) * s[0: N_GRID];
      h_theta[0: N_GRID] = 0.5 * M_SQRT1_2 * (1.0 + tan_theta_1) * s_j1 + 0.5 * M_SQRT1_2 * (1.0 + tan_theta_2) * s[0: N_GRID];
      vdAbs(N_GRID, h_neg_theta, h_neg_theta_to_alpha_less_one);
      vdAbs(N_GRID, h_theta, h_theta_to_alpha_less_one);
      vdPowx(N_GRID, h_neg_theta_to_alpha_less_one, alpha - 1.0, h_neg_theta_to_alpha_less_one);
      vdPowx(N_GRID, h_theta_to_alpha_less_one, alpha - 1.0, h_theta_to_alpha_less_one);
      if (h_neg_theta[0: N_GRID] < 0.0) {
	h_neg_theta_to_alpha_less_one[0: N_GRID] = -h_neg_theta_to_alpha_less_one[0: N_GRID];
      }
      if (h_theta[0: N_GRID] < 0.0) {
	h_theta_to_alpha_less_one[0: N_GRID] = -h_theta_to_alpha_less_one[0: N_GRID];
      }
      h_neg_theta_to_alpha[0: N_GRID] = h_neg_theta_to_alpha_less_one[0: N_GRID] * h_neg_theta[0: N_GRID];
      h_theta_to_alpha[0: N_GRID] = h_theta_to_alpha_less_one[0: N_GRID] * h_theta[0: N_GRID];
      h_neg_theta_to_alpha_less_two[0: N_GRID] = h_neg_theta_to_alpha_less_one[0: N_GRID] / h_neg_theta[0: N_GRID];
      h_theta_to_alpha_less_two[0: N_GRID] = h_theta_to_alpha_less_one[0: N_GRID] / h_theta[0: N_GRID];
      d_theta_1_chi_U_s_[0: N_GRID] = 0.5 * M_SQRT1_2 * alpha * sec_squared_theta_1 * s_j1
	* (h_neg_theta_to_alpha_less_one[0: N_GRID] - h_theta_to_alpha_less_one[0: N_GRID]);
      d_theta_2_chi_U_s_[0: N_GRID] = 0.5 * M_SQRT1_2 * alpha * sec_squared_theta_2 * s[0: N_GRID]
	* (h_neg_theta_to_alpha_less_one[0: N_GRID] - h_theta_to_alpha_less_one[0: N_GRID]);
      d_theta_1_theta_1_chi_U_s[N_GRID * j1: N_GRID] = d_theta_1_chi_U_s_[0: N_GRID] * d_theta_1_chi_U_s_[0: N_GRID]
	+ M_SQRT1_2 * alpha * tan_theta_1 * sec_squared_theta_1 * s_j1
	* (h_neg_theta_to_alpha_less_one[0: N_GRID] - h_theta_to_alpha_less_one[0: N_GRID])
	- 0.125 * alpha * (alpha - 1.0) * sec_squared_theta_1 * sec_squared_theta_1 * s_j1 * s_j1
	* (h_neg_theta_to_alpha_less_two[0: N_GRID] + h_theta_to_alpha_less_two[0: N_GRID]);
      d_theta_1_theta_2_chi_U_s[N_GRID * j1: N_GRID] = d_theta_1_chi_U_s_[0: N_GRID] * d_theta_2_chi_U_s_[0: N_GRID]
	- 0.125 * alpha * (alpha - 1.0) * sec_squared_theta_1 * sec_squared_theta_2 * s_j1 * s[0: N_GRID]
	* (h_neg_theta_to_alpha_less_two[0: N_GRID] + h_theta_to_alpha_less_two[0: N_GRID]);
      d_theta_2_theta_2_chi_U_s[N_GRID * j1: N_GRID] = d_theta_2_chi_U_s_[0: N_GRID] * d_theta_2_chi_U_s_[0: N_GRID]
	+ M_SQRT1_2 * alpha * tan_theta_2 * sec_squared_theta_2 * s[0: N_GRID]
	* (h_neg_theta_to_alpha_less_one[0: N_GRID] - h_theta_to_alpha_less_one[0: N_GRID])
	- 0.125 * alpha * (alpha - 1.0) * sec_squared_theta_2 * sec_squared_theta_2 * s[0: N_GRID] * s[0: N_GRID]
	* (h_neg_theta_to_alpha_less_two[0: N_GRID] + h_theta_to_alpha_less_two[0: N_GRID]);
      d_theta_1_chi_U_s[N_GRID * j1: N_GRID] = d_theta_1_chi_U_s_[0: N_GRID];
      d_theta_2_chi_U_s[N_GRID * j1: N_GRID] = d_theta_2_chi_U_s_[0: N_GRID];
      chi_U_s[N_GRID * j1: N_GRID] = MKL_Complex8(-(h_neg_theta_to_alpha[0: N_GRID] + h_theta_to_alpha[0: N_GRID]),
						    -M_PI * (N_GRID - 1) * (j1 + __sec_implicit_index(0)) / N_GRID);
    }
    vcExp(N_GRID * N_GRID, chi_U_s, chi_U_s);
    for (int j1 = 0; j1 != N_GRID; ++j1) {
      for (int j2 = 0; j2 != N_GRID; ++j2) {
	float const s_j1_s_j2 = s[j1] * s[j2];
	chi_U_s[N_GRID * j1 + j2] = chi_U_s[N_GRID * j1 + j2] / s_j1_s_j2;
      }
    }
    vcMul(N_GRID * N_GRID, d_theta_1_chi_U_s, chi_U_s, d_theta_1_chi_U_s);
    vcMul(N_GRID * N_GRID, d_theta_2_chi_U_s, chi_U_s, d_theta_2_chi_U_s);
    vcMul(N_GRID * N_GRID, d_theta_1_theta_1_chi_U_s, chi_U_s, d_theta_1_theta_1_chi_U_s);
    vcMul(N_GRID * N_GRID, d_theta_1_theta_2_chi_U_s, chi_U_s, d_theta_1_theta_2_chi_U_s);
    vcMul(N_GRID * N_GRID, d_theta_2_theta_2_chi_U_s, chi_U_s, d_theta_2_theta_2_chi_U_s);
    DftiComputeBackward(hand, chi_U_s);
    vcMul(N_GRID * N_GRID, chi_U_s, post_factor, chi_U_s);
    vcMul(N_GRID * N_GRID, d_theta_1_chi_U_s, post_factor, d_theta_1_chi_U_s);
    vcMul(N_GRID * N_GRID, d_theta_2_chi_U_s, post_factor, d_theta_2_chi_U_s);
    vcMul(N_GRID * N_GRID, d_theta_1_theta_1_chi_U_s, post_factor, d_theta_1_theta_1_chi_U_s);
    vcMul(N_GRID * N_GRID, d_theta_1_theta_2_chi_U_s, post_factor, d_theta_1_theta_2_chi_U_s);
    vcMul(N_GRID * N_GRID, d_theta_2_theta_2_chi_U_s, post_factor, d_theta_2_theta_2_chi_U_s);
    for (int j1 = 0; j1 != N_GRID_2; ++j1) {
      buffer[N_GRID_2 * (0 * N_GRID_2 + j1): N_GRID_2] = -chi_U_s[N_GRID * (N_GRID_4 + j1) + N_GRID_4: N_GRID_2].real();
      buffer[N_GRID_2 * (1 * N_GRID_2 + j1): N_GRID_2] = -d_theta_1_chi_U_s[N_GRID * (N_GRID_4 + j1) + N_GRID_4: N_GRID_2].real();
      buffer[N_GRID_2 * (2 * N_GRID_2 + j1): N_GRID_2] = -d_theta_2_chi_U_s[N_GRID * (N_GRID_4 + j1) + N_GRID_4: N_GRID_2].real();
      buffer[N_GRID_2 * (3 * N_GRID_2 + j1): N_GRID_2] = -d_theta_1_theta_1_chi_U_s[N_GRID * (N_GRID_4 + j1) + N_GRID_4: N_GRID_2].real();
      buffer[N_GRID_2 * (4 * N_GRID_2 + j1): N_GRID_2] = -d_theta_1_theta_2_chi_U_s[N_GRID * (N_GRID_4 + j1) + N_GRID_4: N_GRID_2].real();
      buffer[N_GRID_2 * (5 * N_GRID_2 + j1): N_GRID_2] = -d_theta_2_theta_2_chi_U_s[N_GRID * (N_GRID_4 + j1) + N_GRID_4: N_GRID_2].real();
    }
  }
};

struct Num1D {
  float uj;
  float uj_tj;
  float uj_zj;
  float uj_tj_tj;
  float uj_tj_zj;
};

struct Num2D {
  float ujk;
  float ujk_tj;
  float ujk_tk;
  float ujk_zj;
  float ujk_zk;
  float ujk_tj_tj;
  float ujk_tk_tk;
  float ujk_tj_tk;
  float ujk_tj_zj;
  float ujk_tk_zk;
  float ujk_tj_zk;
  float ujk_tk_zj;
};

class SystemicCost {
  float* const darray_1d;
  float* const darray_2d;
public:
  SystemicCost() :
    darray_1d((float*)mkl_malloc(N_PLAYERS * 3 * N_GRID_2 * sizeof(float), 64)),
    darray_2d((float*)mkl_malloc(N_PLAYERS * (N_PLAYERS - 1) / 2 * 6 * N_GRID_2 * N_GRID_2 * sizeof(float), 64))
  {}
  ~SystemicCost() {
    mkl_free(darray_1d);
    mkl_free(darray_2d);
  }
  void populate(double const alpha, double const* const theta) {
    #pragma omp parallel for
    for (int i = 0; i < N_PLAYERS; ++i) {
      FourierTransform1D fft_obj_1;
      float* const buffer = darray_1d + (i * 3) * N_GRID_2;
      fft_obj_1.run(alpha, theta[i], buffer);
    }
    int const number_of_partitions = 12; // N_PLAYERS / 2;
    int const elements_per_partition = 10; // N_PLAYERS * (N_PLAYERS - 1) / (2 * number_of_partitions);
    #pragma omp parallel for
    for (int m1 = 0; m1 < number_of_partitions; ++m1) {
      FourierTransform2D fft_obj_2;
      for (int m2 = 0; m2 != elements_per_partition; ++m2) {
	int const m = elements_per_partition * m1 + m2;
	float* const buffer = darray_2d + (m * 6) * N_GRID_2 * N_GRID_2;
	int const i2 = (1 + (int)std::sqrt(1 + 8 * m)) / 2;
	int const i1 = m - i2 * (i2 - 1) / 2;
	fft_obj_2.run(alpha, theta[i1], theta[i2], buffer);
      }
    }
  }
  float* C_f(float* const r, double const a, float const* const z) const {
    float ii[N_PLAYERS];
    int i[N_PLAYERS];
    float l[N_PLAYERS];
    ii[:] = 0.5f * (N_GRID_2 - 1) + SQRTN_2PI * z[0: N_PLAYERS];
    i[:] = ii[:];
    l[:] = ii[:] - i[:];
    r[0: N_PLAYERS] = 0.0f;
    for (int j = 0; j != N_PLAYERS; ++j) {
      float const* const uj = darray_1d + (j * 3) * N_GRID_2;
      float const mu = 0.5f * (1.0f + (1.0f - l[j]) * uj[i[j]] + l[j] * uj[i[j] + 1]);
      r[j] += mu;
      r[0: N_PLAYERS] += a / N_PLAYERS * mu;
    }
    for (int k = 1; k != N_PLAYERS; ++k) {
      for (int j = 0; j != k; ++j) {
	int const m = j + k * (k - 1) / 2;
	float const* const ujk = darray_2d + (m * 6) * N_GRID_2 * N_GRID_2;
	float const nu = 0.5f * (1.0f - (1.0f - l[j]) * (1.0f - l[k]) * ujk[N_GRID_2 * i[j] + i[k]]
				 - (1.0f - l[j]) * l[k] * ujk[N_GRID_2 * i[j] + i[k] + 1]
				 - l[j] * (1.0f - l[k]) * ujk[N_GRID_2 * (i[j] + 1) + i[k]]
				 - l[j] * l[k] * ujk[N_GRID_2 * (i[j] + 1) + i[k] + 1]);
	r[j] -= a / N_PLAYERS * nu;
	r[k] -= a / N_PLAYERS * nu;
      }
    }
    return r;
  }
  bool solve(double const a, double const beta, double const kappa, float* const z) const {
    float const max_z = 0.5f * std::sqrt(2.0f * M_PI / N_GRID) * (N_GRID_2 - 1);
    z[0: N_PLAYERS] = 0.0f;
    int n = 0;
    bool is_contained = true;
    float r[N_PLAYERS];
    while (is_contained && n < 100) {
      C_f(r, a, z);
      z[0: N_PLAYERS] = kappa + beta * r[0: N_PLAYERS];
      is_contained = __sec_reduce_and(std::abs(z[0: N_PLAYERS]) <= max_z);
      ++n;
    }
    return is_contained;
  }
  Num1D getnum(float const* const z, int const j) const {
    float const iij = 0.5f * (N_GRID_2 - 1) + SQRTN_2PI * z[j];
    int const ij = iij;
    float const lj = iij - ij;
    Num1D r; {
      float const* const uj = darray_1d + (j * 3) * N_GRID_2;
      r.uj = (1.0f - lj) * uj[ij] + lj * uj[ij + 1];
      r.uj_zj = SQRTN_2PI * (uj[ij + 1] - uj[ij]);
    } {
      float const* const uj_tj = darray_1d + (j * 3 + 1) * N_GRID_2;
      r.uj_tj = (1.0f - lj) * uj_tj[ij] + lj * uj_tj[ij + 1];
      r.uj_tj_zj = SQRTN_2PI * (uj_tj[ij + 1] - uj_tj[ij]);
    } {
      float const* const uj_tj_tj = darray_1d + (j * 3 + 2) * N_GRID_2;
      r.uj_tj_tj = (1.0f - lj) * uj_tj_tj[ij] + lj * uj_tj_tj[ij + 1];
    }
    return r;
  }
  Num2D getnum(float const* const z, int const j, int const k) const {
    float const iij = 0.5f * (N_GRID_2 - 1) + SQRTN_2PI * z[j];
    int const ij = iij;
    float const lj = iij - ij;
    float const iik = 0.5f * (N_GRID_2 - 1) + SQRTN_2PI * z[k];
    int const ik = iik;
    float const lk = iik - ik;
    int const m = j + k * (k - 1) / 2;
    Num2D r; {
      float const* const ujk = darray_2d + (m * 6) * N_GRID_2 * N_GRID_2;
      r.ujk = (1.0f - lj) * (1.0f - lk) * ujk[N_GRID_2 * ij + ik] + (1.0f - lj) * lk * ujk[N_GRID_2 * ij + ik + 1]
	+ lj * (1.0f - lk) * ujk[N_GRID_2 * (ij + 1) + ik] + lj * lk * ujk[N_GRID_2 * (ij + 1) + ik + 1];
      r.ujk_zj = SQRTN_2PI * ((1.0f - lk) * (ujk[N_GRID_2 * (ij + 1) + ik] - ujk[N_GRID_2 * ij + ik])
			      + lk * (ujk[N_GRID_2 * (ij + 1) + ik + 1] - ujk[N_GRID_2 * ij + ik + 1]));
      r.ujk_zk = SQRTN_2PI * ((1.0f - lj) * (ujk[N_GRID_2 * ij + ik + 1] - ujk[N_GRID_2 * ij + ik])
			      + lj * (ujk[N_GRID_2 * (ij + 1) + ik + 1] - ujk[N_GRID_2 * (ij + 1) + ik]));
    } {
      float const* const ujk_tj = darray_2d + (m * 6 + 1) * N_GRID_2 * N_GRID_2;
      r.ujk_tj = (1.0f - lj) * (1.0f - lk) * ujk_tj[N_GRID_2 * ij + ik] + (1.0f - lj) * lk * ujk_tj[N_GRID_2 * ij + ik + 1]
	+ lj * (1.0f - lk) * ujk_tj[N_GRID_2 * (ij + 1) + ik] + lj * lk * ujk_tj[N_GRID_2 * (ij + 1) + ik + 1];
      r.ujk_tj_zj = SQRTN_2PI * ((1.0f - lk) * (ujk_tj[N_GRID_2 * (ij + 1) + ik] - ujk_tj[N_GRID_2 * ij + ik])
				 + lk * (ujk_tj[N_GRID_2 * (ij + 1) + ik + 1] - ujk_tj[N_GRID_2 * ij + ik + 1]));
      r.ujk_tj_zk = SQRTN_2PI * ((1.0f - lj) * (ujk_tj[N_GRID_2 * ij + ik + 1] - ujk_tj[N_GRID_2 * ij + ik])
				 + lj * (ujk_tj[N_GRID_2 * (ij + 1) + ik + 1] - ujk_tj[N_GRID_2 * (ij + 1) + ik]));
    } {
      float const* const ujk_tk = darray_2d + (m * 6 + 2) * N_GRID_2 * N_GRID_2;
      r.ujk_tk = (1.0f - lj) * (1.0f - lk) * ujk_tk[N_GRID_2 * ij + ik] + (1.0f - lj) * lk * ujk_tk[N_GRID_2 * ij + ik + 1]
	+ lj * (1.0f - lk) * ujk_tk[N_GRID_2 * (ij + 1) + ik] + lj * lk * ujk_tk[N_GRID_2 * (ij + 1) + ik + 1];
      r.ujk_tk_zj = SQRTN_2PI * ((1.0f - lk) * (ujk_tk[N_GRID_2 * (ij + 1) + ik] - ujk_tk[N_GRID_2 * ij + ik])
				 + lk * (ujk_tk[N_GRID_2 * (ij + 1) + ik + 1] - ujk_tk[N_GRID_2 * ij + ik + 1]));
      r.ujk_tk_zk = SQRTN_2PI * ((1.0f - lj) * (ujk_tk[N_GRID_2 * ij + ik + 1] - ujk_tk[N_GRID_2 * ij + ik])
				 + lj * (ujk_tk[N_GRID_2 * (ij + 1) + ik + 1] - ujk_tk[N_GRID_2 * (ij + 1) + ik]));
    } {
      float const* const ujk_tj_tj = darray_2d + (m * 6 + 3) * N_GRID_2 * N_GRID_2;
      r.ujk_tj_tj = (1.0f - lj) * (1.0f - lk) * ujk_tj_tj[N_GRID_2 * ij + ik] + (1.0f - lj) * lk * ujk_tj_tj[N_GRID_2 * ij + ik + 1]
	+ lj * (1.0f - lk) * ujk_tj_tj[N_GRID_2 * (ij + 1) + ik] + lj * lk * ujk_tj_tj[N_GRID_2 * (ij + 1) + ik + 1];
    } {
      float const* const ujk_tj_tk = darray_2d + (m * 6 + 4) * N_GRID_2 * N_GRID_2;
      r.ujk_tj_tk = (1.0f - lj) * (1.0f - lk) * ujk_tj_tk[N_GRID_2 * ij + ik] + (1.0f - lj) * lk * ujk_tj_tk[N_GRID_2 * ij + ik + 1]
	+ lj * (1.0f - lk) * ujk_tj_tk[N_GRID_2 * (ij + 1) + ik] + lj * lk * ujk_tj_tk[N_GRID_2 * (ij + 1) + ik + 1];
    } {
      float const* const ujk_tk_tk = darray_2d + (m * 6 + 5) * N_GRID_2 * N_GRID_2;
      r.ujk_tk_tk = (1.0f - lj) * (1.0f - lk) * ujk_tk_tk[N_GRID_2 * ij + ik] + (1.0f - lj) * lk * ujk_tk_tk[N_GRID_2 * ij + ik + 1]
	+ lj * (1.0f - lk) * ujk_tk_tk[N_GRID_2 * (ij + 1) + ik] + lj * lk * ujk_tk_tk[N_GRID_2 * (ij + 1) + ik + 1];
    }
    return r;
  }
  double C(double const b2, int const* const k) const {
    double C_z_k = 0.0;
    for (int i = 0; i != N_PLAYERS; ++i) {
      float const* const u_dag = darray_1d + (i * 3) * N_GRID_2;
      C_z_k += 0.5 / N_PLAYERS * (1.0 + u_dag[k[i]]);
    }
    for (int i2 = 1; i2 != N_PLAYERS; ++i2) {
      for (int i1 = 0; i1 != i2; ++i1) {
	int const m = i1 + i2 * (i2 - 1) / 2;
	float const* const u_dag = darray_2d + (m * 6) * N_GRID_2 * N_GRID_2;
	C_z_k -= 0.5 * b2 / (N_PLAYERS * N_PLAYERS) * (1.0 - u_dag[N_GRID_2 * k[i1] + k[i2]]);
      }
    }
    return C_z_k;
  }
  double* C_prime(double const b2, int const* const k, double* const C_prime_z_k) const {
    for (int i = 0; i != N_PLAYERS; ++i) {
      float const* const u_dag = darray_1d + (i * 3) * N_GRID_2;
      float const* const d_theta_u_dag = darray_1d + (i * 3 + 1) * N_GRID_2;
      C_prime_z_k[i] = 0.5 / N_PLAYERS * d_theta_u_dag[k[i]];
      C_prime_z_k[N_PLAYERS + i] = 0.25 / N_PLAYERS * (u_dag[k[i] + 1] - u_dag[k[i] - 1]);
    }
    for (int i2 = 1; i2 != N_PLAYERS; ++i2) {
      for (int i1 = 0; i1 != i2; ++i1) {
        int const m = i1 + i2 * (i2 - 1) / 2;
	float const* const u_dag = darray_2d + (m * 6) * N_GRID_2 * N_GRID_2;
        float const* const d_theta_1_u_dag = darray_2d + (m * 6 + 1) * N_GRID_2 * N_GRID_2;
        float const* const d_theta_2_u_dag = darray_2d + (m * 6 + 2) * N_GRID_2 * N_GRID_2;
	C_prime_z_k[i1] += 0.5 * b2 / (N_PLAYERS * N_PLAYERS) * d_theta_1_u_dag[N_GRID_2 * k[i1] + k[i2]];
	C_prime_z_k[i2] += 0.5 * b2 / (N_PLAYERS * N_PLAYERS) * d_theta_2_u_dag[N_GRID_2 * k[i1] + k[i2]];
	C_prime_z_k[N_PLAYERS + i1] += 0.25 * b2 / (N_PLAYERS * N_PLAYERS)
	  * (u_dag[N_GRID_2 * (k[i1] + 1) + k[i2]] - u_dag[N_GRID_2 * (k[i1] - 1) + k[i2]]);
	C_prime_z_k[N_PLAYERS + i2] += 0.25 * b2 / (N_PLAYERS * N_PLAYERS)
	  * (u_dag[N_GRID_2 * k[i1] + k[i2] + 1] - u_dag[N_GRID_2 * k[i1] + k[i2] - 1]);
      }
    }
    return C_prime_z_k;
  }
  double* C_double_prime_packed(double const b2, int const* const k, double* const C_double_prime_z_k) const {
    for (int i = 0; i != N_PLAYERS; ++i) {
      float const* const u_dag = darray_1d + (i * 3) * N_GRID_2;
      float const* const d_theta_u_dag = darray_1d + (i * 3 + 1) * N_GRID_2;
      float const* const d_theta_theta_u_dag = darray_1d + (i * 3 + 2) * N_GRID_2;
      C_double_prime_z_k[i * (i + 3) / 2] = 0.5 / N_PLAYERS * d_theta_theta_u_dag[k[i]];
      C_double_prime_z_k[i + (N_PLAYERS + i) * (N_PLAYERS + i + 1) / 2]
	= 0.25 / N_PLAYERS * (d_theta_u_dag[k[i] + 1] - d_theta_u_dag[k[i] - 1]);
      C_double_prime_z_k[(N_PLAYERS + i) * (N_PLAYERS + i + 3) / 2]
	= 0.5 / N_PLAYERS * (u_dag[k[i] + 1] - 2 * u_dag[k[i]] + u_dag[k[i] - 1]);
    }
    for (int i2 = 1; i2 != N_PLAYERS; ++i2) {
      for (int i1 = 0; i1 != i2; ++i1) {
        int const m = i1 + i2 * (i2 - 1) / 2;
	float const* const u_dag = darray_2d + (m * 6) * N_GRID_2 * N_GRID_2;
	float const* const d_theta_1_u_dag = darray_2d + (m * 6 + 1) * N_GRID_2 * N_GRID_2;
	float const* const d_theta_2_u_dag = darray_2d + (m * 6 + 2) * N_GRID_2 * N_GRID_2;
        float const* const d_theta_1_theta_1_u_dag = darray_2d + (m * 6 + 3) * N_GRID_2 * N_GRID_2;
        float const* const d_theta_1_theta_2_u_dag = darray_2d + (m * 6 + 4) * N_GRID_2 * N_GRID_2;
        float const* const d_theta_2_theta_2_u_dag = darray_2d + (m * 6 + 5) * N_GRID_2 * N_GRID_2;
	C_double_prime_z_k[i1 + i2 * (i2 + 1) / 2] = 0.5 * b2 / (N_PLAYERS * N_PLAYERS) * d_theta_1_theta_2_u_dag[N_GRID_2 * k[i1] + k[i2]];
	C_double_prime_z_k[i1 * (i1 + 3) / 2] += 0.5 * b2 / (N_PLAYERS * N_PLAYERS) * d_theta_1_theta_1_u_dag[N_GRID_2 * k[i1] + k[i2]];
	C_double_prime_z_k[i2 * (i2 + 3) / 2] += 0.5 * b2 / (N_PLAYERS * N_PLAYERS) * d_theta_2_theta_2_u_dag[N_GRID_2 * k[i1] + k[i2]];
	C_double_prime_z_k[i1 + (N_PLAYERS + i2) * (N_PLAYERS + i2 + 1) / 2] = 0.25 * b2 / (N_PLAYERS * N_PLAYERS)
	  * (d_theta_1_u_dag[N_GRID_2 * k[i1] + k[i2] + 1] - d_theta_1_u_dag[N_GRID_2 * k[i1] + k[i2] - 1]);
	C_double_prime_z_k[i2 + (N_PLAYERS + i1) * (N_PLAYERS + i1 + 1) / 2] = 0.25 * b2 / (N_PLAYERS * N_PLAYERS)
	  * (d_theta_2_u_dag[N_GRID_2 * (k[i1] + 1) + k[i2]] - d_theta_2_u_dag[N_GRID_2 * (k[i1] - 1) + k[i2]]);
	C_double_prime_z_k[i1 + (N_PLAYERS + i1) * (N_PLAYERS + i1 + 1) / 2] += 0.25 * b2 / (N_PLAYERS * N_PLAYERS)
	  * (d_theta_1_u_dag[N_GRID_2 * (k[i1] + 1) + k[i2]] - d_theta_1_u_dag[N_GRID_2 * (k[i1] - 1) + k[i2]]);
	C_double_prime_z_k[i2 + (N_PLAYERS + i2) * (N_PLAYERS + i2 + 1) / 2] += 0.25 * b2 / (N_PLAYERS * N_PLAYERS)
	  * (d_theta_2_u_dag[N_GRID_2 * k[i1] + k[i2] + 1] - d_theta_2_u_dag[N_GRID_2 * k[i1] + k[i2] - 1]);
	C_double_prime_z_k[N_PLAYERS + i1 + (N_PLAYERS + i2) * (N_PLAYERS + i2 + 1) / 2] = 0.125 * b2 / (N_PLAYERS * N_PLAYERS)
	  * (u_dag[N_GRID_2 * (k[i1] + 1) + k[i2] + 1] - u_dag[N_GRID_2 * (k[i1] + 1) + k[i2] - 1]
	     - u_dag[N_GRID_2 * (k[i1] - 1) + k[i2] + 1] + u_dag[N_GRID_2 * (k[i1] - 1) + k[i2] - 1]);
	C_double_prime_z_k[(N_PLAYERS + i1) * (N_PLAYERS + i1 + 3) / 2] += 0.5 * b2 / (N_PLAYERS * N_PLAYERS)
	  * (u_dag[N_GRID_2 * (k[i1] + 1) + k[i2]] - 2 * u_dag[N_GRID_2 * k[i1] + k[i2]] + u_dag[N_GRID_2 * (k[i1] - 1) + k[i2]]);
	C_double_prime_z_k[(N_PLAYERS + i2) * (N_PLAYERS + i2 + 3) / 2] += 0.5 * b2 / (N_PLAYERS * N_PLAYERS)
	  * (u_dag[N_GRID_2 * k[i1] + k[i2] + 1] - 2 * u_dag[N_GRID_2 * k[i1] + k[i2]] + u_dag[N_GRID_2 * k[i1] + k[i2] - 1]);	  
      }
    }
    return C_double_prime_z_k;
  }
  double* C_double_prime(double b2, int const* const k, double* const C_double_prime_z_k)  const {
    double C_double_prime_z_k_packed[2 * N_PLAYERS * (2 * N_PLAYERS + 1) / 2];
    C_double_prime_packed(b2, k, C_double_prime_z_k_packed);
    for (int i = 0; i != 2 * N_PLAYERS; ++i) {
      C_double_prime_z_k[2 * N_PLAYERS * i + i] = C_double_prime_z_k_packed[i * (i + 3) / 2];
    }
    for (int i2 = 1; i2 != 2 * N_PLAYERS; ++i2) {
      for (int i1 = 0; i1 != i2; ++i1) {
	C_double_prime_z_k[2 * N_PLAYERS * i1 + i2] = C_double_prime_z_k_packed[i1 + i2 * (i2 + 1) / 2];
	C_double_prime_z_k[2 * N_PLAYERS * i2 + i1] = C_double_prime_z_k_packed[i1 + i2 * (i2 + 1) / 2];
      }
    }
    return C_double_prime_z_k;
  }
};

int main(int const argc, char** const argv) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <random_seed>" << std::endl;
  } else {
    //  vmlSetMode(VML_LA);
    float const start_time = second();
    int const random_seed = std::atoi(argv[1]);
    VSLStreamStatePtr random_stream;
    vslNewStream(&random_stream, VSL_BRNG_MT19937, random_seed);
    SystemicCost calc;
    for (int i_theta = 0; i_theta != 24; ++i_theta) {
      double theta[N_PLAYERS + 1];
      vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD_ACCURATE, random_stream, N_PLAYERS + 1, theta, 0.0, 1.0);
      LAPACKE_dlasrt('I', N_PLAYERS, theta);
      theta[0: N_PLAYERS] -= 0.5;
      theta[0: N_PLAYERS] *= M_PI * theta[N_PLAYERS];
      std::cout << "DO $$DECLARE _theta_id integer; DECLARE _z_id integer; BEGIN" << std::endl;
      std::cout << "BEGIN;" << std::endl;  // untested
      std::cout << "INSERT INTO theta (bound, t1";
      for (int j = 1; j != N_PLAYERS; ++j) {
	std::cout << ", t" << j + 1;
      }
      std::cout << ") VALUES (" << 0.5 * M_PI * theta[N_PLAYERS] << ", " << theta[0];
      for (int j = 1; j != N_PLAYERS; ++j) {
	std::cout << ", " << theta[j];
      }
      std::cout << ") RETURNING id INTO _theta_id;" << std::endl;
      int param_id = 0;
      for (int i_alpha = 0; i_alpha <= 1; ++i_alpha) {
	double const alpha = 2.0 - 0.4 * i_alpha;
	calc.populate(alpha, theta);
	for (int i_a = 0; i_a <= 4; ++i_a) {
	  double const a = 0.25 * i_a;
	  for (int i_beta = 0; i_beta <= 7; ++i_beta) {
	    double const beta = 1 << i_beta;
	    for (int i_kappa = 0; i_kappa <= 5; ++i_kappa) {
	      double const kappa = 1.0 * i_kappa;
	      ++param_id;
	      float z[N_PLAYERS];
	      bool const solved = calc.solve(a, beta, kappa, z);
	      if (solved) {
		std::cout << "INSERT INTO z (param_id, theta_id, z1";
		for (int j = 1; j != N_PLAYERS; ++j) {
		  std::cout << ", z" << j + 1;
		}
		std::cout << ") VALUES (" << param_id << ", _theta_id, " << z[0];
		for (int j = 1; j != N_PLAYERS; ++j) {
		  std::cout << ", " << z[j];
		}
		std::cout << ") RETURNING id INTO _z_id;" << std::endl;
		std::cout << "INSERT INTO num1d (z_id, j, uj, uj_tj, uj_zj, uj_tj_tj, uj_tj_zj) VALUES ";
		for (int j = 0; j != N_PLAYERS; ++j) {
		  Num1D const s = calc.getnum(z, j);
		  std::cout << "(_z_id, " << j + 1 << ", " << s.uj << ", " << s.uj_tj << ", " << s.uj_zj << ", " << s.uj_tj_tj << ", " << s.uj_tj_zj << ")";
		  if (j < N_PLAYERS - 1) {
		    std::cout << ", ";
		  } else {
		    std::cout << ";" << std::endl;
		  }
		}
		std::cout << "INSERT INTO num2d (z_id, j, k, ujk, ujk_tj, ujk_tk, ujk_zj, ujk_zk, ujk_tj_tj, ujk_tk_tk, ujk_tj_tk, ujk_tj_zj, ujk_tk_zk, ujk_tj_zk, ujk_tk_zj) VALUES ";
		for (int k = 1; k != N_PLAYERS; ++k) {
		  for (int j = 0; j != k; ++j) {
		    Num2D const s = calc.getnum(z, j, k);
		    std::cout << "(_z_id, " << j + 1 << ", " << k + 1 << ", " << s.ujk << ", " << s.ujk_tj << ", " << s.ujk_tk << ", " << s.ujk_zj << ", " << s.ujk_zk << ", " << s.ujk_tj_tj << ", " << s.ujk_tk_tk << ", " << s.ujk_tj_tk << ", " << s.ujk_tj_zj << ", " << s.ujk_tk_zk << ", " << s.ujk_tj_zk  << ", " << s.ujk_tk_zj << ")";
		    if (k < N_PLAYERS - 1 || j < k - 1) {
		      std::cout << ", ";
		    } else {
		      std::cout << ";" << std::endl;
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
      std::cout << "COMMIT;" << std::endl;  // untested
      std::cout << "END;$$ LANGUAGE plpgsql;" << std::endl;
    }
    vslDeleteStream(&random_stream);
    float const end_time = second();
    std::cout << "-- " << random_seed << " " << end_time - start_time << std::endl;
  }
  return 0;
}
