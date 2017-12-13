CREATE DATABASE music;
\c music;

CREATE TABLE param (
  id SERIAL PRIMARY KEY,
  alpha real, a real, beta real, kappa real
);
CREATE TABLE theta (
  id SERIAL PRIMARY KEY,
  bound real,
  t1  real, t2  real, t3  real, t4  real,
  t5  real, t6  real, t7  real, t8  real,
  t9  real, t10 real, t11 real, t12 real,
  t13 real, t14 real, t15 real, t16 real
);
CREATE TABLE z (
  id SERIAL PRIMARY KEY,
  param_id integer REFERENCES param,
  theta_id integer REFERENCES theta,
  z1  real, z2  real, z3  real, z4  real,
  z5  real, z6  real, z7  real, z8  real,
  z9  real, z10 real, z11 real, z12 real,
  z13 real, z14 real, z15 real, z16 real
);
CREATE TABLE num1d (
  z_id integer REFERENCES z,
  j integer,
  uj real,
  uj_tj real,
  uj_zj real,
  uj_tj_tj real,
  uj_tj_zj real
);
CREATE TABLE num2d (
  z_id integer REFERENCES z,
  j integer,
  k integer,
  ujk real,
  ujk_tj real,
  ujk_tk real,
  ujk_zj real,
  ujk_zk real,
  ujk_tj_tj real,
  ujk_tk_tk real,
  ujk_tj_tk real,
  ujk_tj_zj real,
  ujk_tk_zk real,
  ujk_tj_zk real,
  ujk_tk_zj real
);

DO $$BEGIN
  FOR i_alpha IN 0..1 LOOP
    FOR i_a IN 0..4 LOOP
      FOR i_beta IN 0..7 LOOP
      	FOR i_kappa IN 0..5 LOOP
	  INSERT INTO param (alpha, a, beta, kappa) VALUES
	  (2.0 - 0.4 * i_alpha, 0.25 * i_a, 1 << i_beta, i_kappa);
	END LOOP;
      END LOOP;
    END LOOP;
  END LOOP;
END$$ LANGUAGE plpgsql;

export PATH=/Applications/Postgres.app/Contents/Versions/9.4/bin:$PATH
ssh mpx164@login.hpc.qmul.ac.uk 'cat love/jobbie.o3554728.3*' | psql -d love
ssh mpx164@login.hpc.qmul.ac.uk 'cat ~/love/jobbie.o3554728.3* | gzip' | gunzip | psql love

CREATE INDEX num1d_z_id_index ON num1d (z_id);
CREATE INDEX num2d_z_id_index ON num2d (z_id);
CREATE INDEX z_param_id_index ON z (param_id);
