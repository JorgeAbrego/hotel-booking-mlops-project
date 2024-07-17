#!/bin/bash
set -e

# Check if environment variables are defined
if [ -z "$PG_MLFLOW_PWD" ] || [ -z "$PG_AIRFLOW_PWD" ]; then
  echo "ERROR: One or more environment variables for passwords are not defined."
  exit 1
fi

# Create users and databases
PGPASSWORD=$POSTGRES_PASSWORD psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER mlflow_user WITH PASSWORD '$PG_MLFLOW_PWD' CREATEDB;
    CREATE DATABASE mlflow_db
        WITH 
        OWNER = mlflow_user
        ENCODING = 'UTF8'
        LC_COLLATE = 'en_US.utf8'
        LC_CTYPE = 'en_US.utf8'
        TABLESPACE = pg_default;

    CREATE USER airflow_user WITH PASSWORD '$PG_AIRFLOW_PWD' CREATEDB;
    CREATE DATABASE airflow_db
        WITH 
        OWNER = airflow_user
        ENCODING = 'UTF8'
        LC_COLLATE = 'en_US.utf8'
        LC_CTYPE = 'en_US.utf8'
        TABLESPACE = pg_default;

    CREATE USER prediction_user WITH PASSWORD '$PG_PREDICTION_PWD' CREATEDB;
    CREATE DATABASE prediction_db
        WITH 
        OWNER = prediction_user
        ENCODING = 'UTF8'
        LC_COLLATE = 'en_US.utf8'
        LC_CTYPE = 'en_US.utf8'
        TABLESPACE = pg_default;

EOSQL