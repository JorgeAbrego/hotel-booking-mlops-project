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

    CREATE DATABASE reservations_db
        WITH 
        OWNER = prediction_user
        ENCODING = 'UTF8'
        LC_COLLATE = 'en_US.utf8'
        LC_CTYPE = 'en_US.utf8'
        TABLESPACE = pg_default;
EOSQL

# Create table in prediction_db
PGPASSWORD=$PG_PREDICTION_PWD psql -v ON_ERROR_STOP=1 --username "prediction_user" --dbname "prediction_db" <<-EOSQL
    CREATE TABLE prediction_logs (
        id SERIAL PRIMARY KEY,
        hotel VARCHAR,
        meal VARCHAR,
        market_segment VARCHAR,
        distribution_channel VARCHAR,
        reserved_room_type VARCHAR,
        deposit_type VARCHAR,
        customer_type VARCHAR,
        lead_time INTEGER,
        days_in_waiting_list INTEGER,
        adr FLOAT,
        total_stay INTEGER,
        total_people INTEGER,
        is_repeated_guest INTEGER,
        previous_cancellations INTEGER,
        previous_bookings_not_canceled INTEGER,
        booking_changes INTEGER,
        agent INTEGER,
        company INTEGER,
        required_car_parking_spaces INTEGER,
        total_of_special_requests INTEGER,
        is_canceled INTEGER,
        prediction INTEGER,
        probability FLOAT,
        model_name VARCHAR,
        model_version VARCHAR,
        reservation_date TIMESTAMP,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE data_report (
        drift_share FLOAT,
        number_of_columns INTEGER,
        number_of_drifted_columns INTEGER,
        share_of_drifted_columns FLOAT,
        dataset_drift BOOLEAN,
        date_dataset DATE,
        date_report DATE
    );

    CREATE TABLE data_columns_report (
        column_name VARCHAR,
        column_type VARCHAR,
        drift_score FLOAT,
        drift_detected BOOLEAN,
        date_dataset DATE,
        date_report DATE
    );

EOSQL