#!/bin/bash

# Wait until Airflow is ready
airflow db check

# Create connections
airflow connections add 'predictions_db_connection' \
    --conn-type 'postgres' \
    --conn-host 'postgres' \
    --conn-schema 'prediction_db' \
    --conn-login 'prediction_user' \
    --conn-password ${PG_PREDICTION_PWD} \
    --conn-port '5432'

airflow connections add 'reservations_db_connection' \
    --conn-type 'postgres' \
    --conn-host 'postgres' \
    --conn-schema 'reservations_db' \
    --conn-login 'prediction_user' \
    --conn-password ${PG_PREDICTION_PWD} \
    --conn-port '5432'

echo "Config complete!"