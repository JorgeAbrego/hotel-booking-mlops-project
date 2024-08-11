from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Variable
from datetime import datetime, timedelta
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import os

# DAG configuration
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
}

dag = DAG(
    'reservations_prediction_dag',
    default_args=default_args,
    description='DAG for predicting reservation cancellations',
    schedule_interval='0 3 * * *',  # Runs at 3 AM every day
    start_date=days_ago(1),
)

TEMP_FOLDER = "/tmp"  # Path to the temporary folder
os.makedirs(TEMP_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
TEMP_CSV_PATH = os.path.join(TEMP_FOLDER, "reservations_data.csv")

# Function to load data from the database and save it as a CSV file
def load_data(**kwargs):
    # Get the reservation date from the Airflow variable or use yesterday's date
    reservation_date = Variable.get("reservation_date", default_var=None)
    if reservation_date:
        reservation_date = datetime.strptime(reservation_date, '%Y-%m-%d')
    else:
        reservation_date = datetime.now() - timedelta(days=1)

    reservation_date_str = reservation_date.strftime('%Y-%m-%d')

    pg_hook = PostgresHook(postgres_conn_id='reservations_db_connection')
    engine = pg_hook.get_sqlalchemy_engine()   

    # Filter reservations by the reservation date
    query = f"""
    SELECT * 
    FROM reservations
    WHERE reservation_date::date = '{reservation_date_str}'
    """

    df = pd.read_sql_query(query, engine)

    if df.empty:
        return 'no_data'

    df.to_csv(TEMP_CSV_PATH, index=False)
    return 'predict_and_save'

# Function to perform predictions and save the results
def predict_and_save(**kwargs):
    df = pd.read_csv(TEMP_CSV_PATH)  # Load the DataFrame from the CSV file

    # Load preprocessor and model from MLflow
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    client = MlflowClient()

    model_name = "sk-learn-logistic-regression-reg-model"
    model_alias = "champion"
    model_version_details = client.get_model_version_by_alias(model_name, model_alias)

    artifact_path = 'preprocessing/preprocessor_model.pkl'
    local_path = mlflow.artifacts.download_artifacts(run_id=model_version_details.run_id, artifact_path=artifact_path, dst_path='./assets')
    preprocessor = joblib.load('artifacts/preprocessor_model.pkl')

    model_uri = f"models:/{model_name}@{model_alias}"
    model = mlflow.sklearn.load_model(model_uri)

    # Columns in the table
    cat_cols = ['hotel','meal', 'market_segment','distribution_channel',
                'reserved_room_type','deposit_type','customer_type']

    num_cols = ['lead_time','days_in_waiting_list',
                'adr','total_stay','total_people']

    bin_cols = ['is_repeated_guest','previous_cancellations',
                'previous_bookings_not_canceled','booking_changes',
                'agent','company','required_car_parking_spaces',
                'total_of_special_requests']

    dataset =(df
             .drop_duplicates()
             .fillna(0)
             .assign(total_stay=lambda df: df['stays_in_weekend_nights'] + df['stays_in_week_nights'],
                     total_people=lambda df: df['adults'] + df['children'] + df['babies'],
                    )
             [cat_cols + num_cols + bin_cols]
             .assign(total_people=lambda df: df['total_people'].astype('int64'),
                     agent=lambda df: df['agent'].astype('int64'),
                     company=lambda df: df['company'].astype('int64'),
                    )
    )
    print(dataset.columns)
    new_dataset = preprocessor.transform(dataset)

    pred = model.predict(new_dataset)
    pred_prob = model.predict_proba(new_dataset)[:, 1]  # Get the probability of the positive class

    # Add predictions to the DataFrame
    dataset['prediction'] = pred
    dataset['probability'] = pred_prob
    dataset['model_name'] = model_name
    dataset['model_version'] = model_version_details.version
    dataset['prediction_date'] = pd.to_datetime('now')

    # Save the results in the prediction_logs table
    pg_hook = PostgresHook(postgres_conn_id='predictions_db_connection')
    engine = pg_hook.get_sqlalchemy_engine()
    print(df.columns)
    dataset.to_sql('prediction_logs', engine, if_exists='append', index=False)

# Function to delete the temporary file
def delete_temp_file(**kwargs):
    if os.path.exists(TEMP_CSV_PATH):
        os.remove(TEMP_CSV_PATH)  # Delete the temporary CSV file
    else:
        print(f"The file {TEMP_CSV_PATH} does not exist.")

# Function to handle the case where no data is found
def no_data_found(**kwargs):
    print("No data found for the specified reservation date. The DAG will stop.")

# Define tasks
load_data_task = BranchPythonOperator(
    task_id='load_data',
    python_callable=load_data,
    dag=dag,
)

predict_and_save_task = PythonOperator(
    task_id='predict_and_save',
    python_callable=predict_and_save,
    dag=dag,
)

delete_temp_file_task = PythonOperator(
    task_id='delete_temp_file',
    python_callable=delete_temp_file,
    dag=dag,
)

no_data_task = PythonOperator(
    task_id='no_data_found',
    python_callable=no_data_found,
    dag=dag,
)

# Define task dependencies
load_data_task >> [predict_and_save_task, no_data_task]
predict_and_save_task >> delete_temp_file_task
