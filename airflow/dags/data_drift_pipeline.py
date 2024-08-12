from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import os
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from airflow.models import Variable

# DAG configuration
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

with DAG(
    dag_id='data_drift_monitoring',
    default_args=default_args,
    description='DAG for daily data drift monitoring',
    schedule_interval='0 3 * * *',  # Every day at 3:00 AM
    start_date=days_ago(1),
    catchup=False,
) as dag:

    def fetch_reference_data(**kwargs):
        mlflow.set_tracking_uri("http://mlflow-server:5000")
        client = MlflowClient()

        model_name = "sk-learn-logistic-regression-reg-model"
        model_alias = "champion"
        model_version_details = client.get_model_version_by_alias(model_name, model_alias)

        artifact_path = 'reference_dataset/hotel_bookings_reference.csv'
        tmp_path = mlflow.artifacts.download_artifacts(run_id=model_version_details.run_id, artifact_path=artifact_path, dst_path='/tmp')

        df_ref = pd.read_csv(tmp_path).drop(columns='is_canceled')
        df_ref.to_csv('/tmp/reference_data.csv', index=False)

    def fetch_prediction_data(**kwargs):
        # Get the dataset date
        print(datetime.now())
        reservation_date = Variable.get("reservation_date", default_var=(datetime.now() - timedelta(1)).strftime('%Y-%m-%d'))
        kwargs['ti'].xcom_push(key='reservation_date', value=reservation_date)
        print(f"Date: {reservation_date}")
        postgres_hook = PostgresHook(postgres_conn_id='predictions_db_connection')
        engine = postgres_hook.get_sqlalchemy_engine()

        qry = f"""SELECT * 
        FROM public.prediction_logs 
        WHERE DATE(reservation_date) = '{reservation_date}'
        ORDER BY id ASC 
        """
        print(qry)
        df_tst = pd.read_sql(qry, engine).drop(columns=['id', 'prediction', 'probability', 'model_name', 'model_version', 'prediction_date'])
        print(f"Prediction data lenght: {df_tst.shape}")
        if df_tst.empty:
            # If no data is available, end the DAG successfully
            return "No data available for the specified date."

        df_tst.to_csv('/tmp/prediction_data.csv', index=False)

    def run_data_drift_report(**kwargs):
        cat_cols = ['hotel','meal', 'market_segment','distribution_channel',
                    'reserved_room_type','deposit_type','customer_type']

        num_cols = ['lead_time','days_in_waiting_list',
                    'adr','total_stay','total_people']

        bin_cols = ['is_repeated_guest','previous_cancellations',
                    'previous_bookings_not_canceled','booking_changes',
                    'agent','company','required_car_parking_spaces',
                    'total_of_special_requests']

        df_ref = pd.read_csv('/tmp/reference_data.csv')
        df_tst = pd.read_csv('/tmp/prediction_data.csv')

        column_mapping = ColumnMapping(
            numerical_features=num_cols,
            categorical_features=cat_cols + bin_cols,
            target=None
        )

        report = Report(metrics=[DataDriftPreset(drift_share=0.25, cat_stattest_threshold=0.2, num_stattest_threshold=0.2)])
        report.run(reference_data=df_ref, current_data=df_tst, column_mapping=column_mapping)

        report_dict = report.as_dict()

        result = report_dict["metrics"][0]["result"]
        report_date = kwargs['execution_date'].strftime('%Y-%m-%d')
        print(result)
        df_rpt = (pd.DataFrame([result])
                  .assign(date_dataset=kwargs['ti'].xcom_pull(task_ids='fetch_prediction_data', key='reservation_date'),
                          date_report=report_date)
                  )
        df_rpt.to_csv('/tmp/report_data.csv', index=False)
        print(report_dict["metrics"][1]["result"]['drift_by_columns'].keys())
        print(report_dict["metrics"][1]["result"]['drift_by_columns']['adr'])
        lst = []
        for key in report_dict["metrics"][1]["result"]['drift_by_columns'].keys():
            lst.append([report_dict["metrics"][1]["result"]['drift_by_columns'][key]['column_name'],
                        report_dict["metrics"][1]["result"]['drift_by_columns'][key]['column_type'],
                        report_dict["metrics"][1]["result"]['drift_by_columns'][key]['drift_score'],
                        report_dict["metrics"][1]["result"]['drift_by_columns'][key]['drift_detected']]
                        )

        df_cols = (pd.DataFrame(lst)
                   .rename(columns={0:'column_name', 1:'column_type', 2:'drift_score', 3:'drift_detected'})
                   .assign(date_dataset=kwargs['ti'].xcom_pull(task_ids='fetch_prediction_data', key='reservation_date'),
                           date_report=report_date)
                  )
        df_cols.to_csv('/tmp/columns_report_data.csv', index=False)

    def store_reports(**kwargs):
        postgres_hook = PostgresHook(postgres_conn_id='predictions_db_connection')
        engine = postgres_hook.get_sqlalchemy_engine()

        df_rpt = pd.read_csv('/tmp/report_data.csv')
        df_cols = pd.read_csv('/tmp/columns_report_data.csv')

        df_rpt.to_sql('data_report', engine, if_exists='append', index=False)
        df_cols.to_sql('data_columns_report', engine, if_exists='append', index=False)

    fetch_reference_task = PythonOperator(
        task_id='fetch_reference_data',
        python_callable=fetch_reference_data,
        provide_context=True,
    )

    fetch_prediction_task = PythonOperator(
        task_id='fetch_prediction_data',
        python_callable=fetch_prediction_data,
        provide_context=True,
    )

    run_drift_report_task = PythonOperator(
        task_id='run_data_drift_report',
        python_callable=run_data_drift_report,
        provide_context=True,
    )

    store_reports_task = PythonOperator(
        task_id='store_reports',
        python_callable=store_reports,
        provide_context=True,
    )

    fetch_reference_task >> fetch_prediction_task >> run_drift_report_task >> store_reports_task
