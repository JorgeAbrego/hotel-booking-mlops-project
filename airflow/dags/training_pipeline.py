from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Binarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import requests
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import joblib
import os
import shutil

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

artifact_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts') # Change this to your artifacts directory

def check_and_clean_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Directorio {dir_path} creado.")
    else:
        if os.listdir(dir_path):
            for filename in os.listdir(dir_path):
                file_path = os.path.join(dir_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            print(f"Contenido del directorio {dir_path} borrado.")
        else:
            print(f"El directorio {dir_path} ya está vacío.")

def preprocess_and_save_data(artifact_dir):
    print("Downloading dataset...")
    base_url = 'https://raw.githubusercontent.com/JorgeAbrego/hotel-booking-mlops-project/main/data'
    relative_url = 'hotel_booking_dataset.csv'
    docs_url = f'{base_url}/{relative_url}'
    
    try:
        docs_response = requests.get(docs_url)
        docs_response.raise_for_status()
        print(f"HTTP response status code: {docs_response.status_code}")
    except requests.RequestException as e:
        print(f"Error downloading dataset: {e}")
        return

    try:
        base_df = pd.read_csv(docs_url)
    except Exception as e:
        print(f"Error reading CSV data: {e}")
        return
    
    cat_cols = ['hotel','meal', 'market_segment','distribution_channel',
                'reserved_room_type','deposit_type','customer_type']
    num_cols = ['lead_time','days_in_waiting_list',
                'adr','total_stay','total_people']
    bin_cols = ['is_repeated_guest','previous_cancellations',
                'previous_bookings_not_canceled','booking_changes',
                'agent','company','required_car_parking_spaces',
                'total_of_special_requests']
    
    dataset =(base_df
             .drop_duplicates()
             .fillna(0)
             .assign(total_stay=lambda df: df['stays_in_weekend_nights'] + df['stays_in_week_nights'],
                     total_people=lambda df: df['adults'] + df['children'] + df['babies'])
             [cat_cols + num_cols + bin_cols + ['is_canceled']]
             .assign(total_people=lambda df: df['total_people'].astype('int64'),
                     agent=lambda df: df['agent'].astype('int64'),
                     company=lambda df: df['company'].astype('int64'))
            )
    
    dataset.to_csv(f"{artifact_dir}/hotel_bookings_reference.csv", index=False)

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    binary_transformer = Pipeline(steps=[
        ('binarizer', Binarizer())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_cols),
            ('num', numerical_transformer, num_cols),
            ('bin', binary_transformer, bin_cols)
        ])

    processed_data = preprocessor.fit_transform(dataset)
    preprocessor_file = f"{artifact_dir}/preprocessor_model.pkl"
    joblib.dump(preprocessor, preprocessor_file)
    
    np.save(f"{artifact_dir}/processed_data.npy", processed_data)
    dataset['is_canceled'].to_csv(f"{artifact_dir}/is_canceled.csv", index=False)

def train_model(artifact_dir):
    mlflow.set_tracking_uri('http://mlflow-server:5000')
    mlflow.set_experiment('hotel_cancellation_prediction')
    mlflow.sklearn.autolog()

    processed_data = np.load(f"{artifact_dir}/processed_data.npy")
    y = pd.read_csv(f"{artifact_dir}/is_canceled.csv")['is_canceled']
    X_train, X_test, y_train, y_test = train_test_split(processed_data, y, test_size=0.3, random_state=42)

    (pd.DataFrame(X_train)
     .assign(is_canceled=y_train.reset_index(drop=True))
    ).to_csv(artifact_dir + '/train_data.csv', index=False)

    (pd.DataFrame(X_test)
     .assign(is_canceled=y_test.reset_index(drop=True))
    ).to_csv(artifact_dir + '/test_data.csv', index=False)

    model = LogisticRegression()

    with mlflow.start_run(run_name=f"Logistic Regression [{datetime.now().strftime('%Y%m%d-%H%M%S')}]"):
        mlflow.log_artifact(artifact_dir + '/preprocessor_model.pkl', artifact_path='preprocessing')
        mlflow.log_artifact(artifact_dir + '/hotel_bookings_reference.csv', artifact_path='reference_dataset')
        mlflow.log_artifact(artifact_dir + '/train_data.csv', artifact_path='train_data')
        mlflow.log_artifact(artifact_dir + '/test_data.csv', artifact_path='test_data')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        signature = infer_signature(X_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title('Matriz de Confusión - Logistic Regression')
        plt.savefig(f'{artifact_dir}/confusion_matrix_Logistic_Regression.png')
        mlflow.log_artifact(f'{artifact_dir}/confusion_matrix_Logistic_Regression.png')
        plt.close()

        if hasattr(model, 'coef_'):
            feature_importances = np.abs(model.coef_[0])
            importance_df = pd.DataFrame({
                'Feature': [f'feature_{i}' for i in range(len(feature_importances))],
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)
            importance_df.to_csv(f'{artifact_dir}/feature_importances_Logistic_Regression.csv', index=False)
            mlflow.log_artifact(f'{artifact_dir}/feature_importances_Logistic_Regression.csv')

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="sk-learn-logistic-regression-reg-model",
        )
        run = mlflow.active_run()
        print("Active run_id: {}".format(run.info.run_id))
    mlflow.end_run()

def set_model_aliases():
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri='http://mlflow-server:5000')
    model_name = 'sk-learn-logistic-regression-reg-model'
    x = client.get_registered_model(model_name)
    
    if 'challenger' in [alias.alias for alias in x.latest_versions[0].aliases]:
        client.delete_registered_model_alias(model_name, 'challenger')
    if 'champion' in [alias.alias for alias in x.latest_versions[0].aliases]:
        client.delete_registered_model_alias(model_name, 'champion')
    
    client.set_registered_model_alias(model_name, 'champion', x.latest_versions[0].version)

with DAG(
    'hotel_cancellation_training_pipeline',
    default_args=default_args,
    description='A DAG for hotel cancellation prediction training pipeline with MLflow tracking and lineage',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    t1 = PythonOperator(
        task_id='check_and_clean_directory',
        python_callable=check_and_clean_directory,
        op_kwargs={'dir_path': artifact_dir},
    )

    t2 = PythonOperator(
        task_id='preprocess_and_save_data',
        python_callable=preprocess_and_save_data,
        op_kwargs={'artifact_dir': artifact_dir},
    )

    t3 = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs={'artifact_dir': artifact_dir},
    )

    t4 = PythonOperator(
        task_id='set_model_aliases',
        python_callable=set_model_aliases,
    )

    t1 >> t2 >> t3 >> t4
