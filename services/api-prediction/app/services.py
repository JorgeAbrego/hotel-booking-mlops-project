import joblib
import mlflow
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from app.config import settings
from app.logger import logger
from app.database import SessionLocal, engine
from app.models import PredictionLog
import time
from datetime import datetime
from sqlalchemy.exc import OperationalError
from sqlalchemy.sql import text

class ModelService:
    def __init__(self):
        self.initialize_database()
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        self.model_name = settings.MODEL_NAME
        self.model_alias = settings.MODEL_ALIAS
        self.model = None
        self.preprocessor = None
        self.load_model_and_preprocessor()

    def initialize_database(self):
        try:
            # Check if the tables exist
            logger.info("Checking if database tables exist")
            with engine.connect() as connection:
                connection.execute(text("SELECT 1 FROM prediction_logs LIMIT 1;"))
            logger.info("Tables already exist")
        except OperationalError:
            # Create tables if they do not exist
            logger.info("Tables do not exist, creating tables")
            from app.database import Base
            Base.metadata.create_all(bind=engine)
            logger.info("Tables created successfully")

    def load_model_and_preprocessor(self):
        try:
            logger.info("Creating MLflow client")
            model_version_details = self.client.get_model_version_by_alias(self.model_name, self.model_alias)
            logger.info(f"Model version details: {model_version_details}")

            artifact_path = 'preprocessing/preprocessor_model.pkl'
            logger.info(f"Downloading artifact from path: {artifact_path}")
            local_path = mlflow.artifacts.download_artifacts(run_id=model_version_details.run_id, artifact_path=artifact_path, dst_path='./assets')
            logger.info(f"Artifact downloaded to local path: {local_path}")

            logger.info(f"Loading preprocessor from path: {local_path}")
            self.preprocessor = joblib.load(local_path)
            logger.info("Preprocessor loaded successfully")

            model_uri = f"models:/{self.model_name}@{self.model_alias}"
            logger.info(f"Loading model from URI: {model_uri}")
            self.model = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model or preprocessor: {e}")
            raise

    def predict(self, data: pd.DataFrame):
        start_time = time.time()
        try:
            logger.info("Applying preprocessor to input data")
            preprocessed_data = self.preprocessor.transform(data)
            logger.info(f"Preprocessed data: {preprocessed_data}")

            logger.info("Making prediction with the model")
            prediction = self.model.predict(preprocessed_data)
            proba = self.model.predict_proba(preprocessed_data)
            proba_idx = np.argmax(proba, axis=1)

            end_time = time.time()
            execution_time = end_time - start_time

            logger.info(f"Prediction: {prediction[0]}, Probability: {proba[0][proba_idx]}")
            logger.info(f"Prediction execution time: {execution_time:.4f} seconds")

            self.log_prediction(data, int(prediction[0]), float(proba[0][proba_idx]), execution_time)

            return prediction[0], proba[0][proba_idx]
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def log_prediction(self, data: pd.DataFrame, prediction: int, probability: float, execution_time: float):
        db = SessionLocal()
        try:
            logger.info("Saving prediction log to the database")
            prediction_log = PredictionLog(
                hotel=data.iloc[0]["hotel"],
                meal=data.iloc[0]["meal"],
                market_segment=data.iloc[0]["market_segment"],
                distribution_channel=data.iloc[0]["distribution_channel"],
                reserved_room_type=data.iloc[0]["reserved_room_type"],
                deposit_type=data.iloc[0]["deposit_type"],
                customer_type=data.iloc[0]["customer_type"],
                lead_time=int(data.iloc[0]["lead_time"]),
                days_in_waiting_list=int(data.iloc[0]["days_in_waiting_list"]),
                adr=float(data.iloc[0]["adr"]),
                total_stay=int(data.iloc[0]["total_stay"]),
                total_people=int(data.iloc[0]["total_people"]),
                is_repeated_guest=int(data.iloc[0]["is_repeated_guest"]),
                previous_cancellations=int(data.iloc[0]["previous_cancellations"]),
                previous_bookings_not_canceled=int(data.iloc[0]["previous_bookings_not_canceled"]),
                booking_changes=int(data.iloc[0]["booking_changes"]),
                agent=int(data.iloc[0]["agent"]) if data.iloc[0]["agent"] is not None else None,
                company=int(data.iloc[0]["company"]) if data.iloc[0]["company"] is not None else None,
                required_car_parking_spaces=int(data.iloc[0]["required_car_parking_spaces"]),
                total_of_special_requests=int(data.iloc[0]["total_of_special_requests"]),
                is_canceled=int(data.iloc[0]["is_canceled"]),
                prediction=prediction,
                probability=probability,
                model_name=self.model_name,
                model_version=self.model_alias,
                prediction_date=datetime.utcnow()
            )
            db.add(prediction_log)
            db.commit()
            logger.info("Prediction log saved successfully")
        except Exception as e:
            logger.error(f"Error saving prediction log: {e}")
            db.rollback()
        finally:
            db.close()

model_service = ModelService()
