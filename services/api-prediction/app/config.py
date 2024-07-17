import os

class Settings:
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "sk-learn-logistic-regression-reg-model")
    MODEL_ALIAS: str = os.getenv("MODEL_ALIAS", "challenger")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://prediction_user:predapi-p4ss@localhost:5432/prediction_db")

settings = Settings()
