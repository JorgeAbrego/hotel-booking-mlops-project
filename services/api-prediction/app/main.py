from fastapi import FastAPI, HTTPException
from app.schemas import HotelBooking, PredictionResponse
from app.services import model_service
from app.logger import logger
import time
import pandas as pd

app = FastAPI()

@app.post("/predict", response_model=PredictionResponse)
async def predict(booking: HotelBooking):
    start_time = time.time()
    logger.info("Received prediction request")
    
    try:
        data = pd.DataFrame([booking.dict()])
        logger.info(f"Input data: {data.to_dict(orient='records')[0]}")
        
        prediction, probability = model_service.predict(data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Prediction made: {prediction}, Probability: {probability}")
        logger.info(f"Execution time: {execution_time:.4f} seconds")
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "execution_time": execution_time
        }
    except Exception as e:
        logger.error(f"Error in prediction request: {e}")
        raise HTTPException(status_code=500, detail="Error making prediction")
