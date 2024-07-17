from sqlalchemy import Column, Integer, String, Float, DateTime
from app.database import Base
from datetime import datetime

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    hotel = Column(String)
    meal = Column(String)
    market_segment = Column(String)
    distribution_channel = Column(String)
    reserved_room_type = Column(String)
    deposit_type = Column(String)
    customer_type = Column(String)
    lead_time = Column(Integer)
    days_in_waiting_list = Column(Integer)
    adr = Column(Float)
    total_stay = Column(Integer)
    total_people = Column(Integer)
    is_repeated_guest = Column(Integer)
    previous_cancellations = Column(Integer)
    previous_bookings_not_canceled = Column(Integer)
    booking_changes = Column(Integer)
    agent = Column(Integer, nullable=True)
    company = Column(Integer, nullable=True)
    required_car_parking_spaces = Column(Integer)
    total_of_special_requests = Column(Integer)
    is_canceled = Column(Integer)
    prediction = Column(Integer)
    probability = Column(Float)
    model_name = Column(String)
    model_version = Column(String)
    prediction_date = Column(DateTime, default=datetime.utcnow)
