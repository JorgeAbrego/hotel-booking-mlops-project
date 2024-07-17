from pydantic import BaseModel, Field
from typing import Optional

class HotelBooking(BaseModel):
    hotel: str
    meal: str
    market_segment: str
    distribution_channel: str
    reserved_room_type: str
    deposit_type: str
    customer_type: str
    lead_time: int
    days_in_waiting_list: int
    adr: float
    total_stay: int
    total_people: int
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    booking_changes: int
    agent: Optional[int] = Field(None, description="Agent ID can be optional")
    company: Optional[int] = Field(None, description="Company ID can be optional")
    required_car_parking_spaces: int
    total_of_special_requests: int
    is_canceled: int

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    execution_time: float
