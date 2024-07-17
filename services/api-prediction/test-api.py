import requests

url = "http://localhost:8000/predict"

data = {
    'hotel': 'City Hotel',
    'meal': 'BB',
    'market_segment': 'Direct',
    'distribution_channel': 'Direct',
    'reserved_room_type': 'D',
    'deposit_type': 'No Deposit',
    'customer_type': 'Transient',
    'lead_time': 0,
    'days_in_waiting_list': 0,
    'adr': 142.92,
    'total_stay': 4,
    'total_people': 2,
    'is_repeated_guest': 0,
    'previous_cancellations': 0,
    'previous_bookings_not_canceled': 0,
    'booking_changes': 0,
    'agent': 14,
    'company': 0,
    'required_car_parking_spaces': 0,
    'total_of_special_requests': 0,
    'is_canceled': 0
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print("Response JSON:", response.json())
else:
    print(f"Request failed with status code: {response.status_code}")
