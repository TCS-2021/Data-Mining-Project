from kafka import KafkaProducer
import json
import time
import random
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'parkingstream'

def generate_occupancy(time_min):
    """Generate occupancy percentage based on time."""
    if 420 <= time_min <= 600:     # 7 AM – 10 AM
        return random.randint(70, 90)
    elif 720 <= time_min <= 900:   # 12 PM – 3 PM
        return random.randint(85, 100)
    elif 1080 <= time_min <= 1320: # 6 PM – 10 PM
        return random.randint(40, 60)
    else:
        return random.randint(5, 30)

time_of_day = 0

while time_of_day < 1440:
    data = {
        'time_of_day': time_of_day,
        'occupancy': generate_occupancy(time_of_day)
    }
    print(f"Produced: {data}")
    producer.send(topic, value=data)
    time.sleep(2)  # Simulate streaming every second (10 minutes in simulation)
    time_of_day += 10