from kafka import KafkaProducer
import json
import time
from datetime import datetime, timedelta
import random
from collections import defaultdict, deque

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Parameters
num_rows = 1000
user_ids = [f"user_{i:04}" for i in range(1, 201)]
product_categories = ['electronics', 'fashion', 'home', 'books', 'toys']
event_sequence = ['view', 'click', 'add_to_cart', 'purchase']
start_time = datetime.now()

# Track user behavior
user_event_history = defaultdict(lambda: deque(maxlen=4))
user_session_time = defaultdict(float)

print("Starting shopping data producer...")

for i in range(num_rows):
    timestamp = start_time + timedelta(seconds=i * random.randint(1, 3))
    user_id = random.choice(user_ids)
    product_category = random.choice(product_categories)

    # Create behavior: 20% chance of full conversion funnel
    if random.random() < 0.2:
        # Full conversion funnel
        for stage in event_sequence:
            user_event_history[user_id].append(stage)
            user_session_time[user_id] += random.uniform(0.5, 3)
            session_duration = round(user_session_time[user_id], 2)

            label = 1 if stage == 'purchase' and list(user_event_history[user_id]) == event_sequence else 0

            message = {
                "timestamp": str(timestamp),
                "user_id": user_id,
                "event_type": stage,
                "product_category": product_category,
                "session_duration": session_duration,
                "conversion": label
            }
            
            producer.send('shop', message)
            print(f"Sent: {message}")
            
            timestamp += timedelta(seconds=random.randint(1, 2))
            time.sleep(1)  # Small delay between funnel events
    else:
        # Non-conversion behavior
        stage = random.choice(event_sequence[:-1])  # Avoid 'purchase'
        user_event_history[user_id].append(stage)
        user_session_time[user_id] += random.uniform(0.5, 2)
        session_duration = round(user_session_time[user_id], 2)

        message = {
            "timestamp": str(timestamp),
            "user_id": user_id,
            "event_type": stage,
            "product_category": product_category,
            "session_duration": session_duration,
            "conversion": 0  # Not a purchase
        }
        
        producer.send('shopping_events', message)
        print(f"Sent: {message}")
        time.sleep(1)  # Small delay between events

producer.flush()