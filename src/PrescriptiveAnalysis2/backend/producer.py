import json
import random
import time
import logging
from kafka import KafkaProducer

class Producer:
    """Kafka producer for simulated shopping event data."""

    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize Kafka producer
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # Parameters
        self.user_ids = [f"user_{i:04}" for i in range(1, 201)]
        self.product_categories = ['electronics', 'fashion', 'home', 'books', 'toys']
        self.event_types = ['view', 'click', 'add_to_cart']

        # Track per-user purchase frequency
        self.user_purchase_counts = {uid: random.randint(0, 30) for uid in self.user_ids}

    def generate_event(self) -> dict:
        """Generate a simulated shopping event."""
        user_id = random.choice(self.user_ids)
        product_category = random.choice(self.product_categories)
        event_type = random.choice(self.event_types)
        session_duration = round(random.uniform(1, 20), 2)
        user_purchases = self.user_purchase_counts[user_id]

        # Rule-based purchase logic
        purchase = int(
            (event_type == 'view' and session_duration > 12 and user_purchases > 10)
            or (event_type == 'add_to_cart' and session_duration > 5 and user_purchases > 4)
            or (event_type == 'click' and session_duration > 7 and user_purchases > 6)
        )

        # Session duration categorization
        if session_duration <= 6:
            sd = 'low'
        elif 6 < session_duration <= 10:
            sd = 'medium'
        else:
            sd = 'high'

        # User purchases categorization
        if user_purchases <= 5:
            up = 'very few'
        elif 5 < user_purchases <= 10:
            up = 'few'
        elif 10 < user_purchases <= 20:
            up = 'medium'
        else:
            up = 'high'

        return {
            "user_id": user_id,
            "event_type": event_type,
            "session_duration": sd,
            "product_category": product_category,
            "user_total_purchases": up,
            "purchase": purchase
        }

    def run(self):
        """Continuously produce shopping events."""
        self.logger.info("Starting shopping data producer...")
        try:
            while True:
                event = self.generate_event()
                self.producer.send('shopping_events', event)
                self.logger.info(f"Sent: {event}")
                time.sleep(.2)  # Simulate near real-time events

        except KeyboardInterrupt:
            self.logger.info("Producer stopped by user.")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
        finally:
            self.producer.close()
            self.logger.info("Producer connection closed.")

if __name__ == "__main__":
    producer = Producer()
    producer.run()
