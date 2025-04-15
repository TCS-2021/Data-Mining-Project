import json
import logging
import os
import pickle
from kafka import KafkaConsumer
from river import tree, metrics
import io
import base64
import threading
import time
import pandas as pd
from datetime import datetime

class Consumer:
    """Kafka consumer for processing shopping event data and training a Hoeffding tree model."""

    def __init__(self, bootstrap_servers: str = 'localhost:9092', topic: str = 'shopping_events'):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

        # Define categories
        self.product_categories = ['electronics', 'fashion', 'home', 'books', 'toys']
        self.event_types = ['view', 'click', 'add_to_cart']

        # Initialize Hoeffding Tree model
        self.model = tree.HoeffdingTreeClassifier(
            grace_period=100,
            delta=0.99,
            nominal_attributes=[
                f'product_category_{cat}' for cat in self.product_categories
            ] + [
                f'event_type_{et}' for et in self.event_types
            ],
            split_criterion='gini'
        )

        # Initialize metrics
        self.accuracy = metrics.Accuracy()
        self.f1_score = metrics.F1()
        self.sample_count = 0
        
        # Set up data directory
        self.data_dir = './streamlit_data/'
        self.tree_dir = os.path.join(self.data_dir, 'tree')
        self.events_dir = os.path.join(self.data_dir, 'events')
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.tree_dir, exist_ok=True)
        os.makedirs(self.events_dir, exist_ok=True)
        
        # In-memory event store
        self.recent_events = []
        self.max_recent_events = 100
        
        # Last update timestamps
        self.last_tree_update = 0
        self.last_events_update = 0

    def preprocess(self, data: dict) -> dict:
        """Preprocess raw event data into model features."""
        event_mapping = {'view': 0, 'click': 1, 'add_to_cart': 2}
        category_mapping = {cat: idx for idx, cat in enumerate(self.product_categories)}

        return {
            'event_type': event_mapping[data['event_type']],
            'session_duration': data['session_duration'],
            'product_category': category_mapping[data['product_category']],
            'user_total_purchases': data['user_total_purchases']
        }

    def process_event(self, data: dict):
        """Process a single shopping event."""
        features = self.preprocess(data)
        target = data['purchase']

        try:
            # Make prediction
            y_pred = self.model.predict_one(features)

            if y_pred is not None:
                self.accuracy.update(target, y_pred)
                self.f1_score.update(target, y_pred)

            # Learn from the new example
            self.model.learn_one(features, target)
            self.sample_count += 1
            
            # Create readable event record
            event_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user_id": data["user_id"],
                "event_type": data["event_type"],
                "session_duration": data["session_duration"],
                "product_category": data["product_category"],
                "user_total_purchases": data["user_total_purchases"],
                "purchase": "Yes" if target == 1 else "No",
                "prediction": "Yes" if y_pred == 1 else "No" if y_pred == 0 else "N/A"
            }
            
            # Add to recent events
            self.recent_events.append(event_record)
            self.recent_events = self.recent_events[-self.max_recent_events:]  # Keep only most recent
            
            # Create pandas DataFrame for events
            df = pd.DataFrame(self.recent_events)
            
            # Update event data file every 5 processed events
            if self.sample_count % 5 == 0:
                # Save as CSV for easy reading
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(self.events_dir, f"events_{current_time}.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved events to {csv_path}")
            
            # Update tree visualization and accuracy every 20 events
            if self.sample_count % 20 == 0:
                self.update_tree_visualization()
                self.update_accuracy()
                
            self.logger.info(f"Processed event: {data['user_id']}, prediction: {y_pred}, actual: {target}")

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def update_tree_visualization(self):
        """Generate and save tree visualization image"""
        try:
            # Generate tree visualization
            graph = self.model.draw(max_depth=5)
            
            # Save as PNG image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(self.tree_dir, f"tree_{timestamp}")
            graph.render(filename=image_path, format='png', cleanup=True)
            
            self.logger.info(f"Tree visualization saved to {image_path}.png")
            self.last_tree_update = time.time()
                
        except Exception as e:
            self.logger.error(f"Error updating tree visualization: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def update_accuracy(self):
        """Update accuracy metrics file"""
        try:
            # Save accuracy to a simple text file
            accuracy_path = os.path.join(self.data_dir, 'accuracy.txt')
            with open(accuracy_path, 'w') as f:
                f.write(str(self.accuracy.get()))
                
            self.logger.info(f"Accuracy updated: {self.accuracy.get():.4f}")
                
        except Exception as e:
            self.logger.error(f"Error updating accuracy: {e}")

    def run(self):
        """Start consuming messages and processing events."""
        self.logger.info("Starting shopping event consumer and training model...")

        try:
            for message in self.consumer:
                data = message.value
                self.process_event(data)

        except KeyboardInterrupt:
            self.logger.info("Consumer stopped by user.")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        finally:
            self.consumer.close()
            self.logger.info("Consumer connection closed.")

if __name__ == "__main__":
    consumer = Consumer()
    consumer.run()