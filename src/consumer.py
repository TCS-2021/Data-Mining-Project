import json
import logging
import os
from kafka import KafkaConsumer
from river import tree, metrics

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

        self.visualisation_dir = './visualisations/'
        os.makedirs(self.visualisation_dir, exist_ok=True)

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

            self.logger.info(f"Processed features: {features}")

            # Periodic summary and visualisation
            if self.sample_count % 20 == 0:
                self.log_model_summary()
                self.visualise_model()

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    def log_model_summary(self):
        """Log model summary and metrics."""
        self.logger.info("\n=== Model Summary ===")
        self.logger.info(f"Samples processed: {self.sample_count}")
        self.logger.info(f"Model summary: {self.model.summary}")
        self.logger.info(f"Current accuracy: {self.accuracy.get():.4f}")
        self.logger.info("====================\n")

    def visualise_model(self):
        """Generate and save model visualisation."""
        graph = self.model.draw(max_depth=7)
        file_path = os.path.join(self.visualisation_dir, f'hoeffding_tree_{self.sample_count}')
        graph.render(file_path, format='png', cleanup=True)

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
        finally:
            self.consumer.close()
            self.logger.info("Consumer connection closed.")

if __name__ == "__main__":
    consumer = Consumer()
    consumer.run()
