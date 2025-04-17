import json
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kafka import KafkaConsumer
from datetime import datetime
import time

# ========== CluStream Micro-Cluster Code ==========

class MicroCluster:
    def __init__(self, point, timestamp):
        self.N = 1
        self.LS = np.array(point, dtype=float)
        self.SS = np.square(point)
        self.timestamp = timestamp

    def add_point(self, point):
        self.N += 1
        self.LS += point
        self.SS += np.square(point)

    def get_centroid(self):
        return self.LS / self.N

    def get_radius(self):
        centroid = self.get_centroid()
        return np.sqrt(np.sum(self.SS / self.N - centroid ** 2))

class CluStream:
    def __init__(self, max_micro_clusters=3, distance_threshold=50):
        self.micro_clusters = []
        self.max_micro_clusters = max_micro_clusters
        self.distance_threshold = distance_threshold

    def _euclidean(self, a, b):
        return np.linalg.norm(a - b)

    def update(self, point, timestamp):
        point = np.array(point)
        if not self.micro_clusters:
            self.micro_clusters.append(MicroCluster(point, timestamp))
            return

        min_dist = float('inf')
        best_cluster = None

        for cluster in self.micro_clusters:
            dist = self._euclidean(point, cluster.get_centroid())
            if dist < min_dist:
                min_dist = dist
                best_cluster = cluster

        if min_dist <= self.distance_threshold:
            best_cluster.add_point(point)
        elif len(self.micro_clusters) < self.max_micro_clusters:
            self.micro_clusters.append(MicroCluster(point, timestamp))
        else:
            # Replace the oldest cluster
            oldest_idx = np.argmin([c.timestamp for c in self.micro_clusters])
            self.micro_clusters[oldest_idx] = MicroCluster(point, timestamp)

    def get_clusters(self):
        return [cluster.get_centroid() for cluster in self.micro_clusters]

# ========== Kafka + CluStream Consumer ==========

class Consumer:
    """Kafka consumer for processing parking event data and clustering with CluStream."""

    def __init__(self, bootstrap_servers: str = 'localhost:9092', topic: str = 'parkingstream'):
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='parking-group'
        )

        # Initialize CluStream model
        self.clustream = CluStream(max_micro_clusters=3, distance_threshold=100)

        # Define colors for clusters
        self.cluster_colors = ['blue', 'green', 'purple']

        # Initialize counters and storage
        self.sample_count = 0
        self.recent_events = []
        self.max_recent_events = 100

        # Set up data directory
        self.data_dir = './src/PrescriptiveAnalysis2/streamlit_data/'
        self.plot_dir = os.path.join(self.data_dir, 'plots')
        self.events_dir = os.path.join(self.data_dir, 'events')
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.events_dir, exist_ok=True)

        # Last update timestamps
        self.last_plot_update = 0
        self.last_events_update = 0

    def preprocess(self, data: dict) -> list:
        """Preprocess raw event data into model features."""
        return [data['time_of_day'], data['occupancy']]

    def process_event(self, data: dict):
        """Process a single parking event."""
        try:
            # Preprocess data
            point = self.preprocess(data)
            timestamp = data['time_of_day']

            # Update CluStream model
            self.clustream.update(point, timestamp)
            self.sample_count += 1

            # Determine closest cluster
            cluster_centers = np.array(self.clustream.get_clusters())
            if len(cluster_centers) > 0:
                distances = [np.linalg.norm(np.array(point) - c) for c in cluster_centers]
                closest_cluster = np.argmin(distances)
                cluster_label = closest_cluster
            else:
                cluster_label = -1  # No clusters yet

            # Create readable event record
            event_record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "time_of_day": data["time_of_day"],
                "occupancy": data["occupancy"],
                "cluster": cluster_label
            }

            # Add to recent events
            self.recent_events.append(event_record)
            self.recent_events = self.recent_events[-self.max_recent_events:]  # Keep only most recent

            # Create pandas DataFrame for events
            df = pd.DataFrame(self.recent_events)

            # Update event data file every 5 processed events
            if self.sample_count % 5 == 0:
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = os.path.join(self.events_dir, f"events_{current_time}.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved events to {csv_path}")
                self.last_events_update = time.time()

            # Update cluster visualization every 20 events
            if self.sample_count % 20 == 0:
                self.update_cluster_visualization()

            self.logger.info(f"Processed event: time_of_day={data['time_of_day']}, occupancy={data['occupancy']}, cluster={cluster_label}")

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def update_cluster_visualization(self):
        """Generate and save cluster visualization plot."""
        try:
            # Prepare data
            X = np.array([[e['time_of_day'], e['occupancy']] for e in self.recent_events])
            cluster_centers = np.array(self.clustream.get_clusters())

            # Assign colors based on closest cluster
            point_colors = []
            for p in X:
                if len(cluster_centers) > 0:
                    distances = [np.linalg.norm(p - c) for c in cluster_centers]
                    closest_cluster = np.argmin(distances)
                    point_colors.append(self.cluster_colors[closest_cluster])
                else:
                    point_colors.append('gray')  # Default color if no clusters

            # Create plot
            plt.clf()
            if len(X) > 0:
                plt.scatter(X[:, 0], X[:, 1], color=point_colors, alpha=0.5, label='Streamed Points')
            if len(cluster_centers) > 0:
                plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', s=200, marker='X', label='Micro-Clusters')

            plt.xlabel("Time of Day (minutes)")
            plt.ylabel("Occupancy (%)")
            plt.title("CluStream: Streaming Micro-Clustering")
            plt.legend()

            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.plot_dir, f"plot_{timestamp}.png")
            plt.savefig(plot_path)
            plt.close()

            self.logger.info(f"Cluster visualization saved to {plot_path}")
            self.last_plot_update = time.time()

        except Exception as e:
            self.logger.error(f"Error updating cluster visualization: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def run(self):
        """Start consuming messages and processing events."""
        self.logger.info("Starting parking event consumer and clustering with CluStream...")

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


