from kafka import KafkaConsumer
import json
import numpy as np
import matplotlib.pyplot as plt

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

plt.ion()

consumer = KafkaConsumer(
    'parkingstream',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',
    enable_auto_commit=True,
    group_id='parking-group'
)

clustream = CluStream(max_micro_clusters=3, distance_threshold=100)
data_stream = []

# Define three colors for clusters
cluster_colors = ['blue', 'green', 'purple']

print("📡 Waiting for Kafka streaming data...")

for message in consumer:
    val = message.value
    print(f"📥 Received: {val}")

    point = [val['time_of_day'], val['occupancy']]
    data_stream.append(point)

    clustream.update(point, timestamp=val['time_of_day'])

    # Plotting
    X = np.array(data_stream)
    cluster_centers = np.array(clustream.get_clusters())

    # Assign a color to each point based on its proximity to a single cluster
    point_colors = []
    for p in X:
        # Calculate distances to all clusters and find the closest
        distances = [np.linalg.norm(p - c) for c in cluster_centers]
        closest_cluster = np.argmin(distances)  # index of the closest cluster
        # Assign the color corresponding to the closest cluster
        point_colors.append(cluster_colors[closest_cluster])

    plt.clf()
    # Scatter the points, each with a color based on the closest cluster
    plt.scatter(X[:, 0], X[:, 1], color=point_colors, alpha=0.5, label='Streamed Points')

    # Plot the cluster centers
    if len(cluster_centers) > 0:
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', s=200, marker='X', label='Micro-Clusters')



    plt.xlabel("Time of Day (minutes)")
    plt.ylabel("Occupancy (%)")
    plt.title("CluStream: Streaming Micro-Clustering")
    plt.legend()
    plt.savefig('plot.png')  # Save latest plot image
    plt.pause(3)

