from kafka import KafkaConsumer
from river import tree, metrics
import json
from sklearn import tree
import matplotlib.pyplot as plt

# Initialize Kafka consumer
consumer = KafkaConsumer(
    'shop',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)
product_categories = ['electronics', 'fashion', 'home', 'books', 'toys']
# Initialize Hoeffding Tree for conversion prediction
model = tree.HoeffdingTreeClassifier(
    grace_period=5,
    delta=0.99,
    split_criterion='info_gain'
)

# Track metrics
accuracy = metrics.Accuracy()
f1_score = metrics.F1()

# Preprocess message for model
def preprocess_message(message):
    # Convert categorical features to numerical
    event_mapping = {'view': 0, 'click': 1, 'add_to_cart': 2, 'purchase': 3}
    category_mapping = {cat: idx for idx, cat in enumerate(product_categories)}
    
    features = {
        'event_type': event_mapping[message['event_type']],
        'product_category': category_mapping[message['product_category']],
        'session_duration': message['session_duration']
    }
    label = message['conversion']
    
    return features, label

print("Starting shopping data consumer and training model...")

for msg in consumer:
    try:
        message = msg.value
        X, y = preprocess_message(message)
        
        print(f"\nReceived: {message}")
        
        # Make prediction before learning
        y_pred = model.predict_one(X)
        if y_pred is not None:
            accuracy.update(y, y_pred)
            f1_score.update(y, y_pred)
            print(f"Predicted conversion: {y_pred} (Actual: {y})")
            print(f"Accuracy: {accuracy.get():.4f}, F1 Score: {f1_score.get():.4f}")
        else:
            print("No prediction yet (model untrained)")
        
        # Learn incrementally
        model.learn_one(X, y)
        
        # Print model stats
        print(f"Model summary: {model.summary}")
        
                # 3. Plot the tree
        plt.figure(figsize=(12, 10))  # Adjust figure size as needed
        sklearn.tree.plot_tree(
            model,  # Pass the trained classifier
            feature_names=[
                'timestamp', 'user_id', 'event_type', 'product_category',
       'session_duration'
            ],  # Replace with your feature names
            class_names=[
                '0',
                '1',
            ],  # Replace with your class names
            filled=True,
            rounded=True,
            fontsize=10,
        )  # Add desired plot parameters
        plt.title('Hoeffding Tree Visualization')  # Add a title
        plt.show()
        
    except Exception as e:
        print(f"Error processing message: {e}")