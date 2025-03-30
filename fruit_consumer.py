from kafka import KafkaConsumer
from river import tree,metrics
import math

# Initialize Kafka consumer
consumer = KafkaConsumer('fruit', bootstrap_servers='localhost:9092')

# Initialize Hoeffding Tree with parameters
model = tree.HoeffdingTreeClassifier(
    grace_period=3,  
    delta=0.95,      
    split_criterion='info_gain'
)

# Track accuracy
accuracy = metrics.Accuracy()

# Preprocess Kafka message
def preprocess_message(message):
    data = message.decode('utf-8').split(',')
    weight, colour, label = map(int, data)
    X = {'Weight': weight, 'Colour': colour}
    y = label  # 1 for Apple, 2 for Orange
    return X, y

print("Starting consumer and training Hoeffding Tree...")

# Consume and train incrementally
for msg in consumer:
    X, y = preprocess_message(msg.value)
    print(f"Received: Weight={X['Weight']}, Colour={X['Colour']}, Label={y}")
    
    # Predict before learning
    y_pred = model.predict_one(X)
    if y_pred is not None:
        accuracy.update(y, y_pred)
        print(f"Prediction: {y_pred}, Accuracy: {accuracy.get():.4f}")
    else:
        print("No prediction yet (model untrained)")
    
    # Learn incrementally
    model.learn_one(X, y)
    
    # current model stats 
    print(f"Model stats: {model.summary}")