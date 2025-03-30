from kafka import KafkaProducer
import time
import random

producer = KafkaProducer(bootstrap_servers='localhost:9092')

# Sample data from the document
#weight -> 1 - light, 2 - medium, 3- heavy
#color -> 1 - green, 2 - orange, 3 - red
#label -> 1 - apple, 2 - orange, 

# 3 - cherry, 4 - jackfruit, 5 - amla
data_samples = [
    {"Weight": 1, "Colour": 1, "Label": 1},  
    {"Weight": 2, "Colour": 3, "Label": 1},  
    {"Weight": 3, "Colour": 2, "Label": 2},  
    {"Weight": 1, "Colour": 3, "Label": 1},  
    {"Weight": 2, "Colour": 2, "Label": 2}, 
    {"Weight": 3, "Colour": 2, "Label": 2},  
    {"Weight": 1, "Colour": 2, "Label": 2},  
    {"Weight": 3, "Colour": 1, "Label": 1},
    {"Weight": 2, "Colour": 1, "Label": 1},
    {"Weight": 3, "Colour": 3, "Label": 1}
]

print("Starting producer...")

for sample in data_samples:
    message = f"{sample['Weight']},{sample['Colour']},{sample['Label']}".encode('utf-8')
    
    producer.send('fruit', message)
    print(f"Sent: Weight={sample['Weight']}, Colour={sample['Colour']}, Label={sample['Label']}")
    print("\n")
    time.sleep(6)

producer.flush()