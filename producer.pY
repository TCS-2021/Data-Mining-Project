
import requests
from quixstreams import Application
import json



response = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": 11,
        "longitude": 77,
        "current": "temperature_2m",
        "timeout":20,
        "verify":False
})

#fetched data
weather = response.json()

print(weather)

'''
app = Application(
        broker_address="localhost:9092",
        loglevel="DEBUG",
)

#gurantees cleanup with block
with app.get_producer() as producer:
        producer.produce(
                topic="weather_data_demo",
                key="India",
                value=json.dumps(weather),
)
        '''