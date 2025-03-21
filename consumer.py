from quixstreams import Application
import json

app = Application(
        broker_address="localhost:9092",
        loglevel="DEBUG",
        consumer_group="weather_reader",
        auto_offset_reset="earliest",
)

with app.get_consumer() as consumer:
        consumer.subscribe(["weather_data_demo"])

        while True:
                msg = consumer.poll(1)

                if msg is None:
                        print("Waiting...")
                elif msg.error() is not None:
                        raise Exception(msg.error())
                elif msg.value() is None:
                        print(f"Skipping empty message at offset {msg.offset()}")
                else:
                        key = msg.key().decode('utf8')
                        value = json.loads(msg.value())
                        offset = msg.offset()
                        print(f"{offset} {key} {value}")