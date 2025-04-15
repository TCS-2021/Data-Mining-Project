import streamlit as st
import subprocess
import threading
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from glob import glob
from PIL import Image
from datetime import datetime

# Configuration
KAFKA_HOME = "C:/kafka_2.13-3.9.0"  # Update this path if needed
TOPIC_NAME = "parkingstream"
#VISUAL_PATH = './visualisations/'
PLOT_PATH = os.path.join('plot.png')
#os.makedirs(VISUAL_PATH, exist_ok=True)

# Kafka commands
ZK_CMD = f'{KAFKA_HOME}/bin/windows/zookeeper-server-start.bat {KAFKA_HOME}/config/zookeeper.properties'
KAFKA_CMD = f'{KAFKA_HOME}/bin/windows/kafka-server-start.bat {KAFKA_HOME}/config/server.properties'
TOPIC_CMD = f'{KAFKA_HOME}/bin/windows/kafka-topics.bat --create --topic {TOPIC_NAME} --bootstrap-server localhost:9092 --replication-factor 1 --partitions 4'

# Streamlit UI setup
st.set_page_config(page_title="CluStream Kafka Visualizer", layout="wide")
st.title("📊 Real-Time CluStream via Kafka")

# Global process holders
processes = {
    'zookeeper': None,
    'kafka_server': None,
    'producer': None,
    'consumer': None
}

# Initialize session state
if "event_data" not in st.session_state:
    st.session_state.event_data = []
if "running" not in st.session_state:
    st.session_state.running = False

def launch_background(command, process_key=None):
    process = subprocess.Popen(command, shell=True)
    if process_key:
        processes[process_key] = process
    return process

def start_zookeeper():
    st.info("Starting Zookeeper...")
    processes['zookeeper'] = launch_background(ZK_CMD)
    time.sleep(5)
    st.success("Zookeeper started successfully")

def start_kafka_server():
    st.info("Starting Kafka Server...")
    processes['kafka_server'] = launch_background(KAFKA_CMD)
    time.sleep(10)
    st.success("Kafka Server started successfully")

def create_topic():
    st.info("Creating Kafka topic...")
    subprocess.call(TOPIC_CMD, shell=True)
    time.sleep(2)
    st.success(f"Topic '{TOPIC_NAME}' created successfully")

def start_producer():
    st.info("Starting Kafka Producer...")
    processes['producer'] = launch_background(['python', 'producer_clu.py'])
    st.success("Producer started successfully")

def start_consumer():
    st.info("Starting Kafka Consumer...")
    processes['consumer'] = launch_background(['python', 'consumer_clu.py'])
    st.success("Consumer started successfully")
        
    st.title("📊 Live CluStream Plot Viewer")

    plot_path = 'plot.png'
    placeholder = st.empty()

    while True:
        if os.path.exists(plot_path):
            image = Image.open(plot_path)
            placeholder.image(image, caption='Updated CluStream Plot', use_column_width=True)
        else:
            placeholder.warning("Waiting for plot.png...")
        time.sleep(5)

def stop_services():
    st.session_state.running = False
    for name, process in processes.items():
        if process:
            try:
                process.terminate()
            except:
                pass
            processes[name] = None
    st.success("All services stopped successfully")

def start_all_services():
    st.session_state.running = True
    start_zookeeper()
    start_kafka_server()
    create_topic()
    start_producer()
    start_consumer()

# UI Controls
st.subheader("🚀 Pipeline Controls")
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Full Pipeline", disabled=st.session_state.running):
        start_all_services()
with col2:
    if st.button("Stop Pipeline", disabled=not st.session_state.running):
        stop_services()


def update_ui():
    last_plot_time = None
    if os.path.exists(PLOT_PATH):
            current_time = os.path.getmtime(PLOT_PATH)
            if current_time != last_plot_time:
                try:
                    img = Image.open(PLOT_PATH)
                    plot_placeholder.image(img, use_column_width=True)
                    last_plot_time = current_time
                except:
                    pass

    time.sleep(2)

# Background thread
threading.Thread(target=update_ui, daemon=True).start()

# Status Sidebar
st.sidebar.subheader("Pipeline Status")
st.sidebar.write(f"Zookeeper: {'✅ Running' if processes['zookeeper'] else '❌ Stopped'}")
st.sidebar.write(f"Kafka Server: {'✅ Running' if processes['kafka_server'] else '❌ Stopped'}")
st.sidebar.write(f"Producer: {'✅ Running' if processes['producer'] else '❌ Stopped'}")
st.sidebar.write(f"Consumer: {'✅ Running' if processes['consumer'] else '❌ Stopped'}")
