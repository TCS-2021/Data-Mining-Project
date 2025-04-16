import streamlit as st
import subprocess
import time
import os
import glob
import shutil
from PIL import Image
import pandas as pd

# Configuration
KAFKA_HOME = "C:/kafka_2.13-3.9.0"  # Update this path if needed
TOPIC_NAME = "parkingstream"

# Paths (keep relative as in consumer code)
DATA_DIR = './src/PrescriptiveAnalysis2/streamlit_data/'
PLOT_DIR = os.path.join(DATA_DIR, 'plots')
EVENTS_DIR = os.path.join(DATA_DIR, 'events')

# Create directories if they don't exist
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)

# Kafka commands
ZK_CMD = f'{KAFKA_HOME}/bin/windows/zookeeper-server-start.bat {KAFKA_HOME}/config/zookeeper.properties'
KAFKA_CMD = f'{KAFKA_HOME}/bin/windows/kafka-server-start.bat {KAFKA_HOME}/config/server.properties'
TOPIC_CMD = f'{KAFKA_HOME}/bin/windows/kafka-topics.bat --create --topic {TOPIC_NAME} --bootstrap-server localhost:9092 --replication-factor 1 --partitions 4'

# Streamlit UI setup
st.set_page_config(page_title="CluStream Clustering", layout="wide")
st.title("CluStream Clustering")

# Global variables to track processes
processes = {
    'zookeeper': None,
    'kafka_server': None,
    'producer': None,
    'consumer': None
}

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
if "last_plot_refresh" not in st.session_state:
    st.session_state.last_plot_refresh = time.time()
if "last_data_refresh" not in st.session_state:
    st.session_state.last_data_refresh = time.time()

def clear_directories():
    """Clear contents of both PLOT_DIR and EVENTS_DIR"""
    try:
        # Clear plot directory
        for filename in os.listdir(PLOT_DIR):
            file_path = os.path.join(PLOT_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                st.error(f'Failed to delete {file_path}. Reason: {e}')
        
        # Clear events directory
        for filename in os.listdir(EVENTS_DIR):
            file_path = os.path.join(EVENTS_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                st.error(f'Failed to delete {file_path}. Reason: {e}')
        
        st.success("Cleared all plot and event data files")
    except Exception as e:
        st.error(f"Error clearing directories: {e}")

def launch_background(command, process_key=None):
    """Launch a background process and store reference if needed"""
    if isinstance(command, list):
        process = subprocess.Popen(command)
    else:
        process = subprocess.Popen(command, shell=True)
    if process_key:
        processes[process_key] = process
    return process

def start_zookeeper():
    """Start Zookeeper service"""
    st.info("Starting Zookeeper...")
    processes['zookeeper'] = launch_background(ZK_CMD)
    time.sleep(5)
    st.success("Zookeeper started successfully")

def start_kafka_server():
    """Start Kafka server"""
    st.info("Starting Kafka Server...")
    processes['kafka_server'] = launch_background(KAFKA_CMD)
    time.sleep(10)
    st.success("Kafka Server started successfully")

def create_topic():
    """Create Kafka topic"""
    st.info("Creating Kafka topic...")
    try:
        result = subprocess.run(TOPIC_CMD, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            st.success(f"Topic '{TOPIC_NAME}' created successfully")
        else:
            st.warning(f"Topic creation may have failed: {result.stderr}")
    except Exception as e:
        st.warning(f"Topic creation may have failed: {e}")
    time.sleep(2)

def start_producer():
    """Start the producer script"""
    st.info("Starting Kafka Producer...")
    processes['producer'] = launch_background(['python', 'src/PrescriptiveAnalysis2/backend/producer_clu.py'])
    st.success("Producer started successfully")

def start_consumer():
    """Start the consumer script"""
    st.info("Starting Kafka Consumer...")
    processes['consumer'] = launch_background(['python', 'src/PrescriptiveAnalysis2/backend/consumer_clu.py'])
    st.success("Consumer started successfully")

def stop_services():
    """Stop all running services and clear data directories"""
    st.session_state.running = False
    
    # First stop all services
    for name, process in processes.items():
        if process:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                pass
            processes[name] = None
    # Then clear the data directories
    clear_directories()
    
    st.success("All services stopped and data cleared successfully")

def start_all_services():
    """Start the complete pipeline"""
    if not st.session_state.running:
        st.session_state.running = True
        start_zookeeper()
        start_kafka_server()
        create_topic()
        start_producer()
        start_consumer()

# UI Controls
st.subheader("")
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Full Pipeline", disabled=st.session_state.running):
        start_all_services()
with col2:
    if st.button("Stop Pipeline", disabled=not st.session_state.running):
        stop_services()

# Layout containers
col1, col2 = st.columns([2, 3])

# Event data display container
with col1:
    st.subheader("📈 Recent Events")
    events_placeholder = st.empty()

# Cluster visualization container
with col2:
    st.subheader("📊 CluStream Cluster Visualization")
    plot_placeholder = st.empty()

def get_recent_event_data():
    """Get the most recent event data from CSV files"""
    try:
        csv_files = glob.glob(os.path.join(EVENTS_DIR, "*.csv"))
        if not csv_files:
            st.warning("No event CSV files found in ./streamlit_data/events/")
            return None
        csv_files.sort(key=os.path.getmtime, reverse=True)
        df = pd.read_csv(csv_files[0])
        return df
    except Exception as e:
        st.error(f"Error loading event data: {e}")
        return None

def get_latest_plot_image():
    """Get the most recent cluster visualization image"""
    try:
        png_files = glob.glob(os.path.join(PLOT_DIR, "*.png"))
        if not png_files:
            st.warning("No plot PNG files found in ./streamlit_data/plots/")
            return None
        png_files.sort(key=os.path.getmtime, reverse=True)
        return Image.open(png_files[0])
    except Exception as e:
        st.error(f"Error loading plot image: {e}")
        return None

def update_status():
    """Update pipeline status in sidebar"""
    status_text = f"""
    - Zookeeper: {'✅ Running' if processes['zookeeper'] and processes['zookeeper'].poll() is None else '❌ Stopped'}
    - Kafka Server: {'✅ Running' if processes['kafka_server'] and processes['kafka_server'].poll() is None else '❌ Stopped'}
    - Producer: {'✅ Running' if processes['producer'] and processes['producer'].poll() is None else '❌ Stopped'}
    - Consumer: {'✅ Running' if processes['consumer'] and processes['consumer'].poll() is None else '❌ Stopped'}
    """
    st.sidebar.markdown(status_text)

# Sidebar for status
st.sidebar.subheader("Pipeline Status")

# Main UI logic
if st.session_state.running:
    # Update UI when pipeline is running
    update_status()
    
    # Update event data
    df = get_recent_event_data()
    if df is not None:
        events_placeholder.dataframe(df, use_container_width=True)
    else:
        events_placeholder.warning("Waiting for event data...")
    
    # Update cluster visualization
    img = get_latest_plot_image()
    if img:
        plot_placeholder.image(img, use_column_width=True)
    else:
        plot_placeholder.warning("Waiting for cluster visualization...")
    
    time.sleep(2)  # Update every 2 seconds
    st.rerun()  # Refresh UI
else:
    # Display placeholders when pipeline is not running
    update_status()
    events_placeholder.info("Start the pipeline to see recent events.")
    plot_placeholder.info("Start the pipeline to see the cluster visualization.")