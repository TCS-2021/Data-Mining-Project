import streamlit as st
import os
import sys
from pathlib import Path

# Set page configuration first
st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add src directory and subdirectories to sys.path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

# Add each application's directory to sys.path
app_dirs = ['DescriptiveAnalysis', 'PredictiveAnalytics1', 'preprocessing', 'PrescriptiveAnalysis1']
for app_dir in app_dirs:
    app_path = src_path / app_dir
    if app_path.exists():
        sys.path.insert(0, str(app_path))
    else:
        print(f"Directory {app_dir} not found in src/. Please check the directory structure.")

# Add PredictiveAnalytics1/Frontend and PrescriptiveAnalysis1 subdirectories to sys.path
predictive_frontend_path = src_path / 'PredictiveAnalytics1' / 'Frontend'
if predictive_frontend_path.exists():
    sys.path.insert(0, str(predictive_frontend_path))
else:
    print(f"Frontend directory not found at {predictive_frontend_path}.")

prescriptive_frontend_path = src_path / 'PrescriptiveAnalysis1' / 'Frontend'
if prescriptive_frontend_path.exists():
    sys.path.insert(0, str(prescriptive_frontend_path))
else:
    print(f"Frontend directory not found at {prescriptive_frontend_path}.")

prescriptive_backend_path = src_path / 'PrescriptiveAnalysis1' / 'Backend'
if prescriptive_backend_path.exists():
    sys.path.insert(0, str(prescriptive_backend_path))
else:
    print(f"Backend directory not found at {prescriptive_backend_path}.")

# Import main functions from each application
try:
    from DescriptiveAnalysis.frontend import main as descriptive_main
except ModuleNotFoundError as e:
    print(f"Error importing DescriptiveAnalysis: {e}")
    descriptive_main = None

try:
    from PredictiveAnalytics1.Frontend.app import main as predictive_main
except ModuleNotFoundError as e:
    print(f"Error importing PredictiveAnalytics1: {e}")
    predictive_main = None

try:
    from preprocessing.app import main as preprocessing_main
except ModuleNotFoundError as e:
    print(f"Error importing preprocessing: {e}")
    preprocessing_main = None

try:
    from PrescriptiveAnalysis1.Frontend.main import main as prescriptive_main
except ModuleNotFoundError as e:
    print(f"Error importing PrescriptiveAnalysis1: {e}")
    prescriptive_main = None

# Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');

    html, body {
        font-family: 'Montserrat', sans-serif;
        color: #333;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
        border-radius: 8px;
        padding: 1rem;
    }
    .stSelectbox > div > label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
    }
    .main-container {
        background-color: #fff;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        margin: 1rem;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """
    Main function to run the Streamlit dashboard application.
    Provides a sidebar to navigate between different analytics applications.
    """
    st.title("Analytics Dashboard")

    # Display import errors, if any
    if descriptive_main is None:
        st.error("Cannot import DescriptiveAnalysis. Please check the directory structure and files.")
    if predictive_main is None:
        st.error("Cannot import PredictiveAnalytics1. Please check the directory structure and files.")
    if preprocessing_main is None:
        st.error("Cannot import preprocessing. Please check the directory structure and files.")
    if prescriptive_main is None:
        st.error("Cannot import PrescriptiveAnalysis1. Please check the directory structure and files.")

    # Sidebar for application selection
    st.sidebar.title("Navigation")
    app_options = [
        "Descriptive Analysis",
        "Predictive Analytics",
        "Preprocessing",
        "Prescriptive Analysis"
    ]
    selected_app = st.sidebar.selectbox("Select Application", app_options)

    # Map selected app to the corresponding main function
    app_functions = {
        "Descriptive Analysis": descriptive_main,
        "Predictive Analytics": predictive_main,
        "Preprocessing": preprocessing_main,
        "Prescriptive Analysis": prescriptive_main
    }

    # Run the selected application's main function
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        selected_function = app_functions[selected_app]
        if selected_function is None:
            st.error(f"Cannot run {selected_app}. The module could not be imported. Please check the error messages above.")
        else:
            selected_function()
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()