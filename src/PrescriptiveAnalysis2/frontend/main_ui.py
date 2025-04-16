import streamlit as st
import subprocess

st.set_page_config(page_title="PRESCRIPTIVE ANALYSIS-2 : STREAMING DATA ANALYSIS")

st.title("PRESCRIPTIVE ANALYSIS-2 : STREAMING DATA ANALYSIS")
st.markdown("### Select a module to launch:")

option = st.selectbox("Choose Module", ["-- Select --", "Hoeffding Tree", "CluStream"])

if st.button("Run Selected Module"):
    if option == "Hoeffding Tree":
        st.success("Launching Hoeffding Tree UI...")
        subprocess.Popen(["streamlit", "run", "./src/PrescriptiveAnalysis2/frontend/hoeffding_ui.py"])
    elif option == "CluStream":
        st.success("Launching Clustream UI...")
        subprocess.Popen(["streamlit", "run", "./src/PrescriptiveAnalysis2/frontend/clustream_ui.py"])
    else:
        st.warning("Please select a valid module.")
