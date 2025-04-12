import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import time
from Backend.apriori_graph import parse_graph_file, apriori_graph_mining
from Backend.gsp import preprocess_sequences_ordered, gsp_algorithm

def apriori_graph_mining_app():
    st.title("Apriori-Based Graph Mining")
    uploaded_file = st.file_uploader("Upload your graph dataset file ", type=['txt'], key="apriori_file")
    if uploaded_file is not None:
        graphs = parse_graph_file(uploaded_file)
        st.write(f"Number of graphs loaded: {len(graphs)}")
        min_support = st.slider("Minimum Support", 1, len(graphs), 2, key="apriori_min_support")
        if st.button("Run Apriori Algorithm"):
            with st.spinner("Processing..."):
                tables, frequent_edge_sets = apriori_graph_mining(graphs, min_support)
                for k in range(len(tables)):
                    if tables[k]:  
                        st.header(f"{k+1}-Edge Frequent Subgraphs")
                        df_k_edge = pd.DataFrame(tables[k])
                        st.dataframe(df_k_edge)
                        if frequent_edge_sets[k]:
                            st.write(f"Frequent {k+1}-edge sub-graphs: {['[' + ', '.join([f'({e[0]}, {e[1]})' for e in edge_set]) + ']' for edge_set in frequent_edge_sets[k]]}")
                        else:
                            st.write(f"No frequent {k+1}-edge sub-graphs found.")


def gsp_algorithm_app():
    st.title("GSP Algorithm Implementation")
    st.write("This app performs sequence pattern mining using the GSP algorithm.")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="gsp_file")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File successfully uploaded and read!")
            with st.expander("View Uploaded Data"):
                st.dataframe(df)
            min_support = st.slider(
                "Select minimum support threshold (0-1)",
                min_value=0.01,
                max_value=1.0,
                value=0.4,
                step=0.01,
                key="gsp_min_support"
            )
            if st.button("Run GSP Algorithm"):
                with st.spinner("Processing..."):
                    start_time = time.time()
                    customer_sequences = preprocess_sequences_ordered(df)
                    sequences = customer_sequences['SEQUENCE'].tolist()
                    with st.expander("View Processed Sequences"):
                        st.write(sequences)
                    results = gsp_algorithm(sequences, min_support)
                    end_time = time.time()
                    st.success(f"Processing completed in {end_time - start_time:.2f} seconds!")
                    st.header("GSP Algorithm Results")
                    st.subheader("Frequent 1-Item Sequences")
                    frequent_1 = results['1_item']['frequent']
                    df_1 = pd.DataFrame([(str([set(s) for s in seq]), support) for seq, support in sorted(frequent_1, key=lambda x: str(x[0]))],
                                        columns=["Sequence", "Support"])
                    st.dataframe(df_1)
                    k = 2
                    while f'{k}_item' in results:
                        st.subheader(f"{k}-Item Sequences")
                        st.write(f"Candidate {k}-Item Sequences:")
                        candidates = results[f'{k}_item']['candidates']
                        df_candidates = pd.DataFrame([(str([set(s) for s in seq]), support) for seq, support in sorted(candidates, key=lambda x: str(x[0]))],
                                                    columns=["Sequence", "Support"])
                        st.dataframe(df_candidates)
                        st.write(f"Frequent {k}-Item Sequences:")
                        frequent_k = results[f'{k}_item']['frequent']
                        df_frequent = pd.DataFrame([(str([set(s) for s in seq]), support) for seq, support in sorted(frequent_k, key=lambda x: str(x[0]))],
                                                  columns=["Sequence", "Support"])
                        st.dataframe(df_frequent)
                        k += 1
                    st.subheader("All Frequent Sequences")
                    all_frequent = results['all_frequent']
                    df_all = pd.DataFrame([(str([set(s) for s in seq]), support) for seq, support in sorted(all_frequent, key=lambda x: (len(x[0]), str(x[0])))],
                                          columns=["Sequence", "Support"])
                    st.dataframe(df_all)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def main():
    st.sidebar.title("Algorithm Selection")
    algorithm = st.sidebar.selectbox("Choose an algorithm", ["Apriori Graph Mining", "GSP Algorithm"])
    if algorithm == "Apriori Graph Mining":
        apriori_graph_mining_app()
    elif algorithm == "GSP Algorithm":
        gsp_algorithm_app()

if __name__ == "__main__":
    main()