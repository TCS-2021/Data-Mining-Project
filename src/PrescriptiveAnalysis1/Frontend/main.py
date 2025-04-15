import streamlit as st
import sys
import os
import pandas as pd
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Backend.gspan import run_gspan_analysis, construct_dfs_code, load_graphs_from_json
from Backend.apriori_graph import parse_graph_file, apriori_graph_mining
from Backend.gsp import preprocess_sequences_ordered, gsp_algorithm
from Backend.apriori import run_apriori_analysis
from Backend.fp_growth import run_fp_growth_analysis
from Backend.spade import preprocess_data_vertical, get_transaction_table, run_spade_analysis, format_pattern, get_pattern_length

def apriori_graph_mining_app():
    st.title("Apriori-Based Graph Mining")
    uploaded_file = st.file_uploader("Upload your graph dataset file", type=['txt'], key="apriori_file")
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
                    customer_sequences = preprocess_sequences_ordered(df)
                    sequences = customer_sequences['SEQUENCE'].tolist()
                    with st.expander("View Processed Sequences"):
                        st.write(sequences)
                    results = gsp_algorithm(sequences, min_support)
                    st.success("Processing completed!")
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

def gspan_algorithm_app():
    st.title("gSpan Algorithm Implementation")
    uploaded_file = st.file_uploader("Upload your JSON graph dataset file", type=['json'], key="gspan_file")
    if uploaded_file is not None:
        temp_file_path = "temp_graphs.json"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        graphs_dict = load_graphs_from_json(temp_file_path)
        
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        if graphs_dict is not None:
            min_support = st.slider("Minimum Support", 1, len(graphs_dict), 2, key="gspan_min_support")
            if st.button("Run gSpan Algorithm"):
                with st.spinner("Processing..."):
                    st.header("DFS Codes for Each Graph")
                    all_dfs_codes = {}
                    for graph_name, graph in graphs_dict.items():
                        dfs_code, discovery_order = construct_dfs_code(graph, 'A', directed=True)
                        all_dfs_codes[graph_name] = dfs_code
                        st.subheader(f"DFS Code for {graph_name}")
                        if dfs_code:
                            df_dfs = pd.DataFrame(dfs_code, columns=["i", "j", "label_i", "edge_label", "label_j"])
                            st.dataframe(df_dfs)
                        else:
                            st.write("No DFS code generated for this graph.")
                        st.write(f"Discovery Order: {discovery_order}")

                    tables, frequent_edge_sets = run_gspan_analysis(graphs_dict, min_support, directed=True)
                    for k in range(len(tables)):
                        if tables[k]:
                            st.header(f"{k+1}-Edge Frequent Subgraphs")
                            df_k_edge = pd.DataFrame(tables[k])
                            st.dataframe(df_k_edge)
                            if frequent_edge_sets[k]:
                                edge_sets_str = ['[' + ', '.join([f'({e[0]}-{e[1]})' for e in edge_set]) + ']' for edge_set in frequent_edge_sets[k]]
                                st.write(f"Frequent {k+1}-edge sub-graphs: {edge_sets_str}")
                            else:
                                st.write(f"No frequent {k+1}-edge sub-graphs found.")
        else:
            st.error("Failed to load graphs from the uploaded file.")

def apriori_algorithm_app():
    st.title("Apriori Algorithm Implementation")
    st.write("This app performs frequent itemset mining and rule generation using the Apriori algorithm.")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="apriori_csv_file")
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
                value=0.02,
                step=0.01,
                key="apriori_min_support"
            )
            min_confidence = st.slider(
                "Select minimum confidence threshold (0-1)",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="apriori_min_confidence"
            )
            if st.button("Run Apriori Algorithm"):
                with st.spinner("Processing..."):
                    itemsets_df, rules_df, execution_time, error = run_apriori_analysis(df, min_support, min_confidence)
                    if error:
                        st.error(f"Error: {error}")
                    else:
                        st.success("Processing completed!")
                        if not itemsets_df.empty:
                            st.header("Frequent Itemsets")
                            for level in sorted(itemsets_df["Level"].unique()):
                                st.subheader(f"Level {level} Frequent Itemsets")
                                level_df = itemsets_df[itemsets_df["Level"] == level][["Frequent Itemset", "Support"]]
                                st.dataframe(level_df)
                                st.write(f"Number of {level}-itemsets: {len(level_df)}")
                        else:
                            st.write("No frequent itemsets found.")
                        if not rules_df.empty:
                            st.header("Association Rules")
                            st.dataframe(rules_df)
                            st.write(f"Number of rules: {len(rules_df)}")
                        else:
                            st.write("No association rules generated.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def fp_growth_algorithm_app():
    st.title("FP-Growth Algorithm Implementation")
    st.write("This app performs frequent itemset mining and rule generation using the FP-Growth algorithm.")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="fp_growth_csv_file")
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
                value=0.02,
                step=0.01,
                key="fp_growth_min_support"
            )
            min_confidence = st.slider(
                "Select minimum confidence threshold (0-1)",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="fp_growth_min_confidence"
            )
            if st.button("Run FP-Growth Algorithm"):
                with st.spinner("Processing..."):
                    itemsets_df, rules_df, execution_time, error = run_fp_growth_analysis(df, min_support, min_confidence)
                    if error:
                        st.error(f"Error: {error}")
                    else:
                        st.success("Processing completed!")
                        if not itemsets_df.empty:
                            st.header("Frequent Itemsets")
                            for level in sorted(itemsets_df["Level"].unique()):
                                st.subheader(f"Level {level} Frequent Itemsets")
                                level_df = itemsets_df[itemsets_df["Level"] == level][["Frequent Itemset", "Support"]]
                                st.dataframe(level_df)
                                st.write(f"Number of {level}-itemsets: {len(level_df)}")
                        else:
                            st.write("No frequent itemsets found.")
                        if not rules_df.empty:
                            st.header("Association Rules")
                            st.dataframe(rules_df)
                            st.write(f"Number of rules: {len(rules_df)}")
                        else:
                            st.write("No association rules generated.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def spade_algorithm_app():
    st.title("SPADE Algorithm Implementation")
    st.write("This app performs sequential pattern mining using the SPADE algorithm.")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="spade_file")
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
                value=0.5,
                step=0.01,
                key="spade_min_support"
            )
            
            if st.button("Run SPADE Algorithm"):
                with st.spinner("Processing..."):
                    transactions_df, detailed_results, all_frequent_df, error = run_spade_analysis(df, min_support)
                    if error:
                        st.error(f"Error: {error}")
                    else:
                        st.success("Processing completed!")
                        
                        # Display vertical format sample
                        if "vertical_format_sample" in detailed_results:
                            st.header("Vertical Format Sample")
                            st.dataframe(detailed_results["vertical_format_sample"])
                        
                        # Display transaction table
                        if transactions_df is not None and not transactions_df.empty:
                            st.header("Transaction Table")
                            st.dataframe(transactions_df)
                            st.write(f"Total unique sequences (customers): {detailed_results['total_sequences']}")
                            st.write(f"Minimum support threshold: {detailed_results['min_support']}")
                        
                        # Display Frequent 1-Sequences
                        if "frequent_1" in detailed_results:
                            st.header("SPADE Algorithm Results")
                            st.subheader("Frequent 1-Sequences")
                            st.dataframe(detailed_results["frequent_1"])
                        
                        # Display each level of candidate and frequent sequences
                        for k, candidates_df in detailed_results.get("candidates", []):
                            st.subheader(f"Generating {k}-Sequences")
                            st.write(f"Candidate {k}-Sequences:")
                            st.dataframe(candidates_df)
                            
                            # Find the corresponding frequent sequences for this k
                            frequent_df = next((df for level, df in detailed_results.get("frequent", []) if level == k), None)
                            if frequent_df is not None:
                                st.write(f"Frequent {k}-Sequences:")
                                st.dataframe(frequent_df)
                        
                        # Display all frequent sequences
                        if not all_frequent_df.empty:
                            st.subheader("All Frequent Sequences (Ordered by Length)")
                            st.dataframe(all_frequent_df)
                        else:
                            st.write("No frequent sequences found.")
                            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def main():
    st.sidebar.title("Algorithm Selection")
    algorithm = st.sidebar.selectbox("Choose an algorithm", ["Apriori Algorithm", "FP-Growth Algorithm", "SPADE Algorithm", "Apriori Graph Mining", "GSP Algorithm", "gSpan Algorithm"])
    if algorithm == "Apriori Algorithm":
        apriori_algorithm_app()
    elif algorithm == "FP-Growth Algorithm":
        fp_growth_algorithm_app()
    elif algorithm == "SPADE Algorithm":
        spade_algorithm_app()
    elif algorithm == "Apriori Graph Mining":
        apriori_graph_mining_app()
    elif algorithm == "GSP Algorithm":
        gsp_algorithm_app()
    elif algorithm == "gSpan Algorithm":
        gspan_algorithm_app()

if __name__ == "__main__":
    main()