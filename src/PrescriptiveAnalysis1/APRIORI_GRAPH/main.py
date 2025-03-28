import streamlit as st
import pandas as pd
from itertools import combinations

def parse_graph_file(file):
    graphs = []
    current_graph = []
    for line in file:
        line = line.decode('utf-8').strip() if isinstance(line, bytes) else line.strip()
        if not line:
            continue
        if line.startswith('#'):
            if current_graph:
                graphs.append(current_graph)
            current_graph = []
        else:
            edge = tuple(sorted(line.split()))
            current_graph.append(edge)
    if current_graph:
        graphs.append(current_graph)
    return graphs

def get_all_edges(graphs):
    all_edges = set()
    for graph in graphs:
        for edge in graph:
            all_edges.add(edge)
    return sorted(list(all_edges))

def compute_support(edge_set, graphs):
    support = 0
    for graph in graphs:
        if all(edge in graph for edge in edge_set):
            support += 1
    return support

def apriori_graph_mining(graphs, min_support):
    num_graphs = len(graphs)
    tables = []  # tables for each k
    frequent_edge_sets = []  # Frequent edge sets for each k

    # Frequent 1-edge subgraphs
    all_edges = get_all_edges(graphs)
    frequent_1_edge = []
    table_1_edge = []
    for edge in all_edges:
        support = compute_support([edge], graphs)
        qualifies = 'Y' if support >= min_support else 'N'
        entry = {'Edge': f"({edge[0]}, {edge[1]})"}
        for i in range(num_graphs):
            entry[f'Graph {i+1}'] = 'Y' if edge in graphs[i] else 'N'
        entry['Support'] = support
        entry['Qualify'] = qualifies
        table_1_edge.append(entry)
        if support >= min_support:
            frequent_1_edge.append([edge])
    tables.append(table_1_edge)
    frequent_edge_sets.append(frequent_1_edge)

    # Frequent k-edge subgraphs for k >= 2
    k = 2
    while frequent_edge_sets[-1]:  # as long as there are frequent (k-1)-edge subgraphs
        frequent_k_minus_1 = frequent_edge_sets[-1]
        frequent_1_edges = [edge_set[0] for edge_set in frequent_edge_sets[0]]  
        candidate_k_edge = set()  
        table_k_edge = []

        # Special case for k=2: Generate all pairs of frequent 1-edges
        if k == 2:
            for edge1, edge2 in combinations(frequent_1_edges, 2):
                new_edge_set = [edge1, edge2]
                edge_set_tuple = tuple(sorted(new_edge_set))
                if edge_set_tuple in candidate_k_edge:
                    continue
                candidate_k_edge.add(edge_set_tuple)
        else:
            # For k > 2, combine (k-1)-edge subgraphs with 1-edges
            for edge_set in frequent_k_minus_1:
                for edge in frequent_1_edges:

                    if edge in edge_set:
                        continue
                    new_edge_set = edge_set + [edge]

                    edge_set_tuple = tuple(sorted(new_edge_set))
                    if edge_set_tuple in candidate_k_edge:
                        continue
                    candidate_k_edge.add(edge_set_tuple)

        for edge_set_tuple in candidate_k_edge:
            edge_set = list(edge_set_tuple)
            support = compute_support(edge_set, graphs)
            qualifies = 'Y' if support >= min_support else 'N'
            edge_pairs_str = ' '.join([f"({e[0]}, {e[1]})" for e in edge_set])
            entry = {'Edge Pairs': edge_pairs_str}
            for i in range(num_graphs):
                entry[f'Graph {i+1}'] = 'Y' if all(e in graphs[i] for e in edge_set) else 'N'
            entry['Support'] = support
            entry['Qualify'] = qualifies
            table_k_edge.append(entry)

        # Filtering frequent k-edge subgraphs
        frequent_k_edge = [list(edge_set) for edge_set in candidate_k_edge if compute_support(list(edge_set), graphs) >= min_support]
        tables.append(table_k_edge)
        frequent_edge_sets.append(frequent_k_edge)
        k += 1

    return tables, frequent_edge_sets

def apriori_graph_mining_app():
    st.title("Apriori-Based Graph Mining")

    uploaded_file = st.file_uploader("Upload your graph dataset file ", type=['txt'])

    if uploaded_file is not None:
        graphs = parse_graph_file(uploaded_file)
        st.write(f"Number of graphs loaded: {len(graphs)}")

        min_support = st.slider("Minimum Support", 1, len(graphs), 2)

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
    else:
        st.write("Please upload a graph dataset file to proceed.")


if __name__ == "__main__":
    apriori_graph_mining_app()