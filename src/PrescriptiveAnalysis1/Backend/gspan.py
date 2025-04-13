import json
from collections import defaultdict

def load_graphs_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            graphs = json.load(file)
        graphs = {str(key): {str(v): neighbors for v, neighbors in graph.items()}
                 for key, graph in graphs.items()}
        return graphs
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def construct_dfs_code(graph, start_vertex, directed=True):
    visited = set()
    discovery_order = {}
    dfs_tree_edges = []
    back_edges = []
    vertex_index = [0]

    def dfs(vertex):
        if vertex in visited:
            return
        discovery_order[vertex] = vertex_index[0]
        visited.add(vertex)
        vertex_index[0] += 1
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                dfs_tree_edges.append((vertex, neighbor))
                dfs(neighbor)
            elif (neighbor, vertex) not in dfs_tree_edges and vertex != neighbor:
                if discovery_order.get(neighbor, float('inf')) < discovery_order.get(vertex, float('inf')):
                    if directed or neighbor < vertex:
                        back_edges.append((vertex, neighbor))

    dfs(start_vertex)
    dfs_code = [(discovery_order[u], discovery_order[v], u, 1, v) for u, v in dfs_tree_edges]
    dfs_code.extend([(discovery_order[u], discovery_order[v], u, 1, v) for u, v in back_edges])
    return dfs_code, discovery_order

def normalize_edge(label_i, label_j, is_forward, directed):
    if directed:
        return (label_i, label_j, is_forward)
    if label_i < label_j:
        return (label_i, label_j, True)
    return (label_j, label_i, True)

def is_subgraph_present(subgraph_edges, graph_code, directed):
    if directed:
        graph_edges = {(li, lj, i < j) for i, j, li, el, lj in graph_code}
        subgraph_edges_set = {(li, lj, is_forward) for li, lj, is_forward in subgraph_edges}
    else:
        graph_edges = {frozenset([li, lj]) for i, j, li, el, lj in graph_code}
        subgraph_edges_set = {frozenset([li, lj]) for li, lj, _ in subgraph_edges}
    return subgraph_edges_set.issubset(graph_edges)

def enumerate_subgraphs(graphs, min_support=2, directed=True):
    dfs_codes = {}
    for graph_name, graph in graphs.items():
        dfs_code, discovery_order = construct_dfs_code(graph, 'A', directed)
        dfs_codes[graph_name] = dfs_code

    edge_support = defaultdict(list)
    for graph_name, dfs_code in dfs_codes.items():
        for edge in dfs_code:
            i, j, label_i, edge_label, label_j = edge
            is_forward = i < j
            edge_key = normalize_edge(label_i, label_j, is_forward, directed)
            if graph_name not in edge_support[edge_key]:
                edge_support[edge_key].append(graph_name)

    frequent_subgraphs = defaultdict(dict)
    infrequent_subgraphs = defaultdict(dict)
    seen_structures = set()

    def grow_subgraph(current_edges, graph_codes):
        edge_set = frozenset({(li, lj, is_f) if directed else frozenset([li, lj]) for li, lj, is_f in current_edges})
        if edge_set in seen_structures:
            return
        support = sum(1 for g_code in graph_codes.values() if is_subgraph_present(current_edges, g_code, directed))
        size = len(current_edges)
        edge_str = ", ".join([f"({li}-{lj})" if (is_f or not directed) else f"({lj}-{li})" for li, lj, is_f in current_edges])

        if support >= min_support:
            frequent_subgraphs[size][edge_str] = (current_edges, support, "forward")
            seen_structures.add(edge_set)
        else:
            infrequent_subgraphs[size][edge_str] = (current_edges, support, "forward")
            return

        vertices = set()
        for li, lj, _ in current_edges:
            vertices.add(li)
            vertices.add(lj)

        for graph_name, graph_code in graph_codes.items():
            if not is_subgraph_present(current_edges, graph_code, directed):
                continue
            for next_i, next_j, nlabel_i, nedge_label, nlabel_j in graph_code:
                next_is_forward = next_i < next_j
                if not next_is_forward:
                    continue
                next_edge = normalize_edge(nlabel_i, nlabel_j, next_is_forward, directed)
                if (next_edge not in current_edges and
                    (nlabel_i in vertices or nlabel_j in vertices)):
                    new_edges = current_edges + [next_edge]
                    grow_subgraph(new_edges, graph_codes)

    for edge_key, graph_list in edge_support.items():
        label_i, label_j, is_forward = edge_key
        edges = [(label_i, label_j, is_forward)]
        support = len(graph_list)
        edge_str = f"({label_i}-{label_j})" if (is_forward or not directed) else f"({label_j}-{label_i})"
        size = 1
        if support >= min_support:
            frequent_subgraphs[size][edge_str] = (edges, support, "forward" if is_forward or not directed else "backward")
            if is_forward or not directed:
                grow_subgraph(edges, dfs_codes)
        else:
            infrequent_subgraphs[size][edge_str] = (edges, support, "forward" if is_forward or not directed else "backward")
    return frequent_subgraphs, infrequent_subgraphs, dfs_codes

def run_gspan_analysis(graphs, min_support=2, directed=True):
    frequent_subgraphs, _, dfs_codes = enumerate_subgraphs(graphs, min_support, directed)
    
    result_tables = []
    frequent_edge_sets = []
    for size in sorted(frequent_subgraphs.keys()):
        table_k_edge = []
        frequent_k_edge = []
        for edge_str, (edges, support, _) in frequent_subgraphs[size].items():
            entry = {
                'Edge Pairs': edge_str,
                'Support': support,
                'Qualify': 'Y'
            }
            for i, graph_name in enumerate(graphs.keys()):
                entry[f'Graph {i+1}'] = 'Y' if is_subgraph_present(edges, dfs_codes[graph_name], directed) else 'N'
            table_k_edge.append(entry)
            frequent_k_edge.append(edges)
        result_tables.append(table_k_edge)
        frequent_edge_sets.append(frequent_k_edge)
    
    return result_tables, frequent_edge_sets
