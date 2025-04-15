import unittest
import io
import sys

sys.path.append('../../src')  # Relative path from tests/PrescriptiveAnalysis1/ to src/
from PrescriptiveAnalysis1.Backend.apriori_graph import parse_graph_file, get_all_edges, compute_support, apriori_graph_mining

class TestAprioriGraph(unittest.TestCase):
    def setUp(self):
        self.graph_data = """
# Graph 1
A B
B C
A D
B E
C E
C F
# Graph 2
A B
B C
A D
B E
# Graph 3
A C
C D
B E
E F
A F
"""
        # Create a file-like object
        self.graph_file = io.BytesIO(self.graph_data.encode('utf-8'))
        
        # Parse graphs for use in tests
        self.graph_file.seek(0)
        self.graphs = parse_graph_file(self.graph_file)
        
        # Expected unique edges (sorted tuples)
        self.expected_edges = [
            ('A', 'B'), ('A', 'C'), ('A', 'D'), ('A', 'F'),
            ('B', 'C'), ('B', 'E'), ('C', 'D'), ('C', 'E'),
            ('C', 'F'), ('E', 'F')
        ]

    def test_parse_graph_file(self):
        self.graph_file.seek(0)
        graphs = parse_graph_file(self.graph_file)
        self.assertEqual(len(graphs), 3)
        # Graph 1: {A-B, B-C, A-D, B-E, C-E, C-F}
        self.assertEqual(set(graphs[0]), {
            ('A', 'B'), ('B', 'C'), ('A', 'D'), ('B', 'E'), ('C', 'E'), ('C', 'F')
        })
        # Graph 2: {A-B, B-C, A-D, B-E}
        self.assertEqual(set(graphs[1]), {
            ('A', 'B'), ('B', 'C'), ('A', 'D'), ('B', 'E')
        })
        # Graph 3: {A-C, C-D, B-E, E-F, A-F}
        self.assertEqual(set(graphs[2]), {
            ('A', 'C'), ('C', 'D'), ('B', 'E'), ('E', 'F'), ('A', 'F')
        })

    def test_parse_graph_file_empty(self):
        empty_file = io.BytesIO(b"")
        graphs = parse_graph_file(empty_file)
        self.assertEqual(graphs, [])

    def test_parse_graph_file_single_graph(self):
        single_graph_data = """
# Graph 1
A B
B C
"""
        single_file = io.BytesIO(single_graph_data.encode('utf-8'))
        graphs = parse_graph_file(single_file)
        self.assertEqual(len(graphs), 1)
        self.assertEqual(set(graphs[0]), {('A', 'B'), ('B', 'C')})

    def test_get_all_edges(self):
        edges = get_all_edges(self.graphs)
        self.assertEqual(edges, self.expected_edges)
        self.assertEqual(len(edges), 10)

    def test_get_all_edges_empty(self):
        edges = get_all_edges([])
        self.assertEqual(edges, [])

    def test_compute_support(self):
        # Single edge support
        self.assertEqual(compute_support([('A', 'B')], self.graphs), 2)  # G1, G2
        self.assertEqual(compute_support([('B', 'E')], self.graphs), 3)  # G1, G2, G3
        self.assertEqual(compute_support([('A', 'F')], self.graphs), 1)  # G3
        # Multi-edge support
        self.assertEqual(compute_support([('A', 'B'), ('B', 'C')], self.graphs), 2)  # G1, G2
        self.assertEqual(compute_support([('A', 'C'), ('C', 'D')], self.graphs), 1)  # G3
        self.assertEqual(compute_support([('A', 'B'), ('B', 'E'), ('A', 'D')], self.graphs), 2)  # G1, G2

    def test_compute_support_empty_graphs(self):
        support = compute_support([('A', 'B')], [])
        self.assertEqual(support, 0)

    def test_apriori_graph_mining_min_support_2(self):
        tables, frequent_edge_sets = apriori_graph_mining(self.graphs, min_support=2)
        self.assertTrue(len(tables) >= 3)  # At least k=1, k=2, k=3
        self.assertTrue(len(frequent_edge_sets) >= 3)

        # k=1 table
        table_1 = tables[0]
        self.assertEqual(len(table_1), 10)  # All 10 edges
        expected_edges = {
            '(A, B)': {'support': 2, 'graphs': [0, 1]},
            '(A, C)': {'support': 1, 'graphs': [2]},
            '(A, D)': {'support': 2, 'graphs': [0, 1]},
            '(A, F)': {'support': 1, 'graphs': [2]},
            '(B, C)': {'support': 2, 'graphs': [0, 1]},
            '(B, E)': {'support': 3, 'graphs': [0, 1, 2]},
            '(C, D)': {'support': 1, 'graphs': [2]},
            '(C, E)': {'support': 1, 'graphs': [0]},
            '(C, F)': {'support': 1, 'graphs': [0]},
            '(E, F)': {'support': 1, 'graphs': [2]}
        }
        for entry in table_1:
            edge = entry['Edge']
            self.assertIn(edge, expected_edges)
            self.assertEqual(entry['Support'], expected_edges[edge]['support'])
            self.assertEqual(entry['Qualify'], 'Y' if expected_edges[edge]['support'] >= 2 else 'N')
            for i in range(3):
                expected = 'Y' if i in expected_edges[edge]['graphs'] else 'N'
                self.assertEqual(entry[f'Graph {i+1}'], expected)

        # k=1 frequent edge sets
        self.assertEqual(len(frequent_edge_sets[0]), 4)  # (A,B), (A,D), (B,C), (B,E)
        expected_frequent_1 = [[('A', 'B')], [('A', 'D')], [('B', 'C')], [('B', 'E')]]
        self.assertTrue(all(edge_set in frequent_edge_sets[0] for edge_set in expected_frequent_1))

        # k=2 table
        table_2 = tables[1]
        expected_k2 = {
            '(A, B) (A, D)': {'support': 2, 'graphs': [0, 1]},
            '(A, B) (B, C)': {'support': 2, 'graphs': [0, 1]},
            '(A, B) (B, E)': {'support': 2, 'graphs': [0, 1]},
            '(A, D) (B, C)': {'support': 2, 'graphs': [0, 1]},
            '(A, D) (B, E)': {'support': 2, 'graphs': [0, 1]},
            '(B, C) (B, E)': {'support': 2, 'graphs': [0, 1]}
        }
        self.assertEqual(len(table_2), len(expected_k2))
        for entry in table_2:
            edge_pairs = entry['Edge Pairs']
            self.assertIn(edge_pairs, expected_k2)
            self.assertEqual(entry['Support'], expected_k2[edge_pairs]['support'])
            self.assertEqual(entry['Qualify'], 'Y')
            for i in range(3):
                expected = 'Y' if i in expected_k2[edge_pairs]['graphs'] else 'N'
                self.assertEqual(entry[f'Graph {i+1}'], expected)

        # k=2 frequent edge sets
        self.assertEqual(len(frequent_edge_sets[1]), 6)
        expected_frequent_2 = [
            [('A', 'B'), ('A', 'D')],
            [('A', 'B'), ('B', 'C')],
            [('A', 'B'), ('B', 'E')],
            [('A', 'D'), ('B', 'C')],
            [('A', 'D'), ('B', 'E')],
            [('B', 'C'), ('B', 'E')]
        ]
        self.assertTrue(all(sorted(edge_set) in [sorted(es) for es in frequent_edge_sets[1]] for edge_set in expected_frequent_2))

        # k=3 table
        table_3 = tables[2]
        expected_k3 = {
            '(A, B) (A, D) (B, C)': {'support': 2, 'graphs': [0, 1]},
            '(A, B) (A, D) (B, E)': {'support': 2, 'graphs': [0, 1]},
            '(A, B) (B, C) (B, E)': {'support': 2, 'graphs': [0, 1]},
            '(A, D) (B, C) (B, E)': {'support': 2, 'graphs': [0, 1]}
        }
        self.assertEqual(len(table_3), len(expected_k3))
        for entry in table_3:
            edge_pairs = entry['Edge Pairs']
            self.assertIn(edge_pairs, expected_k3)
            self.assertEqual(entry['Support'], expected_k3[edge_pairs]['support'])
            self.assertEqual(entry['Qualify'], 'Y')
            for i in range(3):
                expected = 'Y' if i in expected_k3[edge_pairs]['graphs'] else 'N'
                self.assertEqual(entry[f'Graph {i+1}'], expected)

        # k=3 frequent edge sets
        self.assertEqual(len(frequent_edge_sets[2]), 4)
        expected_frequent_3 = [
            [('A', 'B'), ('A', 'D'), ('B', 'C')],
            [('A', 'B'), ('A', 'D'), ('B', 'E')],
            [('A', 'B'), ('B', 'C'), ('B', 'E')],
            [('A', 'D'), ('B', 'C'), ('B', 'E')]
        ]
        self.assertTrue(all(sorted(edge_set) in [sorted(es) for es in frequent_edge_sets[2]] for edge_set in expected_frequent_3))

    def test_apriori_graph_mining_min_support_3(self):
        tables, frequent_edge_sets = apriori_graph_mining(self.graphs, min_support=3)
        self.assertEqual(len(tables), 2)  # k=1, k=2 (k=2 is empty)
        self.assertEqual(len(frequent_edge_sets), 2)
        # k=1: Only (B,E) has support 3
        table_1 = tables[0]
        self.assertEqual(len(frequent_edge_sets[0]), 1)
        self.assertEqual(frequent_edge_sets[0], [[('B', 'E')]])
        for entry in table_1:
            if entry['Edge'] == '(B, E)':
                self.assertEqual(entry['Support'], 3)
                self.assertEqual(entry['Qualify'], 'Y')
                self.assertEqual(entry['Graph 1'], 'Y')
                self.assertEqual(entry['Graph 2'], 'Y')
                self.assertEqual(entry['Graph 3'], 'Y')
            else:
                self.assertEqual(entry['Qualify'], 'N')
        # k=2: Empty
        self.assertEqual(frequent_edge_sets[1], [])

    def test_apriori_graph_mining_empty_graphs(self):
        tables, frequent_edge_sets = apriori_graph_mining([], min_support=2)
        self.assertEqual(tables, [[]])
        self.assertEqual(frequent_edge_sets, [[]])

if __name__ == '__main__':
    unittest.main()