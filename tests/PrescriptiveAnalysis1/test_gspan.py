import unittest
import json
import os
import sys

sys.path.append('../../src')  # Relative path from tests/PrescriptiveAnalysis1/ to src/
from PrescriptiveAnalysis1.Backend.gspan import load_graphs_from_json, construct_dfs_code, normalize_edge, is_subgraph_present, enumerate_subgraphs, run_gspan_analysis


class TestGSpan(unittest.TestCase):
    def setUp(self):
        self.test_json_content = {
            "G1": {
                "A": ["B", "C"],
                "B": ["A"],
                "C": ["A", "D"],
                "D": ["C", "A"]
            },
            "G2": {
                "A": ["B", "C"],
                "B": ["A", "D"],
                "C": ["A", "E"],
                "D": ["B"],
                "E": ["C"]
            },
            "G3": {
                "A": ["B", "C"],
                "B": ["A", "D"],
                "C": ["D", "A"],
                "D": ["B", "C"]
            }
        }
        self.test_json_file = "test_gspan_graphs.json"
        with open(self.test_json_file, 'w') as f:
            json.dump(self.test_json_content, f)

        self.graphs = load_graphs_from_json(self.test_json_file)
        self.directed = True
        self.min_support = 2

    def tearDown(self):
        if os.path.exists(self.test_json_file):
            os.remove(self.test_json_file)

    def test_load_graphs_from_json(self):
        graphs = load_graphs_from_json(self.test_json_file)
        self.assertIsNotNone(graphs)
        self.assertEqual(len(graphs), 3)
        self.assertIn("G1", graphs)
        self.assertIn("G2", graphs)
        self.assertIn("G3", graphs)
        self.assertEqual(set(graphs["G1"].keys()), {"A", "B", "C", "D"})
        self.assertEqual(graphs["G1"]["A"], ["B", "C"])

    def test_load_graphs_from_json_invalid_file(self):
        result = load_graphs_from_json("non_existent.json")
        self.assertIsNone(result)

    def test_load_graphs_from_json_invalid_json(self):
        with open("invalid.json", "w") as f:
            f.write("invalid json")
        result = load_graphs_from_json("invalid.json")
        self.assertIsNone(result)
        os.remove("invalid.json")

    def test_construct_dfs_code(self):
        graph = self.graphs["G1"]
        dfs_code, discovery_order = construct_dfs_code(graph, "A", directed=True)
        self.assertTrue(dfs_code)
        self.assertTrue(discovery_order)
        self.assertEqual(len(discovery_order), len(graph))
        for code in dfs_code:
            self.assertEqual(len(code), 5)
            self.assertIn(code[2], graph.keys())
            self.assertIn(code[4], graph.keys())
            self.assertEqual(code[3], 1)

    def test_normalize_edge_directed(self):
        edge = normalize_edge("A", "B", True, directed=True)
        self.assertEqual(edge, ("A", "B", True))
        edge = normalize_edge("B", "A", False, directed=True)
        self.assertEqual(edge, ("B", "A", False))

    def test_normalize_edge_undirected(self):
        edge = normalize_edge("A", "B", True, directed=False)
        self.assertEqual(edge, ("A", "B", True))
        edge = normalize_edge("B", "A", False, directed=False)
        self.assertEqual(edge, ("A", "B", True))

    def test_is_subgraph_present_directed(self):
        dfs_code, _ = construct_dfs_code(self.graphs["G1"], "A", directed=True)
        subgraph_edges = [("A", "B", True), ("A", "C", True)]
        self.assertTrue(is_subgraph_present(subgraph_edges, dfs_code, directed=True))
        subgraph_edges = [("A", "E", True)]
        self.assertFalse(is_subgraph_present(subgraph_edges, dfs_code, directed=True))

    def test_is_subgraph_present_undirected(self):
        dfs_code, _ = construct_dfs_code(self.graphs["G1"], "A", directed=False)
        subgraph_edges = [("A", "B", True), ("A", "C", True)]
        self.assertTrue(is_subgraph_present(subgraph_edges, dfs_code, directed=False))
        subgraph_edges = [("A", "E", True)]
        self.assertFalse(is_subgraph_present(subgraph_edges, dfs_code, directed=False))

    def test_enumerate_subgraphs_directed(self):
        frequent_subgraphs, infrequent_subgraphs, dfs_codes = enumerate_subgraphs(self.graphs, self.min_support, directed=True)
        self.assertTrue(frequent_subgraphs)
        self.assertTrue(dfs_codes)
        for size, subgraphs in frequent_subgraphs.items():
            for edge_str, (edges, support, _) in subgraphs.items():
                self.assertGreaterEqual(support, self.min_support)
                supporting_graphs = [g for g, code in dfs_codes.items() if is_subgraph_present(edges, code, directed=True)]
                self.assertEqual(len(supporting_graphs), support)
        self.assertIn("(A-B)", frequent_subgraphs[1])
        self.assertIn("(A-C)", frequent_subgraphs[1])
        self.assertEqual(frequent_subgraphs[1]["(A-B)"][1], 3)
        

    def test_enumerate_subgraphs_undirected(self):
        frequent_subgraphs, infrequent_subgraphs, dfs_codes = enumerate_subgraphs(self.graphs, self.min_support, directed=False)
        self.assertTrue(frequent_subgraphs)
        self.assertIn("(A-B)", frequent_subgraphs[1])
        self.assertNotIn("(B-A)", frequent_subgraphs[1])

    def test_run_gspan_analysis(self):
        result_tables, frequent_edge_sets = run_gspan_analysis(self.graphs, self.min_support, directed=True)
        self.assertTrue(result_tables)
        self.assertTrue(frequent_edge_sets)
        for table in result_tables:
            for entry in table:
                self.assertIn("Edge Pairs", entry)
                self.assertIn("Support", entry)
                self.assertIn("Qualify", entry)
                self.assertIn("Graph 1", entry)
                self.assertIn("Graph 2", entry)
                self.assertIn("Graph 3", entry)
                self.assertEqual(entry["Qualify"], "Y")
                self.assertGreaterEqual(entry["Support"], self.min_support)
        found_ab = False
        for table in result_tables:
            for entry in table:
                if entry["Edge Pairs"] == "(A-B)":
                    found_ab = True
                    self.assertEqual(entry["Support"], 3)
                    self.assertEqual(entry["Graph 1"], "Y")
                    self.assertEqual(entry["Graph 2"], "Y")
                    self.assertEqual(entry["Graph 3"], "Y")
        self.assertTrue(found_ab)

    def test_run_gspan_analysis_high_min_support(self):
        result_tables, frequent_edge_sets = run_gspan_analysis(self.graphs, min_support=4, directed=True)
        self.assertEqual(result_tables, [])
        self.assertEqual(frequent_edge_sets, [])

if __name__ == '__main__':
    unittest.main()