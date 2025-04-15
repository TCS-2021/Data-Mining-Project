import unittest
import pandas as pd
import sys

sys.path.append('../../src')  # Relative path from tests/PrescriptiveAnalysis1/ to src/
from PrescriptiveAnalysis1.Backend.gsp import preprocess_sequences_ordered, is_subsequence, gsp_algorithm

class TestGSPAlgorithm(unittest.TestCase):
    def setUp(self):
        # Sample DataFrame for testing
        data = {
            'NAME': ['Customer1', 'Customer1', 'Customer1', 'Customer2', 'Customer2', 'Customer3'],
            'INVOICEDATE': ['01/01/2025', '02/01/2025', '03/01/2025', '01/01/2025', '02/01/2025', '01/01/2025'],
            'PRODUCTNAME': ['A', 'B', 'C', 'A', 'B', 'C']
        }
        self.df = pd.DataFrame(data)
        self.sequences = preprocess_sequences_ordered(self.df)['SEQUENCE'].tolist()
        self.min_support_threshold = 0.5  # 50% (2 out of 3 customers)

    def test_preprocess_sequences_ordered_single_customer(self):
        single_df = pd.DataFrame({
            'NAME': ['Customer1', 'Customer1'],
            'INVOICEDATE': ['01/01/2025', '02/01/2025'],
            'PRODUCTNAME': ['A', 'B']
        })
        result = preprocess_sequences_ordered(single_df)
        self.assertEqual(len(result), 1)
        self.assertListEqual(result['SEQUENCE'].tolist(), [[{'A'}, {'B'}]])

    def test_is_subsequence(self):
        # Test basic subsequence
        self.assertTrue(is_subsequence([{'A'}], [{'A'}, {'B'}]))
        self.assertTrue(is_subsequence([{'A'}, {'B'}], [{'A'}, {'B'}, {'C'}]))
        # Test non-subsequence
        self.assertFalse(is_subsequence([{'B'}], [{'A'}, {'C'}]))
        # Test empty candidate
        self.assertTrue(is_subsequence([], [{'A'}, {'B'}]))
        # Test partial match
        self.assertFalse(is_subsequence([{'A'}, {'C'}], [{'A'}, {'B'}]))

    def test_gsp_algorithm_empty(self):
        results = gsp_algorithm([], self.min_support_threshold)
        self.assertEqual(results['1_item']['frequent'], [])
        self.assertNotIn('2_item', results)
        self.assertEqual(results['all_frequent'], [])

if __name__ == '__main__':
    unittest.main()