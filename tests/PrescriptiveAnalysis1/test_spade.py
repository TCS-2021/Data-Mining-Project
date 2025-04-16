import unittest
import pandas as pd
import sys
import os
from collections import defaultdict

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.PrescriptiveAnalysis1.Backend.spade import (
    preprocess_data_vertical,
    get_transaction_table,
    create_idlists,
    format_pattern,
    get_pattern_length,
    run_spade_analysis
)

class TestSPADE(unittest.TestCase):
    def setUp(self):
        # Load sample data for testing
        data = {
            'NAME': [1, 1, 1, 1, 2, 2, 3, 4, 4, 4],
            'INVOICEDATE': ['1/1/2025', '1/3/2025', '1/4/2025', '1/4/2025', '1/1/2025', '1/1/2025', '1/1/2025', '1/2/2025', '1/2/2025', '1/3/2025'],
            'PRODUCTNAME': ['C,D', 'A,B,C', 'A,B,F', 'A,C,D,F', 'A,B,F', 'E', 'A,B,F', 'D,H,G', 'B,F', 'A,G,H']
        }
        self.df = pd.DataFrame(data)
        self.min_support = 0.5  # 50% (2 out of 4 sequences)
        # Preprocessed vertical format for use in tests
        self.vertical_df, _ = preprocess_data_vertical(self.df)

    def test_get_transaction_table(self):
        transactions_df, error = get_transaction_table(self.vertical_df)
        self.assertIsNone(error)
        self.assertIsNotNone(transactions_df)
        self.assertEqual(list(transactions_df.columns), ['Customer ID (SID)', 'Event ID (EID)', 'Items'])
        self.assertGreater(len(transactions_df), 0)  # Ensure non-empty

    def test_create_idlists(self):
        idlists, error = create_idlists(self.vertical_df)
        self.assertIsNone(error)
        self.assertIsInstance(idlists, defaultdict)
        # Check some items
        self.assertIn('A', idlists)
        self.assertIn('B', idlists)

    def test_format_pattern(self):
        self.assertEqual(format_pattern(frozenset(['A', 'B'])), '{A, B}')
        self.assertEqual(format_pattern(('A', 'B')), '<A -> B>')
        self.assertEqual(format_pattern(frozenset(['C'])), '{C}')

    def test_get_pattern_length(self):
        self.assertEqual(get_pattern_length(frozenset(['A', 'B'])), 2)
        self.assertEqual(get_pattern_length(('A', 'B')), 2)
        self.assertEqual(get_pattern_length(frozenset(['C'])), 1)

    def test_run_spade_analysis(self):
        transactions_df, detailed_results, all_frequent_df, error = run_spade_analysis(self.df, self.min_support)
        self.assertIsNone(error)
        self.assertIsNotNone(transactions_df)
        self.assertIsNotNone(detailed_results)
        self.assertIsNotNone(all_frequent_df)
        # Check basic structure
        self.assertEqual(list(all_frequent_df.columns), ['Pattern', 'Support', 'Pattern Type', 'Length'])
        self.assertGreater(len(all_frequent_df), 0)  # Ensure non-empty results

if __name__ == '__main__':
    unittest.main()