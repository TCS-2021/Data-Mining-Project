import unittest
import pandas as pd
import sys
import os
from collections import defaultdict


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from src.PrescriptiveAnalysis1.Backend.spade import (
    preprocess_data_vertical,
    get_transaction_table,
    create_idlists,
    calculate_support,
    join_idlists,
    generate_candidate_k_sequences,
    filter_frequent_sequences,
    format_pattern,
    get_pattern_length,
    run_spade_analysis
)

class TestSPADE(unittest.TestCase):
    def setUp(self):
        # Load example2.csv data
        data = {
            'NAME': [1, 1, 1, 1, 2, 2, 3, 4, 4, 4],
            'INVOICEDATE': ['1/1/2025', '1/3/2025', '1/4/2025', '1/4/2025', '1/1/2025', '1/1/2025', '1/1/2025', '1/2/2025', '1/2/2025', '1/3/2025'],
            'PRODUCTNAME': ['C,D', 'A,B,C', 'A,B,F', 'A,C,D,F', 'A,B,F', 'E', 'A,B,F', 'D,H,G', 'B,F', 'A,G,H']
        }
        self.df = pd.DataFrame(data)
        self.min_support = 0.5  # 50% (2 out of 4 sequences)
        # Preprocessed vertical format for use in tests
        self.vertical_df, _ = preprocess_data_vertical(self.df)
        self.total_sequences = self.vertical_df['SID'].nunique() if self.vertical_df is not None else 0

    def test_get_transaction_table(self):
        transactions_df, error = get_transaction_table(self.vertical_df)
        self.assertIsNone(error)
        self.assertIsNotNone(transactions_df)
        self.assertEqual(list(transactions_df.columns), ['Customer ID (SID)', 'Event ID (EID)', 'Items'])
        self.assertEqual(len(transactions_df), 10)  # 4 for SID=1, 2 for SID=2, 1 for SID=3, 3 for SID=4
        # Verify a transaction
        sid_1_eid_1 = transactions_df[(transactions_df['Customer ID (SID)'] == 1) & (transactions_df['Event ID (EID)'] == 1)]
        self.assertEqual(sid_1_eid_1['Items'].iloc[0], 'C, D')

    def test_create_idlists(self):
        idlists, error = create_idlists(self.vertical_df)
        self.assertIsNone(error)
        self.assertIsInstance(idlists, defaultdict)
        # Check some items
        self.assertIn('A', idlists)
        self.assertIn('B', idlists)
        # Verify A's ID-list
        expected_a = [(1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 3)]
        self.assertEqual(sorted(idlists['A']), sorted(expected_a))

    def test_calculate_support(self):
        idlists, _ = create_idlists(self.vertical_df)
        support = calculate_support(idlists['A'], self.total_sequences)
        self.assertAlmostEqual(support, 4/4)  # A appears in all 4 sequences
        support = calculate_support(idlists['E'], self.total_sequences)
        self.assertAlmostEqual(support, 1/4)  # E appears in 1 sequence
        support = calculate_support([], self.total_sequences)
        self.assertEqual(support, 0)

    def test_join_idlists_itemset(self):
        idlists, _ = create_idlists(self.vertical_df)
        result = join_idlists(idlists['A'], idlists['B'], join_type='itemset')
        # A and B in same EID: SID=1 (EID=2,3), SID=2 (EID=1), SID=3 (EID=1)
        expected = [(1, 2), (1, 3), (2, 1), (3, 1)]
        self.assertEqual(sorted(result), sorted(expected))

    def test_generate_candidate_k_sequences_k2(self):
        idlists, _ = create_idlists(self.vertical_df)
        frequent_1 = [(frozenset(['A']), 4), (frozenset(['B']), 4), (frozenset(['C']), 2), (frozenset(['F']), 3)]
        candidates, error = generate_candidate_k_sequences(frequent_1, 2, idlists)
        self.assertIsNone(error)
        self.assertTrue(candidates)
        # Check some candidates
        patterns = [pattern for pattern, _ in candidates]
        # Itemset: {A,B}
        self.assertIn(frozenset(['A', 'B']), patterns)
        # Sequence: <A->B>
        self.assertIn(('A', 'B'), patterns)
        # Verify A,B itemset support
        for pattern, idlist in candidates:
            if pattern == frozenset(['A', 'B']):
                self.assertEqual(sorted(idlist), [(1, 2), (1, 3), (2, 1), (3, 1)])

    def test_filter_frequent_sequences(self):
        idlists, _ = create_idlists(self.vertical_df)
        candidates = [(frozenset(['A', 'B']), [(1, 2), (1, 3), (2, 1), (3, 1)]),
                      (('A', 'B'), [(1, 3), (1, 4)])]
        frequent, error = filter_frequent_sequences(candidates, self.min_support, self.total_sequences)
        self.assertIsNone(error)
        self.assertEqual(len(frequent), 1)  # Only {A,B} has support >= 0.5 (3/4)
        self.assertEqual(frequent[0][0], frozenset(['A', 'B']))
        self.assertAlmostEqual(frequent[0][1], 3)

    def test_format_pattern(self):
        self.assertEqual(format_pattern(frozenset(['A', 'B'])), '{A, B}')
        self.assertEqual(format_pattern(('A', 'B')), '<A -> B>')
        self.assertEqual(format_pattern(frozenset(['C'])), '{C}')

    def test_get_pattern_length(self):
        self.assertEqual(get_pattern_length(frozenset(['A', 'B'])), 2)
        self.assertEqual(get_pattern_length(('A', 'B')), 2)
        self.assertEqual(get_pattern_length(frozenset(['C'])), 1)

    def test_run_spade_analysis(self):
        transactions_df, results, all_frequent_df, error = run_spade_analysis(self.df, self.min_support)
        self.assertIsNone(error)
        self.assertIsNotNone(transactions_df)
        self.assertIsNotNone(results)
        self.assertIsNotNone(all_frequent_df)
        # Check transaction table
        self.assertEqual(len(transactions_df), 10)
        # Check frequent 1-sequences
        frequent_1, candidates_all, all_frequent = results
        self.assertEqual(len(frequent_1), 4)  # A, B, C, F
        # Check all_frequent_df
        self.assertEqual(list(all_frequent_df.columns), ['Pattern', 'Support', 'Pattern Type', 'Length'])
        self.assertTrue('{A, B}' in all_frequent_df['Pattern'].values)
        # Verify support for {A}
        a_row = all_frequent_df[all_frequent_df['Pattern'] == '{A}']
        self.assertAlmostEqual(a_row['Support'].iloc[0], 4)

if __name__ == '__main__':
    unittest.main()