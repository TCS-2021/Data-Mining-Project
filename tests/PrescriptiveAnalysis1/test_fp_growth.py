import unittest
import pandas as pd
import sys
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from src.PrescriptiveAnalysis1.Backend.fp_growth import FPNode, FPTree, FPGrowth, BusinessRuleGenerator, run_fp_growth_analysis


class TestFPGrowth(unittest.TestCase):
    def setUp(self):
        # Sample transactions for testing
        self.transactions = [
            {'A', 'B', 'C'},
            {'A', 'B'},
            {'B', 'C'},
            {'A', 'C'},
            {'A', 'B', 'C', 'D'}
        ]
        self.min_support = 0.4  # 40% (2 out of 5 transactions)
        self.min_confidence = 0.5
        # Sample DataFrame for run_fp_growth_analysis
        data = {
            'INVOICENO': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5],
            'PRODUCTNAME': ['A', 'B', 'C', 'A', 'B', 'B', 'C', 'A', 'C', 'A', 'B', 'C', 'D']
        }
        self.df = pd.DataFrame(data)

    def test_fp_node_initialization(self):
        node = FPNode('A', 2, None)
        self.assertEqual(node.item, 'A')
        self.assertEqual(node.count, 2)
        self.assertIsNone(node.parent)
        self.assertEqual(node.children, {})
        self.assertIsNone(node.next_link)

    def test_fp_tree_build(self):
        tree = FPTree(self.transactions, self.min_support, len(self.transactions))
        self.assertIsNotNone(tree.root)
        self.assertEqual(tree.root.item, None)
        self.assertTrue(tree.item_support)  # Ensure item_support is populated
        # Check if frequent items meet min_support (2 transactions)
        expected_items = {'A', 'B', 'C'}  # D should be excluded (appears in 1 transaction)
        self.assertEqual(set(tree.item_support.keys()), expected_items)

    def test_fp_tree_insert_transaction(self):
        tree = FPTree([], self.min_support, 5)  # Empty tree
        tree.item_support = {'A': [3, None], 'B': [2, None]}
        transaction = ['A', 'B']
        tree.insert_transaction(transaction, tree.root)
        # Check if nodes were created
        self.assertIn('A', tree.root.children)
        self.assertIn('B', tree.root.children['A'].children)
        # Check counts
        self.assertEqual(tree.root.children['A'].count, 1)
        self.assertEqual(tree.root.children['A'].children['B'].count, 1)
        # Check header table linkage
        self.assertIsNotNone(tree.item_support['A'][1])
        self.assertIsNotNone(tree.item_support['B'][1])

    def test_business_rule_generator(self):
        fp_growth = FPGrowth(self.transactions, self.min_support)
        patterns, _ = fp_growth.find_frequent_patterns()
        rule_generator = BusinessRuleGenerator(patterns, self.transactions, self.min_confidence)
        rules = rule_generator.derive_rules()
        self.assertTrue(rules)  # Ensure rules are generated
        # Check a sample rule, e.g., {A, B} => {C}
        for antecedent, consequent, support, confidence in rules:
            if antecedent == 'A, B' and consequent == 'C':
                self.assertGreaterEqual(confidence, self.min_confidence)
                self.assertAlmostEqual(support, 2/5)  # {A, B, C} appears in 2 transactions

    def test_run_fp_growth_analysis(self):
        itemsets_df, rules_df, execution_time, error = run_fp_growth_analysis(
            self.df, self.min_support, self.min_confidence
        )
        self.assertIsNone(error)
        self.assertIsNotNone(itemsets_df)
        self.assertIsNotNone(rules_df)
        self.assertGreaterEqual(execution_time, 0)  # Modified to allow zero
        # Check if itemsets_df has expected columns
        self.assertEqual(list(itemsets_df.columns), ['Level', 'Frequent Itemset', 'Support'])
        # Check if rules_df has expected columns
        self.assertEqual(list(rules_df.columns), ['Antecedent', 'Consequent', 'Support', 'Confidence'])
        # Verify some frequent itemsets
        self.assertTrue(any('A, B' in itemset for itemset in itemsets_df['Frequent Itemset']))

    def test_empty_transactions(self):
        df = pd.DataFrame({'INVOICENO': [], 'PRODUCTNAME': []})
        itemsets_df, rules_df, execution_time, error = run_fp_growth_analysis(
            df, self.min_support, self.min_confidence
        )
        self.assertEqual(error, "No valid transactions found.")
        self.assertIsNone(itemsets_df)
        self.assertIsNone(rules_df)
        self.assertIsNone(execution_time)

    def test_low_support(self):
        fp_growth = FPGrowth(self.transactions, 0.9)
        patterns, _ = fp_growth.find_frequent_patterns()
        self.assertEqual(patterns, {})  # No patterns should be found

if __name__ == '__main__':
    unittest.main()