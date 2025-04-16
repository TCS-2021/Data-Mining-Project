import unittest
import pandas as pd
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.append('../../src')  # Relative path from tests/PrescriptiveAnalysis1/ to src/
from src.PrescriptiveAnalysis1.Backend.apriori import AprioriAlgorithm, BusinessRuleGenerator, run_apriori_analysis

class TestApriori(unittest.TestCase):
    def setUp(self):
        # Sample transactional data
        self.transactions = [
            {'A', 'B', 'C'},
            {'A', 'B'},
            {'B', 'C'},
            {'A', 'C'},
            {'A', 'B', 'D'}
        ]
        self.min_support = 0.4  # 40% (2 out of 5 transactions)
        self.min_confidence = 0.5
        # Sample DataFrame for run_apriori_analysis
        data = {
            'INVOICENO': [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5],
            'PRODUCTNAME': ['A', 'B', 'C', 'A', 'B', 'B', 'C', 'A', 'C', 'A', 'B', 'D']
        }
        self.df = pd.DataFrame(data)

    def test_apriori_algorithm_initialization(self):
        apriori = AprioriAlgorithm(self.transactions, self.min_support)
        self.assertEqual(apriori.transactions, self.transactions)
        self.assertEqual(apriori.min_support, self.min_support)
        self.assertEqual(apriori.frequent_patterns, {})

    def test_count_item_frequencies(self):
        apriori = AprioriAlgorithm(self.transactions, self.min_support)
        candidates = [frozenset({'A'}), frozenset({'B'}), frozenset({'C'}), frozenset({'D'})]
        frequent_items = apriori.count_item_frequencies(candidates)
        expected = [
            (frozenset({'A'}), 4/5),
            (frozenset({'B'}), 4/5),
            (frozenset({'C'}), 3/5),
        ]
        self.assertEqual(len(frequent_items), 3)  # D has support 1/5 < 0.4
        for item, support in frequent_items:
            self.assertTrue((item, support) in expected)

    def test_create_new_combinations(self):
        apriori = AprioriAlgorithm(self.transactions, self.min_support)
        prev_frequent = [frozenset({'A'}), frozenset({'B'}), frozenset({'C'})]
        new_combinations = apriori.create_new_combinations(prev_frequent, 2)
        expected = {frozenset({'A', 'B'}), frozenset({'A', 'C'}), frozenset({'B', 'C'})}
        self.assertEqual(new_combinations, expected)

    def test_find_frequent_itemsets(self):
        apriori = AprioriAlgorithm(self.transactions, self.min_support)
        frequent_patterns = apriori.find_frequent_itemsets()
        self.assertIn(1, frequent_patterns)
        self.assertIn(2, frequent_patterns)
        # Level 1: A, B, C
        level_1 = frequent_patterns[1]
        self.assertEqual(len(level_1), 3)
        expected_1 = {frozenset({'A'}), frozenset({'B'}), frozenset({'C'})}
        self.assertTrue(all(item in [x[0] for x in level_1] for item in expected_1))
        # Level 2: A,B; A,C; B,C
        level_2 = frequent_patterns[2]
        self.assertEqual(len(level_2), 3)
        expected_2 = {frozenset({'A', 'B'}), frozenset({'A', 'C'}), frozenset({'B', 'C'})}
        self.assertTrue(all(item in [x[0] for x in level_2] for item in expected_2))

    def test_execute(self):
        apriori = AprioriAlgorithm(self.transactions, self.min_support)
        patterns, execution_time = apriori.execute()
        self.assertEqual(patterns, apriori.frequent_patterns)
        self.assertGreaterEqual(execution_time, 0)
        self.assertIn(1, patterns)
        self.assertIn(2, patterns)
        self.assertEqual(len(patterns[1]), 3)  # A, B, C
        self.assertEqual(len(patterns[2]), 3)  # A,B; A,C; B,C

    def test_business_rule_generator(self):
        apriori = AprioriAlgorithm(self.transactions, self.min_support)
        frequent_patterns = apriori.find_frequent_itemsets()
        rule_generator = BusinessRuleGenerator(frequent_patterns, self.transactions, self.min_confidence)
        rules = rule_generator.derive_rules()
        self.assertTrue(rules)
        # Check a sample rule: A => B
        for antecedent, consequent, support, confidence in rules:
            if antecedent == 'A' and consequent == 'B':
                self.assertAlmostEqual(support, 3/5)  # A,B appears in 3 transactions
                self.assertAlmostEqual(confidence, (3/5) / (4/5))  # Support(A,B) / Support(A)
                self.assertGreaterEqual(confidence, self.min_confidence)

    def test_compute_confidence(self):
        apriori = AprioriAlgorithm(self.transactions, self.min_support)
        frequent_patterns = apriori.find_frequent_itemsets()
        rule_generator = BusinessRuleGenerator(frequent_patterns, self.transactions, self.min_confidence)
        confidence = rule_generator.compute_confidence(frozenset({'A'}), frozenset({'B'}))
        self.assertAlmostEqual(confidence, (3/5) / (4/5))  # Support(A,B) / Support(A)
        confidence = rule_generator.compute_confidence(frozenset({'D'}), frozenset({'A'}))
        self.assertEqual(confidence, 0)  # D not frequent

    def test_fetch_support(self):
        apriori = AprioriAlgorithm(self.transactions, self.min_support)
        frequent_patterns = apriori.find_frequent_itemsets()
        rule_generator = BusinessRuleGenerator(frequent_patterns, self.transactions, self.min_confidence)
        support = rule_generator.fetch_support(frozenset({'A', 'B'}))
        self.assertAlmostEqual(support, 3/5)
        support = rule_generator.fetch_support(frozenset({'A', 'D'}))
        self.assertEqual(support, 0)  # A,D not frequent

    def test_run_apriori_analysis(self):
        itemsets_df, rules_df, execution_time, error = run_apriori_analysis(self.df, self.min_support, self.min_confidence)
        self.assertIsNone(error)
        self.assertIsNotNone(itemsets_df)
        self.assertIsNotNone(rules_df)
        self.assertGreaterEqual(execution_time, 0)
        # Check DataFrame columns
        self.assertEqual(list(itemsets_df.columns), ['Level', 'Frequent Itemset', 'Support'])
        self.assertEqual(list(rules_df.columns), ['Antecedent', 'Consequent', 'Support', 'Confidence'])
        # Verify some frequent itemsets
        self.assertTrue(any('A, B' in itemset for itemset in itemsets_df['Frequent Itemset']))
        # Verify a rule
        self.assertTrue(any((row['Antecedent'] == 'A') & (row['Consequent'] == 'B') 
                           for _, row in rules_df.iterrows()))

    def test_run_apriori_analysis_empty(self):
        empty_df = pd.DataFrame({'INVOICENO': [], 'PRODUCTNAME': []})
        itemsets_df, rules_df, execution_time, error = run_apriori_analysis(empty_df, self.min_support, self.min_confidence)
        self.assertEqual(error, "No valid transactions found.")
        self.assertIsNone(itemsets_df)
        self.assertIsNone(rules_df)
        self.assertIsNone(execution_time)

    def test_run_apriori_analysis_high_support(self):
        apriori = AprioriAlgorithm(self.transactions, 0.9)
        patterns = apriori.find_frequent_itemsets()
        self.assertEqual(patterns, {})  # No itemsets with support >= 0.9

if __name__ == '__main__':
    unittest.main()