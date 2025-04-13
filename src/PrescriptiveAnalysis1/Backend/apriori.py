import pandas as pd
from collections import defaultdict
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

class AprioriAlgorithm:
    def __init__(self, transactions, min_support):
        self.transactions = transactions
        self.min_support = min_support
        self.frequent_patterns = {}

    @timer
    def execute(self):
        self.frequent_patterns = self.find_frequent_itemsets()
        return self.frequent_patterns

    def count_item_frequencies(self, item_combinations):
        item_count = defaultdict(int)
        for data in self.transactions:
            for item_set in item_combinations:
                if item_set.issubset(data):
                    item_count[frozenset(item_set)] += 1

        valid_items = []
        total_transactions = len(self.transactions)
        for item, count in item_count.items():
            support = count / total_transactions
            if support >= self.min_support:
                valid_items.append((item, support))

        return valid_items

    def create_new_combinations(self, prev_frequent_sets, length):
        return {a.union(b) for a in prev_frequent_sets for b in prev_frequent_sets if len(a.union(b)) == length}

    def find_frequent_itemsets(self):
        freq_itemsets = {}
        unique_items = {item for transaction in self.transactions for item in transaction}
        current_candidates = [frozenset([item]) for item in unique_items]
        level = 1

        while current_candidates:
            frequent_items = self.count_item_frequencies(current_candidates)
            if not frequent_items:
                break
            freq_itemsets[level] = frequent_items
            current_candidates = self.create_new_combinations([item for item, _ in frequent_items], level + 1)
            level += 1

        return freq_itemsets

class BusinessRuleGenerator:
    def __init__(self, frequent_patterns, transactions, min_confidence):
        self.frequent_patterns = frequent_patterns
        self.transactions = transactions
        self.min_confidence = min_confidence

    def derive_rules(self):
        rules = []
        for level, itemsets in self.frequent_patterns.items():
            for itemset, support in itemsets:
                if len(itemset) > 1:
                    for item in itemset:
                        antecedent = itemset - {item}
                        consequent = {item}
                        confidence = self.compute_confidence(antecedent, consequent)
                        if confidence >= self.min_confidence:
                            rules.append((', '.join(sorted(antecedent)), ', '.join(sorted(consequent)), support, confidence))
        return rules

    def compute_confidence(self, antecedent, consequent):
        antecedent_support = self.fetch_support(antecedent)
        if antecedent_support == 0:
            return 0
        combined_support = self.fetch_support(antecedent | consequent)
        return combined_support / antecedent_support

    def fetch_support(self, itemset):
        itemset = frozenset(itemset)
        for level, itemsets in self.frequent_patterns.items():
            for stored_itemset, support in itemsets:
                if stored_itemset == itemset:
                    return support
        return 0

def run_apriori_analysis(df, min_support, min_confidence):
    try:
        transactions = df.groupby("INVOICENO")["PRODUCTNAME"].apply(set).tolist()
        if not transactions:
            return None, None, None, "No valid transactions found."

        apriori = AprioriAlgorithm(transactions, min_support)
        frequent_patterns, execution_time = apriori.execute()
        
        data = []
        for level, itemsets in frequent_patterns.items():
            for itemset, support in sorted(itemsets, key=lambda x: x[1], reverse=True):
                data.append([level, ", ".join(sorted(itemset)), support])

        itemsets_df = pd.DataFrame(data, columns=["Level", "Frequent Itemset", "Support"]) if data else pd.DataFrame()

        rule_generator = BusinessRuleGenerator(frequent_patterns, transactions, min_confidence)
        rules = rule_generator.derive_rules()
        rules_df = pd.DataFrame(
            rules,
            columns=["Antecedent", "Consequent", "Support", "Confidence"]
        ) if rules else pd.DataFrame()

        return itemsets_df, rules_df, execution_time, None

    except Exception as e:
        return None, None, None, str(e)