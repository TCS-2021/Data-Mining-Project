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

class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next_link = None

class FPTree:
    def __init__(self, transactions, min_support, total_transactions):
        self.min_support = min_support * total_transactions
        self.item_support = {}
        self.root = FPNode(None, 1, None)
        self.patterns = {}
        self.build_tree(transactions)

    def build_tree(self, transactions):
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        item_counts = {k: v for k, v in item_counts.items() if v >= self.min_support}
        if not item_counts:
            return

        sorted_items = sorted(item_counts, key=item_counts.get, reverse=True)
        self.item_support = {item: [count, None] for item, count in item_counts.items()}

        for transaction in transactions:
            filtered_items = [item for item in transaction if item in item_counts]
            filtered_items.sort(key=lambda item: item_counts[item], reverse=True)
            self.insert_transaction(filtered_items, self.root)

    def insert_transaction(self, items, parent_node):
        if not items:
            return

        first_item = items[0]
        if first_item in parent_node.children:
            parent_node.children[first_item].count += 1
        else:
            new_node = FPNode(first_item, 1, parent_node)
            parent_node.children[first_item] = new_node

            if not self.item_support[first_item][1]:
                self.item_support[first_item][1] = new_node
            else:
                last_linked_node = self.item_support[first_item][1]
                while last_linked_node.next_link:
                    last_linked_node = last_linked_node.next_link
                last_linked_node.next_link = new_node

        self.insert_transaction(items[1:], parent_node.children[first_item])

    def extract_patterns(self, suffix=frozenset(), level=1):
        for item, (count, node) in sorted(self.item_support.items(), key=lambda x: x[1][0]):
            new_suffix = suffix | frozenset([item])
            self.patterns[new_suffix] = (count, level)

            conditional_transactions = []
            while node:
                path = []
                parent = node.parent
                while parent and parent.item:
                    path.append(parent.item)
                    parent = parent.parent
                path.reverse()
                for _ in range(node.count):
                    conditional_transactions.append(path)
                node = node.next_link

            if conditional_transactions:
                conditional_tree = FPTree(conditional_transactions, self.min_support / len(conditional_transactions), len(conditional_transactions))
                conditional_tree.extract_patterns(new_suffix, level + 1)
                self.patterns.update(conditional_tree.patterns)

class FPGrowth:
    def __init__(self, transactions, min_support):
        self.transactions = transactions
        self.min_support = min_support

    @timer
    def find_frequent_patterns(self):
        tree = FPTree(self.transactions, self.min_support, len(self.transactions))
        tree.extract_patterns()
        return tree.patterns

class BusinessRuleGenerator:
    def __init__(self, frequent_patterns, transactions, min_confidence):
        self.frequent_patterns = frequent_patterns
        self.transactions = transactions
        self.min_confidence = min_confidence

    def derive_rules(self):
        rules = []
        total_transactions = len(self.transactions)
        for itemset, (freq, _) in self.frequent_patterns.items():
            if len(itemset) > 1:
                for item in itemset:
                    antecedent = itemset - {item}
                    consequent = {item}
                    confidence = self.compute_confidence(antecedent, consequent)
                    if confidence >= self.min_confidence:
                        rules.append((', '.join(sorted(antecedent)), ', '.join(sorted(consequent)), freq / total_transactions, confidence))
        return rules

    def compute_confidence(self, antecedent, consequent):
        antecedent_support = self.fetch_support(antecedent)
        if antecedent_support == 0:
            return 0
        combined_support = self.fetch_support(antecedent | consequent)
        return combined_support / antecedent_support

    def fetch_support(self, itemset):
        itemset = frozenset(itemset)
        return self.frequent_patterns.get(itemset, (0,))[0]

def run_fp_growth_analysis(df, min_support, min_confidence):
    try:
        transactions = df.groupby("INVOICENO")["PRODUCTNAME"].apply(set).tolist()
        if not transactions:
            return None, None, None, "No valid transactions found."

        fp_growth = FPGrowth(transactions, min_support)
        frequent_patterns, execution_time = fp_growth.find_frequent_patterns()

        total_transactions = len(transactions)
        data = []
        for pattern, (count, level) in frequent_patterns.items():
            support = count / total_transactions
            data.append([level, ", ".join(sorted(pattern)), support])

        itemsets_df = pd.DataFrame(data, columns=["Level", "Frequent Itemset", "Support"]) if data else pd.DataFrame()
        if not itemsets_df.empty:
            itemsets_df = itemsets_df.sort_values(by=["Level", "Support"], ascending=[True, False])

        rule_generator = BusinessRuleGenerator(frequent_patterns, transactions, min_confidence)
        rules = rule_generator.derive_rules()
        rules_df = pd.DataFrame(
            rules,
            columns=["Antecedent", "Consequent", "Support", "Confidence"]
        ) if rules else pd.DataFrame()

        return itemsets_df, rules_df, execution_time, None

    except Exception as e:
        return None, None, None, str(e)