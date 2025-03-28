import pandas as pd
from itertools import combinations
from collections import defaultdict
from prettytable import PrettyTable
from collections import OrderedDict
from itertools import product

# Preprocessing function (unchanged)
def preprocess_sequences_ordered(df):
    df['INVOICEDATE'] = pd.to_datetime(df['INVOICEDATE'], dayfirst=True)
    df_sorted = df.sort_values(['NAME', 'INVOICEDATE'])
    transactions = df_sorted.groupby(['NAME', 'INVOICEDATE'])['PRODUCTNAME'].apply(set).reset_index()
    customer_sequences = transactions.groupby('NAME', observed=False).apply(
        lambda x: x['PRODUCTNAME'].tolist(),
        include_groups=False
    ).reset_index(name='SEQUENCE')
    return customer_sequences

# Generate 1-item sequences (unchanged)
def generate_gsp_1_itemsets(sequences, min_support_threshold):
    item_counts = {}
    itemset_counts = {}
    for sequence in sequences:
        sequence_items = set()
        sequence_itemsets = set()
        for transaction in sequence:
            for item in transaction:
                if item not in sequence_items:
                    item_counts[item] = item_counts.get(item, 0) + 1
                    sequence_items.add(item)
            if len(transaction) > 1:
                for combo_size in range(2, len(transaction) + 1):
                    for itemset in combinations(sorted(transaction), combo_size):
                        if itemset not in sequence_itemsets:
                            itemset_counts[itemset] = itemset_counts.get(itemset, 0) + 1
                            sequence_itemsets.add(itemset)
    total_sequences = len(sequences)
    candidate_sequences = {}
    for item, count in item_counts.items():
        candidate_sequences[item] = count
    for itemset, count in itemset_counts.items():
        candidate_sequences[itemset] = count
    frequent_sequences = {
        item: support for item, support in candidate_sequences.items()
        if support / total_sequences >= min_support_threshold
    }
    return candidate_sequences, frequent_sequences

# Helper function to check if a candidate is a subsequence (unchanged)
def is_subsequence(candidate, sequence):
    if not candidate:
        return True
    cand_idx = 0
    for event in sequence:
        if cand_idx >= len(candidate):
            break
        if candidate[cand_idx].issubset(event):
            cand_idx += 1
    return cand_idx == len(candidate)

# Function to generate k-item sequences (unchanged)
def generate_gsp_k_itemsets(sequences, frequent_prev, k, min_support_threshold):
    frequent_prev_sequences = [seq for seq, _ in frequent_prev]
    candidate_k_sequences = {}
    if k == 2:
        for seq1, seq2 in product(frequent_prev_sequences, repeat=2):
            candidate = seq1 + seq2
            candidate_k_sequences[tuple(candidate)] = 0
    else:
        for seq1 in frequent_prev_sequences:
            for seq2 in frequent_prev_sequences:
                if seq1[-(k-2):] == seq2[:(k-2)]:
                    candidate = seq1 + [seq2[-1]]
                    candidate_k_sequences[tuple(candidate)] = 0
    total_sequences = len(sequences)
    for sequence in sequences:
        for candidate in candidate_k_sequences:
            cand_list = list(candidate)
            if is_subsequence(cand_list, sequence):
                candidate_k_sequences[candidate] += 1
    min_support = min_support_threshold * total_sequences
    frequent_k_sequences = [
        (list(candidate), support)
        for candidate, support in candidate_k_sequences.items()
        if support >= min_support
    ]
    return candidate_k_sequences, frequent_k_sequences

# Main GSP algorithm with improved printing
def gsp_algorithm(sequences, min_support_threshold):
    # Step 1: Generate frequent 1-item sequences
    candidate_sequences, frequent_sequences = generate_gsp_1_itemsets(sequences, min_support_threshold)

    # Convert frequent 1-item sequences to GSP format
    frequent_1 = []
    for seq, support in frequent_sequences.items():
        if isinstance(seq, str):
            frequent_1.append(([frozenset([seq])], support))
        else:
            frequent_1.append(([frozenset(seq)], support))

    # Print Frequent 1-Item Sequences using PrettyTable
    print("Frequent 1-Item Sequences:")
    table_1 = PrettyTable()
    table_1.field_names = ["Sequence", "Support"]
    for seq, support in sorted(frequent_1, key=lambda x: str(x[0])):
        table_1.add_row([str([set(s) for s in seq]), support])
    print(table_1)

    all_frequent = frequent_1.copy()
    k = 2

    while True:
        print(f"\nGenerating {k}-Item Sequences:")
        frequent_prev = [seq for seq in all_frequent if len(seq[0]) == k-1]
        if not frequent_prev:
            break

        candidate_k_sequences, frequent_k_sequences = generate_gsp_k_itemsets(
            sequences, frequent_prev, k, min_support_threshold
        )

        # Print Candidate k-Item Sequences using PrettyTable
        print(f"Candidate {k}-Item Sequences:")
        table_candidates = PrettyTable()
        table_candidates.field_names = ["Sequence", "Support"]
        for seq, support in sorted(candidate_k_sequences.items(), key=lambda x: str(x[0])):
            table_candidates.add_row([str([set(s) for s in seq]), support])
        print(table_candidates)

        # Print Frequent k-Item Sequences using PrettyTable
        print(f"\nFrequent {k}-Item Sequences:")
        table_frequent = PrettyTable()
        table_frequent.field_names = ["Sequence", "Support"]
        for seq, support in sorted(frequent_k_sequences, key=lambda x: str(x[0])):
            table_frequent.add_row([str([set(s) for s in seq]), support])
        print(table_frequent)

        if not frequent_k_sequences:
            break

        all_frequent.extend(frequent_k_sequences)
        k += 1

    # Print All Frequent Sequences using PrettyTable
    print("\nAll Frequent Sequences:")
    table_all = PrettyTable()
    table_all.field_names = ["Sequence", "Support"]
    for seq, support in sorted(all_frequent, key=lambda x: (len(x[0]), str(x[0]))):
        table_all.add_row([str([set(s) for s in seq]), support])
    print(table_all)

    return all_frequent



df=pd.read_csv("mydata2.csv")
print(df)


# Get the sequences as a list for GSP
customer_sequences = preprocess_sequences_ordered(df)
sequences = customer_sequences['SEQUENCE'].tolist()

print("Sequences for GSP:")
print(sequences)
print("\n\n")

# Run the GSP algorithm with a minimum support threshold of 0.2
min_support_threshold = 0.5
all_frequent_sequences = gsp_algorithm(sequences, min_support_threshold)