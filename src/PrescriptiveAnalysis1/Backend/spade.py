import pandas as pd
from collections import defaultdict

def preprocess_data_vertical(df):
    """
    Convert horizontal data format to vertical format (SID, EID, item).
    SID = Sequence ID (customer ID)
    EID = Event ID (timestamp/order of events)
    """
    try:
        # Convert dates to datetime
        try:
            df['INVOICEDATE'] = pd.to_datetime(df['INVOICEDATE'], errors='coerce')
        except:
            df['INVOICEDATE'] = pd.to_datetime(df['INVOICEDATE'], errors='coerce', dayfirst=True)
        
        df_sorted = df.sort_values(['NAME', 'INVOICEDATE'])
        df_sorted['EID'] = df_sorted.groupby('NAME').cumcount() + 1

        vertical_format = []
        for _, row in df_sorted.iterrows():
            if isinstance(row['PRODUCTNAME'], str) and ',' in row['PRODUCTNAME']:
                for item in row['PRODUCTNAME'].split(','):
                    vertical_format.append({
                        'SID': row['NAME'],
                        'EID': row['EID'],
                        'item': item.strip()
                    })
            else:
                vertical_format.append({
                    'SID': row['NAME'],
                    'EID': row['EID'],
                    'item': str(row['PRODUCTNAME']).strip()
                })

        return pd.DataFrame(vertical_format), None
    except Exception as e:
        return None, f"Error in preprocessing data: {str(e)}"

def get_transaction_table(vertical_df):
    """
    Create a transaction table by grouping items by SID and EID.
    """
    try:
        transactions = vertical_df.groupby(['SID', 'EID'])['item'].apply(lambda x: ', '.join(sorted(set(x)))).reset_index()
        transactions.columns = ['Customer ID (SID)', 'Event ID (EID)', 'Items']
        return transactions, None
    except Exception as e:
        return None, f"Error in creating transaction table: {str(e)}"

def create_idlists(vertical_df):
    """Create ID-lists for each item (item, SID, EID)."""
    try:
        idlists = defaultdict(list)
        for _, row in vertical_df.iterrows():
            idlists[row['item']].append((row['SID'], row['EID']))
        return idlists, None
    except Exception as e:
        return None, f"Error in creating ID-lists: {str(e)}"

def calculate_support(idlist, total_sequences):
    """Calculate support as number of unique sequences / total sequences."""
    unique_sids = len(set(sid for sid, _ in idlist))
    return unique_sids / total_sequences if total_sequences > 0 else 0

def generate_1_sequences(idlists, min_support, total_sequences):
    """Generate frequent 1-sequences."""
    try:
        frequent_1_sequences = []
        for item, idlist in idlists.items():
            support = calculate_support(idlist, total_sequences)
            if support >= min_support:
                frequent_1_sequences.append((frozenset([item]), support * total_sequences))
        return frequent_1_sequences, None
    except Exception as e:
        return None, f"Error in generating 1-sequences: {str(e)}"

def join_idlists(idlist1, idlist2, join_type='temporal'):
    """
    Join two ID-lists based on join type:
    - 'temporal': for sequence extension (different events)
    - 'itemset': for itemset extension (same event)
    """
    result = []
    dict1 = defaultdict(list)
    for sid, eid in idlist1:
        dict1[sid].append(eid)

    for sid, eid in idlist2:
        if sid in dict1:
            if join_type == 'temporal':
                for eid1 in dict1[sid]:
                    if eid > eid1:
                        result.append((sid, eid))
                        break
            else:
                if eid in dict1[sid]:
                    result.append((sid, eid))
    return result

def generate_candidate_k_sequences(frequent_sequences_k_minus_1, k, idlists):
    """Generate candidate k-sequences from frequent (k-1)-sequences."""
    try:
        candidates = []
        items = [seq for seq, _ in frequent_sequences_k_minus_1]
        
        if k == 2:
            # Generate unique itemsets and sequences
            seen_itemsets = set()
            for i, item_i in enumerate(items):
                for j, item_j in enumerate(items[i+1:], start=i+1):  # Ensure i < j to avoid duplicates
                    item_i_str = list(item_i)[0]
                    item_j_str = list(item_j)[0]
                    if item_i_str == item_j_str:
                        continue
                    
                    idlist_i = idlists[item_i_str]
                    idlist_j = idlists[item_j_str]
                    
                    # Itemset extension: only generate in canonical order
                    itemset_tuple = tuple(sorted([item_i_str, item_j_str]))
                    if itemset_tuple not in seen_itemsets:
                        new_itemset = frozenset(itemset_tuple)
                        new_idlist = join_idlists(idlist_i, idlist_j, join_type='itemset')
                        candidates.append((new_itemset, new_idlist))
                        seen_itemsets.add(itemset_tuple)
                    
                    # Sequence extension: both orders are valid
                    new_sequence = (item_i_str, item_j_str)
                    new_idlist = join_idlists(idlist_i, idlist_j, join_type='temporal')
                    candidates.append((new_sequence, new_idlist))
                    
                    new_sequence = (item_j_str, item_i_str)
                    new_idlist = join_idlists(idlist_j, idlist_i, join_type='temporal')
                    candidates.append((new_sequence, new_idlist))
        else:
            sequence_patterns = [(p, s) for p, s in frequent_sequences_k_minus_1 if isinstance(p, tuple) and len(p) == k-1]
            for i, (seq_i, _) in enumerate(sequence_patterns):
                for j, (seq_j, _) in enumerate(sequence_patterns):
                    if i == j:
                        continue
                    if seq_i[:-1] == seq_j[:-1]:
                        new_sequence = seq_i + (seq_j[-1],)
                        idlist_i = idlists[seq_i[-1]]
                        idlist_j = idlists[seq_j[-1]]
                        new_idlist = join_idlists(idlist_i, idlist_j, join_type='temporal')
                        candidates.append((new_sequence, new_idlist))

        return candidates, None
    except Exception as e:
        return None, f"Error in generating candidate {k}-sequences: {str(e)}"

def filter_frequent_sequences(candidates, min_support, total_sequences):
    """Filter candidates to get frequent sequences."""
    try:
        frequent_sequences = []
        seen_patterns = set()
        for pattern, idlist in candidates:
            support = calculate_support(idlist, total_sequences)
            if support >= min_support:
                pattern_key = pattern if isinstance(pattern, tuple) else tuple(sorted(pattern))
                if pattern_key not in seen_patterns:
                    frequent_sequences.append((pattern, support * total_sequences))
                    seen_patterns.add(pattern_key)
        return frequent_sequences, None
    except Exception as e:
        return None, f"Error in filtering frequent sequences: {str(e)}"

def format_pattern(pattern):
    """Format a pattern for readability."""
    if isinstance(pattern, frozenset):
        return f"{{{', '.join(sorted(pattern))}}}"
    elif isinstance(pattern, tuple):
        return f"<{' -> '.join(pattern)}>"
    return str(pattern)

def get_pattern_length(pattern):
    """Get length of a pattern (number of items)."""
    if isinstance(pattern, frozenset):
        return len(pattern)
    elif isinstance(pattern, tuple):
        return len(pattern)
    return 1

def run_spade_analysis(df, min_support):
    """
    Main SPADE algorithm implementation with enhanced output.
    Returns: transactions_df, detailed_results, all_frequent_df, error
    """
    try:
        vertical_df, error = preprocess_data_vertical(df)
        if error:
            return None, None, None, error

        transactions_df, error = get_transaction_table(vertical_df)
        if error:
            return None, None, None, error

        idlists, error = create_idlists(vertical_df)
        if error:
            return None, None, None, error
        
        total_sequences = vertical_df['SID'].nunique()
        frequent_1, error = generate_1_sequences(idlists, min_support, total_sequences)
        if error:
            return None, None, None, error

        frequent_1_df = pd.DataFrame([
            (format_pattern(seq), support)
            for seq, support in sorted(frequent_1, key=lambda x: str(x[0]))
        ], columns=["Pattern", "Support"])

        all_frequent = list(frequent_1)
        all_frequent_by_level = {1: frequent_1}
        
        detailed_results = {
            "vertical_format_sample": vertical_df.head(10),
            "transactions": transactions_df,
            "total_sequences": total_sequences,
            "min_support": min_support,
            "frequent_1": frequent_1_df,
            "candidates": [],  # Store candidates as a list of (k, df) tuples
            "frequent": []     # Store frequent sequences as a list of (k, df) tuples
        }

        k = 2
        while True:
            candidates_k, error = generate_candidate_k_sequences(all_frequent_by_level.get(k-1, []), k, idlists)
            if error:
                return None, None, None, error
            
            if not candidates_k:
                break
                
            candidates_df = pd.DataFrame([
                (format_pattern(seq), len(idlist))
                for seq, idlist in sorted(candidates_k, key=lambda x: str(x[0]))
            ], columns=["Pattern", "ID-List Length"])
            detailed_results["candidates"].append((k, candidates_df))
            
            frequent_k, error = filter_frequent_sequences(candidates_k, min_support, total_sequences)
            if error:
                return None, None, None, error

            if not frequent_k:
                break

            all_frequent_by_level[k] = frequent_k
            frequent_k_df = pd.DataFrame([
                (format_pattern(seq), support)
                for seq, support in sorted(frequent_k, key=lambda x: str(x[0]))
            ], columns=["Pattern", "Support"])
            detailed_results["frequent"].append((k, frequent_k_df))
            
            all_frequent.extend(frequent_k)
            k += 1

        all_frequent_df = pd.DataFrame(
            [(format_pattern(seq), support, "Itemset" if isinstance(seq, frozenset) else "Sequence", get_pattern_length(seq))
             for seq, support in sorted(all_frequent, key=lambda x: (get_pattern_length(x[0]), isinstance(x[0], frozenset), str(x[0])))],
            columns=["Pattern", "Support", "Pattern Type", "Length"]
        )
        
        detailed_results["all_frequent"] = all_frequent_df
        return transactions_df, detailed_results, all_frequent_df, None

    except Exception as e:
        error_msg = f"Error in SPADE analysis: {str(e)}"
        return None, None, None, error_msg