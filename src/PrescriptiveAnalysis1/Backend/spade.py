import pandas as pd
from collections import defaultdict
import traceback

def preprocess_data_vertical(df):
    """
    Convert horizontal data format to vertical format (SID, EID, item).
    SID = Sequence ID (from NAME column)
    EID = Event ID (timestamp/order of events)
    """
    try:
        if 'NAME' not in df.columns:
            return None, "Error: NAME column missing in dataset"
        df = df.copy()
        df['SID'] = df['NAME'].astype(str)

        if df['SID'].isnull().any():
            return None, "Error: Invalid or missing NAME values"

        if 'INVOICEDATE' in df.columns:
            try:
                df['INVOICEDATE'] = pd.to_datetime(df['INVOICEDATE'], errors='coerce')
            except:
                df['INVOICEDATE'] = pd.to_datetime(df['INVOICEDATE'], errors='coerce', dayfirst=True)
            df_sorted = df.sort_values(['SID', 'INVOICEDATE'])
        else:
            df_sorted = df.sort_values(['SID'])

        df_sorted['EID'] = df_sorted.groupby('SID').cumcount() + 1

        vertical_format = []
        for _, row in df_sorted.iterrows():
            if isinstance(row['PRODUCTNAME'], str) and ',' in row['PRODUCTNAME']:
                for item in row['PRODUCTNAME'].split(','):
                    vertical_format.append({
                        'SID': row['SID'],
                        'EID': row['EID'],
                        'item': item.strip()
                    })
            else:
                vertical_format.append({
                    'SID': row['SID'],
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
        transactions = vertical_df.groupby(['SID', 'EID'])['item'].apply(lambda x: set(x)).reset_index()
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

def calculate_support(pattern, transactions_df):
    """
    Calculate support by checking pattern in transaction table.
    Support = (number of SIDs containing pattern) / (total SIDs)
    """
    try:
        total_sids = transactions_df['Customer ID (SID)'].nunique()
        if total_sids == 0:
            return 0

        matching_sids = set()
        grouped = transactions_df.groupby('Customer ID (SID)')

        if isinstance(pattern, frozenset):
            pattern_items = set(pattern)
            for sid, group in grouped:
                for _, row in group.iterrows():
                    if pattern_items.issubset(row['Items']):
                        matching_sids.add(sid)
                        break
        elif isinstance(pattern, tuple):
            for sid, group in grouped:
                group = group.sort_values('Event ID (EID)')
                found = [False] * len(pattern)
                current_pos = 0
                for _, row in group.iterrows():
                    items = row['Items']
                    if current_pos < len(pattern):
                        current_element = pattern[current_pos]
                        element_items = set(current_element) if isinstance(current_element, frozenset) else {current_element}
                        if element_items.issubset(items):
                            found[current_pos] = True
                            current_pos += 1
                if all(found):
                    matching_sids.add(sid)

        return len(matching_sids) / total_sids if total_sids > 0 else 0
    except Exception as e:
        return 0

def generate_1_sequences(transactions_df, min_support):
    """Generate frequent 1-sequences using transaction table."""
    try:
        unique_items = set()
        for items in transactions_df['Items']:
            unique_items.update(items)
        
        frequent_1_sequences = []
        for item in unique_items:
            pattern = frozenset([item])
            support = calculate_support(pattern, transactions_df)
            if support >= min_support:
                frequent_1_sequences.append((pattern, support * transactions_df['Customer ID (SID)'].nunique()))
        return frequent_1_sequences, None
    except Exception as e:
        return None, f"Error in generating 1-sequences: {str(e)}"

def join_idlists(idlist1=None, idlist2=None, join_type='temporal', first_itemset=None, second_itemset=None, idlists=None):
    """
    Join ID-lists based on join type:
    - 'temporal': sequence extension (different events)
    - 'itemset': itemset extension (same event)
    - 'sequence_itemset': sequence -> itemset or itemset -> itemset
    """
    result = []
    
    if join_type == 'sequence_itemset' and first_itemset is not None and second_itemset is not None and idlists is not None:
        first_items = sorted(list(first_itemset)) if isinstance(first_itemset, (frozenset, set)) else [first_itemset]
        second_items = sorted(list(second_itemset)) if isinstance(second_itemset, (frozenset, set)) else [second_itemset]

        first_idlist = [(sid, eid) for sid, eid in idlists[first_items[0]]]
        for item in first_items[1:]:
            next_idlist = [(sid, eid) for sid, eid in idlists[item]]
            first_idlist = [(sid, eid) for sid, eid in first_idlist if (sid, eid) in next_idlist]

        second_idlist = [(sid, eid) for sid, eid in idlists[second_items[0]]]
        for item in second_items[1:]:
            next_idlist = [(sid, eid) for sid, eid in idlists[item]]
            second_idlist = [(sid, eid) for sid, eid in second_idlist if (sid, eid) in next_idlist]

        first_by_sid = defaultdict(list)
        for sid, eid in sorted(first_idlist, key=lambda x: (x[0], x[1])):
            first_by_sid[sid].append(eid)

        sid_added = set()
        for sid, eid2 in sorted(second_idlist, key=lambda x: (x[0], x[1])):
            if sid in first_by_sid:
                for eid1 in first_by_sid[sid]:
                    if eid2 > eid1 and sid not in sid_added:
                        result.append((sid, eid2))
                        sid_added.add(sid)
                        break
    elif join_type == 'temporal':
        first_by_sid = defaultdict(list)
        for sid, eid in sorted(idlist1, key=lambda x: (x[0], x[1])):
            first_by_sid[sid].append(eid)
        sid_added = set()
        for sid, eid2 in sorted(idlist2, key=lambda x: (x[0], x[1])):
            if sid in first_by_sid and sid not in sid_added:
                for eid1 in first_by_sid[sid]:
                    if eid2 > eid1:
                        result.append((sid, eid2))
                        sid_added.add(sid)
                        break
    elif join_type == 'itemset':
        sid_eid_set = set(idlist2)
        for sid, eid in idlist1:
            if (sid, eid) in sid_eid_set:
                result.append((sid, eid))

    return result

def generate_candidate_k_sequences(frequent_sequences_k_minus_1, k, idlists, transactions_df):
    """Generate candidate k-sequences from frequent (k-1)-sequences."""
    try:
        candidates = []
        seen_itemsets = set()
        seen_sequences = set()

        itemsets = [(p, s) for p, s in frequent_sequences_k_minus_1 if isinstance(p, frozenset)]
        sequences = [(p, s) for p, s in frequent_sequences_k_minus_1 if isinstance(p, tuple)]

        # Collect single frequent items
        single_items = []
        for p, _ in frequent_sequences_k_minus_1:
            if isinstance(p, frozenset) and len(p) == 1:
                single_items.append(list(p)[0])

        if k == 2:
            items = [seq for seq, _ in frequent_sequences_k_minus_1]
            for i, item_i in enumerate(items):
                for j, item_j in enumerate(items[i+1:], start=i+1):
                    item_i_str = list(item_i)[0]
                    item_j_str = list(item_j)[0]
                    if item_i_str == item_j_str:
                        continue
                    
                    idlist_i = idlists[item_i_str]
                    idlist_j = idlists[item_j_str]
                    
                    itemset_tuple = tuple(sorted([item_i_str, item_j_str]))
                    if itemset_tuple not in seen_itemsets:
                        new_itemset = frozenset(itemset_tuple)
                        new_idlist = join_idlists(idlist_i, idlist_j, join_type='itemset')
                        if new_idlist:
                            candidates.append((new_itemset, new_idlist))
                        seen_itemsets.add(itemset_tuple)
                    
                    new_sequence = (item_i_str, item_j_str)
                    new_idlist = join_idlists(idlist_i, idlist_j, join_type='temporal')
                    if new_sequence not in seen_sequences and new_idlist:
                        candidates.append((new_sequence, new_idlist))
                        seen_sequences.add(new_sequence)
                    
                    new_sequence = (item_j_str, item_i_str)
                    new_idlist = join_idlists(idlist_j, idlist_i, join_type='temporal')
                    if new_sequence not in seen_sequences and new_idlist:
                        candidates.append((new_sequence, new_idlist))
                        seen_sequences.add(new_sequence)
        else:
            # Itemset joins
            for i, (itemset_i, _) in enumerate(itemsets):
                for j, (itemset_j, _) in enumerate(itemsets[i+1:], start=i+1):
                    items_i = sorted(list(itemset_i))
                    items_j = sorted(list(itemset_j))
                    if items_i[:-1] == items_j[:-1]:
                        new_items = sorted(list(itemset_i) + [items_j[-1]])
                        new_itemset = frozenset(new_items)
                        itemset_tuple = tuple(new_items)
                        if itemset_tuple not in seen_itemsets:
                            new_idlist = join_idlists(idlists[items_i[0]], idlists[items_j[-1]], join_type='itemset')
                            for item in new_items[1:-1]:
                                next_idlist = idlists[item]
                                new_idlist = [(sid, eid) for sid, eid in new_idlist if (sid, eid) in next_idlist]
                            if new_idlist:
                                candidates.append((new_itemset, new_idlist))
                            seen_itemsets.add(itemset_tuple)
            
            # Sequence joins
            for i, (seq_i, _) in enumerate(sequences):
                for j, (seq_j, _) in enumerate(sequences):
                    if i == j:
                        continue
                    if seq_i[:-1] == seq_j[:-1]:
                        new_sequence = seq_i + (seq_j[-1],)
                        if new_sequence not in seen_sequences:
                            last_item_i = seq_i[-1] if isinstance(seq_i[-1], str) else sorted(seq_i[-1])[0]
                            last_item_j = seq_j[-1] if isinstance(seq_j[-1], str) else sorted(seq_j[-1])[0]
                            new_idlist = join_idlists(idlists[last_item_i], idlists[last_item_j], join_type='temporal')
                            if new_idlist:
                                candidates.append((new_sequence, new_idlist))
                            seen_sequences.add(new_sequence)
            
            # Sequence -> Itemset
            for seq, _ in sequences:
                last_seq_element = seq[-1]
                last_items = [last_seq_element] if isinstance(last_seq_element, str) else sorted(last_seq_element)
                for itemset, _ in itemsets:
                    if len(seq) == 1:
                        new_sequence = (last_seq_element, itemset)
                    else:
                        new_sequence = seq[:-1] + (itemset,)
                    sequence_tuple = new_sequence
                    if sequence_tuple not in seen_sequences:
                        new_idlist = join_idlists(
                            idlists[last_items[0]], None,
                            join_type='sequence_itemset',
                            first_itemset=frozenset(last_items),
                            second_itemset=itemset,
                            idlists=idlists
                        )
                        if new_idlist:
                            candidates.append((new_sequence, new_idlist))
                        seen_sequences.add(sequence_tuple)
            
            # Itemset -> Sequence
            for itemset, _ in itemsets:
                itemset_items = sorted(itemset)
                for seq, _ in sequences:
                    first_seq_element = seq[0]
                    first_items = [first_seq_element] if isinstance(first_seq_element, str) else sorted(first_seq_element)
                    new_sequence = (itemset,) + seq[1:]
                    sequence_tuple = new_sequence
                    if sequence_tuple not in seen_sequences:
                        new_idlist = join_idlists(
                            idlists[itemset_items[0]], None,
                            join_type='sequence_itemset',
                            first_itemset=frozenset(itemset_items),
                            second_itemset=frozenset(first_items),
                            idlists=idlists
                        )
                        if new_idlist:
                            candidates.append((new_sequence, new_idlist))
                        seen_sequences.add(sequence_tuple)
            
            # Itemset -> Single Item
            if k == 3:
                for itemset, _ in itemsets:
                    if len(itemset) >= 2:
                        itemset_items = sorted(itemset)
                        for single_item in single_items:
                            new_sequence = (itemset, single_item)
                            sequence_tuple = new_sequence
                            if sequence_tuple not in seen_sequences:
                                new_idlist = join_idlists(
                                    idlists[itemset_items[0]], None,
                                    join_type='sequence_itemset',
                                    first_itemset=frozenset(itemset_items),
                                    second_itemset=frozenset([single_item]),
                                    idlists=idlists
                                )
                                if new_idlist:
                                    candidates.append((new_sequence, new_idlist))
                                seen_sequences.add(sequence_tuple)

        return candidates, None
    except Exception as e:
        return None, f"Error in generating candidate {k}-sequences: {str(e)}"

def filter_frequent_sequences(candidates, min_support, transactions_df):
    """Filter candidates to get frequent sequences using transaction table."""
    try:
        frequent_sequences = []
        seen_patterns = set()
        for pattern, idlist in candidates:
            support = calculate_support(pattern, transactions_df)
            if support >= min_support:
                pattern_key = pattern if isinstance(pattern, tuple) else tuple(sorted(pattern))
                if pattern_key not in seen_patterns:
                    frequent_sequences.append((pattern, support * transactions_df['Customer ID (SID)'].nunique()))
                    seen_patterns.add(pattern_key)
        return frequent_sequences, None
    except Exception as e:
        return None, f"Error in filtering frequent sequences: {str(e)}"

def format_pattern(pattern):
    """Format a pattern for readability."""
    if isinstance(pattern, frozenset):
        return f"{{{', '.join(sorted(pattern))}}}"
    elif isinstance(pattern, tuple):
        return f"<{' -> '.join([format_pattern(p) if isinstance(p, frozenset) else p for p in pattern])}>"
    return str(pattern)

def get_pattern_length(pattern):
    """Get length of a pattern (number of items)."""
    if isinstance(pattern, frozenset):
        return len(pattern)
    elif isinstance(pattern, tuple):
        return sum(1 if isinstance(p, str) else len(p) for p in pattern)
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
        
        frequent_1, error = generate_1_sequences(transactions_df, min_support)
        if error:
            return None, None, None, error

        frequent_1_df = pd.DataFrame([
            (format_pattern(seq), support)
            for seq, support in sorted(frequent_1, key=lambda x: str(x[0]))
        ], columns=["Pattern", "Support"])

        all_frequent = list(frequent_1)
        all_frequent_by_level = {1: frequent_1}
        
        detailed_results = {
            "vertical_format_sample": vertical_df,
            "transactions": transactions_df,
            "total_sequences": transactions_df['Customer ID (SID)'].nunique(),
            "min_support": min_support,
            "frequent_1": frequent_1_df,
            "candidates": [],
            "frequent": []
        }

        k = 2
        while True:
            candidates_k, error = generate_candidate_k_sequences(all_frequent_by_level.get(k-1, []), k, idlists, transactions_df)
            if error:
                return None, None, None, error
            
            if not candidates_k:
                break
                
            candidates_df = pd.DataFrame([
                (format_pattern(seq), len(set(sid for sid, _ in idlist)))
                for seq, idlist in sorted(candidates_k, key=lambda x: str(x[0]))
            ], columns=["Pattern", "ID-List Length"])
            detailed_results["candidates"].append((k, candidates_df))
            
            frequent_k, error = filter_frequent_sequences(candidates_k, min_support, transactions_df)
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