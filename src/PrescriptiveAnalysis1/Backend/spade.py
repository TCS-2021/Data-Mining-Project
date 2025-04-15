import pandas as pd
from collections import defaultdict
import traceback

def preprocess_data_vertical(df):
    """
    Convert horizontal data format to vertical format (SID, EID, item).
    SID = Sequence ID (customer ID)
    EID = Event ID (timestamp/order of events)
    """
    try:
        # Convert dates to datetime
        df['INVOICEDATE'] = pd.to_datetime(df['INVOICEDATE'], errors='coerce', dayfirst=True)
        
        # Sort data by customer and date
        df_sorted = df.sort_values(['NAME', 'INVOICEDATE'])

        # Create event IDs for each customer
        df_sorted['EID'] = df_sorted.groupby('NAME').cumcount() + 1

        # Handle comma-separated values in PRODUCTNAME
        vertical_format = []
        for _, row in df_sorted.iterrows():
            if isinstance(row['PRODUCTNAME'], str) and ',' in row['PRODUCTNAME']:
                # Split by comma and process each item
                for item in row['PRODUCTNAME'].split(','):
                    vertical_format.append({
                        'SID': row['NAME'],
                        'EID': row['EID'],
                        'item': item.strip()
                    })
            else:
                # Single item
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
                # For sequence extension, EID2 > EID1
                for eid1 in dict1[sid]:
                    if eid > eid1:
                        result.append((sid, eid))
                        break
            else:  # itemset extension
                # For itemset extension, identical EIDs
                if eid in dict1[sid]:
                    result.append((sid, eid))
    return result

def generate_candidate_k_sequences(frequent_sequences_k_minus_1, k, idlists):
    """Generate candidate k-sequences from frequent (k-1)-sequences."""
    try:
        candidates = []
        
        # Extract patterns from sequences
        items = [seq for seq, _ in frequent_sequences_k_minus_1]
        
        if k == 2:
            # Try all pairs for both sequence and itemset extensions
            for i, item_i in enumerate(items):
                for j, item_j in enumerate(items):
                    if i == j:
                        continue
                    
                    # Extract item strings from frozensets
                    item_i_str = list(item_i)[0]
                    item_j_str = list(item_j)[0]
                    
                    if item_i_str == item_j_str:
                        continue
                    
                    # Get ID lists for the items
                    idlist_i = idlists[item_i_str]
                    idlist_j = idlists[item_j_str]
                    
                    # Itemset extension (both items in same event)
                    new_itemset = frozenset([item_i_str, item_j_str])
                    new_idlist = join_idlists(idlist_i, idlist_j, join_type='itemset')
                    if new_idlist:  # Only add if the join produced results
                        candidates.append((new_itemset, new_idlist))
                    
                    # Sequence extension (sequential events)
                    new_sequence = (item_i_str, item_j_str)
                    new_idlist = join_idlists(idlist_i, idlist_j, join_type='temporal')
                    if new_idlist:  # Only add if the join produced results
                        candidates.append((new_sequence, new_idlist))
        else:
            # For k > 2, use prefix-based join for sequences
            # Create a lookup dictionary for pattern -> idlist
            idlist_lookup = {}
            for pattern, _ in frequent_sequences_k_minus_1:
                if isinstance(pattern, tuple):
                    # This is for sequence patterns
                    idlist_lookup[pattern] = None  # We'll fill this later
            
            # This is a simplified version that needs to be expanded for full implementation
            for i, seq_i in enumerate(items):
                for j, seq_j in enumerate(items):
                    if i == j:
                        continue
                    
                    # Only handle tuple patterns (sequences) for k > 2
                    if isinstance(seq_i, tuple) and isinstance(seq_j, tuple) and len(seq_i) == len(seq_j) == k-1:
                        # Check if they share the same prefix (all but last item)
                        if seq_i[:-1] == seq_j[:-1] and seq_i[-1] != seq_j[-1]:
                            # Create new sequence by adding the last item of seq_j to seq_i
                            new_sequence = seq_i + (seq_j[-1],)
                            
                            # For k > 2, we would need more complex ID-list joining logic here
                            # In a full implementation, you'd need to track ID-lists for all k-1 patterns
                            
                            # Placeholder - in actual implementation, this would require proper ID-list joining
                            # This is where the algorithm needs expansion
                            # For now, we'll return an empty candidates list for k > 2
                            pass
        
        return candidates, None
    except Exception as e:
        return None, f"Error in generating candidate {k}-sequences: {str(e)}\n{traceback.format_exc()}"

def filter_frequent_sequences(candidates, min_support, total_sequences):
    """Filter candidates to get frequent sequences."""
    try:
        frequent_sequences = []
        for pattern, idlist in candidates:
            support = calculate_support(idlist, total_sequences)
            if support >= min_support:
                frequent_sequences.append((pattern, support * total_sequences))
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
    Main SPADE algorithm implementation.
    Returns: transactions_df, results (frequent_1, candidates, all_frequent), all_frequent_df, error
    """
    try:
        # Step 1: Preprocess data to vertical format
        vertical_df, error = preprocess_data_vertical(df)
        if error:
            return None, None, None, error

        # Step 2: Create transaction table
        transactions_df, error = get_transaction_table(vertical_df)
        if error:
            return None, None, None, error

        # Step 3: Create ID-lists
        idlists, error = create_idlists(vertical_df)
        if error:
            return None, None, None, error
        total_sequences = vertical_df['SID'].nunique()

        # Step 4: Generate frequent 1-sequences
        frequent_1, error = generate_1_sequences(idlists, min_support, total_sequences)
        if error:
            return None, None, None, error

        # Step 5: Generate frequent k-sequences (k ≥ 2)
        all_frequent = [(pattern, support) for pattern, support in frequent_1]
        candidates_all = []
        k = 2

        while True:
            # Generate candidates
            candidates_k, error = generate_candidate_k_sequences(frequent_1 if k == 2 else [], k, idlists)
            if error:
                return None, None, None, error
            
            if not candidates_k:
                break
                
            # Filter frequent sequences
            frequent_k, error = filter_frequent_sequences(candidates_k, min_support, total_sequences)
            if error:
                return None, None, None, error

            if not frequent_k:
                break

            # Update all_frequent and candidates
            all_frequent.extend(frequent_k)
            candidates_all.extend(frequent_k)
            k += 1

        # Create DataFrame for all frequent sequences
        all_frequent_df = pd.DataFrame(
            [(format_pattern(seq), support, "Itemset" if isinstance(seq, frozenset) else "Sequence", get_pattern_length(seq))
             for seq, support in sorted(all_frequent, key=lambda x: (get_pattern_length(x[0]), str(x[0])))],
            columns=["Pattern", "Support", "Pattern Type", "Length"]
        )

        results = (frequent_1, candidates_all, all_frequent)
        return transactions_df, results, all_frequent_df, None

    except Exception as e:
        error_msg = f"Error in SPADE analysis: {str(e)}\n{traceback.format_exc()}"
        return None, None, None, error_msg