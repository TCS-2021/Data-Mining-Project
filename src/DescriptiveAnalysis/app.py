"""Module containing data processing functions for the data warehouse application."""
import itertools
import streamlit as st
import pandas as pd

# Custom CSS for styling
st.markdown("""
    <style>
    .main-container { padding: 1rem; }
    .stButton>button { width: 100%; }
    .dimension-box { border: 1px solid #ddd; padding: 1rem; margin-bottom: 1rem; border-radius: 5px; }
    .section-title { margin-top: 1.5rem; margin-bottom: 0.5rem; }
    .tab-content { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

def create_data_cube(df: pd.DataFrame, dimensions: list, measures: list) -> pd.DataFrame:
    """Create a data cube by grouping data by dimensions and aggregating measures."""
    # Check if all dimensions and measures exist in the DataFrame
    missing_dims = [dim for dim in dimensions if dim not in df.columns]
    missing_measures = [measure for measure in measures if measure not in df.columns]
    if missing_dims:
        st.error(f"Missing dimension columns: {', '.join(missing_dims)}")
        return pd.DataFrame()
    if missing_measures:
        st.error(f"Missing measure columns: {', '.join(missing_measures)}")
        return pd.DataFrame()
    return df.groupby(dimensions).agg({measure: 'sum' for measure in measures}).reset_index()

def load_data():
    """Handle file upload and return DataFrame and columns."""
    with st.container():
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        if not uploaded_file:
            st.warning("Please upload a CSV file to proceed")
            return None, None
        try:
            data_frame = pd.read_csv(uploaded_file)
            return data_frame, data_frame.columns.tolist()
        except (pd.errors.EmptyDataError, pd.errors.ParserError,
                UnicodeDecodeError, ValueError, PermissionError) as e:
            st.error(f"Error loading file: {str(e)}")
            return None, None

def configure_dimensions(all_columns: list) -> dict:
    """Configure dimension tables and return dimension configurations."""
    st.subheader("Dimension Tables")
    num_dimensions = st.number_input(
        "Number of Dimensions",
        min_value=1,
        max_value=10,
        value=3,
        key="num_dims"
    )

    dimension_tables = {}
    for i in range(int(num_dimensions)):
        with st.expander(f"Dimension {i+1}", expanded=False):
            dim_name = st.text_input(
                "Table Name",
                value=f"dim_{i+1}",
                key=f"dim_name_{i}"
            )
            # First select all dimension columns
            available_cols = [
                col for col in all_columns
                if col not in [c for dim in dimension_tables.values()
                            for c in dim.get('columns', [])]
            ]
            selected_cols = st.multiselect(
                "Select Dimension Columns",
                available_cols,
                key=f"dim_cols_{i}"
            )
            # Then separately select hierarchy columns in order
            if selected_cols:
                st.write("Configure Hierarchy Levels (Select one column at a time in order):")
                hierarchy = []
                level = 1
                while True:
                    remaining_cols = [col for col in selected_cols if col not in hierarchy]
                    if not remaining_cols:
                        break
                    # For each level, show only columns not already in hierarchy
                    level_col = st.selectbox(
                        f"Hierarchy Level {level} (Select from remaining columns)",
                        [None] + remaining_cols,
                        key=f"hierarchy_{i}_level_{level}"
                    )
                    if not level_col or level_col == "None":
                        break
                    hierarchy.append(level_col)
                    level += 1
                    # Option to add more levels
                    if not st.checkbox(f"Add another hierarchy level for Dimension {i+1}",
                                      value=True,
                                      key=f"add_level_{i}_{level}"):
                        break
            else:
                hierarchy = []
            primary_key = st.selectbox(
                "Primary Key",
                selected_cols if selected_cols else [None],
                key=f"dim_pk_{i}"
            )
            if selected_cols and primary_key:
                dimension_tables[dim_name] = {
                    'columns': selected_cols,
                    'primary_key': primary_key,
                    'hierarchy': hierarchy
                }
                # Show the configured hierarchy
                if hierarchy:
                    st.write("Configured Hierarchy:")
                    st.write(" → ".join(hierarchy))
                else:
                    st.warning("No hierarchy configured for this dimension")
    return dimension_tables

def configure_fact_table(all_columns: list, dimension_tables: dict) -> tuple[list, list]:
    """Configure fact table measures and columns."""
    st.subheader("Fact Table")
    remaining_cols = [
        col for col in all_columns
        if col not in [c for dim in dimension_tables.values() for c in dim.get('columns', [])]
    ]
    fact_measures = st.multiselect(
        "Measures",
        remaining_cols,
        key="fact_cols"
    )
    fact_table_cols = fact_measures.copy()
    primary_keys = [dim.get('primary_key') for dim in dimension_tables.values()]
    fact_table_cols.extend(primary_keys)
    return fact_measures, fact_table_cols

def display_results(
    df: pd.DataFrame,
    dimension_tables: dict,
    fact_df: pd.DataFrame,
    fact_measures: list
):
    """Display generated tables and data cubes."""
    with st.expander("Generated Tables", expanded=True):
        for dim_name, config in dimension_tables.items():
            st.write(f"**{dim_name}**")
            dim_df = df[config['columns']].drop_duplicates()
            st.dataframe(dim_df, height=200)
        st.write("**Fact Table**")
        st.dataframe(fact_df, height=200)

    with st.expander("Data Cubes", expanded=True):
        dim_keys = list(dimension_tables.keys())
        for n in range(1, len(dim_keys) + 1):
            st.write(f"**{n}D Cubes**")
            dim_combinations = list(itertools.combinations(dim_keys, n))
            for combo in dim_combinations:
                dim_columns = [
                    dimension_tables[dim_name]['primary_key']
                    for dim_name in combo
                ]
                cube = create_data_cube(fact_df, dim_columns, fact_measures)
                if not cube.empty:
                    st.write(f"Cube: {', '.join(combo)}")
                    st.dataframe(cube, height=200)

def olap_operations(fact_df: pd.DataFrame,
                    dimension_tables: dict,
                    fact_measures: list,
                    original_df: pd.DataFrame):
    """Handle drill-down, roll-up on the data."""
    st.header("Drill-Down and Roll-Up")
    if not dimension_tables:
        st.warning("No dimensions configured. Please set up dimensions first.")
        return
    if not fact_measures:
        st.warning("No measures configured. Please set up measures first.")
        return
    # Select dimension for Drill-Down and Roll-Up
    selected_dim = st.selectbox(
        "Select Dimension for Drill-Down and Roll-Up",
        list(dimension_tables.keys()),
        key="olap_dim"
    )
    dim_config = dimension_tables[selected_dim]
    if len(dim_config['hierarchy']) <= 1:
        st.warning(
            "This dimension doesn't have hierarchy levels defined. "
            "Add more columns to the dimension to enable drill-down/roll-up."
        )
        return
    # Select measure for Drill-Down and Roll-Up
    selected_measure = st.selectbox(
        "Select Measure",
        fact_measures,
        key="olap_measure"
    )
    # Get current level from session state or set to highest level
    if 'current_level' not in st.session_state:
        st.session_state.current_level = 0
    # Get the dimension data
    dim_df = original_df[dim_config['columns']].drop_duplicates()
    # Merge fact table with dimension table to get all hierarchy levels
    merged_df = pd.merge(
        fact_df,
        dim_df,
        left_on=dim_config['primary_key'],
        right_on=dim_config['primary_key'],
        how='left'
    )
    # Get current hierarchy column
    current_level_col = dim_config['hierarchy'][st.session_state.current_level]
    st.write(f"**Current Level:** {current_level_col}")
    # Create a cube at the current level (only showing hierarchy column and measure)
    cube = merged_df.groupby([current_level_col]).agg({selected_measure: 'sum'}).reset_index()
    if not cube.empty:
        st.dataframe(cube[[current_level_col, selected_measure]], height=300)
    # Drill-down and Roll-up buttons
    col1, col2 = st.columns(2)
    with col1:
        if (st.button("Drill Down") and
            st.session_state.current_level < len(dim_config['hierarchy']) - 1):
            st.session_state.current_level += 1
            st.rerun()
    with col2:
        if st.button("Roll Up") and st.session_state.current_level > 0:
            st.session_state.current_level -= 1
            st.rerun()
    # Display hierarchy path
    hierarchy_path = " → ".join(
        [f"**{level}**" if i == st.session_state.current_level else level
         for i, level in enumerate(dim_config['hierarchy'])]
    )
    st.write(f"**Hierarchy Path:** {hierarchy_path}")

def main():
    """Main function to run the Streamlit application."""
    st.title("Data Warehouse & Cube Generator")

    # Load data
    data_frame, all_columns = load_data()
    if data_frame is None:
        return

    # Create tabs
    tab1, tab2 = st.tabs(["Data Warehouse Setup", "Drill-Down and Roll-Up"])
    with tab1:
        # Layout for first tab
        col1, col2 = st.columns([1, 2])

        # Configure dimensions and fact table
        with col1:
            dimension_tables = configure_dimensions(all_columns)
        with col2:
            fact_measures, fact_table_cols = configure_fact_table(all_columns, dimension_tables)
            if st.button("Generate Data Warehouse", key="generate_btn"):
                try:
                    fact_df = data_frame[fact_table_cols]
                    display_results(data_frame, dimension_tables, fact_df, fact_measures)
                    # Store in session state for Drill-Down and Roll-Up
                    st.session_state.fact_df = fact_df
                    st.session_state.dimension_tables = dimension_tables
                    st.session_state.fact_measures = fact_measures
                except KeyError as e:
                    st.error(f"Column not found in data: {str(e)}")
    with tab2:
        if 'fact_df' in st.session_state:
            olap_operations(st.session_state.fact_df,
            st.session_state.dimension_tables,
            st.session_state.fact_measures,
            data_frame
        )
        else:
            st.warning("Please generate the data warehouse first in the Setup tab.")

if __name__ == "__main__":
    main()
    