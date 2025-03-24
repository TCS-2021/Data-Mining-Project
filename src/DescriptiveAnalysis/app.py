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
    </style>
""", unsafe_allow_html=True)

def create_data_cube(df: pd.DataFrame, dimensions: list, measures: list) -> pd.DataFrame:
    """Create a data cube by grouping data by dimensions and aggregating measures."""
    return df.groupby(dimensions).agg({measure: 'sum' for measure in measures}).reset_index()

def load_data():
    """Handle file upload and return DataFrame and columns."""
    with st.container():
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        if not uploaded_file:
            st.warning("Please upload a CSV file to proceed")
            return None, None
        data_frame = pd.read_csv(uploaded_file)
        return data_frame, data_frame.columns.tolist()

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
            available_cols = [
                col for col in all_columns
                if col not in [c for dim in dimension_tables.values() for c in dim['columns']]
            ]
            selected_cols = st.multiselect(
                "Columns",
                available_cols,
                key=f"dim_cols_{i}"
            )
            primary_key = st.selectbox(
                "Primary Key",
                selected_cols,
                key=f"dim_pk_{i}"
            )
            if selected_cols and primary_key:
                dimension_tables[dim_name] = {
                    'columns': selected_cols,
                    'primary_key': primary_key
                }
    return dimension_tables

def configure_fact_table(all_columns: list, dimension_tables: dict) -> tuple[list, list]:
    """Configure fact table measures and columns."""
    st.subheader("Fact Table")
    remaining_cols = [
        col for col in all_columns
        if col not in [c for dim in dimension_tables.values() for c in dim['columns']]
    ]
    fact_measures = st.multiselect(
        "Measures",
        remaining_cols,
        key="fact_cols"
    )
    fact_table_cols = fact_measures.copy()
    primary_keys = [dim['primary_key'] for dim in dimension_tables.values()]
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
                st.write(f"Cube: {', '.join(combo)}")
                st.dataframe(cube, height=200)

def main():
    """Main function to run the Streamlit application."""
    st.title("Data Warehouse & Cube Generator")

    # Load data
    data_frame, all_columns = load_data()
    if data_frame is None:
        return

    # Layout
    col1, col2 = st.columns([1, 2])

    # Configure dimensions and fact table
    with col1:
        dimension_tables = configure_dimensions(all_columns)
    with col2:
        fact_measures, fact_table_cols = configure_fact_table(all_columns, dimension_tables)
        if st.button("Generate", key="generate_btn"):
            fact_df = data_frame[fact_table_cols]
            display_results(data_frame, dimension_tables, fact_df, fact_measures)

if __name__ == "__main__":
    main()
    