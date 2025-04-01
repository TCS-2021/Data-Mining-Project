"""Module containing data processing functions for the"
" data warehouse application with Batch Processing."""
import itertools
import time
import io
import zipfile
import base64
import math
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

st.set_page_config(
    layout="wide",
)

# Custom CSS for a fancy and modern look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');

    html, body {
        background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
        font-family: 'Montserrat', sans-serif;
        color: #333;
    }
    .main-container {
        background-color: #fff;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        margin: 1rem;
    }

    .css-1e5imcs, .css-1d391kg {
        background-color: #ffffff !important;
        border-radius: 8px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        padding: 1rem;
    }
    .css-1aumxhk {
        font-size: 1.2rem;
        font-weight: 600;
    }
    .css-1d391kg .stTabs {
        background-color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

def create_data_cube(df: pd.DataFrame, dimensions: list, measures: list) -> pd.DataFrame:
    """Create a data cube by grouping data by dimensions and aggregating measures."""
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
            available_cols = [
                col for col in all_columns
                if( col not in
                    [c for dim in dimension_tables.values()
                    for c in dim.get('columns', [])]
                )
            ]
            selected_cols = st.multiselect(
                "Select Dimension Columns",
                available_cols,
                key=f"dim_cols_{i}"
            )
            if selected_cols:
                st.write("Configure Hierarchy Levels (Select one column at a time in order):")
                hierarchy = []
                level = 1
                while True:
                    remaining_cols = [col for col in selected_cols if col not in hierarchy]
                    if not remaining_cols:
                        break
                    level_col = st.selectbox(
                        f"Hierarchy Level {level} (Select from remaining columns)",
                        [None] + remaining_cols,
                        key=f"hierarchy_{i}_level_{level}"
                    )
                    if not level_col or level_col == "None":
                        break
                    hierarchy.append(level_col)
                    level += 1
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
    """Display generated tables and data cubes in tabs."""
    with st.expander("Generated Tables", expanded=True):
        for dim_name, config in dimension_tables.items():
            st.write(f"**{dim_name}**")
            st.dataframe(df[config['columns']].drop_duplicates(), height=200)
        st.write("**Fact Table**")
        st.dataframe(fact_df, height=200)

    with st.expander("Data Cubes", expanded=True):
        # Generate cubes using the helper function
        all_cubes = generate_data_cubes(fact_df, dimension_tables, fact_measures)
        if all_cubes:
            tabs = st.tabs(list(all_cubes.keys()))
            for tab, (cube_name, cube) in zip(tabs, all_cubes.items()):
                with tab:
                    st.write(f"**{cube_name} Cube**")
                    st.dataframe(cube, height=400)
                    csv = cube.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label=f"Download {cube_name} as CSV",
                        data=csv,
                        file_name=f'cube_{cube_name.replace(" × ", "_")}.csv',
                        mime='text/csv',
                        key=f'dl_{cube_name}'
                    )
        else:
            st.warning("No cubes were generated. Please check your configuration.")

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
    selected_measure = st.selectbox(
        "Select Measure",
        fact_measures,
        key="olap_measure"
    )
    if 'current_level' not in st.session_state:
        st.session_state.current_level = 0
    dim_df = original_df[dim_config['columns']].drop_duplicates()
    merged_df = pd.merge(
        fact_df,
        dim_df,
        left_on=dim_config['primary_key'],
        right_on=dim_config['primary_key'],
        how='left'
    )
    current_level_col = dim_config['hierarchy'][st.session_state.current_level]
    st.write(f"**Current Level:** {current_level_col}")
    cube = merged_df.groupby([current_level_col]).agg({selected_measure: 'sum'}).reset_index()
    if not cube.empty:
        st.dataframe(cube[[current_level_col, selected_measure]], height=300)
    col1, col2 = st.columns(2)
    with col1:
        if( st.button("Drill Down") and
            st.session_state.current_level < len(dim_config['hierarchy']) - 1):
            st.session_state.current_level += 1
            st.rerun()
    with col2:
        if st.button("Roll Up") and st.session_state.current_level > 0:
            st.session_state.current_level -= 1
            st.rerun()
    hierarchy_path = " → ".join(
        [f"**{level}**" if i == st.session_state.current_level else level
         for i, level in enumerate(dim_config['hierarchy'])]
    )
    st.write(f"**Hierarchy Path:** {hierarchy_path}")

def generate_data_cubes(fact_df: pd.DataFrame, dimension_tables: dict, fact_measures: list) -> dict:
    """Generate all possible data cubes and return as a dictionary."""
    all_cubes = {}
    dim_keys = list(dimension_tables.keys())
    for n in range(1, len(dim_keys)+1):
        dim_combinations = list(itertools.combinations(dim_keys, n))
        for combo in dim_combinations:
            dim_columns = [dimension_tables[dim_name]['primary_key'] for dim_name in combo]
            cube = create_data_cube(fact_df, dim_columns, fact_measures)
            if not cube.empty:
                cube_name = " × ".join(combo)
                all_cubes[cube_name] = cube
    return all_cubes

def package_batch(batch_cubes, batch_number):
    """Package a batch of cubes into a zip file and return its base64 string and filename."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for cube_name, cube in batch_cubes:
            csv_data = cube.to_csv(index=False)
            filename = f"cube_{cube_name.replace(' × ', '_')}.csv"
            zf.writestr(filename, csv_data)
    zip_data = zip_buffer.getvalue()
    zip_data_base64 = base64.b64encode(zip_data).decode('utf-8')
    zip_filename = f"data_cubes_batch_{batch_number}.zip"
    return zip_data_base64, zip_filename

def download_zip_files(cubes, batch_size):
    """Package cubes into multiple zip files based on the batch size and trigger downloads."""
    cube_items = list(cubes.items())
    num_batches = math.ceil(len(cube_items) / batch_size)
    # Package batches using a list comprehension
    zip_files = [package_batch(cube_items[i * batch_size:(i + 1) * batch_size], i + 1)
                 for i in range(num_batches)]
    # Build HTML with JavaScript to trigger the downloads
    download_html = "<html><body><script>"
    for zip_data_base64, zip_filename in zip_files:
        download_html += f"""
        var a = document.createElement('a');
        a.href = "data:application/zip;base64,{zip_data_base64}";
        a.download = "{zip_filename}";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        """
    download_html += """
        document.write('<p style="font-weight:600; color:#28a745;">Data cubes are downloaded in batches.</p>');
        </script></body></html>
    """
    components.html(download_html, height=150)


def manage_timer_controls(chosen_minutes):
    """Handle the Schedule, Stop, and Restart buttons for the timer."""
    col1, col2, col3 = st.columns(3)
    if col1.button("Schedule"):
        st.session_state.batch_running = True
        st.session_state.batch_total_seconds = chosen_minutes * 60
        st.session_state.batch_remaining_seconds = st.session_state.batch_total_seconds
    if col2.button("Stop Schedule"):
        st.session_state.batch_running = False
    if col3.button("Restart Schedule"):
        st.session_state.batch_running = True
        st.session_state.batch_remaining_seconds = chosen_minutes * 60

def countdown_timer(placeholder):
    """Display the countdown and update the timer."""
    placeholder.write(f"**Time left:** {st.session_state.batch_remaining_seconds} seconds")
    time.sleep(1)
    st.session_state.batch_remaining_seconds -= 1
    st.rerun()

def process_download(batch_size):
    """Generate data cubes and trigger the download of zip files."""
    cubes = generate_data_cubes(
        st.session_state.fact_df,
        st.session_state.dimension_tables,
        st.session_state.fact_measures
    )
    if cubes:
        download_zip_files(cubes, batch_size)
    else:
        st.error("No data cubes generated. Please check your configuration.")

def batch_processing():
    """Handle scheduled batch processing for downloading data cubes."""
    st.header("Batch Processing: Scheduled Data Cube Download")
    if ('fact_df' not in st.session_state or
        'dimension_tables' not in st.session_state or
        'fact_measures' not in st.session_state):
        st.warning("Please generate the data warehouse first in the Setup tab.")
        return

    # Get schedule time
    schedule_option = st.selectbox("Select Schedule Time",
                                   ["1 min", "2 min", "5 min", "Custom"],
                                   key="batch_schedule_option")
    if schedule_option == "Custom":
        chosen_minutes = st.number_input("Enter number of minutes",
                                         min_value=1, value=1,
                                         key="batch_custom_time")
    else:
        chosen_minutes = int(schedule_option.split()[0])
    st.write(f"Scheduled time: **{chosen_minutes} minute(s)**")

    # Get batch size input
    batch_size = st.number_input("Batch Size (Number of data cubes per zip)",
                                 min_value=1, value=1,
                                 key="batch_size")

    # Initialize timer variables if not already set
    if "batch_total_seconds" not in st.session_state:
        st.session_state.batch_total_seconds = chosen_minutes * 60
        st.session_state.batch_remaining_seconds = st.session_state.batch_total_seconds
        st.session_state.batch_running = False

    # Manage timer control buttons
    manage_timer_controls(chosen_minutes)
    placeholder = st.empty()

    # Countdown logic with automatic download and timer restart
    if st.session_state.batch_running:
        if st.session_state.batch_remaining_seconds > 0:
            countdown_timer(placeholder)
        else:
            st.session_state.batch_running = False
            placeholder.write("Time is up. Downloading data cubes...")
            process_download(batch_size)
            st.session_state.batch_running = True
            st.session_state.batch_remaining_seconds = st.session_state.batch_total_seconds
            st.rerun()
    else:
        placeholder.write("Batch processing is stopped.")

def main():
    """Main function to run the Streamlit application."""
    st.title("Data Warehouse & Cube Generator")

    # Load data
    data_frame, all_columns = load_data()
    if data_frame is None:
        return

    # Create three tabs: Setup, OLAP Operations, and Batch Processing
    tab1, tab2, tab3 = st.tabs(["Data Warehouse Setup",
                                "Drill-Down and Roll-Up",
                                "Batch Processing"])
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            dimension_tables = configure_dimensions(all_columns)
        with col2:
            fact_measures, fact_table_cols = configure_fact_table(all_columns, dimension_tables)
            if st.button("Generate Data Warehouse", key="generate_btn"):
                try:
                    fact_df = data_frame[fact_table_cols]
                    display_results(data_frame, dimension_tables, fact_df, fact_measures)
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
                            data_frame)
        else:
            st.warning("Please generate the data warehouse first in the Setup tab.")
    with tab3:
        batch_processing()

if __name__ == "__main__":
    main()
