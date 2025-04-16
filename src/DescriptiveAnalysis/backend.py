"""Module containing backend part of data processing functions for"
" the data warehouse application with Batch Processing."""
import itertools
import io
import zipfile
import base64
import math
import pandas as pd
import streamlit as st  # Used here for caching and error messaging

def create_data_cube(df: pd.DataFrame, dimensions: list, measures: list) -> pd.DataFrame:
    """
    Create a data cube by grouping data by dimensions and aggregating measures.
    
    - Checks for missing dimension or measure columns.
    - Returns the aggregated DataFrame.
    """
    missing_dims = [dim for dim in dimensions if dim not in df.columns]
    missing_measures = [measure for measure in measures if measure not in df.columns]
    if missing_dims:
        st.error(f"Missing dimension columns: {', '.join(missing_dims)}")
        return pd.DataFrame()
    if missing_measures:
        st.error(f"Missing measure columns: {', '.join(missing_measures)}")
        return pd.DataFrame()
    return df.groupby(dimensions).agg({measure: 'sum' for measure in measures}).reset_index()

@st.cache_data
def generate_data_cubes(fact_df: pd.DataFrame, dimension_tables: dict, fact_measures: list) -> dict:
    """
    Generate all possible data cubes based on different combinations of dimension tables.
    
    For each combination of dimensions, the function creates a cube using
    the `create_data_cube` function.
    """
    all_cubes = {}
    dim_keys = list(dimension_tables.keys())
    for n in range(1, len(dim_keys) + 1):
        dim_combinations = list(itertools.combinations(dim_keys, n))
        for combo in dim_combinations:
            # Use primary keys of selected dimensions
            dim_columns = [dimension_tables[dim_name]['primary_key'] for dim_name in combo]
            cube = create_data_cube(fact_df, dim_columns, fact_measures)
            if not cube.empty:
                cube_name = " × ".join(combo)
                all_cubes[cube_name] = cube
    return all_cubes

def package_batch(batch_cubes, batch_number):
    """
    Package a batch of cubes into a zip file and return its base64 string and filename.
    
    This is useful for splitting large numbers of cubes into smaller batches.
    """
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
    """
    Package cubes into multiple zip files based on the batch size and
    build HTML with JavaScript to trigger downloads in the browser.
    """
    cube_items = list(cubes.items())
    num_batches = math.ceil(len(cube_items) / batch_size)
    zip_files = [
        package_batch(cube_items[i * batch_size:(i + 1) * batch_size], i + 1)
        for i in range(num_batches)
    ]
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
    return download_html

def process_download(batch_size, fact_df, dimension_tables, fact_measures):
    """
    Generate data cubes and prepare the HTML content required to download the cubes as ZIP files.
    
    Returns the download HTML that is rendered on the UI.
    """
    cubes = generate_data_cubes(fact_df, dimension_tables, fact_measures)
    if cubes:
        return download_zip_files(cubes, batch_size)
    st.error("No data cubes generated. Please check your configuration.")
    return None
