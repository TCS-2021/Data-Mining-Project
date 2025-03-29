import streamlit as st
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess
import plotly.express as px


# Configure page
st.set_page_config(
    page_title="Data Preprocessor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stDataFrame {
        border: 1px solid #444;
        border-radius: 5px;
    }
    .stExpander {
        background: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
    }
    .highlight {
        background-color: #fffd80;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Handle missing values with various imputation methods
def handle_missing_values(df):
    st.subheader("2. Handle Missing Values")
    with st.expander("Handle Missing Values"):

        if df.isna().sum().sum() == 0:
            st.success("No missing values found in the dataset!")
            return df

        all_cols = df.columns[df.isna().any()].tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        method = st.selectbox("Select imputation method", [
                "Simple Imputation (Mean/Median/Mode)",
                "Regression Imputation",
                "Decision Tree Imputation",
                "Drop Rows with Missing Values"
            ])
        
        if method == "Drop Rows with Missing Values":
            df.dropna(subset=all_cols, inplace=True)
            st.write(f"Dropped rows with missing values in all the columns")

        else:
            if numeric_cols:
                if method == "Simple Imputation (Mean/Median/Mode)":
                    strategy = st.selectbox("Select strategy", ["Mean", "Median", "Mode"])

                    for col in numeric_cols:
                        if strategy == "Mean":
                            fill_value = df[col].mean()
                        elif strategy == "Median":
                            fill_value = df[col].median()
                        else:  # Mode
                            fill_value = df[col].mode()[0]

                        df[col].fillna(fill_value, inplace=True)
                        st.write(f"Filled missing values in the numerical column {col} with {fill_value}")

                elif method in ["Regression Imputation", "Decision Tree Imputation"]:
                    if method == "Regression Imputation":
                        imputer = IterativeImputer(random_state=42, max_iter=50, tol=1e-3)
                        st.write("Performed regression imputation on all numeric columns")
                    else:
                        imputer = IterativeImputer(estimator=DecisionTreeRegressor(), random_state=42)
                        st.write("Performed decision tree imputation on all numeric columns")

                    # Impute numeric columns
                    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

            else:
                st.warning("No numeric columns found!")

            # Impute non-numeric columns using mode
            non_numeric_cols = [col for col in all_cols if col not in numeric_cols]
            for col in non_numeric_cols:
                fill_value = df[col].mode()[0] if not df[col].mode().empty else ""
                df[col].fillna(fill_value, inplace=True)
                st.write(f"Filled missing values in the categorical column {col} with mode {fill_value}")

        st.write("Updated Data Preview:")
        st.write(df.head())
    return df

#Smooth data using various methods
def smooth_data(df):
    st.subheader("3. Smooth Data") 
    with st.expander("Smooth Your Data"): 
        
        # Check for numeric columns 
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.error("Oops! No numeric columns found in your data!")
            return df  

        method = st.selectbox("Select smoothing method", [
            "Moving Average",
            "Exponential",
            "Gaussian",
            "LOESS"
        ], key="smoothing_method")

        # Window size selection for smoothing
        window_size = st.slider(
            "Smoothing amount:",
            3,    # Minimum smoothing
            15,   # Maximum smoothing
            5,    # Default setting
            key="smoothing_window"
        )

        with st.spinner('Smoothing your data...'):
            # Apply smoothing to ALL numeric columns
            for col in numeric_cols:
                if method == "Moving Average":
                    df[f"{col}_smoothed"] = df[col].rolling(window=window_size).mean()
                elif method == "Exponential":
                    df[f"{col}_smoothed"] = df[col].ewm(span=window_size).mean()
                elif method == "Gaussian":
                    df[f"{col}_smoothed"] = gaussian_filter1d(df[col], sigma=window_size/3)
                else:  # LOESS
                    smoothed = lowess(df[col], np.arange(len(df)), frac=window_size/len(df))
                    df[f"{col}_smoothed"] = smoothed[:, 1]

            # Store and show preview
            st.session_state.smoothing_df = df.copy()
            
            st.write("Smoothed Data Preview:")
            st.write(df.head())  
        
            fig = px.line(
                df,
                y=[f"{col}_smoothed" for col in numeric_cols],
                title="Visualization of Smoothed Data"
            )
            st.plotly_chart(fig, use_container_width=True)

    return df  

def handle_outliers(df):
    st.subheader("4. Handle Outliers")
    with st.expander("Outlier Handling", expanded=False):
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.error("No numeric columns found!")
            return df

        col1, col2 = st.columns(2)
        method = st.selectbox("Detection method:", ["IQR", "Z-Score", "Modified Z-Score", "Percentile"], key="detection_method")

        if method == "IQR":
            multiplier = st.slider("IQR multiplier:", 1.0, 5.0, 1.5)
        elif method in ["Z-Score", "Modified Z-Score"]:
            threshold = st.slider("Threshold value:", 1.0, 5.0, 3.0)
        else:  # Percentile-based
            col1, col2 = st.columns(2)
            with col1:
                lower_pct = st.slider("Lower percentile:", 0.0, 10.0, 1.0)
            with col2:
                upper_pct = st.slider("Upper percentile:", 90.0, 100.0, 99.0)

        treatment = st.radio("Treatment method:", ["Remove", "Cap", "Replace with median"], horizontal=True)

        with st.spinner('Processing...'):
            total_outliers = 0
            for col in numeric_cols:
                if method == "IQR":
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - (multiplier * iqr)
                    upper_bound = q3 + (multiplier * iqr)
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                elif method == "Z-Score":
                    z_scores = (df[col] - df[col].mean()) / df[col].std()
                    outliers = df[np.abs(z_scores) > threshold]
                elif method == "Modified Z-Score":
                    median = df[col].median()
                    mad = np.median(np.abs(df[col] - median))
                    modified_z_scores = 0.6745 * (df[col] - median) / mad
                    outliers = df[np.abs(modified_z_scores) > threshold]
                else:  # Percentile-based
                    lower_bound = df[col].quantile(lower_pct/100)
                    upper_bound = df[col].quantile(upper_pct/100)
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

                num_outliers = len(outliers)
                total_outliers += num_outliers

                if num_outliers > 0:
                    if treatment == "Remove":
                        df = df[~df.index.isin(outliers.index)]
                    elif treatment == "Cap":
                        if method in ["IQR", "Percentile"]:
                            df[col] = np.where(df[col] > upper_bound, upper_bound,
                                                    np.where(df[col] < lower_bound, lower_bound,
                                                            df[col]))
                        else:
                            median_val = df[col].median()
                            df[col] = np.where(np.abs((df[col] - median_val)/df[col].std()) > threshold,
                                                    median_val, df[col])
                    else:  # Replace with median
                        median_val = df[col].median()
                        df.loc[outliers.index, col] = median_val

                    st.write(f"Would process {num_outliers} outliers in {col}")

            st.write(f"Total outliers that would be processed: {total_outliers}")

            st.write("Updated Preview:")
            st.write(df.head())  
        
            fig = px.box(df, title="After Outlier Treatment Preview")
            st.plotly_chart(fig, use_container_width=True)

    return df

def main():
    st.title("Data Preprocessing Tool")

    # File upload section
    with st.container():
        st.subheader("1. Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed"
        )

    if uploaded_file:
        # Read data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Store in session state
            st.session_state.df = df

            # Show basic info
            st.success(f"✅ Successfully loaded {len(df)} rows × {len(df.columns)} columns")
            
            # Display raw data
            with st.expander("🔍 View Raw Data", expanded=False):
                st.dataframe(
                    df.style.highlight_null(color='#ffcccb'), 
                    height=300,
                    use_container_width=True
                )

            # Column information
            with st.expander("📋 Column Information", expanded=True):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str).values,
                    'Missing Values': df.isna().sum(),
                    'Unique Values': df.nunique(),
                    'Sample Value': df.iloc[0].astype(str).values
                })
                
                col_info['Type'] = col_info['Type'].replace('object', 'string')
                st.dataframe(col_info, use_container_width=True)

                # Quick stats
                st.markdown(f"""
                - **Total Columns:** {len(df.columns)}
                - **Total Rows:** {len(df)}
                - **Missing Values:** {df.isna().sum().sum()}
                - **Duplicate Rows:** {df.duplicated().sum()}
                """)

            # Handle missing values
            processed_df = handle_missing_values(df)

            # Smoothing data
            processed_df = smooth_data(processed_df)

            # Handle outliers
            processed_df = handle_outliers(processed_df)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()