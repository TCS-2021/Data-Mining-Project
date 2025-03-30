import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from scipy.ndimage import gaussian_filter1d
from statsmodels.nonparametric.smoothers_lowess import lowess


# Page config and custom UI styling
st.set_page_config(page_title="Data Preprocessor", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stDataFrame {
        border: 1px solid #444;
        border-radius: 5px;
    }
    .stExpander {
        border-radius: 5px;
        padding: 10px;
    }
    .highlight {
        background-color: #fffd80;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values using multiple imputation strategies.
    Returns the modified DataFrame.
    """
    st.subheader("2. Handle Missing Values")
    with st.expander("Handle Missing Values"):
        if data.isna().sum().sum() == 0:
            st.success("No missing values found in the dataset!")
            return data

        columns_with_na = data.columns[data.isna().any()].tolist()
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

        # Select method for imputation
        imputation_method = st.selectbox("Select imputation method", [
            "Simple Imputation (Mean/Median/Mode)",
            "Regression Imputation",
            "Decision Tree Imputation",
            "Drop Rows with Missing Values"
        ])

        if imputation_method == "Drop Rows with Missing Values":
            data.dropna(subset=columns_with_na, inplace=True)
            st.write("Dropped rows with missing values.")
        else:
            if numeric_columns:
                if imputation_method == "Simple Imputation (Mean/Median/Mode)":
                    strategy = st.selectbox("Select strategy", ["Mean", "Median", "Mode"])
                    
                    for column in numeric_columns:
                        # Compute value based on strategy
                        imputed_value = (
                            data[column].mean() if strategy == "Mean"
                            else data[column].median() if strategy == "Median"
                            else data[column].mode()[0]
                        )
                        data[column] = data[column].fillna(imputed_value)
                        st.write(f"Filled missing values in {column} with {strategy.lower()}: {imputed_value:.2f}")
                else:
                    # Use regression or tree-based imputation
                    imputer = IterativeImputer(
                        estimator=DecisionTreeRegressor() if imputation_method == "Decision Tree Imputation" else None,
                        random_state=42
                    )
                    data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
                    st.write(f"{imputation_method} applied on numeric columns.")

            # Fill categorical columns with mode
            categorical_columns = [col for col in columns_with_na if col not in numeric_columns]
            for column in categorical_columns:
                mode_value = data[column].mode()[0] if not data[column].mode().empty else ""
                data[column].fillna(mode_value, inplace=True)
                st.write(f"Filled missing values in {column} with mode: {mode_value}")

        st.write("**Preview after missing value handling:**")
        st.dataframe(data.head())
        return data


def smooth_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applies smoothing to all numeric columns using selected technique.
    Returns the modified DataFrame with new smoothed columns.
    """
    st.subheader("3. Smooth Data")
    with st.expander("Smooth Your Data"):
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        if not numeric_columns:
            st.error("No numeric columns available.")
            return data

        smoothing_method = st.selectbox("Select smoothing method", ["Moving Average", "Exponential", "Gaussian", "LOESS"])
        smoothing_window = st.slider("Smoothing intensity", 3, 15, 5)

        for column in numeric_columns:
            # Apply selected smoothing technique
            if smoothing_method == "Moving Average":
                data[f"{column}_smoothed"] = data[column].rolling(window=smoothing_window).mean()
            elif smoothing_method == "Exponential":
                data[f"{column}_smoothed"] = data[column].ewm(span=smoothing_window).mean()
            elif smoothing_method == "Gaussian":
                data[f"{column}_smoothed"] = gaussian_filter1d(data[column], sigma=smoothing_window / 3)
            else:
                loess_result = lowess(data[column], np.arange(len(data)), frac=smoothing_window / len(data))
                data[f"{column}_smoothed"] = loess_result[:, 1]

        # Store a backup and display results
        st.session_state.smoothing_df = data.copy()
        st.write("Preview after smoothing:")
        st.dataframe(data.head())

        # Plot all smoothed columns
        smoothed_cols = [f"{col}_smoothed" for col in numeric_columns if f"{col}_smoothed" in data.columns]
        fig = px.line(data[smoothed_cols], title="Smoothed Data Visualization")

        st.plotly_chart(fig, use_container_width=True)
        return data


def handle_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Detects and treats outliers in numeric columns using selected detection and treatment strategies.
    Returns the modified DataFrame.
    """
    st.subheader("4. Handle Outliers")
    with st.expander("Outlier Handling"):
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        if not numeric_columns:
            st.error("No numeric columns available.")
            return data

        detection_method = st.selectbox("Detection method", ["IQR", "Z-Score", "Modified Z-Score", "Percentile"])
        treatment_strategy = st.radio("Treatment", ["Remove", "Cap", "Replace with median"], horizontal=True)

        threshold, lower_percentile, upper_percentile = 3.0, 1.0, 99.0
        iqr_multiplier = 1.5

        # Get user input based on method
        if detection_method == "IQR":
            iqr_multiplier = st.slider("IQR Multiplier", 1.0, 5.0, 1.5)

        elif detection_method in ["Z-Score", "Modified Z-Score"]:
            threshold = st.slider("Threshold", 1.0, 5.0, 3.0)        
        else:
            lower_percentile = st.slider("Lower percentile", 0.0, 10.0, 1.0)
            upper_percentile = st.slider("Upper percentile", 90.0, 100.0, 99.0)

        total_outlier_count = 0

        for column in numeric_columns:
            # Detect outliers based on selected method
            if detection_method == "IQR":
                q1 = data[column].quantile(0.25)
                q3 = data[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            
            elif detection_method == "Z-Score":
                z_scores = (data[column] - data[column].mean()) / data[column].std()
                outliers = data[np.abs(z_scores) > threshold]
            
            elif detection_method == "Modified Z-Score":
                median = data[column].median()
                median_abs_dev = np.median(np.abs(data[column] - median))
                modified_z_scores = 0.6745 * (data[column] - median) / median_abs_dev
                outliers = data[np.abs(modified_z_scores) > threshold]
            
            else:
                lower_bound = data[column].quantile(lower_percentile / 100)
                upper_bound = data[column].quantile(upper_percentile / 100)
                outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

            num_outliers = len(outliers)
            total_outlier_count += num_outliers

            # Apply chosen treatment to outliers
            if num_outliers > 0:
                if treatment_strategy == "Remove":
                    data = data[~data.index.isin(outliers.index)]
                
                elif treatment_strategy == "Cap":
                    data[column] = np.clip(data[column], lower_bound, upper_bound)
                
                else:
                    median_value = data[column].median()
                    data.loc[outliers.index, column] = median_value

                st.write(f"{num_outliers} outliers processed in '{column}'")

        st.write(f"Total outliers handled: {total_outlier_count}")
        st.dataframe(data.head())

        # Show box plot for updated data
        fig = px.box(data.select_dtypes(include=np.number), title="Box Plot After Outlier Treatment")
        st.plotly_chart(fig, use_container_width=True)
        return data


def analyse_variance(data: pd.DataFrame) -> pd.DataFrame:
    """
    Performs variance analysis and feature selection on numeric columns.
    Allows selection by threshold, top-N, or manual choice.
    Optionally filters the dataset and downloads selected features.
    Returns the modified DataFrame (optionally filtered).
    """
    st.subheader("5. Variance Analysis")
    with st.expander("Analyse Feature Variance", expanded=False):
        numeric_columns = data.select_dtypes(include=np.number).columns.tolist()

        if not numeric_columns:
            st.error("No numeric columns found for variance analysis!")
            return data

        # Compute variance, standard deviation, and coefficient of variation
        variance_summary = pd.DataFrame({
            'Feature': numeric_columns,
            'Variance': [data[column].var() for column in numeric_columns],
            'Standard Deviation': [data[column].std() for column in numeric_columns],
            'Coefficient of Variation': [
                data[column].std() / data[column].mean() if data[column].mean() != 0 else np.nan
                for column in numeric_columns
            ]
        }).sort_values('Variance', ascending=False).reset_index(drop=True)

        # Tabs for chart view and table view
        tab_visuals, tab_table = st.tabs(["Visualization", "Data"])

        with tab_visuals:
            st.subheader("Feature Variance Distribution")

            # Variance bar chart
            fig_variance = px.bar(
                variance_summary,
                x='Feature',
                y='Variance',
                title="Feature Variance Comparison",
                color='Variance',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_variance, use_container_width=True)

            # Coefficient of Variation chart
            cv_data = variance_summary.dropna(subset=['Coefficient of Variation'])
            if not cv_data.empty:
                fig_cv = px.bar(
                    cv_data,
                    x='Feature',
                    y='Coefficient of Variation',
                    title="Coefficient of Variation (Higher values = more variability relative to mean)",
                    color='Coefficient of Variation',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_cv, use_container_width=True)

        with tab_table:
            st.dataframe(variance_summary, use_container_width=True)

        st.subheader("Feature Selection")

        # Choose selection method
        selection_strategy = st.radio(
            "Selection method:",
            ["Variance Threshold", "Top N Features", "Manual Selection"],
            horizontal=True
        )

        if selection_strategy == "Variance Threshold":
            min_variance = st.slider(
                "Minimum variance threshold:",
                min_value=0.0,
                max_value=float(variance_summary['Variance'].max()),
                value=0.1,
                step=0.05
            )
            selected_features = variance_summary[variance_summary['Variance'] >= min_variance]['Feature'].tolist()

        elif selection_strategy == "Top N Features":
            top_n = st.slider(
                "Number of top features to keep:",
                min_value=1,
                max_value=len(numeric_columns),
                value=min(5, len(numeric_columns)),
                step=1
            )
            selected_features = variance_summary.head(top_n)['Feature'].tolist()

        else:
            # Manual feature picker
            selected_features = st.multiselect(
                "Select features to keep:",
                options=numeric_columns,
                default=numeric_columns[:min(5, len(numeric_columns))]
            )

        if selected_features:
            st.success(f"Selected {len(selected_features)} features based on variance analysis")
            st.write("Selected features:")
            st.write(", ".join(selected_features))

            # Optionally create new dataset with selected features
            if st.checkbox("Create dataset with only selected features"):
                include_categorical = st.checkbox("Include non-numeric columns", value=True)
                non_numeric_columns = [col for col in data.columns if col not in numeric_columns]

                final_columns = selected_features + non_numeric_columns if include_categorical else selected_features
                selected_data = data[final_columns].copy()

                st.write("Preview of dataset with selected features:")
                st.dataframe(selected_data.head(), use_container_width=True)

                # Download option
                csv_data = selected_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download selected features dataset",
                    data=csv_data,
                    file_name="selected_features_dataset.csv",
                    mime="text/csv"
                )

                # Optionally update original dataset
                if st.checkbox("Update main dataset to only include selected features"):
                    data = selected_data.copy()
                    st.info("Main dataset updated to include only selected features")
        else:
            st.warning("No features selected. Please adjust your selection criteria.")

        # Correlation analysis among selected features
        if len(selected_features) > 1:
            st.subheader("Correlation Analysis for Selected Features")
            correlation_matrix = data[selected_features].corr()

            fig_corr = px.imshow(
                correlation_matrix,
                title="Correlation Matrix for Selected Features",
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                width=700, height=600
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            correlation_threshold = st.slider("Correlation threshold for highlighting:", 0.0, 1.0, 0.8, 0.05)

            # Find pairs with high correlation
            high_correlation_pairs = correlation_matrix.where(
                (np.abs(correlation_matrix) > correlation_threshold) & (np.abs(correlation_matrix) < 1.0)
            ).stack().reset_index()

            high_correlation_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']

            if not high_correlation_pairs.empty:
                st.write(f"Feature pairs with correlation above {correlation_threshold}:")
                st.dataframe(
                    high_correlation_pairs.sort_values('Correlation', ascending=False),
                    use_container_width=True
                )
            else:
                st.info(f"No feature pairs with correlation above {correlation_threshold} found.")

    return data


def main():
    """
    Streamlit app entry point for preprocessing tool.
    Handles file upload, data preview, and sequential preprocessing steps.
    """
    st.title("📊 Data Preprocessing Tool")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        try:
            # Read the uploaded file
            raw_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.df = raw_data

            st.success(f"✅ Loaded {len(raw_data)} rows × {len(raw_data.columns)} columns")

            with st.expander("🔍 View Raw Data"):
                try:
                    styled = raw_data.style.applymap(lambda v: 'background-color: #c94444' if pd.isna(v) else '')
                    st.dataframe(styled, height=300, use_container_width=True)
                except Exception:
                    st.dataframe(raw_data, height=300, use_container_width=True)

            with st.expander("📋 Column Summary", expanded=True):
                column_summary = pd.DataFrame({
                    'Column': raw_data.columns,
                    'Type': raw_data.dtypes.astype(str),
                    'Missing': raw_data.isna().sum(),
                    'Unique': raw_data.nunique(),
                    'Example': raw_data.iloc[0].astype(str)
                })
                column_summary['Type'] = column_summary['Type'].replace('object', 'string')
                st.dataframe(column_summary, use_container_width=True)

                st.markdown(f"""
                - **Columns:** {len(raw_data.columns)}
                - **Rows:** {len(raw_data)}
                - **Missing Values:** {raw_data.isna().sum().sum()}
                - **Duplicate Rows:** {raw_data.duplicated().sum()}
                """)

            # Apply preprocessing steps sequentially
            processed_data = handle_missing_values(raw_data)
            processed_data = smooth_data(processed_data)
            processed_data = handle_outliers(processed_data)
            processed_data = analyse_variance(processed_data)

            # Downloading feature :  Preprocessed dataset
            st.subheader("📥 Download Final Preprocessed Dataset")
            csv_final = processed_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Final Dataset as CSV",
                data=csv_final,
                file_name="preprocessed_dataset.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Error loading file: {str(error)}")

if __name__ == "__main__":
    main()
