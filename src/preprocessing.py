import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("📊 Data Preprocessing Tool")
    st.write("Upload your CSV/TSV file and perform preprocessing steps")

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "tsv"])

    if uploaded_file is not None:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep='\t')

        st.session_state.df = df.copy()
        st.success("File successfully uploaded!")

        # Show original data
        with st.expander("View Raw Data"):
            st.write(df)

        # Display columns
        with st.expander("1. List Columns"):
            st.write("Columns in the dataset:")
            st.table(pd.DataFrame(df.columns, columns=["Column Names"]))

        # Missing values handling
        with st.expander("2. Handle Missing Values"):
            st.subheader("Handle Missing Values")
            cols_with_missing = df.columns[df.isna().any()].tolist()

            if cols_with_missing:
                selected_col = st.selectbox("Select column with missing values", cols_with_missing)
                method = st.radio("Select handling method",
                                ["Mean/Median Imputation", "Drop Rows with Missing Values"])

                if method == "Mean/Median Imputation":
                    if np.issubdtype(df[selected_col].dtype, np.number):
                        impute_method = st.selectbox("Select imputation method", ["Mean", "Median"])
                        if impute_method == "Mean":
                            fill_value = df[selected_col].mean()
                        else:
                            fill_value = df[selected_col].median()
                    else:
                        fill_value = df[selected_col].mode()[0]

                    df[selected_col].fillna(fill_value, inplace=True)
                    st.write(f"Filled missing values in {selected_col} with {fill_value}")
                else:
                    df.dropna(subset=[selected_col], inplace=True)
                    st.write(f"Dropped rows with missing values in {selected_col}")

                st.write("Updated Data Preview:")
                st.write(df.head())
            else:
                st.success("No missing values found in the dataset!")

        # Data smoothing
        with st.expander("3. Data Smoothing"):
            st.subheader("Data Smoothing")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                smooth_col = st.selectbox("Select column to smooth", numeric_cols)
                window_size = st.slider("Select window size for smoothing", 3, 15, 5)

                # Moving average smoothing
                df[f"{smooth_col}_smoothed"] = df[smooth_col].rolling(window=window_size).mean()
                st.line_chart(df[[smooth_col, f"{smooth_col}_smoothed"]])
                st.write("Smoothed column added to dataset")
            else:
                st.warning("No numeric columns found for smoothing")

        # Outlier handling
        with st.expander("4. Handle Outliers"):
            st.subheader("Outlier Detection & Removal")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                outlier_col = st.selectbox("Select column for outlier detection", numeric_cols)

                # IQR Method
                q1 = df[outlier_col].quantile(0.25)
                q3 = df[outlier_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)

                outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
                st.write(f"Found {len(outliers)} outliers in {outlier_col}")

                if st.checkbox("Remove outliers"):
                    df = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                    st.write(f"Removed {len(outliers)} outliers from {outlier_col}")
                    st.write("Updated Data Preview:")
                    st.write(df.head())
            else:
                st.warning("No numeric columns found for outlier detection")

        # Variance calculation
        with st.expander("5. Field Variance Analysis"):
            st.subheader("Field Variance Analysis")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                variances = df[numeric_cols].var().sort_values(ascending=False)
                st.write("Variance of numerical fields (sorted):")
                st.table(variances)

                selected_features = st.multiselect("Select critical features to keep",
                                                 numeric_cols, default=numeric_cols)
                df = df[selected_features + list(df.select_dtypes(exclude=np.number).columns)]
                st.write("Selected features updated in dataset")
            else:
                st.warning("No numeric columns found for variance calculation")

        # Download processed data
        st.subheader("Download Processed Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Preprocessed Data",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
