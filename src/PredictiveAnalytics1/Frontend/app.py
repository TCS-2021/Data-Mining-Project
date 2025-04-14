import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.backend import (
    load_data, preprocess_data, apply_pca, determine_problem_type, train_model)
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


st.set_page_config(page_title="Model Comparison Tool", layout="wide")


@st.cache_data
def cached_load_data(file_path):
    return load_data(file_path)


def main():
    st.title("Machine Learning Model Comparison Tool")

    data_choice = st.radio("Choose how you'd like to provide data:", ["Upload your own CSV", "Use a predefined dataset"])

    df = None
    uploaded_file = None

    if data_choice == "Upload your own CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = cached_load_data(uploaded_file)
    else:
        current_file = Path(__file__).resolve()
        dataset_dir = current_file.parents[3] / 'Datasets' / 'predictive-analytics-1'
        dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]

        if dataset_files:
            selected_dataset = st.selectbox("Select a dataset", dataset_files)
            dataset_path = os.path.join(dataset_dir, selected_dataset)
            df = cached_load_data(dataset_path)
        else:
            st.warning("No datasets found.")
            return

    if df is not None:
        with st.expander("View Raw Data"):
            st.dataframe(df.head())

        clustering_mode = st.checkbox("Enable clustering mode (no target variable)")

        if clustering_mode:
            target_col = None
            df = preprocess_data(df)
            X = df.values
            problem_type = "clustering"
        else:
            target_col = st.selectbox("Select the target variable", df.columns)
            df = preprocess_data(df, target_col)
            X = df.drop(columns=[target_col]).values
            y = df[target_col].values
            problem_type = determine_problem_type(y)

            if problem_type == "classification":
                le = LabelEncoder()
                y = le.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_pca = apply_pca(X_scaled)


        st.subheader("Model Selection")

        if problem_type == "regression":
            model_options = [
                "Linear Regression", "Decision Tree Regressor",
                "Random Forest", "SVM Regressor", "AdaBoost",
                "Lasso", "Ridge"
            ]
        elif problem_type == "classification":
            model_options = [
                "Logistic Regression", "Decision Tree Classifier",
                "Random Forest", "SVM Classifier", "Naive Bayes"
            ]
        else:  # clustering
            model_options = [
                "DBSCAN", "Spectral Clustering"
            ]

        col1, col2 = st.columns(2)
        with col1:
            model1 = st.selectbox("Select first model", model_options, index=0)
        with col2:
            other_models = [m for m in model_options if m != model1]
            model2 = st.selectbox("Select second model", other_models, index=0)

        if problem_type == "clustering":
            if model1 == "DBSCAN" or model2 == "DBSCAN":
                eps = st.slider("DBSCAN eps parameter", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("DBSCAN min_samples parameter", 1, 10, 5)
            if model1 == "Spectral Clustering" or model2 == "Spectral Clustering":
                n_clusters = st.slider("Number of clusters for Spectral Clustering", 2, 10, 3)

        if st.button("Compare Models"):
            with st.spinner("Training models..."):
                if problem_type == "clustering":
                    if model1 == "DBSCAN":
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                    elif model1 == "Spectral Clustering":
                        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')

                    labels = model.fit_predict(X_pca)
                    metrics1 = {
                        "Model": model1,
                        "Silhouette Score": silhouette_score(X_pca, labels) if len(np.unique(labels)) > 1 else "N/A",
                        "Number of Clusters": len(np.unique(labels))
                    }

                    if model2 == "DBSCAN":
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                    elif model2 == "Spectral Clustering":
                        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')

                    labels = model.fit_predict(X_pca)
                    metrics2 = {
                        "Model": model2,
                        "Silhouette Score": silhouette_score(X_pca, labels) if len(np.unique(labels)) > 1 else "N/A",
                        "Number of Clusters": len(np.unique(labels))
                    }
                else:
                    metrics1 = train_model(X_pca, y, model1, problem_type)
                    metrics2 = train_model(X_pca, y, model2, problem_type)

                comparison_df = pd.DataFrame([metrics1, metrics2])

                st.subheader("Model Comparison Results")

                cols = st.columns(2)
                with cols[0]:
                    if problem_type == "clustering":
                        st.metric(label=f"{model1} Silhouette Score",
                                value=f"{metrics1['Silhouette Score']:.4f}" if metrics1['Silhouette Score'] != "N/A" else "N/A")
                        st.metric(label=f"{model1} Clusters", value=metrics1['Number of Clusters'])
                    else:
                        if problem_type == "classification":
                            st.metric(label=f"{model1} Accuracy",
                                value=f"{metrics1['Accuracy']:.4f}" if metrics1['Accuracy'] != "-" else "N/A")
                        if problem_type == "regression":
                            st.metric(label=f"{model1} MAE", value=f"{metrics1['MAE']:.4f}")

                with cols[1]:
                    if problem_type == "clustering":
                        st.metric(label=f"{model2} Silhouette Score",
                                value=f"{metrics2['Silhouette Score']:.4f}" if metrics2['Silhouette Score'] != "N/A" else "N/A")
                        st.metric(label=f"{model2} Clusters", value=metrics2['Number of Clusters'])
                    else:
                        if problem_type == "classification":
                            st.metric(label=f"{model2} Accuracy",
                                value=f"{metrics2['Accuracy']:.4f}" if metrics2['Accuracy'] != "-" else "N/A")
                        if problem_type == "regression":
                            st.metric(label=f"{model2} MAE", value=f"{metrics2['MAE']:.4f}")

                st.dataframe(comparison_df.set_index('Model').T)


                st.subheader("Performance Visualization")
                chart_data = comparison_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
                chart_data['Score'] = pd.to_numeric(chart_data['Score'], errors='coerce')
                chart_data = chart_data.dropna(subset=['Score'])


                if not chart_data.empty:
                    st.bar_chart(chart_data, x='Metric', y='Score', color='Model', stack=False)
                else:
                    st.warning("No comparable metrics available for these model types")

                if problem_type == "clustering":
                    st.subheader("Cluster Visualization")
                    cluster_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

                    if model1 == "DBSCAN":
                        model = DBSCAN(eps=eps, min_samples=min_samples)
                    elif model1 == "Spectral Clustering":
                        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')

                    cluster_df['Cluster'] = model.fit_predict(X_pca)
                    st.scatter_chart(cluster_df, x='PC1', y='PC2', color='Cluster')

if __name__ == "__main__":
    main()