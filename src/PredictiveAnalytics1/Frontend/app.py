#app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from ..Backend.backend import load_data, preprocess_data, apply_pca, determine_problem_type, train_model
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

@st.cache_data
def cached_load_data(file_path):
    return load_data(file_path)

def get_hyperparameters_ui(model_name, problem_type, key_prefix=""):
    params = {}

    if problem_type == "regression":
        if model_name == "Random Forest":
            params['n_estimators'] = st.number_input(f"{model_name} - n_estimators", 10, 500, 100, key=key_prefix + "_n_estimators")
            params['max_depth'] = st.number_input(f"{model_name} - max_depth", 1, 50, 10, key=key_prefix + "_max_depth")
        elif model_name == "Decision Tree Regressor":
            params['max_depth'] = st.number_input(f"{model_name} - max_depth", 1, 50, 5, key=key_prefix + "_max_depth")
        elif model_name == "SVM Regressor":
            params['C'] = st.number_input(f"{model_name} - C", 0.01, 10.0, 1.0, key=key_prefix + "_C")
            params['kernel'] = st.selectbox(f"{model_name} - kernel", ['linear', 'rbf', 'poly'], key=key_prefix + "_kernel")
        elif model_name == "AdaBoost":
            params['n_estimators'] = st.number_input(f"{model_name} - n_estimators", 10, 500, 100, key=key_prefix + "_n_estimators")
        elif model_name == "Lasso":
            params['alpha'] = st.number_input(f"{model_name} - alpha", 0.01, 1.0, 0.1, key=key_prefix + "_alpha")
        elif model_name == "Ridge":
            params['alpha'] = st.number_input(f"{model_name} - alpha", 0.01, 1.0, 0.1, key=key_prefix + "_alpha")

    elif problem_type == "classification":
        if model_name == "Random Forest":
            params['n_estimators'] = st.number_input(f"{model_name} - n_estimators", 10, 500, 100, key=key_prefix + "_n_estimators")
            params['max_depth'] = st.number_input(f"{model_name} - max_depth", 1, 50, 10, key=key_prefix + "_max_depth")
        elif model_name == "Decision Tree Classifier":
            params['max_depth'] = st.number_input(f"{model_name} - max_depth", 1, 50, 5, key=key_prefix + "_max_depth")
        elif model_name == "SVM Classifier":
            params['C'] = st.number_input(f"{model_name} - C", 0.01, 10.0, 1.0, key=key_prefix + "_C")
            params['kernel'] = st.selectbox(f"{model_name} - kernel", ['linear', 'rbf', 'poly'], key=key_prefix + "_kernel")

    elif problem_type == "clustering":
        if model_name == "DBSCAN":
            params['eps'] = st.slider(f"{model_name} - eps", 0.1, 2.0, 0.5, 0.1, key=key_prefix + "_eps")
            params['min_samples'] = st.slider(f"{model_name} - min_samples", 1, 20, 5, key=key_prefix + "_min_samples")
        elif model_name == "Spectral Clustering":
            params['n_clusters'] = st.slider(f"{model_name} - n_clusters", 2, 10, 3, key=key_prefix + "_n_clusters")
            params['affinity'] = st.selectbox(f"{model_name} - affinity", ['nearest_neighbors', 'rbf'], key=key_prefix + "_affinity")

    return params

def render_cluster_chart(X_pca, hyperparams, model_name):
    import streamlit as st
    import pandas as pd
    from sklearn.cluster import DBSCAN, SpectralClustering

    cluster_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

    if model_name == "DBSCAN":
        model = DBSCAN(**hyperparams)
    elif model_name == "Spectral Clustering":
        model = SpectralClustering(**hyperparams)

    cluster_df['Cluster'] = model.fit_predict(X_pca)

    st.scatter_chart(cluster_df, x='PC1', y='PC2', color='Cluster')

def render_metrics_and_chart(comparison_df, problem_type, model1, model2, metrics1, metrics2):
    import streamlit as st

    st.subheader("Model Comparison Results")
    cols = st.columns(2)
    with cols[0]:
        if problem_type == "clustering":
            st.metric(label=f"{model1} Silhouette Score", value=metrics1['Silhouette Score'])
            st.metric(label=f"{model1} Clusters", value=metrics1['Number of Clusters'])
        elif problem_type == "classification":
            st.metric(label=f"{model1} Accuracy", value=f"{metrics1['Accuracy']:.4f}")
        elif problem_type == "regression":
            st.metric(label=f"{model1} MAE", value=f"{metrics1['MAE']:.4f}")

    with cols[1]:
        if problem_type == "clustering":
            st.metric(label=f"{model2} Silhouette Score", value=metrics2['Silhouette Score'])
            st.metric(label=f"{model2} Clusters", value=metrics2['Number of Clusters'])
        elif problem_type == "classification":
            st.metric(label=f"{model2} Accuracy", value=f"{metrics2['Accuracy']:.4f}")
        elif problem_type == "regression":
            st.metric(label=f"{model2} MAE", value=f"{metrics2['MAE']:.4f}")

    st.dataframe(comparison_df.set_index('Model').T)

    chart_data = comparison_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    chart_data['Score'] = pd.to_numeric(chart_data['Score'], errors='coerce')
    chart_data = chart_data.dropna(subset=['Score'])

    if not chart_data.empty:
        st.bar_chart(chart_data, x='Metric', y='Score', color='Model', stack=False)
    else:
        st.warning("No comparable metrics available for these model types")


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
        if not dataset_dir.exists():
            st.warning("Dataset directory not found. Please upload a CSV file.")
            dataset_files = []
        else:
            dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]

        if dataset_files:
            selected_dataset = st.selectbox("Select a dataset", dataset_files)
            dataset_path = os.path.join(dataset_dir, selected_dataset)
            df = cached_load_data(dataset_path)
        else:
            st.warning("No datasets found.")
            return

    if df is not None:

        if (len(df)>10000):
            st.write(f"Original dataset shape: {df.shape}")

            max_sample = 10000
            sample_size = st.slider("Sample size (for performance)", 1000, max_sample, 3000, step=500)
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

            st.write(f"Sampled dataset shape: {df.shape}")

        with st.expander("View Sampled Raw Data"):
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

        st.subheader("Tune Model Hyperparameters")

        with st.expander(f"{model1} Hyperparameters", expanded=True):
            hyperparams1 = get_hyperparameters_ui(model1, problem_type, key_prefix="model1")

        with st.expander(f"{model2} Hyperparameters", expanded=True):
            hyperparams2 = get_hyperparameters_ui(model2, problem_type, key_prefix="model2")

        if st.button("Compare Models"):
            with st.spinner("Training models..."):
                if problem_type == "clustering":
                    if model1 == "DBSCAN":
                        model = DBSCAN(**hyperparams1)
                    elif model1 == "Spectral Clustering":
                        model = SpectralClustering(**hyperparams1)

                    labels = model.fit_predict(X_pca)
                    metrics1 = {
                        "Model": model1,
                        "Silhouette Score": silhouette_score(X_pca, labels) if len(np.unique(labels)) > 1 else "N/A",
                        "Number of Clusters": len(np.unique(labels))
                    }

                    if model2 == "DBSCAN":
                        model = DBSCAN(**hyperparams2)
                    elif model2 == "Spectral Clustering":
                        model = SpectralClustering(**hyperparams2)

                    labels = model.fit_predict(X_pca)
                    metrics2 = {
                        "Model": model2,
                        "Silhouette Score": silhouette_score(X_pca, labels) if len(np.unique(labels)) > 1 else "N/A",
                        "Number of Clusters": len(np.unique(labels))
                    }
                else:
                    metrics1 = train_model(X_pca, y, model1, problem_type, hyperparams1)
                    metrics2 = train_model(X_pca, y, model2, problem_type, hyperparams2)

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
                        model = DBSCAN(**hyperparams1)
                    elif model1 == "Spectral Clustering":
                        model = SpectralClustering(**hyperparams1)

                    cluster_df['Cluster'] = model.fit_predict(X_pca)
                    st.scatter_chart(cluster_df, x='PC1', y='PC2', color='Cluster')

if __name__ == "__main__":
    main()