#backend.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, silhouette_score,
    precision_score, recall_score, f1_score
)
from sklearn.cluster import DBSCAN, SpectralClustering


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df, target_col=None):
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    le = LabelEncoder()
    categorical = df.select_dtypes(include=['object'])
    numeric = df.select_dtypes(include=['int64', 'float64'])

    for col in categorical:
        if target_col is None or col != target_col:
            df[col] = le.fit_transform(df[col])

    MinMax = MinMaxScaler()
    for col in numeric:
        if target_col is None or col != target_col:
            df[col] = MinMax.fit_transform(df[[col]])

    return df


def determine_problem_type(y):
    if y is None:
        return "clustering"
    if pd.api.types.is_object_dtype(y) or len(np.unique(y)) <= 20:
        return "classification"
    return "regression"


def apply_pca(X, n_components=2):
    n_components = min(n_components, X.shape[1])
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)


def train_model(X, y, model_type, problem_type, hyperparams=None):
    hyperparams = hyperparams or {}

    if problem_type == "clustering":
        if model_type == "DBSCAN":
            model = DBSCAN(**hyperparams)
        elif model_type == "Spectral Clustering":
            model = SpectralClustering(**hyperparams)
        else:
            raise ValueError("Unsupported clustering model.")

        labels = model.fit_predict(X)
        metrics = {
            "Model": model_type,
            "Silhouette Score": silhouette_score(X, labels) if len(np.unique(labels)) > 1 else "N/A",
            "Number of Clusters": len(np.unique(labels))
        }
        return metrics

    # For supervised learning
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_classes = {
        "regression": {
            "Linear Regression": LinearRegression,
            "Decision Tree Regressor": DecisionTreeRegressor,
            "Random Forest": RandomForestRegressor,
            "SVM Regressor": SVR,
            "AdaBoost": AdaBoostRegressor,
            "Lasso": Lasso,
            "Ridge": Ridge
        },
        "classification": {
            "Logistic Regression": LogisticRegression,
            "Decision Tree Classifier": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "SVM Classifier": SVC,
            "Naive Bayes": GaussianNB
        }
    }

    model_map = model_classes.get(problem_type)
    if not model_map or model_type not in model_map:
        raise ValueError("Unsupported model type or problem type.")

    # Instantiate model with custom hyperparameters
    model_class = model_map[model_type]
    model = model_class(**hyperparams)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        return {
            "Model": model_type,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }
    else:  # regression
        return {
            "Model": model_type,
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R² Score": r2_score(y_test, y_pred),
        }