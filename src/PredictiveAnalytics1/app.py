import streamlit as st
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

st.set_page_config(page_title="Model Comparison Tool", layout="wide")

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, target_col):
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
        if col != target_col:
            df[col] = le.fit_transform(df[col])

    MinMax = MinMaxScaler()
    for col in numeric:
        if col != target_col:
            df[col] = MinMax.fit_transform(df[[col]])

    return df

def determine_problem_type(y):
    if len(np.unique(y)) > 20 or pd.api.types.is_float_dtype(y):
        return "regression"
    return "classification"

def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def train_model(X, y, model_type, problem_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if problem_type == "regression":
        if model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Decision Tree Regressor":
            model = DecisionTreeRegressor()
        elif model_type == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "SVM Regressor":
            model = SVR()
        elif model_type == "AdaBoost":
            model = AdaBoostRegressor(n_estimators=100, random_state=42)
        elif model_type == "Lasso":
            model = Lasso(alpha=0.1)
        elif model_type == "Ridge":
            model = Ridge(alpha=0.1)

    else:  # classification
        if model_type == "Logistic Regression":
            model = LogisticRegression()
        elif model_type == "Decision Tree Classifier":
            model = DecisionTreeClassifier()
        elif model_type == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "SVM Classifier":
            model = SVC()
        elif model_type == "Naive Bayes":
            model = GaussianNB()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if problem_type == "classification":
        metrics = {
            "Model": model_type,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": "-",
            "Recall": "-",
            "F1": "-"
        }
    else:
        metrics = {
            "Model": model_type,
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R² Score": r2_score(y_test, y_pred),
            "Accuracy": "-"
        }

    return metrics

def main():
    st.title("Machine Learning Model Comparison Tool")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        with st.expander("View Raw Data"):
            st.dataframe(df.head())

        target_col = st.selectbox("Select the target variable", df.columns)

        df = preprocess_data(df, target_col)

        X = df.drop(columns=[target_col])
        y = df[target_col]

        problem_type = determine_problem_type(y)
        if problem_type == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.info("This appears to be a classification problem")
        else:
            st.info("This appears to be a regression problem")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_pca = apply_pca(X_scaled)

        # Model selection based on problem type
        st.subheader("Model Selection")

        if problem_type == "regression":
            model_options = [
                "Linear Regression", "Decision Tree Regressor",
                "Random Forest", "SVM Regressor", "AdaBoost",
                "Lasso", "Ridge"
            ]
        else:
            model_options = [
                "Logistic Regression", "Decision Tree Classifier",
                "Random Forest", "SVM Classifier", "Naive Bayes"
            ]

        col1, col2 = st.columns(2)
        with col1:
            model1 = st.selectbox("Select first model", model_options, index=0)
        with col2:
            other_models = [m for m in model_options if m != model1]
            model2 = st.selectbox("Select second model", other_models, index=0)

        if st.button("Compare Models"):
            with st.spinner("Training models..."):
                metrics1 = train_model(X_pca, y, model1, problem_type)
                metrics2 = train_model(X_pca, y, model2, problem_type)

                comparison_df = pd.DataFrame([metrics1, metrics2])

                # Display results
                st.subheader("Model Comparison Results")

                cols = st.columns(2)
                with cols[0]:
                    st.metric(label=f"{model1} Accuracy",
                            value=f"{metrics1['Accuracy']:.4f}" if metrics1['Accuracy'] != "-" else "N/A")
                    if problem_type == "regression":
                        st.metric(label=f"{model1} MAE", value=f"{metrics1['MAE']:.4f}")

                with cols[1]:
                    st.metric(label=f"{model2} Accuracy",
                            value=f"{metrics2['Accuracy']:.4f}" if metrics2['Accuracy'] != "-" else "N/A")
                    if problem_type == "regression":
                        st.metric(label=f"{model2} MAE", value=f"{metrics2['MAE']:.4f}")

                st.dataframe(comparison_df.set_index('Model').T)

                # Visualization
                st.subheader("Performance Visualization")
                chart_data = comparison_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
                chart_data = chart_data[chart_data['Score'] != '-']

                if not chart_data.empty:
                    st.bar_chart(chart_data, x='Metric', y='Score', color='Model')
                else:
                    st.warning("No comparable metrics available for these model types")

if __name__ == "__main__":
    main()