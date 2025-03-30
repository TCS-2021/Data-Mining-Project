import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from unittest.mock import patch
from Preprocessing.app import handle_missing_values
from Preprocessing.app import smooth_data 
from Preprocessing.app import handle_outliers 
from Preprocessing.app import analyse_variance


def load_csv(file_name):
    return pd.read_csv(f"datasets/preprocessing/test_csvs/{file_name}")


# Test mean imputation on numeric columns
def test_mean_imputation():
    df = load_csv("numeric_missing.csv")
    with patch("streamlit.selectbox") as mock_selectbox:
        mock_selectbox.side_effect = [
            "Simple Imputation (Mean/Median/Mode)",
            "Mean"
        ]
        result = handle_missing_values(df.copy())
        assert result.select_dtypes(include=np.number).isna().sum().sum() == 0


# Test mode imputation on categorical columns
def test_mode_imputation():
    df = load_csv("categorical_data.csv")
    with patch("streamlit.selectbox") as mock_selectbox:
        mock_selectbox.side_effect = [
            "Simple Imputation (Mean/Median/Mode)",
            "Mode"
        ]
        result = handle_missing_values(df.copy())
        assert result.select_dtypes(include='object').isna().sum().sum() == 0


# Test numeric missing value imputation using regression
def test_regression_imputation():
    df = load_csv("numeric_missing.csv")
    with patch("streamlit.selectbox") as mock_selectbox:
        mock_selectbox.side_effect = ["Regression Imputation"]
        result = handle_missing_values(df.copy())
        assert result.isna().sum().sum() == 0


# Test numeric imputation using decision tree
def test_decision_tree_imputation():
    df = load_csv("numeric_missing.csv")
    with patch("streamlit.selectbox") as mock_selectbox:
        mock_selectbox.side_effect = ["Decision Tree Imputation"]
        result = handle_missing_values(df.copy())
        assert result.isna().sum().sum() == 0


# Test row removal when missing values exist
def test_drop_rows():
    df = load_csv("mixed_missing.csv")
    original_len = len(df)
    with patch("streamlit.selectbox") as mock_selectbox:
        mock_selectbox.return_value = "Drop Rows with Missing Values"
        result = handle_missing_values(df.copy())
        assert len(result) < original_len


# Test smoothing (Moving Average) adds new columns
def test_smoothing_moving_average():
    df = load_csv("numeric_missing.csv").fillna(0)
    with patch("streamlit.selectbox") as mock_selectbox, patch("streamlit.slider") as mock_slider:
        mock_selectbox.return_value = "Moving Average"
        mock_slider.return_value = 3
        result = smooth_data(df.copy())
        assert any(col.endswith("_smoothed") for col in result.columns)


# Test IQR outlier detection with capping strategy
def test_outlier_detection_iqr():
    df = pd.DataFrame({"X": [1, 2, 3, 100]})
    with patch("streamlit.selectbox") as mock_selectbox, patch("streamlit.radio") as mock_radio, patch("streamlit.slider") as mock_slider:
        mock_selectbox.return_value = "IQR"
        mock_radio.return_value = "Cap"
        mock_slider.return_value = 1.5
        result = handle_outliers(df.copy())
        assert result["X"].max() < 100  # Capped


# Test variance analysis feature selection runs without crashing
def test_variance_analysis_executes():
    df = pd.DataFrame({
        "X1": [1, 2, 3, 4],
        "X2": [10, 10, 10, 10],
        "X3": [5, 15, 10, 20]
    })
    with patch("streamlit.radio") as mock_radio, patch("streamlit.slider") as mock_slider:
        mock_radio.return_value = "Top N Features"
        mock_slider.return_value = 2
        result = analyse_variance(df.copy())
        assert isinstance(result, pd.DataFrame)
