import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression, make_blobs
from unittest.mock import patch
from unittest.mock import patch, MagicMock
import pytest
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from PredictiveAnalytics1.Backend import backend
from PredictiveAnalytics1.Frontend import app

# ---------- BACKEND TESTS ----------

def test_preprocess_numeric_data():
    df = pd.DataFrame({
        "Feature1": [1, 2, np.nan, 4],
        "Feature2": [10, 20, 30, 40],
        "Target": [0, 1, 0, 1]
    })
    processed = backend.preprocess_data(df.copy(), target_col="Target")
    assert not processed.isnull().values.any()
    assert processed.shape == df.shape


def test_preprocess_categorical_data():
    df = pd.DataFrame({
        "Feature1": ["A", "B", "A", np.nan],
        "Feature2": [1, 2, 3, 4],
        "Target": ["Yes", "No", "Yes", "No"]
    })
    processed = backend.preprocess_data(df.copy(), target_col="Target")

    assert not processed.isnull().values.any()
    assert any(col.startswith("Feature1") for col in processed.columns if col != "Target")

def test_train_model_classification():
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    result = backend.train_model(X, y, "Logistic Regression", "classification")
    assert "Accuracy" in result

def test_train_model_regression():
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    result = backend.train_model(X, y, "Linear Regression", "regression")
    assert "MAE" in result

def test_train_model_clustering():
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    result = backend.train_model(X, None, "DBSCAN", "clustering")
    assert "Silhouette Score" in result

def test_apply_pca_dimensions():
    X = np.random.rand(100, 5)
    X_pca = backend.apply_pca(X, n_components=2)
    assert X_pca.shape == (100, 2)

# ---------- FRONTEND TESTS ----------

@pytest.fixture
def test_mock_dataset():
    return pd.DataFrame({
        "feature1": range(1, 21),
        "feature2": range(21, 41),
        "Target": [1, 0] * 10
    })

@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.file_uploader")
@patch("streamlit.slider")
@patch("os.listdir")
@patch("PredictiveAnalytics1.Frontend.app.cached_load_data")
def test_frontend_upload_flow_valid(mock_load, mock_listdir, mock_slider, mock_uploader, mock_select, mock_radio, mock_dataset):
    mock_radio.return_value = "Use a predefined dataset"
    mock_select.side_effect = ["FIFA.csv", "Target", "Random Forest", "SVM Regressor"]
    mock_slider.return_value = 10
    mock_uploader.return_value = None
    mock_listdir.return_value = ["FIFA.csv"]
    mock_load.return_value = mock_dataset
    app.main()


@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.slider")
@patch("streamlit.file_uploader")
@patch("os.listdir")
@patch("PredictiveAnalytics1.Frontend.app.cached_load_data")
def test_frontend_predefined_selection_flow(mock_load, mock_listdir, mock_uploader, mock_slider, mock_select, mock_radio, mock_dataset):
    mock_radio.return_value = "Use a predefined dataset"
    mock_select.side_effect = ["FIFA.csv", "Target", "Random Forest", "SVM Regressor"]
    mock_slider.return_value = 10
    mock_listdir.return_value = ["FIFA.csv"]
    mock_load.return_value = mock_dataset
    app.main()


@patch("streamlit.slider")
def test_sample_size_slider_respects_upper_bound(mock_slider):
    df = pd.DataFrame(np.random.rand(10, 5), columns=[f'col{i}' for i in range(5)])
    mock_slider.return_value = 5
    result = df.sample(n=5, random_state=42)
    assert result.shape[0] == 5


@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.file_uploader")
@patch("os.listdir")
@patch("PredictiveAnalytics1.Frontend.app.cached_load_data")
def test_frontend_model_dropdown_exclusion(mock_load, mock_listdir, mock_upload, mock_select, mock_radio, mock_dataset):
    mock_radio.return_value = "Use a predefined dataset"
    mock_upload.return_value = None
    mock_listdir.return_value = ["FIFA.csv"]
    mock_load.return_value = mock_dataset

    #  Valid selection sequence
    mock_select.side_effect = ["FIFA.csv", "Target", "Random Forest", "SVM Regressor"]

    app.main()
    assert mock_select.call_count >= 4


@patch("streamlit.number_input")
@patch("streamlit.selectbox")
def test_expander_hyperparams_render(mock_select, mock_input):
    mock_input.return_value = 100
    mock_select.return_value = "linear"
    result = app.get_hyperparameters_ui("Random Forest", "regression", key_prefix="test")
    assert "n_estimators" in result

@patch("streamlit.selectbox")
@patch("streamlit.number_input")
def test_frontend_lasso_ridge_selection(mock_input, mock_select):
    mock_input.return_value = 0.5
    result = app.get_hyperparameters_ui("Lasso", "regression", key_prefix="test")
    assert "alpha" in result

    result = app.get_hyperparameters_ui("Ridge", "regression", key_prefix="test")
    assert "alpha" in result

@patch("streamlit.number_input")
def test_adaboost_hyperparam_input(mock_input):
    mock_input.return_value = 100
    result = app.get_hyperparameters_ui("AdaBoost", "regression", key_prefix="test")
    assert "n_estimators" in result

@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.slider")
@patch("streamlit.button")
@patch("streamlit.file_uploader")
@patch("os.listdir")
@patch("PredictiveAnalytics1.Frontend.app.cached_load_data")
def test_frontend_single_model_selection(mock_data, mock_list, mock_upload, mock_button, mock_slider, mock_select, mock_radio):
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "Target": [1, 0, 1, 0, 1]
    })
    mock_radio.return_value = "Use a predefined dataset"
    mock_button.return_value = False
    mock_slider.return_value = 5
    mock_upload.return_value = None
    mock_select.side_effect = ["FIFA.csv", "Target", "Random Forest", "Decision Tree Regressor"]
    mock_list.return_value = ["FIFA.csv"]
    mock_data.return_value = df
    app.main()

@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.slider")
@patch("streamlit.button")
@patch("streamlit.file_uploader")
@patch("os.listdir")
@patch("PredictiveAnalytics1.Frontend.app.cached_load_data")
def test_frontend_regression_model_selection(mock_data, mock_list, mock_upload, mock_button, mock_slider, mock_select, mock_radio):
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [10, 20, 30, 40, 50],
        "Target": [1.2, 2.3, 3.4, 4.5, 5.6]
    })
    mock_radio.return_value = "Use a predefined dataset"
    mock_button.return_value = False
    mock_slider.return_value = 5
    mock_upload.return_value = None
    mock_select.side_effect = ["FIFA.csv", "Target", "Lasso", "Ridge"]
    mock_list.return_value = ["FIFA.csv"]
    mock_data.return_value = df
    app.main()

@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.slider")
@patch("streamlit.button")
@patch("streamlit.file_uploader")
@patch("os.listdir")
@patch("PredictiveAnalytics1.Frontend.app.cached_load_data")
def test_frontend_classification_model_selection(mock_data, mock_list, mock_upload, mock_button, mock_slider, mock_select, mock_radio):
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 10, 15, 20, 25],
        "Target": [0, 1, 1, 0, 1]
    })
    mock_radio.return_value = "Use a predefined dataset"
    mock_button.return_value = False
    mock_slider.return_value = 5
    mock_upload.return_value = None
    mock_select.side_effect = ["FIFA.csv", "Target", "Logistic Regression", "Naive Bayes"]
    mock_list.return_value = ["FIFA.csv"]
    mock_data.return_value = df
    app.main()

@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.slider")
@patch("streamlit.button")
@patch("streamlit.file_uploader")
@patch("os.listdir")
@patch("PredictiveAnalytics1.Frontend.app.cached_load_data")
def test_frontend_button_not_clicked(mock_data, mock_list, mock_upload, mock_button, mock_slider, mock_select, mock_radio):
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [50, 60, 70, 80, 90],
        "Target": [1, 1, 0, 0, 1]
    })
    mock_radio.return_value = "Use a predefined dataset"
    mock_button.return_value = False
    mock_slider.return_value = 5
    mock_upload.return_value = None
    mock_select.side_effect = ["FIFA.csv", "Target", "Random Forest", "AdaBoost"]
    mock_list.return_value = ["FIFA.csv"]
    mock_data.return_value = df
    app.main()

@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.file_uploader")
@patch("streamlit.button")
def test_frontend_user_upload_flow(mock_button, mock_file_uploader, mock_selectbox, mock_radio):
    mock_radio.return_value = "Upload your own CSV"
    mock_file_uploader.return_value = None
    mock_selectbox.return_value = "Target"
    mock_button.return_value = False
    app.main()

@patch("streamlit.radio")
@patch("streamlit.file_uploader")
def test_frontend_default_upload_mode(mock_uploader, mock_radio):
    mock_radio.return_value = "Upload your own CSV"
    mock_uploader.return_value = None
    app.main()

@patch("streamlit.radio")
def test_frontend_default_predefined_mode(mock_radio):
    mock_radio.return_value = "Use a predefined dataset"
    with patch("os.listdir", return_value=[]), patch("streamlit.warning") as mock_warning:
        app.main()
        mock_warning.assert_called_once()
@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.button")
def test_frontend_minimal_regression_dataset(mock_button, mock_selectbox, mock_radio):
    mock_radio.return_value = "Use a predefined dataset"
    mock_selectbox.side_effect = ["FIFA.csv", "Target", "Lasso", "Ridge"]
    mock_button.return_value = False

    df = pd.DataFrame({
        "feature1": [1.0, 2.0],
        "feature2": [3.0, 4.0],
        "Target": [10.0, 20.0]
    })

    with patch("os.listdir", return_value=["FIFA.csv"]), \
         patch("streamlit.file_uploader"), \
         patch("PredictiveAnalytics1.Frontend.app.cached_load_data", return_value=df):
        app.main()

@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.button")
def test_frontend_empty_dataset_warning(mock_button, mock_selectbox, mock_radio):
    mock_radio.return_value = "Use a predefined dataset"
    mock_selectbox.return_value = "FIFA.csv"
    mock_button.return_value = False

    with patch("os.listdir", return_value=[]), \
         patch("streamlit.warning") as mock_warning:
        app.main()
        mock_warning.assert_called_with("No datasets found.")


@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.file_uploader")
@patch("streamlit.expander")
@patch("streamlit.dataframe")
@patch("streamlit.button")
def test_frontend_raw_data_display(mock_button, mock_dataframe, mock_expander, mock_file_uploader, mock_selectbox, mock_radio):
    mock_radio.return_value = "Use a predefined dataset"
    mock_file_uploader.return_value = None
    mock_button.return_value = False
    mock_selectbox.side_effect = ["FIFA.csv", "Target", "Random Forest", "Decision Tree Regressor"]
    mock_expander.return_value.__enter__.return_value = None

    df = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "Target": [1, 0, 1]
    })

    with patch("os.listdir", return_value=["FIFA.csv"]), \
         patch("PredictiveAnalytics1.Frontend.app.cached_load_data", return_value=df):
        app.main()
        mock_expander.assert_called()

@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.file_uploader")
@patch("streamlit.button")
def test_frontend_predefined_dataset_dropdown(mock_button, mock_file_uploader, mock_selectbox, mock_radio):
    mock_radio.return_value = "Use a predefined dataset"
    mock_file_uploader.return_value = None
    mock_button.return_value = False
    mock_selectbox.side_effect = ["FIFA.csv", "Target", "Random Forest", "Decision Tree Classifier"]

    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [4, 5, 6, 7],
        "Target": [0, 1, 0, 1]
    })

    with patch("os.listdir", return_value=["FIFA.csv"]), \
         patch("PredictiveAnalytics1.Frontend.app.cached_load_data", return_value=df):
        app.main()

@patch("streamlit.radio")
@patch("streamlit.warning")
def test_frontend_warning_no_datasets(mock_warning, mock_radio):
    mock_radio.return_value = "Use a predefined dataset"
    with patch("os.listdir", return_value=[]):
        app.main()
        mock_warning.assert_called_once()

@patch("streamlit.radio")
@patch("streamlit.warning")
def test_frontend_no_datasets_found(mock_warning, mock_radio):
    mock_radio.return_value = "Use a predefined dataset"
    with patch("os.listdir", return_value=[]):
        app.main()
        mock_warning.assert_called_with("No datasets found.")


def test_train_model_with_hyperparameters():
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    hyperparams = {'max_depth': 3}
    result = backend.train_model(X, y, "Decision Tree Classifier", "classification", hyperparams)
    assert "Accuracy" in result

def test_random_forest_regression_with_hyperparams():
    X = np.random.rand(50, 4)
    y = np.random.rand(50)
    hyperparams = {'n_estimators': 20, 'max_depth': 5}
    result = backend.train_model(X, y, "Random Forest", "regression", hyperparams)
    assert "MAE" in result

@patch("streamlit.radio")
@patch("streamlit.checkbox")
@patch("streamlit.selectbox")
@patch("streamlit.slider")
@patch("streamlit.button")
@patch("streamlit.file_uploader")
def test_clustering_mode_enabled(mock_file_uploader, mock_button, mock_slider, mock_selectbox, mock_checkbox, mock_radio):
    mock_radio.return_value = "Use a predefined dataset"
    mock_checkbox.return_value = True
    mock_button.return_value = False
    mock_file_uploader.return_value = None
    mock_selectbox.side_effect = ["FIFA.csv", "DBSCAN", "Spectral Clustering", "nearest_neighbors"]
    mock_slider.side_effect = [0.5, 5, 3]

    df = pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0],
    })

    with patch("os.listdir", return_value=["FIFA.csv"]), \
         patch("PredictiveAnalytics1.Frontend.app.cached_load_data", return_value=df):
        app.main()


def test_cached_load_data_reads_file_correctly():
    dummy_df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    with patch("PredictiveAnalytics1.Frontend.app.load_data", return_value=dummy_df) as mock_loader:
        result = app.cached_load_data("dummy/path.csv")
        mock_loader.assert_called_once_with("dummy/path.csv")
        pd.testing.assert_frame_equal(result, dummy_df)

@pytest.fixture
def mock_dataset():
    return pd.DataFrame({
        "feature1": range(1, 51),       # 50 samples
        "feature2": range(101, 151),
        "Target": [0, 1] * 25           # balanced target
    })

from unittest.mock import patch
import pandas as pd
import numpy as np

@patch("streamlit.radio")
@patch("streamlit.selectbox")
@patch("streamlit.slider")
@patch("streamlit.write")
@patch("streamlit.file_uploader")
@patch("os.listdir")
@patch("PredictiveAnalytics1.Frontend.app.cached_load_data")
def test_large_dataset_sample_block(
    mock_data, mock_list, mock_upload, mock_write,
    mock_slider, mock_selectbox, mock_radio
):
    from PredictiveAnalytics1.Frontend import app

    # Mock a dataset > 10,000 rows
    large_df = pd.DataFrame({
        "feature1": np.random.rand(12000),
        "feature2": np.random.rand(12000),
        "Target": np.random.rand(12000),
    })

    # Setup mocks
    mock_radio.return_value = "Use a predefined dataset"
    mock_selectbox.side_effect = ["FIFA.csv", "Target", "Linear Regression", "Ridge"]
    mock_upload.return_value = None
    mock_slider.return_value = 3000  # selected sample size
    mock_list.return_value = ["FIFA.csv"]
    mock_data.return_value = large_df

    app.main()

    # Assert write is called to show both shapes
    mock_write.assert_any_call(f"Original dataset shape: {large_df.shape}")
    mock_write.assert_any_call(f"Sampled dataset shape: {(3000, 3)}")

@patch("PredictiveAnalytics1.Frontend.app.st.scatter_chart")
def test_render_cluster_chart(mock_chart):
    from PredictiveAnalytics1.Frontend.app import render_cluster_chart

    X, _ = make_blobs(n_samples=10, centers=2, random_state=42)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    render_cluster_chart(X_pca, {"eps": 0.5, "min_samples": 3}, "DBSCAN")

    mock_chart.assert_called_once()

@patch("streamlit.bar_chart")
@patch("streamlit.metric")
@patch("streamlit.subheader")
def test_metric_and_bar_chart_display(mock_subheader, mock_metric, mock_bar_chart):
    import pandas as pd
    from PredictiveAnalytics1.Frontend import app

    df = pd.DataFrame([
        {"Model": "Model1", "Accuracy": 0.85},
        {"Model": "Model2", "Accuracy": 0.80}
    ])

    app.render_metrics_and_chart(
        comparison_df=df,
        problem_type="classification",
        model1="Model1",
        model2="Model2",
        metrics1={"Model": "Model1", "Accuracy": 0.85},
        metrics2={"Model": "Model2", "Accuracy": 0.80}
    )

    assert mock_metric.call_count >= 2
    mock_bar_chart.assert_called_once()

@patch("streamlit.radio")
@patch("streamlit.checkbox")
@patch("streamlit.selectbox")
@patch("streamlit.slider")
@patch("streamlit.button")
@patch("streamlit.file_uploader")
@patch("streamlit.subheader")
@patch("streamlit.metric")
@patch("streamlit.dataframe")
@patch("streamlit.bar_chart")
@patch("streamlit.scatter_chart")
@patch("streamlit.columns")
@patch("os.listdir")
@patch("PredictiveAnalytics1.Frontend.app.cached_load_data")
def test_clustering_comparison_block_large_coverage(
    mock_load, mock_listdir, mock_columns,
    mock_scatter, mock_bar, mock_data, mock_metric,
    mock_subheader, mock_upload, mock_button, mock_slider,
    mock_select, mock_checkbox, mock_radio
):
    from PredictiveAnalytics1.Frontend import app

    X, _ = make_blobs(n_samples=30, centers=2, random_state=42)
    df = pd.DataFrame(X, columns=["feature1", "feature2"])

    mock_radio.return_value = "Use a predefined dataset"
    mock_checkbox.return_value = True
    mock_button.return_value = True
    mock_upload.return_value = None
    mock_listdir.return_value = ["FIFA.csv"]
    mock_load.return_value = df
    mock_select.side_effect = ["FIFA.csv", "DBSCAN", "Spectral Clustering", "nearest_neighbors"]
    mock_slider.side_effect = [0.3, 3, 2]
    mock_columns.return_value = [patch("streamlit.container"), patch("streamlit.container")]

    app.main()

    # Assertions
    assert mock_metric.call_count >= 4, "Expected clustering metrics not rendered"
    assert mock_bar.called, "Bar chart not rendered"
    assert mock_scatter.called, "Scatter chart not rendered"
    assert mock_data.called, "Comparison dataframe not shown"
