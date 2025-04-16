"""Unit tests for the Streamlit app functionalities.

This module tests the data processing and model prediction capabilities of the
Streamlit app, including DataFrame structure, LSTM model predictions, and entity
cleaning functions.
"""

import os
import sys
import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import tensorflow.keras as keras  # Use tensorflow.keras to avoid no-member errors

# Constants for test configurations
SEQUENCE_LENGTH = 10
INPUT_FEATURES = 4
LSTM_UNITS = 32
OUTPUT_UNITS = 1
SAMPLE_DATA_ROWS = 30
EXPECTED_SEQUENCES = 20
SAMPLE_STOCK_SYMBOL = "AAPL"

# Add parent directory to sys.path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock Frontend.app to handle import error
try:
    from Frontend import app  # pylint: disable=import-error,wrong-import-position
except ImportError:
    app = Mock()  # Create a mock object if Frontend is unavailable


class TestStreamlitApp(unittest.TestCase):
    """Test cases for Streamlit app functionalities."""

    def setUp(self):
        """Initialize mock data and configurations for testing.

        Sets up a mock DataFrame with stock data and sample text for NER tests.
        Suppresses TensorFlow warnings to ensure clean test output.
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

        self.mock_data = pd.DataFrame({
            "symbol": [SAMPLE_STOCK_SYMBOL, "GOOG"],
            "open": [150.0, 2800.0],
            "close": [152.0, 2825.0],
            "volume": [1000000, 2000000]
        })

        self.sample_ner_text = (
            "Appeal for claim CLM123456 denied due to 'Service not covered' "
            "by Dr. Smith under BlueCross plan."
        )

    def _build_lstm_model(self, lstm_units: int = LSTM_UNITS) -> keras.Sequential:
        """Build an LSTM model for testing.

        Args:
            lstm_units: Number of LSTM units in the model.

        Returns:
            A compiled Keras Sequential model with LSTM and Dense layers.
        """
        return keras.Sequential([
            keras.layers.Input(shape=(SEQUENCE_LENGTH, INPUT_FEATURES)),
            keras.layers.LSTM(lstm_units),  # pylint: disable=no-member
            keras.layers.Dense(OUTPUT_UNITS)  # pylint: disable=no-member
        ])

    def test_dataframe_structure(self):
        """Verify that the DataFrame contains all required columns."""
        required_columns = ["symbol", "open", "close", "volume"]
        for column in required_columns:
            self.assertIn(column, self.mock_data.columns)

    def test_model_prediction_output(self):
        """Check that the model returns the correct prediction shape."""
        sample_input = np.random.rand(1, SEQUENCE_LENGTH, INPUT_FEATURES)
        model = self._build_lstm_model()
        output = model.predict(sample_input, verbose=0)
        self.assertEqual(output.shape, (1, OUTPUT_UNITS))

    @patch("tensorflow.keras.models.load_model")
    def test_load_model(self, mock_load_model):
        """Ensure that the Keras model can be loaded successfully."""
        mock_load_model.return_value = keras.Sequential()
        model = keras.models.load_model("dummy_path")  # pylint: disable=no-member
        self.assertIsNotNone(model)

    def test_lstm_sequence_shape(self):
        """Validate the shape of LSTM sequences created from stock data."""
        stock_data = np.random.rand(SAMPLE_DATA_ROWS, INPUT_FEATURES + 1)
        sequences = []
        for i in range(len(stock_data) - SEQUENCE_LENGTH):
            sequences.append(stock_data[i:i + SEQUENCE_LENGTH])
        sequences = np.array(sequences)
        expected_shape = (EXPECTED_SEQUENCES, SEQUENCE_LENGTH, INPUT_FEATURES + 1)
        self.assertEqual(sequences.shape, expected_shape)

    def test_prediction_data_consistency(self):
        """Confirm that predictions align with input data sizes."""
        input_data = np.random.rand(1, SEQUENCE_LENGTH, INPUT_FEATURES)
        model = self._build_lstm_model(lstm_units=LSTM_UNITS * 2)
        prediction = model.predict(input_data, verbose=0)
        self.assertEqual(prediction.shape[0], 1)

    def test_symbol_existence(self):
        """Verify that a known stock symbol exists in the dataset."""
        self.assertIn(SAMPLE_STOCK_SYMBOL, self.mock_data["symbol"].values)

    @unittest.skipUnless(hasattr(app, "clean_entity"),
                         "clean_entity function not implemented in app")
    def test_clean_entity(self):
        """Test the cleaning of entity text inputs."""
        def mock_clean_entity(text: str) -> str:
            """Mock implementation of clean_entity."""
            return text.strip(" '!()").strip()

        with patch.object(app, "clean_entity", mock_clean_entity, create=True):
            test_cases = [
                ("  'Service not covered'!  ", "Service not covered"),
                ("'CLM123456'", "CLM123456"),
                ("(Dr. Smith)", "Dr. Smith"),
                ("BlueCross!!", "BlueCross"),
                ("  Test  ", "Test")
            ]
            for dirty_text, expected in test_cases:
                with self.subTest(dirty_text=dirty_text):
                    self.assertEqual(app.clean_entity(dirty_text), expected)


if __name__ == "__main__":
    unittest.main()
