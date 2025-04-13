from datetime import timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as keras_backend
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


def load_full_data(file):
    """Load and preprocess historical stock data from CSV."""
    data_frame = pd.read_csv(file, parse_dates=['Date'], dayfirst=True)
    data_frame.columns = data_frame.columns.str.strip()
    data_frame['Date'] = pd.to_datetime(data_frame['Date'], format='%d-%m-%Y', errors='coerce')
    data_frame = data_frame.dropna(subset=['Date']).sort_values('Date')
    data_frame.set_index('Date', inplace=True)

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    data_frame[numeric_cols] = data_frame[numeric_cols].apply(pd.to_numeric, errors='coerce')
    data_frame = data_frame.dropna(subset=numeric_cols)

    return data_frame[['Stock Name'] + numeric_cols]


def create_sequences(data, sequence_length, feature_index):
    """Create LSTM sequences from time series data."""
    x_seq, y_seq = [], []
    for i in range(len(data) - sequence_length):
        window = data[i:i + sequence_length]
        target = data[i + sequence_length, feature_index]
        if not np.any(np.isnan(window)) and not np.isnan(target):
            x_seq.append(window)
            y_seq.append(target)
    return np.array(x_seq), np.array(y_seq)


def build_lstm_model(input_shape):
    """Build a simple stacked LSTM model."""
    keras_backend.clear_session()
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def build_bidirectional_lstm(sequence_length):
    """Build improved bidirectional LSTM model."""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(sequence_length, 1)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


def rolling_forecast(model, normalized_data, sequence_length):
    """Perform rolling forecast using trained model."""
    predictions = []
    if len(normalized_data) < sequence_length:
        return predictions

    last_sequence = normalized_data[:sequence_length].reshape(1, sequence_length, -1)

    for i in range(sequence_length, len(normalized_data)):
        prediction = model.predict(last_sequence, verbose=0)[0][0]
        predictions.append(prediction)
        next_input = normalized_data[i].reshape(1, 1, -1)
        last_sequence = np.append(last_sequence[:, 1:, :], next_input, axis=1)

    return predictions


@st.cache_data
def load_and_cache_file(uploaded_file):
    """Cache uploaded CSV file content."""
    return load_full_data(uploaded_file)


@st.cache_data(ttl=300)
def get_realtime_data():
    """Fetch recent weekly NIFTY50 data."""
    ticker_data = yf.Ticker("^NSEI")
    return ticker_data.history(period="5y", interval="1wk")


@st.cache_data(ttl=300)
def get_realtime_daily_data():
    """Fetch recent daily NIFTY50 data."""
    ticker_data = yf.Ticker("^NSEI")
    return ticker_data.history(period="2y", interval="1d")


def create_main_chart(df, selected_metric, chart_type):
    """Create main plotly chart for real-time analysis."""
    fig = go.Figure()
    if chart_type == "Line Chart":
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[selected_metric],
            mode='lines',
            name=selected_metric,
            line={"color": '#00CC96', "width": 2}
        ))
    else:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC',
            increasing_line_color='#00CC96',
            decreasing_line_color='#FF6666'
        ))
    return fig


def handle_predictions(df, selected_metric, prediction_days, sequence_length):
    """Handle prediction logic and return forecast data."""
    accuracy_warnings = []
    forecast_values = []
    prediction_dates = []
    try:
        raw_data = df[selected_metric].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(raw_data)

        x_data, y_data = [], []
        for i in range(len(scaled_data) - sequence_length - prediction_days):
            x_data.append(scaled_data[i:i+sequence_length])
            y_data.append(scaled_data[i+sequence_length])

        x_data = np.array(x_data)
        y_data = np.array(y_data).reshape(-1)

        split_index = int(0.8 * len(x_data))
        x_train, x_test = x_data[:split_index], x_data[split_index:]
        y_train, y_test = y_data[:split_index], y_data[split_index:]

        model = build_bidirectional_lstm(sequence_length)
        model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
        y_pred = model.predict(x_test).flatten()
        r2 = r2_score(y_test, y_pred)

        if r2 < 0.85:
            accuracy_warnings.append(
                f"Model accuracy ({r2 * 100:.1f}%) below 85% threshold"
            )
        else:
            last_seq = scaled_data[-sequence_length:]
            future_forecast = []
            for _ in range(prediction_days):
                next_pred = model.predict(
                    last_seq.reshape(1, sequence_length, 1)
                )[0][0]
                future_forecast.append(next_pred)
                last_seq = np.append(last_seq[1:], [[next_pred]], axis=0)

            forecast_values = scaler.inverse_transform(
                np.array(future_forecast).reshape(-1, 1)
            ).flatten()
            prediction_dates = pd.date_range(
                df.index[-1] + timedelta(days=1),
                periods=prediction_days
            )

    except Exception as prediction_error:  # pylint: disable=broad-except
        accuracy_warnings.append(f"Prediction failed: {str(prediction_error)}")

    return forecast_values, prediction_dates, accuracy_warnings

#Streamlit app
def main():
    st.set_page_config(page_title="Stock Analysis", layout="wide")
    st.title("MarketVision: NIFTY 50 Analysis & Forecasting")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Historical Data CSV", type=["csv"]
    )
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Data Viewer", "Realtime Analysis", "LSTM Predictions"]
    )

    full_df = pd.DataFrame()
    if uploaded_file:
        try:
            full_df = load_and_cache_file(uploaded_file)
        except Exception as error:  # pylint: disable=broad-except
            st.error(f"Error loading file: {str(error)}")
            return

    with tab1:
        st.header("Application Overview")
        st.markdown("""
            This application provides analysis and predictions for NIFTY 50 stock prices.
            **Features**:
            - View stock-wise historical and real-time data
            - LSTM-based price prediction
            - Visualization
            **Data Requirements**:
            - Columns: `Date`, `Stock Name`, `Open`, `High`, `Low`, `Close`, `Volume`
        """)

    with tab2:
        st.header("Data Viewer")
        realtime_df = get_realtime_data()
        if not realtime_df.empty:
            st.dataframe(realtime_df.style.format("{:.2f}"), use_container_width=True, height=500)
        else:
            st.warning("Could not fetch real-time data")

        if not full_df.empty:
            formatter = {col: "{:.2f}"
                for col in full_df.select_dtypes(include=np.number).columns
            }
            pd.set_option("styler.render.max_elements", 400000)
            st.dataframe(full_df.style.format(formatter), use_container_width=True)
        else:
            st.warning("Please upload a historical data file.")

    with tab3:
        st.header("Real-time Nifty50 Data with Forecast")
        df = get_realtime_daily_data()
        if not df.empty:
            metric_options = ['Open', 'High', 'Low', 'Close']
            selected_metric = st.selectbox(
                "Select Metric to Plot", metric_options, index=3
            )
            predict_mode = st.checkbox("Enable Predictions", key="predict_toggle")
            prediction_days = st.slider(
                "Select Prediction Days", 1, 30, 7
            ) if predict_mode else 0

            current_price = df[selected_metric].iloc[-1]
            initial_price = df[selected_metric].iloc[0]
            delta_price = current_price - initial_price
            percent_change = (delta_price / initial_price) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric(f"Current {selected_metric}", f"₹{current_price:.2f}")
            col2.metric("Change", f"₹{delta_price:.2f}")
            col3.metric("% Change", f"{percent_change:.2f}%")

            chart_type = st.radio(
                "Chart Type", ["Line Chart", "Candlestick Chart"], horizontal=True
            )
            fig = create_main_chart(df, selected_metric, chart_type)

            if predict_mode and prediction_days > 0:
                st.info("Predictions use LSTM model with minimum 85% accuracy requirement")
                with st.spinner("Training prediction model..."):
                    sequence_length = 60
                    forecast_values, prediction_dates, warnings = handle_predictions(
                        df, selected_metric, prediction_days, sequence_length
                    )

                    if forecast_values:
                        fig.add_trace(go.Scatter(
                            x=prediction_dates,
                            y=forecast_values,
                            mode='lines+markers',
                            name=f'Predicted {selected_metric}',
                            line={"color": '#FFA500', "width": 2, "dash": 'dot'}
                        ))

                for warning in warnings:
                    st.warning(warning)

            fig.update_layout(
                title=f"Nifty50 {selected_metric} Analysis",
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font={"color": 'white'},
                xaxis={"gridcolor": '#34495E'},
                yaxis={"gridcolor": '#34495E'},
                height=600,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Nifty-50 Stock Predictions")
        if not full_df.empty:
            all_stocks = full_df['Stock Name'].unique()
            selected_stocks = st.multiselect("Select Stocks", all_stocks)
            available_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            selected_features = st.multiselect("Select Metrics", available_features)

            for stock in selected_stocks:
                st.subheader(f"Analysis for {stock}")
                stock_df = full_df[full_df['Stock Name'] == stock][available_features].dropna()
                if len(stock_df) < 150:  # sequence_length * 3 = 50*3
                    st.warning(f"Need at least 150 data points for {stock}")
                    continue

                try:
                    data_values = stock_df.values
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    data_scaled = scaler.fit_transform(data_values)
                    predictions_dict = {}
                    sequence_length = 50

                    for feature in selected_features:
                        feature_index = available_features.index(feature)
                        x_data, y_data = create_sequences(
                            data_scaled, sequence_length, feature_index
                        )
                        if len(x_data) < 10:
                            st.error(f"Insufficient training data for {feature}")
                            continue

                        split_idx = int(len(x_data) * 0.8)
                        model = build_lstm_model((sequence_length, len(available_features)))
                        model.fit(
                            x_data[:split_idx], y_data[:split_idx],
                            epochs=20, batch_size=16, verbose=0
                        )

                        if feature != "Volume":
                            y_pred = model.predict(x_data[split_idx:])
                            y_true = y_data[split_idx:]
                            r2 = r2_score(y_true, y_pred)
                            mae = mean_absolute_error(y_true, y_pred)

                            col1, col2 = st.columns(2)
                            col1.metric(f"R² Score of {feature}", f"{r2 * 100:.2f}%")
                            col2.metric(f"MAE of {feature}", f"{mae:.4f}")

                        predicted_scaled = rolling_forecast(
                            model, data_scaled, sequence_length
                        )
                        if predicted_scaled:
                            predictions_scaled = np.zeros(
                                (len(predicted_scaled), len(available_features))
                            )
                            predictions_scaled[:, feature_index] = predicted_scaled
                            predictions = scaler.inverse_transform(
                                predictions_scaled
                            )[:, feature_index]
                            predictions_dict[feature] = predictions

                    if predictions_dict:
                        length_next_predictions = len(next(iter(predictions_dict.values())))
                        prediction_dates = stock_df.index[
                            sequence_length:sequence_length + length_next_predictions
                        ]
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                        for feature_idx, feature in enumerate(selected_features):
                            if feature in predictions_dict:
                                color_idx = feature_idx % len(color_palette)
                                use_secondary_y = 'Volume' if feature=='Volume' else None

                                # Actual values
                                fig.add_trace(go.Scatter(
                                    x=stock_df.index,
                                    y=stock_df[feature],
                                    mode='lines',
                                    name=f'Actual {feature}',
                                    line={"color": color_palette[color_idx], "width": 2}
                                ), secondary_y=use_secondary_y)

                                # Predicted values
                                fig.add_trace(go.Scatter(
                                    x=prediction_dates,
                                    y=predictions_dict[feature],
                                    mode='lines',
                                    name=f'Predicted {feature}',
                                    line={"color": color_palette[color_idx],
                                        "width": 2, "dash": 'dot'}
                                ), secondary_y=use_secondary_y)

                        fig.update_layout(
                            title=f'{stock} Predictions',
                            plot_bgcolor='#000000',
                            paper_bgcolor='#000000',
                            font={"color": 'white'},
                            xaxis={"gridcolor": '#34495E'},
                            yaxis={"gridcolor": '#34495E'},
                            height=600,
                            margin={"r": 120}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:  # pylint: disable=broad-except
                    st.error(f"Error processing {stock}: {str(e)}")


if __name__ == "__main__":
    main()
