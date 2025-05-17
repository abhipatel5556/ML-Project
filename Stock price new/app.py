import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import os

app = Flask(__name__)

# Load model
model = load_model('stock_dl_model.h5')

# Setup SQLite database
DB_PATH = 'predictions.db'
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False)

# Ensure database table exists
from sqlalchemy import text  # ⬅️ Add at the top

def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_symbol TEXT,
                predicted_price REAL,
                prediction_range INTEGER,
                timestamp DATETIME
            )
        """))


init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock = request.form['stock_symbol'].upper().strip()
        prediction_range = int(request.form['prediction_range'])

        if len(stock) <= 5 and not stock.endswith(".NS"):
            stock += ".NS"

        df = yf.download(stock, period='90d', interval='1d')

        if df.empty or 'Close' not in df.columns:
            return render_template('index.html', prediction_text="Error: Invalid symbol or no data.")

        close_prices = df['Close'].values[-60:]

        if len(close_prices) < 60:
            return render_template('index.html', prediction_text="Error: Less than 60 days of data available.")

        # Normalize data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))
        input_data = scaled_data.reshape(1, 60, 1)

        # Predict for given range
        predictions = []
        last_input = input_data.copy()

        for _ in range(prediction_range):
            next_pred = model.predict(last_input)[0][0]
            predictions.append(next_pred)

            last_input = np.append(last_input[:, 1:, :], [[[next_pred]]], axis=1)

        # Inverse transform predictions
        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        final_predicted_price = predicted_prices[-1]

        # Save to DB
        pred_df = pd.DataFrame({
            'stock_symbol': [stock],
            'predicted_price': [final_predicted_price],
            'prediction_range': [prediction_range],
            'timestamp': [pd.to_datetime('now')]
        })
        pred_df.to_sql('predictions', con=engine, if_exists='append', index=False)

        # Plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual'))
        future_dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(1, prediction_range + 1)]
        fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices, mode='lines+markers', name='Predicted', line=dict(dash='dot')))

        graph_html = fig.to_html(full_html=False)

        return render_template(
            'index.html',
            prediction_text=f"Predicted price for {stock} after {prediction_range} day(s): ₹{final_predicted_price:.2f}",
            graph_html=graph_html
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
