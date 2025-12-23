import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go


# =======================
# CSS ANIMATIONS
# =======================
def load_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #141E30, #243B55);
        color: white;
        animation: fadeIn 1.2s ease-in;
    }

    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    h1 {
        text-align: center;
        text-shadow: 0 0 20px #00f2ff;
        animation: glow 2s infinite alternate;
    }

    @keyframes glow {
        from { text-shadow: 0 0 10px #00f2ff; }
        to { text-shadow: 0 0 25px #00f2ff; }
    }

    div[data-testid="stSelectbox"], div[data-testid="stNumberInput"] {
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 8px;
        transition: transform 0.3s ease;
    }

    div[data-testid="stSelectbox"]:hover,
    div[data-testid="stNumberInput"]:hover {
        transform: scale(1.03);
    }

    div.stButton > button {
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white;
        border-radius: 30px;
        padding: 0.6em 1.5em;
        font-size: 16px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 0 20px rgba(255, 36, 118, 0.8);
    }
    </style>
    """, unsafe_allow_html=True)

# =======================
# PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="centered"
)

load_css()

# =======================
# TITLE
# =======================
st.markdown("<h1>📈 Stock Price Prediction using LSTM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered stock trend forecasting dashboard</p>", unsafe_allow_html=True)
st.markdown("---")

# =======================
# USER INPUT
# =======================
stock = st.selectbox("📊 Select Stock", ["AAPL", "GOOGL", "MSFT", "TSLA"])
days = st.slider("📅 Days to Display", 30, 200, 60)

# =======================
# LOAD MODEL
# =======================
model = load_model("models/lstm_model.keras")

# =======================
# FETCH DATA
# =======================
data = yf.download(stock, start="2018-01-01")
close_prices = data[['Close']]

scaler = MinMaxScaler()
scaled = scaler.fit_transform(close_prices)

# =======================
# CREATE SEQUENCES
# =======================
X = []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i, 0])

X = np.array(X)
X = X.reshape(X.shape[0], X.shape[1], 1)

# =======================
# PREDICT
# =======================
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)

# =======================
# DISPLAY RESULTS
# =======================
st.markdown("### 📉 Actual vs Predicted Prices")

fig = go.Figure()

fig.add_trace(go.Scatter(
    y=close_prices[-days:]['Close'],
    mode='lines',
    name='Actual Price',
    line=dict(color='cyan')
))

fig.add_trace(go.Scatter(
    y=predicted_prices[-days:].flatten(),
    mode='lines',
    name='Predicted Price',
    line=dict(color='red')
))

fig.update_layout(
    template="plotly_dark",
    height=450,
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.success("✅ Prediction completed successfully")
st.toast("Model prediction ready 🚀")
