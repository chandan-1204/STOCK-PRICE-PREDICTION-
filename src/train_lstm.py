import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model,Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# 1. DOWNLOAD STOCK DATA
# =========================
stock_symbol = "AAPL"
data = yf.download(stock_symbol, start="2015-01-01", end="2024-01-01")

data = data[['Close']]
data.to_csv("data/stock.csv")

print("Stock data downloaded")

# =========================
# 2. PREPROCESS DATA
# =========================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data)

# Reshape for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# 3. BUILD LSTM MODEL
# =========================
model = load_model("models/lstm_model.keras")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# =========================
# 4. TRAIN MODEL
# =========================
model.fit(X_train, y_train, epochs=10, batch_size=32)

# =========================
# 5. SAVE MODEL
# =========================
model.save("models/lstm_model.keras")
print("Model saved successfully")

# =========================
# 6. PLOT PREDICTIONS
# =========================
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(10,5))
plt.plot(real_prices, color='green', label='Actual Price')
plt.plot(predictions, color='red', label='Predicted Price')
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
