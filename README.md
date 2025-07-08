# 📈 Crypto Price Prediction using LSTM

This project predicts future cryptocurrency prices (e.g., BTC-USD or NVDA stock) using an LSTM deep learning model trained on historical price data. The model is built with TensorFlow/Keras and visualized with Matplotlib inside a Jupyter Notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uosOQldK6TqYTm04od_Yc9aIGm6kNhj4?usp=sharing)

---

## 🚀 Project Overview

- Predicts the **next 10-day prices** of a given stock or cryptocurrency
- Uses **Yahoo Finance** data via `yfinance`
- Trains an **LSTM neural network** with scaled closing prices
- Visualizes:
  - Closing price over time
  - Predicted vs actual prices
  - Future price forecast

---

## 📊 Tech Stack

- Python
- Jupyter Notebook
- TensorFlow / Keras
- yfinance
- sklearn (MinMaxScaler)
- Matplotlib

---

## 🧠 Model

- **2 LSTM layers**
- **1 Dense hidden layer (25 units)**
- **1 Output layer (1 unit)**
- Trained with `mean_squared_error` loss and `adam` optimizer

---

Clone this repository  
   ```bash
   git clone https://github.com/adrijaa291/Crypto_Price_Prediction.git
   cd Crypto_Price_Prediction
   ```

  
