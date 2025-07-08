# 📈 Crypto Price Prediction Web App

A minimal web-based interface that uses deep learning to predict future cryptocurrency prices (e.g., Bitcoin) based on historical data from Yahoo Finance. Built with Python, LSTM models, and data science libraries — the interface allows users to view graphs and predict the next N days of prices.

> 🔗 **[Demo Video](https://drive.google.com/file/d/1VS0ZDXjCfUozm-cRo8UqLDJRel47OXzG/view?usp=sharing)  


---

## 🚀 Features

- 📊 Fetches real-time crypto data using Yahoo Finance (`yfinance`)
- 🔁 Prepares time series for LSTM model training and prediction
- 🤖 Loads a pre-trained `model.keras` for:
  - Validating on historical test data
  - Predicting future prices (custom N days)
- 📉 Visual output includes:
  - Closing price history
  - Model predictions vs actuals
  - Future price forecast graph

---

## 🧰 Tech Stack

- **Core Libraries**:  
  `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow`, `keras`, `yfinance`
  
- **Basic Web UI**:  
  A lightweight Flask app is used to display results and accept user input.

---

## 📂 Project Structure
```bash
Crypto_Price_Prediction/
├── app.py # Main application script
├── model.keras # Trained LSTM model
├── requirements.txt # Required dependencies
└── templates/
├── index.html # Input form
└── result.html # Output graphs and predictions
```
### Clone the repo
```bash
git clone https://github.com/your-username/Crypto_Price_Prediction.git
cd Crypto_Price_Prediction
```
### Run the application
```bash
python app.py
```
