from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

matplotlib.use('Agg')

app = Flask(__name__)
model = load_model("model.keras")

def plot_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{data}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock = request.form.get("stock")
        no_of_days = int(request.form.get("no_of_days"))
        return redirect(url_for("predict", stock=stock, no_of_days=no_of_days))
    return render_template("index.html")

@app.route("/predict")
def predict():
    stock = request.args.get("stock", "BTC-USD")
    no_of_days = int(request.args.get("no_of_days", 10))

    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    stock_data = yf.download(stock, start, end)
    if stock_data.empty:
        return render_template("result.html", error="Invalid stock ticker or no data available.")

    closing_prices = stock_data[['Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(closing_prices)

    x_data = []
    y_data = []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    predictions = model.predict(x_data)
    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    plotting_data = pd.DataFrame({
        'Original': inv_y_test.flatten(),
        'Predicted': inv_predictions.flatten()
    }, index=closing_prices.index[-len(inv_y_test):])

    # Plot 1: Closing Prices
    fig1 = plt.figure(figsize=(15, 5))
    plt.plot(closing_prices, label='Close Price', color='blue')
    plt.title("Historical Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plot1 = plot_to_html(fig1)

    # Plot 2: Original vs Predicted
    fig2 = plt.figure(figsize=(15, 5))
    plt.plot(plotting_data['Original'], label="Original")
    plt.plot(plotting_data['Predicted'], label="Predicted", linestyle="--")
    plt.title("Original vs Predicted")
    plt.legend()
    plot2 = plot_to_html(fig2)

    # Future prediction
    last_100 = scaled_data[-100:].reshape(1, -1, 1)
    future_preds = []
    for _ in range(no_of_days):
        next_day = model.predict(last_100)
        future_preds.append(scaler.inverse_transform(next_day)[0][0])
        last_100 = np.append(last_100[:, 1:, :], next_day.reshape(1, 1, -1), axis=1)

    fig3 = plt.figure(figsize=(15, 5))
    plt.plot(range(1, no_of_days+1), future_preds, marker='o', color='purple')
    plt.title("Next {} Day Predictions".format(no_of_days))
    plt.xlabel("Day")
    plt.ylabel("Predicted Price")
    plot3 = plot_to_html(fig3)

    return render_template("result.html", stock=stock, plot1=plot1, plot2=plot2, plot3=plot3, predictions=future_preds)

if __name__ == '__main__':
    app.run(debug=True)