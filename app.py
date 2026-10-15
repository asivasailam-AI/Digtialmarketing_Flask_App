from datetime import datetime, timedelta
import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from nselib import capital_market as cm

app = Flask(__name__)

model_path = Path(__file__).resolve().parent / "models" / "TSA_Share_Prediction_Model.pkl"
with model_path.open("rb") as model_file:
    model_assets = pickle.load(model_file)


def clean_nselib_price_volume(data):
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], format="%d-%b-%Y", errors="coerce")

    numeric_columns = [
        "OpenPrice",
        "HighPrice",
        "LowPrice",
        "ClosePrice",
    ]
    for column in numeric_columns:
        data[column] = (
            data[column]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.sort_values("Date").set_index("Date")


def parse_date(value):
    for date_format in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue
    raise ValueError("Dates must use YYYY-MM-DD or DD-MM-YYYY format.")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) if request.is_json else request.form
    if payload is None or not hasattr(payload, "get"):
        return jsonify({"error": "A valid JSON request body is required."}), 400

    symbol = str(payload.get("symbol", "")).strip().upper()
    if not symbol:
        return jsonify({"error": "Company symbol is required."}), 400

    try:
        start_date = parse_date(str(payload.get("start_date", "")).strip())
        end_date = parse_date(str(payload.get("end_date", "")).strip())
        days_to_predict = int(payload.get("days_to_predict", ""))
        if days_to_predict < 1:
            raise ValueError("Prediction days must be a positive integer.")
        if start_date > end_date:
            raise ValueError("Start date must be on or before end date.")
    except (TypeError, ValueError) as error:
        message = str(error) or "Prediction days must be a positive integer."
        if request.is_json:
            return jsonify({"error": message}), 400
        return render_template("result.html", error=message), 400

    try:
        raw_data = cm.price_volume_data(
            symbol=symbol,
            from_date=start_date.strftime("%d-%m-%Y"),
            to_date=end_date.strftime("%d-%m-%Y"),
        )
        if raw_data is None or raw_data.empty:
            message = f"No share-price data found for {symbol} in the selected date range."
            if request.is_json:
                return jsonify({"error": message}), 404
            return render_template("result.html", error=message), 404

        columns = model_assets["columns"]
        cleaned_data = clean_nselib_price_volume(raw_data)
        selected_data = cleaned_data[columns].dropna().reset_index(drop=True)
        if len(selected_data) < 5:
            message = "At least five complete trading days are required to make a forecast."
            if request.is_json:
                return jsonify({"error": message}), 422
            return render_template("result.html", error=message), 422

        scaler = model_assets["scaler"]
        model_fit = model_assets["model"]
        scaled_data = scaler.transform(selected_data)
        updated_model = model_fit.append(scaled_data, refit=False)
        forecast_scaled = updated_model.forecast(steps=days_to_predict)
        forecast_values = scaler.inverse_transform(forecast_scaled)
        forecast = pd.DataFrame(forecast_values, columns=columns).round(2)
        forecast["HighPrice"] = forecast[
            ["OpenPrice", "HighPrice", "ClosePrice"]
        ].max(axis=1)
        forecast["LowPrice"] = forecast[
            ["OpenPrice", "LowPrice", "ClosePrice"]
        ].min(axis=1)
        
        # Generate upcoming trading dates (excluding weekends)
        last_date = cleaned_data.index.max()
        trading_dates = []
        current_date = last_date
        while len(trading_dates) < days_to_predict:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:  # Monday=0, Sunday=6
                trading_dates.append(current_date.strftime("%d-%b-%Y"))
        
        forecast.insert(0, "Day", trading_dates)
        forecast_records = forecast.to_dict(orient="records")

    except Exception:
        app.logger.exception("Share-price prediction failed for symbol %s.", symbol)
        message = "Unable to generate a forecast. Check the symbol and date range, then try again."
        if request.is_json:
            return jsonify({"error": message}), 500
        return render_template("result.html", error=message), 500

    if request.is_json:
        return jsonify({"symbol": symbol, "predictions": forecast_records})
    return render_template(
        "result.html",
        symbol=symbol,
        predictions=forecast_records,
        columns=columns,
    )


if __name__ == "__main__":
    app.run(debug=True)
