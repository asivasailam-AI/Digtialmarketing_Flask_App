from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Ignore the specific InconsistentVersionWarning from sklearn
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)

# Load model
model = pickle.load(open("models/Digital_Marketing_Dataset_GridRandomForest.pkl", "rb"))

# Load scaler
scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if the request is JSON (for API calls) or Form data (for index.html)
        if request.is_json:
            data = request.get_json(force=True)
            sample = data.get('features')
        else:
            # Retrieve individual fields from the HTML form
            # Ensure the names match the 'name' attribute in index.html
            sample = [
                float(request.form.get('TimeOnSite', 0)),
                float(request.form.get('AdSpend', 0)),
                float(request.form.get('ClickThroughRate', 0)),
                float(request.form.get('EmailClicks', 0)),
                float(request.form.get('EmailOpens', 0)),
                float(request.form.get('PagesPerVisit', 0)),
                float(request.form.get('PreviousPurchases', 0)),
                0 # Default for CampaignType_Conversion as it's not in the form
            ]

        if sample is None:
            return jsonify({"error": "No features provided"}), 400

        # Convert to NumPy array and reshape
        sample_array = np.array(sample).reshape(1, -1)

        # The "All Zero" Condition Logic
        check_metrics = sample_array[:, 0:7]
        all_zero = (check_metrics == 0).all(axis=1)[0]

        if all_zero:
            prob_val = 0.0
            prediction_val = 0
        else:
            # FIXED: Changed 'loaded_model' to 'model' to match the variable defined at the top
            prediction_val = int(model.predict(sample_array)[0])
            prob_val = float(model.predict_proba(sample_array)[0][1])

        prob_percentage = prob_val * 100

        # If request came from form, render the result page. If JSON, return JSON.
        if request.is_json:
            return jsonify({
                "conversion_prediction": "Yes" if prediction_val == 1 else "No",
                "conversion_probability": f"{prob_percentage:.2f}%",
                "status": "success"
            })
        else:
            return render_template("result.html", 
                                   prediction="Yes" if prediction_val == 1 else "No", 
                                   probability=f"{prob_percentage:.2f}%")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)