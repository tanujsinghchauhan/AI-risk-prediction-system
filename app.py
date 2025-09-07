# app.py
import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import shap

# -------------------------------
# 1. Load model and artifacts
# -------------------------------
MODEL_PATH = "artifacts/lgb_model.joblib"
FEATURES_PATH = "artifacts/features_list.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    raise FileNotFoundError("Ensure artifacts/lgb_model.joblib and features_list.joblib exist!")

model = joblib.load(MODEL_PATH)
features_list = joblib.load(FEATURES_PATH)

# SHAP explainer
explainer = shap.TreeExplainer(model)

# -------------------------------
# 2. Initialize Flask
# -------------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "AI Risk Prediction System is running!"})

# -------------------------------
# 3. Prediction endpoint
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON payload like:
    {
        "hba1c_last": 7.2,
        "hba1c_mean": 6.8,
        ...
        "med_adherence_prop": 0.9
    }
    """
    try:
        data = request.json
        # Convert input to dataframe
        df = pd.DataFrame([data], columns=features_list)
        
        # Predict probability
        pred_prob = model.predict_proba(df)[0, 1]
        
        # Compute SHAP values for top 5 risk drivers
        shap_vals = explainer.shap_values(df)[0]  # binary classifier
        shap_df = pd.DataFrame({
            "feature": features_list,
            "shap_value": shap_vals
        }).sort_values("shap_value", key=abs, ascending=False).head(5)
        
        # Response with the correct key for Streamlit
        response = {
            "risk_prob": float(pred_prob),
            "top_risk_drivers": shap_df.to_dict(orient="records")
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# -------------------------------
# 4. Run app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
