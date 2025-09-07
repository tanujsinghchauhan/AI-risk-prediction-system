# Patient Deterioration Prediction using LightGBM

## Project Overview

This project predicts the risk of patient deterioration within the next 90 days using daily patient-level data including vitals, lab results, medication adherence, lifestyle, and demographics. The model leverages gradient-boosted trees (LightGBM) and provides explainable predictions using SHAP values for clinicians.

The workflow is modular, allowing replacement of synthetic data with real clinical data for practical deployment.

---

## Features

- **Input Data**

  - Patient vitals (e.g., systolic blood pressure)
  - Lab results (e.g., HbA1c)
  - Medication adherence (daily dose taken or missed)
  - Lifestyle data (steps, sleep hours)
  - Demographics (age, sex)
  - Label: `deteriorated_in_next_90d` (binary)

- **Feature Engineering**

  - Recent trends (slopes of lab/vital values)
  - Variability (standard deviation)
  - Delta from baseline
  - Medication adherence aggregates (proportion of days taken, max consecutive misses)
  - Lifestyle summaries
  - Missingness patterns

- **Model**

  - **LightGBM Classifier** with class balancing
  - Calibrated probabilities
  - Predicts risk score between 0–1

- **Explainability**

  - Global explanations: feature importance across all patients
  - Local explanations: patient-level SHAP values highlighting top risk drivers

- **Evaluation**

  - AUROC, AUPRC
  - Confusion matrix, sensitivity, specificity
  - Calibration curves and Brier score

- **Dashboard Support**
  - Predictions saved to `artifacts/predictions.csv` for visualization
  - Columns: patient ID, predicted probability, true label

---

## Project Structure

├── data/
│ └── synthetic_patient_data.csv # Simulated patient-level timeline data
├── artifacts/
│ ├── lgb_model.joblib # Trained LightGBM model
│ ├── features_list.joblib # Feature list used in model
│ └── predictions.csv # Test set predictions for dashboard
├── notebooks/
│ └── patient_deterioration.ipynb # Jupyter notebook with full pipeline
├── app.py # Flask API for predictions
├── dashboard.py # Streamlit dashboard
├── README.md
└── requirements.txt

---

## Getting Started

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
Run the Jupyter Notebook

Open notebooks/patient_deterioration.ipynb and execute cells sequentially to:

Load or simulate data

Perform feature engineering

Split into train/test sets

Train LightGBM model

Evaluate model

Generate SHAP explanations

Save predictions (artifacts/predictions.csv)

Visualize predictions

API Usage (Flask)

The project provides a REST API to serve predictions.

Run the API
python app.py
Endpoints

Home

URL: /

Method: GET

Response:
{
  "message": "AI Risk Prediction System is running!"
}
Predict Risk

URL: /predict

Method: POST

JSON Payload: patient features matching features_list.joblib

Example:
{
  "hba1c_last": 7.2,
  "hba1c_mean": 6.8,
  "hba1c_std": 0.4,
  "systolic_bp_last": 135,
  "systolic_bp_mean": 132,
  "systolic_bp_std": 5,
  "weight_last": 75,
  "weight_mean": 74,
  "weight_std": 2,
  "steps_last": 6000,
  "steps_mean": 6200,
  "steps_std": 400,
  "sleep_hours_last": 7,
  "sleep_hours_mean": 7.1,
  "sleep_hours_std": 0.5,
  "med_adherence_prop": 0.9,
  "med_max_miss_streak": 1,
  "frac_days_recorded": 0.95,
  "num_records": 90,
  "age": 60,
  "sex_M": 1
}
Response:
{
  "predicted_risk": 0.475,
  "top_risk_drivers": [
    {"feature": "hba1c_last", "shap_value": 0.12},
    {"feature": "systolic_bp_mean", "shap_value": 0.08},
    ...
  ]
}
Dashboard (Streamlit)

The Streamlit dashboard provides an interactive interface for clinicians to input patient data and visualize risk predictions along with top contributing features.

Run the Dashboard
streamlit run dashboard.py
Features

Enter patient-level features in the sidebar

View predicted risk probability

View top 5 risk drivers (SHAP values)

Load multiple predictions from artifacts/predictions.csv for batch visualization

Dependencies

Python 3.8+

pandas

numpy

scikit-learn

lightgbm

shap

matplotlib

seaborn

Flask

Streamlit

joblib

Notes

Current dataset is synthetic for prototyping.

Replace with real patient data following the same schema for production use.

SHAP plots can be large; consider using a subset of data for interactive visualizations.
```
