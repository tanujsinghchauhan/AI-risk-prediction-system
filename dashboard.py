# dashboard.py
import streamlit as st
import requests
import pandas as pd

# -------------------------------
# 1. Page config
# -------------------------------
st.set_page_config(page_title="AI Risk Prediction Dashboard", layout="wide")
st.title("AI Risk Prediction System")
st.markdown(
    "Enter patient features below to predict risk and view top risk drivers."
)

# -------------------------------
# 2. Input form
# -------------------------------
with st.form("patient_form"):
    st.subheader("Patient Features")
    
    # Example numeric inputs
    hba1c_last = st.number_input("HbA1c (last)", value=7.2, step=0.1)
    hba1c_mean = st.number_input("HbA1c (mean)", value=6.8, step=0.1)
    hba1c_std = st.number_input("HbA1c (std)", value=0.4, step=0.1)
    
    systolic_bp_last = st.number_input("Systolic BP (last)", value=135)
    systolic_bp_mean = st.number_input("Systolic BP (mean)", value=132)
    systolic_bp_std = st.number_input("Systolic BP (std)", value=5)
    
    weight_last = st.number_input("Weight (last)", value=75)
    weight_mean = st.number_input("Weight (mean)", value=74)
    weight_std = st.number_input("Weight (std)", value=2)
    
    steps_last = st.number_input("Steps (last)", value=6000)
    steps_mean = st.number_input("Steps (mean)", value=6200)
    steps_std = st.number_input("Steps (std)", value=400)
    
    sleep_hours_last = st.number_input("Sleep hours (last)", value=7.0, step=0.1)
    sleep_hours_mean = st.number_input("Sleep hours (mean)", value=7.1, step=0.1)
    sleep_hours_std = st.number_input("Sleep hours (std)", value=0.5, step=0.1)
    
    med_adherence_prop = st.number_input("Medication adherence proportion", value=0.9, min_value=0.0, max_value=1.0, step=0.01)
    med_max_miss_streak = st.number_input("Max consecutive missed days", value=1, step=1)
    
    frac_days_recorded = st.number_input("Fraction of days recorded", value=0.95, min_value=0.0, max_value=1.0, step=0.01)
    num_records = st.number_input("Number of records in window", value=90, step=1)
    
    age = st.number_input("Age", value=60, step=1)
    sex_M = st.selectbox("Sex", options=[1, 0], index=0, format_func=lambda x: "Male" if x==1 else "Female")
    
    submitted = st.form_submit_button("Predict Risk")

# -------------------------------
# 3. Call API when form submitted
# -------------------------------
if submitted:
    payload = {
        "hba1c_last": hba1c_last,
        "hba1c_mean": hba1c_mean,
        "hba1c_std": hba1c_std,
        "systolic_bp_last": systolic_bp_last,
        "systolic_bp_mean": systolic_bp_mean,
        "systolic_bp_std": systolic_bp_std,
        "weight_last": weight_last,
        "weight_mean": weight_mean,
        "weight_std": weight_std,
        "steps_last": steps_last,
        "steps_mean": steps_mean,
        "steps_std": steps_std,
        "sleep_hours_last": sleep_hours_last,
        "sleep_hours_mean": sleep_hours_mean,
        "sleep_hours_std": sleep_hours_std,
        "med_adherence_prop": med_adherence_prop,
        "med_max_miss_streak": med_max_miss_streak,
        "frac_days_recorded": frac_days_recorded,
        "num_records": num_records,
        "age": age,
        "sex_M": sex_M
    }
    
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=payload)
        result = response.json()
        
        if "risk_prob" in result:
            st.success(f"Predicted Risk Probability: {result['risk_prob']:.3f}")
            
            st.subheader("Top 5 Risk Drivers (SHAP values)")
            shap_df = pd.DataFrame(result["top_risk_drivers"])
            st.table(shap_df)
        else:
            st.error("Unexpected API response. Make sure 'risk_prob' key is returned.")
    
    except Exception as e:
        st.error(f"Error calling API: {e}")
