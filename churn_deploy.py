import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('churn_model.pkl')

st.title("🔍 Customer Churn Prediction")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Feature engineering (sample)
contract_dict = {"Month-to-month": 0, "One year": 1, "Two year": 2}
features = pd.DataFrame({
    'gender': [1 if gender == "Male" else 0],
    'SeniorCitizen': [senior],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'Contract': [contract_dict[contract]]
})

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(features)[0]
    st.success("🔴 Customer is likely to churn" if prediction else "🟢 Customer is not likely to churn")
