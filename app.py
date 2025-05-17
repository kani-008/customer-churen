import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model = joblib.load('best_rf_model.joblib')

st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn:")

# Example inputs - replace with your actual features
feature_1 = st.number_input("Feature 1 (e.g., Monthly Charges)", min_value=0.0)
feature_2 = st.number_input("Feature 2 (e.g., Tenure in months)", min_value=0)
# Add all required features here

# When user clicks predict
if st.button("Predict"):
    # Prepare input as DataFrame or array (ensure order matches model training)
    input_data = np.array([[feature_1, feature_2]])  # Add all features in correct order
    
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"Customer is likely to churn with probability {proba:.2f}")
    else:
        st.success(f"Customer is not likely to churn with probability {1 - proba:.2f}")
