import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define features (replace these with actual feature names)
features = [  # Example (replace with your final df.columns without 'Churn')
    'Age', 'Number of Dependents', 'Number of Referrals',
       'Tenure in Months', 'Avg Monthly Long Distance Charges',
       'Avg Monthly GB Download', 'Monthly Charge', 'Total Charges',
       'Total Refunds', 'Total Extra Data Charges',
       'Total Long Distance Charges', 'Total Revenue', 'Gender_Male',
       'Married_Yes', 'Offer_Offer B', 'Offer_Offer C', 'Offer_Offer D',
       'Offer_Offer E', 'Phone Service_Yes', 'Multiple Lines_Yes',
       'Internet Service_Yes', 'Internet Type_DSL',
       'Internet Type_Fiber Optic', 'Online Security_Yes', 'Online Backup_Yes',
       'Device Protection Plan_Yes', 'Premium Tech Support_Yes',
       'Streaming TV_Yes', 'Streaming Movies_Yes', 'Streaming Music_Yes',
       'Unlimited Data_Yes', 'Contract_One Year', 'Contract_Two Year',
       'Paperless Billing_Yes', 'Payment Method_Credit Card',
       'Payment Method_Mailed Check'
]

st.title("Telecom Customer Churn Prediction")

# Input form
input_data = {}
for feat in features:
    if 'Yes' in feat or 'No' in feat or feat.startswith("Gender") or feat.endswith("_Yes"):
        input_data[feat] = st.selectbox(f"{feat.replace('_', ' ')}", ['No', 'Yes']) == 'Yes'
    elif 'Payment' in feat or 'Contract' in feat or 'Internet Service' in feat:
        input_data[feat] = st.selectbox(f"{feat.replace('_', ' ')}", ['No', 'Yes']) == 'Yes'
    else:
        input_data[feat] = st.number_input(f"Enter {feat.replace('_', ' ')}", min_value=0.0)

# Predict button
if st.button("Predict"):
    df_input = pd.DataFrame([input_data])
    df_input = df_input.astype(float)  # Ensure all are floats for scaler
    df_scaled = scaler.transform(df_input)
    prediction = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    st.subheader("Prediction Result")
    st.write("✅ Likely to Stay" if prediction == 0 else "⚠️ Likely to Churn")
    st.write(f"🔍 Confidence: {prob:.2%}")
