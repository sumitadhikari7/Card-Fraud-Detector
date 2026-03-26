import streamlit as st
import numpy as np
import joblib
from datetime import datetime

@st.cache_resource
def load_assets():
    model = joblib.load("fraud_detection_model.pkl")
    return model

model = load_assets()

#App UI
st.set_page_config(page_title="Fraud Guard", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")

st.sidebar.header("Transaction Inputs")
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=10.0)
time_input = st.sidebar.time_input("Time of Transaction", datetime.now().time())

st.subheader("Model Features")
col1, col2 = st.columns(2)
with col1:
    v17 = st.slider("V17 (High Impact)", -30.0, 10.0, 0.0)
    v14 = st.slider("V14 (High Impact)", -20.0, 10.0, 0.0)
with col2:
    v12 = st.slider("V12", -20.0, 10.0, 0.0)
    v10 = st.slider("V10", -20.0, 10.0, 0.0)

# Feature Engineering
def process_input():
    # Time Risk
    hour = time_input.hour
    time_risk = 3 if 2 <= hour <= 5 else (2 if (hour > 5 and hour <= 8) or (hour >= 22) else 1)

    # Is Zero Amount
    is_zero = 1 if amount == 0 else 0

    features = [0.0] * 28
    features[16] = v17
    features[13] = v14
    features[11] = v12
    features[9] = v10

    final_input = features + [time_risk, is_zero, amount]
    return np.array(final_input).reshape(1, -1)

# Prediction
if st.button("Predict Fraud"):
    input_data = process_input()
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.divider()