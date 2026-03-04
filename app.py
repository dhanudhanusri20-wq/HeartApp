# app.py
import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="logo.png",   # <-- put your custom logo here (png format)
    layout="centered",      # optional, "wide" if you want full width
    initial_sidebar_state="expanded"
)

# ----------------------------
# App Title
# ----------------------------
st.title("❤️ Heart Disease Prediction App")
st.write("Enter your health details below to predict the risk of heart disease.")

# ----------------------------
# User Input Section
# ----------------------------
st.header("Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
ex_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

# ----------------------------
# Prediction Logic (Dummy)
# ----------------------------
# Replace this with your actual ML model later
def predict_heart_disease(age, sex, bp, chol, fbs, max_hr, ex_angina):
    # Dummy rule-based logic
    risk = 0
    if age > 50:
        risk += 1
    if bp > 140:
        risk += 1
    if chol > 240:
        risk += 1
    if fbs == "Yes":
        risk += 1
    if max_hr < 120:
        risk += 1
    if ex_angina == "Yes":
        risk += 1
    return "High Risk" if risk >= 3 else "Low Risk"

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict Heart Disease"):
    result = predict_heart_disease(age, sex, bp, chol, fbs, max_hr, ex_angina)
    st.subheader("Prediction Result:")
    if result == "High Risk":
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

# ----------------------------
# Footer
# ----------------------------
st.write("---")
st.write("Created with ❤️ using Streamlit")
