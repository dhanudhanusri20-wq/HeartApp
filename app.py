import streamlit as st
import sqlite3
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="logo.png",
    layout="wide"
)

# ---------------- DATABASE ---------------- #
conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    result TEXT,
    risk_score REAL,
    created_at TEXT
)
""")
conn.commit()

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- LOGIN ---------------- #
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

def login():
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "🫀 Single Prediction", "📂 Bulk Prediction", "📊 Analytics", "🕘 History"]
)

# ================= DASHBOARD ================= #
if menu == "🏠 Dashboard":

    st.title("🏥 Medical Dashboard")

    df = pd.read_sql_query("SELECT * FROM history", conn)

    total = len(df)
    high = len(df[df["risk_score"] >= 0.7])
    medium = len(df[(df["risk_score"] >= 0.3) & (df["risk_score"] < 0.7)])
    low = len(df[df["risk_score"] < 0.3])
    avg = df["risk_score"].mean() if total > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total Patients", total)
    col2.metric("High Risk", high)
    col3.metric("Medium Risk", medium)
    col4.metric("Low Risk", low)
    col5.metric("Avg Risk Score", f"{avg:.2f}")

# ================= SINGLE PREDICTION ================= #
elif menu == "🫀 Single Prediction":

    st.title("🫀 Single Patient Prediction")

    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120", ["Yes", "No"])
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("Major Vessels", [0,1,2,3])
    thal = st.selectbox("Thal", [1,2,3])

    sex = 1 if sex=="Male" else 0
    fbs = 1 if fbs=="Yes" else 0
    exang = 1 if exang=="Yes" else 0

    if st.button("Predict"):

        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]])

        scaled = scaler.transform(input_data)
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        result = "Heart Disease" if pred==1 else "No Heart Disease"

        st.success(result)
        st.info(f"Risk Score: {prob:.2f}")

        cursor.execute(
            "INSERT INTO history (age, result, risk_score, created_at) VALUES (?,?,?,?)",
            (age, result, float(prob), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()

# ================= BULK ================= #
elif menu == "📂 Bulk Prediction":

    st.title("📂 Bulk Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()

        required = [
            "age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak",
            "slope","ca","thal"
        ]

        try:
            df = df[required]
            df["sex"] = df["sex"].replace({"Male":1,"Female":0})
            df["fbs"] = df["fbs"].replace({"Yes":1,"No":0})
            df["exang"] = df["exang"].replace({"Yes":1,"No":0})

            scaled = scaler.transform(df)
            preds = model.predict(scaled)
            probs = model.predict_proba(scaled)[:,1]

            df["Prediction"] = preds
            df["Risk Score"] = probs

            st.success("Bulk Prediction Done")
            st.write(df)

        except:
            st.error("CSV format incorrect")

# ================= ANALYTICS ================= #
elif menu == "📊 Analytics":

    st.title("📊 Analytics")

    df = pd.read_sql_query("SELECT * FROM history", conn)

    if len(df) > 0:
