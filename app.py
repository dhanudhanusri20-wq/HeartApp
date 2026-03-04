import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import sqlite3

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="logo.png",
    layout="centered"
)

# ---------------- DATABASE ---------------- #
conn = sqlite3.connect("predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
   id INTEGER PRIMARY KEY,
age INTEGER,
result TEXT,
risk_score REAL,
timestamp TEXT

)
""")
conn.commit()

# ---------------- BACKGROUND ---------------- #
def set_bg(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()

        page_bg_img = f"""
        <style>
        .stApp {{
           background-image: url("data:image/jpeg;base64,{encoded}");
           background-size: cover;
           background-position: center;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except:
        pass

set_bg("bg.jpg")

# ---------------- SESSION ---------------- #
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# ---------------- LOGIN ---------------- #
def login():
    st.title("🔐 Heart Disease Prediction - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🫀 Heart Disease Risk Prediction System")
st.caption("AI-Based Clinical Decision Support Prototype")

# =====================================================
# SINGLE PREDICTION
# =====================================================

st.header("Single Patient Prediction")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (1-3)", [1,2,3])

sex_val = 1 if sex=="Male" else 0
fbs_val = 1 if fbs=="Yes" else 0
exang_val = 1 if exang=="Yes" else 0

if st.button("Predict"):

    input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs_val,
                            restecg, thalach, exang_val, oldpeak,
                            slope, ca, thal]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    risk = "Low Risk" if probability < 0.3 else \
           "Medium Risk" if probability < 0.7 else \
           "High Risk"

    result_text = "Heart Disease" if prediction==1 else "No Heart Disease"

    st.success(f"Prediction: {result_text}")
    st.info(f"Risk Score: {probability:.2f} → {risk}")

    # Save to database
    cursor.execute(
        "INSERT INTO history (age, result, risk_score) VALUES (?, ?, ?)",
        (age, result_text, probability)
    )
    conn.commit()

    # ---------------- AI EXPLANATION ---------------- #
    explanation = []

    if age > 60:
        explanation.append("Advanced age increases cardiovascular risk.")

    if chol > 240:
        explanation.append("High cholesterol level significantly raises heart disease risk.")

    if trestbps > 140:
        explanation.append("Elevated resting blood pressure is a major risk factor.")

    if exang_val == 1:
        explanation.append("Exercise induced angina indicates cardiac stress.")

    if oldpeak > 2:
        explanation.append("High ST depression suggests abnormal heart function.")

    if cp == 3:
        explanation.append("Asymptomatic chest pain type is associated with higher risk.")

    if probability < 0.3:
        explanation.append("Overall risk is low based on current clinical inputs.")
    elif probability < 0.7:
        explanation.append("Moderate risk detected. Lifestyle modification recommended.")
    else:
        explanation.append("High cardiovascular risk detected. Medical consultation advised.")

    st.subheader("AI Clinical Explanation")
    for point in explanation:
        st.write("•", point)

    # ---------------- PDF REPORT ---------------- #
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Heart Disease Prediction Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Paragraph(f"Result: {result_text}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Score: {probability:.2f}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    st.download_button(
        label="Download Prediction Report (PDF)",
        data=buffer,
        file_name="prediction_report.pdf",
        mime="application/pdf"
    )

# =====================================================
# GRAPH FROM DATABASE
# =====================================================

data = pd.read_sql_query("SELECT * FROM history ORDER BY id ASC", conn)

if not data.empty:
    st.subheader("Risk Score Trend")

    fig, ax = plt.subplots()
    ax.plot(data["risk_score"], marker='o')
    ax.set_xlabel("Prediction Number")
    ax.set_ylabel("Risk Score")
    ax.set_ylim(0,1)
    ax.grid(True)

    st.pyplot(fig)

# =====================================================
# BULK CSV PREDICTION
# =====================================================

st.header("Bulk Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    expected_columns = [
        "age","sex","cp","trestbps","chol","fbs",
        "restecg","thalach","exang","oldpeak",
        "slope","ca","thal"
    ]

    try:
        df = df[expected_columns]
        df = df.fillna(df.median(numeric_only=True))

        df["sex"] = df["sex"].replace({"Male":1,"Female":0})
        df["fbs"] = df["fbs"].replace({"Yes":1,"No":0})
        df["exang"] = df["exang"].replace({"Yes":1,"No":0})

        scaled = scaler.transform(df)
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:,1]

        df["Prediction"] = preds
        df["Risk Score"] = probs

        st.success("Bulk Prediction Completed")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode()

        st.download_button(
            "Download Results CSV",
            csv,
            "bulk_prediction_results.csv",
            "text/csv"
        )

    except:
        st.error("CSV format incorrect. Check column names.")

# =====================================================
# HISTORY TABLE
# =====================================================

st.subheader("Prediction History")
st.dataframe(data)

# =====================================================
# CLEAR & LOGOUT
# =====================================================

col1, col2 = st.columns(2)

with col1:
    if st.button("Clear History"):
        cursor.execute("DELETE FROM history")
        conn.commit()
        st.success("History Cleared")
        st.rerun()

with col2:
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

