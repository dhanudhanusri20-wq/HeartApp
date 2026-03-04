import streamlit as st
import sqlite3
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="AI Heart Disease Prediction",
    page_icon="logo.png",
    layout="wide"
)

# ---------------- BACKGROUND ---------------- #
def set_bg(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """, unsafe_allow_html=True)
    except:
        pass

set_bg("bg.jpg")

# ---------------- CARD STYLE ---------------- #
st.markdown("""
<style>
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
}
.card h2 {
    margin: 0;
    font-size: 28px;
}
.card p {
    margin: 5px 0 0 0;
    font-size: 15px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ---------------- DATABASE ---------------- #
conn = sqlite3.connect("predictions.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    patient_name TEXT,
    age INTEGER,
    gender TEXT,
    result TEXT,
    risk_score REAL,
    created_at TEXT
)
""")
conn.commit()
conn.close()

# ---------------- LOGIN ---------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 AI Clinical Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Credentials")

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SIDEBAR ---------------- #
menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Single Prediction", "Bulk Prediction", "Doctor Dashboard", "Logout"]
)

# ---------------- HOME ---------------- #
if menu == "Home":
    st.title("🫀 AI Powered Heart Disease Prediction System")
    st.write("Advanced Clinical Decision Support using Machine Learning")

# ---------------- SINGLE PREDICTION ---------------- #
elif menu == "Single Prediction":

    st.header("Single Patient AI Prediction")

    patient_id = st.text_input("Patient ID")
    patient_name = st.text_input("Patient Name")

    age = st.number_input("Age", 20, 100, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    sex = 1 if gender == "Male" else 0

    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("FBS >120", ["Yes", "No"])
    fbs = 1 if fbs == "Yes" else 0
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max HR", 60, 220, 150)
    exang = st.selectbox("Exercise Angina", ["Yes", "No"])
    exang = 1 if exang == "Yes" else 0
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("Major Vessels", [0,1,2,3])
    thal = st.selectbox("Thal", [1,2,3])

    if st.button("Predict"):

        data = np.array([[age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak,
                          slope, ca, thal]])

        scaled = scaler.transform(data)
        prediction = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0][1]

        result = "Heart Disease" if prediction == 1 else "No Heart Disease"

        # ---------------- AI RESULT DISPLAY ---------------- #
        if prob >= 0.7:
            st.error(f"🔴 {result} (High Risk)")
        elif prob >= 0.3:
            st.warning(f"🟡 {result} (Medium Risk)")
        else:
            st.success(f"🟢 {result} (Low Risk)")

        st.markdown(f"### 🧠 AI Confidence Score: {prob:.2f}")
        st.progress(int(prob * 100))

        # ---------------- AI EXPLANATION ---------------- #
        st.markdown("### 🤖 AI Explanation")
        reasons = []

        if age > 60:
            reasons.append("Age above 60 increases cardiovascular risk.")
        if chol > 240:
            reasons.append("High cholesterol level detected.")
        if trestbps > 140:
            reasons.append("Elevated resting blood pressure.")
        if exang == 1:
            reasons.append("Exercise induced angina present.")
        if oldpeak > 2:
            reasons.append("Significant ST depression observed.")

        if reasons:
            for r in reasons:
                st.write("•", r)
        else:
            st.write("No major abnormal indicators detected.")

        # ---------------- SAVE TO DATABASE ---------------- #
        conn = sqlite3.connect("predictions.db")
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO history (patient_id, patient_name, age, gender, result, risk_score, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (patient_id, patient_name, age, gender, result, prob, datetime.now()))
        conn.commit()
        conn.close()

        # ---------------- PDF ---------------- #
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("<b>AI Heart Disease Prediction Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.5 * inch))
        elements.append(Paragraph(f"Patient ID: {patient_id}", styles["Normal"]))
        elements.append(Paragraph(f"Name: {patient_name}", styles["Normal"]))
        elements.append(Paragraph(f"Result: {result}", styles["Normal"]))
        elements.append(Paragraph(f"Confidence Score: {prob:.2f}", styles["Normal"]))

        doc.build(elements)
        buffer.seek(0)

        st.download_button(
            "Download PDF Report",
            buffer,
            "prediction_report.pdf",
            "application/pdf"
        )

# ---------------- BULK PREDICTION ---------------- #
elif menu == "Bulk Prediction":

    st.header("Bulk AI Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        expected = ["age","sex","cp","trestbps","chol","fbs",
                    "restecg","thalach","exang","oldpeak",
                    "slope","ca","thal"]

        df = df[expected]
        scaled = scaler.transform(df)

        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:,1]

        df["Prediction"] = preds
        df["Risk Score"] = probs

        st.write(df)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download Results CSV", csv,
                           "bulk_prediction_results.csv", "text/csv")

# ---------------- DOCTOR DASHBOARD ---------------- #
elif menu == "Doctor Dashboard":

    st.header("Doctor Dashboard")

    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql_query("SELECT * FROM history", conn)
    conn.close()

    if not df.empty:

        total = len(df)
        high = len(df[df["risk_score"] >= 0.7])
        medium = len(df[(df["risk_score"] >= 0.3) & (df["risk_score"] < 0.7)])
        low = len(df[df["risk_score"] < 0.3])

        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(f"<div class='card'><h2>{total}</h2><p>Total Patients</p></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='card'><h2>{high}</h2><p>High Risk</p></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='card'><h2>{medium}</h2><p>Medium Risk</p></div>", unsafe_allow_html=True)
        col4.markdown(f"<div class='card'><h2>{low}</h2><p>Low Risk</p></div>", unsafe_allow_html=True)

        st.markdown("### Risk Distribution")
        fig, ax = plt.subplots()
        ax.pie([high, medium, low],
               labels=["High","Medium","Low"],
               autopct="%1.1f%%")
        st.pyplot(fig)

        st.dataframe(df)

    else:
        st.info("No records found")

# ---------------- LOGOUT ---------------- #
elif menu == "Logout":
    st.session_state.logged_in = False
    st.rerun()
