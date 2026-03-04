import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import sqlite3
import hashlib
from datetime import datetime

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
def create_tables():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS history(
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

create_tables()

# ---------------- DEFAULT USER ---------------- #
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_default_user():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users VALUES (?,?)",
                  ("admin", hash_password("1234")))
        conn.commit()
    conn.close()

create_default_user()

# ---------------- LOGIN ---------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        conn = sqlite3.connect("predictions.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?",
                  (username, hash_password(password)))
        data = c.fetchone()
        conn.close()

        if data:
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

# ---------------- TITLE ---------------- #
st.title("🫀 Heart Disease Risk Prediction System")

# =====================================================
# SINGLE PREDICTION
# =====================================================

st.header("Single Patient Prediction")

patient_id = st.text_input("Patient ID")
patient_name = st.text_input("Patient Name")

age = st.number_input("Age", 20, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar >120", ["Yes", "No"])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Major Vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (1-3)", [1,2,3])

sex = 1 if gender=="Male" else 0
fbs = 1 if fbs=="Yes" else 0
exang = 1 if exang=="Yes" else 0

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol,
                            fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    risk = "Low Risk" if probability < 0.3 else \
           "Medium Risk" if probability < 0.7 else \
           "High Risk"

    result_text = "Heart Disease" if prediction==1 else "No Heart Disease"

    st.success(f"Prediction: {result_text}")
    st.info(f"Risk Score: {probability:.2f} → {risk}")

    # SAVE TO DATABASE
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)",
              (patient_id, patient_name, age,
               gender, result_text,
               probability, str(datetime.now())))
    conn.commit()
    conn.close()

    # PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Heart Disease Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f"Patient: {patient_name}", styles["Normal"]))
    elements.append(Paragraph(f"Result: {result_text}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Score: {probability:.2f}", styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)

    st.download_button("Download PDF",
                       buffer,
                       "report.pdf",
                       "application/pdf")

# =====================================================
# BULK PREDICTION
# =====================================================

st.header("Bulk CSV Prediction")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower().str.strip()

    expected = ["age","sex","cp","trestbps","chol","fbs",
                "restecg","thalach","exang","oldpeak",
                "slope","ca","thal"]

    try:
        df = df[expected]
        df = df.fillna(df.median(numeric_only=True))

        scaled = scaler.transform(df)
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:,1]

        df["Prediction"] = preds
        df["Risk Score"] = probs

        st.success("Bulk Prediction Done")
        st.write(df)

    except:
        st.error("CSV format incorrect.")

# =====================================================
# DOCTOR DASHBOARD
# =====================================================

st.header("Doctor Dashboard")

conn = sqlite3.connect("predictions.db")
df = pd.read_sql_query("SELECT * FROM history", conn)
conn.close()

if not df.empty:

    total = len(df)
    high = len(df[df["risk_score"]>=0.7])
    medium = len(df[(df["risk_score"]>=0.3)&(df["risk_score"]<0.7)])
    low = len(df[df["risk_score"]<0.3])

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total", total)
    col2.metric("High", high)
    col3.metric("Medium", medium)
    col4.metric("Low", low)

    st.write(df)

    fig, ax = plt.subplots()
    ax.pie([high,medium,low],
           labels=["High","Medium","Low"],
           autopct="%1.1f%%")
    st.pyplot(fig)

else:
    st.info("No records found.")

# =====================================================
# LOGOUT
# =====================================================

if st.button("Logout"):
    st.session_state.logged_in=False
    st.rerun()
