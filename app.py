import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

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

# ---------------- SESSION INIT ---------------- #
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "history" not in st.session_state:
    st.session_state["history"] = []

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

# ---------------- TITLE ---------------- #
st.title("🫀 Heart Disease Risk Prediction System")
st.caption("AI-Based Clinical Decision Support Prototype")

# ======================================================
# SINGLE PATIENT PREDICTION
# ======================================================

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

sex = 1 if sex=="Male" else 0
fbs = 1 if fbs=="Yes" else 0
exang = 1 if exang=="Yes" else 0

if st.button("Predict"):

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
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

    st.session_state["history"].append(probability)

    # ---------------- PDF GENERATION ---------------- #
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

# ======================================================
# GRAPH
# ======================================================

if len(st.session_state["history"]) > 0:

    st.subheader("Risk Score Trend")

    fig, ax = plt.subplots()
    ax.plot(range(1, len(st.session_state["history"]) + 1),
            st.session_state["history"],
            marker='o')
    ax.set_xlabel("Prediction Number")
    ax.set_ylabel("Risk Score")
    ax.set_ylim(0,1)
    ax.grid(True)

    st.pyplot(fig)

else:
    st.info("No predictions yet.")

# ======================================================
# CSV BULK PREDICTION
# ======================================================

st.header("Bulk Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.columns = df.columns.str.strip().str.lower()
    df = df.fillna(df.median(numeric_only=True))

    if "sex" in df.columns:
        df["sex"] = df["sex"].map({"Male":1,"Female":0}).fillna(df["sex"])

    if "fbs" in df.columns:
        df["fbs"] = df["fbs"].map({"Yes":1,"No":0}).fillna(df["fbs"])

    if "exang" in df.columns:
        df["exang"] = df["exang"].map({"Yes":1,"No":0}).fillna(df["exang"])

    try:
        scaled = scaler.transform(df)
        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:,1]

        df["Prediction"] = preds
        df["Risk Score"] = probs

        st.success("Bulk Prediction Completed")
        st.write(df)

        csv = df.to_csv(index=False).encode()

        st.download_button(
            "Download Results CSV",
            csv,
            "bulk_prediction_results.csv",
            "text/csv"
        )

    except:
        st.error("Column mismatch or invalid format!")

# ======================================================
# HISTORY
# ======================================================

st.subheader("Prediction History")
st.write(st.session_state["history"])

# ======================================================
# CLEAR & LOGOUT
# ======================================================

col1, col2 = st.columns(2)

with col1:
    if st.button("Clear History"):
        st.session_state["history"] = []
        st.success("History Cleared")

with col2:
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

        

