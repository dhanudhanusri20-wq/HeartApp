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
st.set_page_config(page_title="Heart Disease Prediction", page_icon="logo.png", layout="centered")

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

# ---------------- DATABASE ---------------- #
conn = sqlite3.connect("predictions.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    name TEXT,
    prediction TEXT,
    risk_score REAL
)
""")
conn.commit()

# ---------------- SESSION INIT ---------------- #
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- SIDEBAR NAVIGATION ---------------- #
st.sidebar.title("Navigation")
if st.session_state["logged_in"]:
    page = st.sidebar.radio("Go to", ["Home","Single Prediction","Bulk Prediction","Doctor Dashboard","Logout"])
else:
    page = "Login"

# ================= LOGIN PAGE ================= #
def login_page():
    st.title("🔐 Heart Disease Prediction - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.experimental_rerun()
        else:
            st.error("Invalid Credentials")

# ================= HOME PAGE ================= #
def home_page():
    st.title("🫀 Heart Disease Prediction System")
    st.write("Welcome to the AI-based Heart Disease Risk Prediction App.")
    st.write("Use the sidebar to navigate between pages.")
    
# ================= SINGLE PATIENT PREDICTION ================= #
def single_prediction():
    st.header("Single Patient Prediction")
    patient_id = st.text_input("Patient ID")
    patient_name = st.text_input("Patient Name")
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
        risk = "Low Risk" if probability < 0.3 else "Medium Risk" if probability < 0.7 else "High Risk"
        result_text = "Heart Disease" if prediction==1 else "No Heart Disease"

        # Animated card
        st.markdown(f"""
        <div style='
            background-color: {"#90ee90" if risk=="Low Risk" else "#ffff99" if risk=="Medium Risk" else "#ff7f7f"};
            padding: 15px; border-radius: 10px; width:300px; text-align:center; margin:auto;'>
            <h3>{result_text}</h3>
            <p>Risk Score: {probability:.2f} → {risk}</p>
        </div>
        """, unsafe_allow_html=True)

        # Save history in session & DB
        st.session_state["history"].append(probability)
        c.execute("INSERT INTO history(patient_id,name,prediction,risk_score) VALUES (?,?,?,?)",
                  (patient_id, patient_name, result_text, float(probability)))
        conn.commit()

        # PDF report
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        elements.append(Paragraph("<b>Heart Disease Prediction Report</b>", styles["Title"]))
        elements.append(Spacer(1,0.5*inch))
        elements.append(Paragraph(f"Patient ID: {patient_id}", styles["Normal"]))
        elements.append(Paragraph(f"Patient Name: {patient_name}", styles["Normal"]))
        elements.append(Paragraph(f"Result: {result_text}", styles["Normal"]))
        elements.append(Paragraph(f"Risk Score: {probability:.2f}", styles["Normal"]))
        doc.build(elements)
        buffer.seek(0)
        st.download_button("Download Prediction Report (PDF)", buffer, "prediction_report.pdf", "application/pdf")

        # Graph
        if len(st.session_state["history"])>0:
            st.subheader("Risk Score Trend")
            fig, ax = plt.subplots()
            ax.plot(range(1,len(st.session_state["history"])+1),
                    st.session_state["history"], marker='o')
            ax.set_xlabel("Prediction Number")
            ax.set_ylabel("Risk Score")
            ax.set_ylim(0,1)
            ax.grid(True)
            st.pyplot(fig)

# ================= BULK PREDICTION ================= #
def bulk_prediction():
    st.header("Bulk Prediction (CSV Upload)")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        expected_columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
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
            st.success("Bulk Prediction Completed ✅")
            st.write(df)
            csv = df.to_csv(index=False).encode()
            st.download_button("Download Results CSV", csv, "bulk_prediction_results.csv","text/csv")
        except:
            st.error("CSV format incorrect. Check column names.")

# ================= DOCTOR DASHBOARD ================= #
def doctor_dashboard():
    st.header("Doctor Dashboard - All Predictions")
    df_db = pd.read_sql("SELECT * FROM history", conn)
    st.write(df_db)

# ================= LOGOUT / CLEAR ================= #
def logout_clear():
    col1,col2 = st.columns(2)
    with col1:
        if st.button("Clear History"):
            st.session_state["history"]=[]
            c.execute("DELETE FROM history")
            conn.commit()
            st.success("History Cleared")
            st.experimental_rerun()
    with col2:
        if st.button("Logout"):
            st.session_state["logged_in"]=False
            st.experimental_rerun()

# ================= PAGE RENDER ================= #
if page=="Login":
    login_page()
elif page=="Home":
    home_page()
elif page=="Single Prediction":
    single_prediction()
elif page=="Bulk Prediction":
    bulk_prediction()
elif page=="Doctor Dashboard":
    doctor_dashboard()
elif page=="Logout":
    logout_clear()
