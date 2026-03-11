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

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# ---------------- DATABASE ---------------- #

conn = sqlite3.connect("predictions.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS history (
id INTEGER PRIMARY KEY AUTOINCREMENT,
patient_id TEXT,
patient_name TEXT,
prediction TEXT,
risk_score REAL
)
""")

conn.commit()

# ---------------- PASSWORD HASH ---------------- #

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------- PDF REPORT FUNCTION ---------------- #

def generate_pdf(patient_id, patient_name, age, result, probability):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("Heart Disease Prediction Report", styles["Title"]))
    story.append(Spacer(1,20))

    story.append(Paragraph(f"Patient ID: {patient_id}", styles["Normal"]))
    story.append(Paragraph(f"Patient Name: {patient_name}", styles["Normal"]))
    story.append(Paragraph(f"Age: {age}", styles["Normal"]))

    story.append(Spacer(1,20))

    story.append(Paragraph(f"Prediction Result: {result}", styles["Normal"]))
    story.append(Paragraph(f"Risk Score: {probability*100:.2f}%", styles["Normal"]))

    story.append(Spacer(1,20))

    if probability < 0.3:
        advice = "Low Risk: Maintain healthy lifestyle."
    elif probability < 0.7:
        advice = "Moderate Risk: Improve diet and exercise."
    else:
        advice = "High Risk: Please consult a cardiologist immediately."

    story.append(Paragraph(f"Health Advice: {advice}", styles["Normal"]))

    doc.build(story)

    buffer.seek(0)

    return buffer


# ---------------- SESSION ---------------- #

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LOGIN ---------------- #

def login():

    st.title("🔐 Heart Disease Prediction System")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):

        if user == "admin" and hash_password(pwd) == hash_password("1234"):
            st.session_state.logged_in = True
            st.success("Login Successful")

        else:
            st.error("Invalid Login")


if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- SIDEBAR ---------------- #

page = st.sidebar.radio(
    "Navigation",
    ["Home","Single Prediction","Bulk Prediction","Doctor Dashboard","Logout"]
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- HOME ---------------- #

if page == "Home":

    st.title("❤️ Heart Disease Prediction System")

    st.write("""
Welcome to the AI Powered Heart Disease Prediction System.

This system helps doctors and healthcare professionals
predict heart disease risk using machine learning.
""")

    st.markdown("""
Features:

❤️ Single Patient Prediction  
📂 Bulk Prediction  
🏥 Doctor Dashboard  
📄 PDF Report Generation
""")

    st.info("Exercise daily and maintain healthy diet.")

# ---------------- SINGLE PREDICTION ---------------- #

elif page == "Single Prediction":

    st.header("Single Patient Prediction")

    patient_name = st.text_input("Patient Name")
    patient_id = st.text_input("Patient ID")

    age = st.number_input("Age",20,100,50)

    sex = st.selectbox("Sex",["Male","Female"])
    cp = st.selectbox("Chest Pain Type",[0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure",80,200,120)
    chol = st.number_input("Cholesterol",100,400,200)
    fbs = st.selectbox("Fasting Blood Sugar >120",["Yes","No"])
    restecg = st.selectbox("Rest ECG",[0,1,2])
    thalach = st.number_input("Max Heart Rate",60,220,150)
    exang = st.selectbox("Exercise Angina",["Yes","No"])
    oldpeak = st.number_input("ST Depression",0.0,10.0,1.0)
    slope = st.selectbox("Slope",[0,1,2])
    ca = st.selectbox("Major Vessels",[0,1,2,3])
    thal = st.selectbox("Thal",[1,2,3])

    sex = 1 if sex=="Male" else 0
    fbs = 1 if fbs=="Yes" else 0
    exang = 1 if exang=="Yes" else 0

    if st.button("Predict"):

        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                          thalach,exang,oldpeak,slope,ca,thal]])

        scaled = scaler.transform(data)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        result = "Heart Disease Detected" if prediction==1 else "No Heart Disease"

        st.success(result)

        st.progress(float(probability))

        st.write(f"Risk Score: {probability*100:.2f}%")

        # graph
        labels=["No Disease","Disease"]
        values=model.predict_proba(scaled)[0]

        fig,ax=plt.subplots()
        ax.bar(labels,values)

        st.pyplot(fig)

        # feature importance
        st.subheader("Feature Importance")

        features=["Age","Sex","CP","BP","Chol","FBS","ECG",
                  "HR","Angina","ST","Slope","Vessels","Thal"]

        if hasattr(model,"feature_importances_"):
            imp=model.feature_importances_

            fig2,ax2=plt.subplots()

            ax2.barh(features,imp)

            st.pyplot(fig2)

        # save database
        c.execute(
        "INSERT INTO history (patient_id,patient_name,prediction,risk_score) VALUES (?,?,?,?)",
        (patient_id,patient_name,result,probability))

        conn.commit()

        # PDF
        st.subheader("Download Report")

        pdf=generate_pdf(patient_id,patient_name,age,result,probability)

        st.download_button(
        "Download PDF",
        pdf,
        "heart_report.pdf",
        "application/pdf"
        )

# ---------------- BULK ---------------- #

elif page=="Bulk Prediction":

    st.header("Bulk Prediction")

    file=st.file_uploader("Upload CSV")

    if file:

        df=pd.read_csv(file)

        features=df.iloc[:,2:]

        scaled=scaler.transform(features)

        preds=model.predict(scaled)

        probs=model.predict_proba(scaled)[:,1]

        df["Prediction"]=preds
        df["Risk Score"]=probs

        st.dataframe(df)

# ---------------- DASHBOARD ---------------- #

elif page=="Doctor Dashboard":

    st.header("Doctor Dashboard")

    data=pd.read_sql_query("SELECT * FROM history",conn)

    st.dataframe(data)

# ---------------- LOGOUT ---------------- #

elif page=="Logout":

    st.session_state.logged_in=False
    st.success("Logged Out")
















































