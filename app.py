import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import sqlite3
import hashlib

# Gemini AI
import google.generativeai as genai

# Configure API Key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load Gemini Model
model = genai.GenerativeModel("gemini-1.5-flash")

# PDF
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

# ---------------- BACKGROUND ---------------- #
def set_bg(image_file):
    try:
        with open(image_file,"rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
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

# ---------------- SESSION INIT ---------------- #
if "logged_in" not in st.session_state: st.session_state["logged_in"] = False
if "page" not in st.session_state: st.session_state["page"] = "Home"
if "history" not in st.session_state: st.session_state["history"] = []

# ---------------- DATABASE ---------------- #
conn = sqlite3.connect("predictions.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT,
    patient_name TEXT,
    prediction TEXT,
    risk_score REAL
)''')
conn.commit()

# ---------------- LOGIN ---------------- #
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_page():
    st.title("🔐 Heart Disease Prediction - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username=="admin" and hash_password(password)==hash_password("1234"):
            st.session_state["logged_in"] = True
            st.success("Login Successful ✅")
        else:
            st.error("Invalid Credentials")

# ---------------- LOGIN CHECK ---------------- #
if not st.session_state["logged_in"]:
    login_page()
    st.stop()  # Stop everything until login is done

# ---------------- SIDEBAR NAVIGATION ---------------- #
st.sidebar.image("logo.png", width=150)
st.session_state["page"] = st.sidebar.radio(
    "Navigation",
    ["Home","Single Prediction","Bulk Prediction","Doctor Dashboard","Chatbot","Logout"]
)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- AI ADVICE ---------------- #
def ai_advice(risk_score):
    if risk_score < 0.3:
        return "Maintain healthy lifestyle, regular check-ups."
    elif risk_score < 0.7:
        return "Consult doctor, monitor diet and exercise."
    else:
        return "Immediate medical attention recommended."

# ---------------- PDF GENERATION ---------------- #
def generate_pdf(patient_id, patient_name, age, result_text, risk_score):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("<b>Heart Disease Prediction Report</b>", styles["Title"]))
    elements.append(Spacer(1,0.5*inch))
    elements.append(Paragraph(f"Patient ID: {patient_id}", styles["Normal"]))
    elements.append(Paragraph(f"Patient Name: {patient_name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Paragraph(f"Result: {result_text}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Score: {risk_score:.2f}", styles["Normal"]))
    elements.append(Paragraph(f"AI Advice: {ai_advice(risk_score)}", styles["Normal"]))
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------- HOME PAGE ---------------- #
if st.session_state["page"]=="Home":
    st.title("🏠 Welcome to Heart Disease Prediction System")
    st.write("Use the sidebar to navigate: Single Prediction, Bulk Prediction, Doctor Dashboard, or Logout.")

# ---------------- SINGLE PREDICTION ---------------- #
if st.session_state["page"]=="Single Prediction":
    st.header("Single Patient Prediction")
    patient_id = st.text_input("Patient ID")
    patient_name = st.text_input("Patient Name")
    age = st.number_input("Age",20,100,50)
    sex = st.selectbox("Sex",["Male","Female"])
    cp = st.selectbox("Chest Pain Type (0-3)",[0,1,2,3])
    trestbps = st.number_input("Resting BP",80,200,120)
    chol = st.number_input("Cholesterol",100,400,200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",["Yes","No"])
    restecg = st.selectbox("Rest ECG (0-2)",[0,1,2])
    thalach = st.number_input("Max Heart Rate",60,220,150)
    exang = st.selectbox("Exercise Induced Angina",["Yes","No"])
    oldpeak = st.number_input("ST Depression",0.0,10.0,1.0)
    slope = st.selectbox("Slope (0-2)",[0,1,2])
    ca = st.selectbox("Major Vessels (0-3)",[0,1,2,3])
    thal = st.selectbox("Thal (1-3)",[1,2,3])

    sex = 1 if sex=="Male" else 0
    fbs = 1 if fbs=="Yes" else 0
    exang = 1 if exang=="Yes" else 0

    if st.button("Predict"):
        input_data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        risk = "Low Risk" if probability<0.3 else "Medium Risk" if probability<0.7 else "High Risk"
        result_text = "Heart Disease" if prediction==1 else "No Heart Disease"
        advice = ai_advice(probability)

        st.success(f"Prediction: {result_text}")
        st.info(f"Risk Score: {probability:.2f} → {risk}")
        st.info(f"AI Advice: {advice}")  # <-- AI advice shown

        # Save to session & DB
        st.session_state["history"].append(probability)
        if patient_id and patient_name:
            c.execute("INSERT INTO history (patient_id,patient_name,prediction,risk_score) VALUES (?,?,?,?)",
                      (patient_id,patient_name,result_text,probability))
            conn.commit()

        # PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elements = []
        styles = getSampleStyleSheet()
        elements.append(Paragraph("<b>Heart Disease Prediction Report</b>", styles["Title"]))
        elements.append(Spacer(1,0.5*inch))
        elements.append(Paragraph(f"Patient ID: {patient_id}", styles["Normal"]))
        elements.append(Paragraph(f"Patient Name: {patient_name}", styles["Normal"]))
        elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
        elements.append(Paragraph(f"Result: {result_text}", styles["Normal"]))
        elements.append(Paragraph(f"Risk Score: {probability:.2f}", styles["Normal"]))
        elements.append(Paragraph(f"AI Advice: {advice}", styles["Normal"]))  # <-- AI advice in PDF
        doc.build(elements)
        buffer.seek(0)
        st.download_button("Download PDF Report",buffer,f"{patient_name}_report.pdf","application/pdf")

        # Graph
        if len(st.session_state["history"])>0:
            fig, ax = plt.subplots()
            ax.plot(range(1,len(st.session_state["history"])+1), st.session_state["history"], marker='o')
            ax.set_xlabel("Prediction Number")
            ax.set_ylabel("Risk Score")
            ax.set_ylim(0,1)
            ax.set_title("Risk Score Trend")
            ax.grid(True)
            st.pyplot(fig)

# ---------------- BULK PREDICTION ---------------- #
if st.session_state["page"] == "Bulk Prediction":
    st.header("Bulk Prediction (CSV Upload)")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            required_cols = ["patient_id","patient_name","age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
            
            # Check CSV has all required columns
            if all(col in df.columns for col in required_cols):
                df = df[required_cols]
                
                # Convert categorical to numeric
                df["sex"] = df["sex"].replace({"Male":1,"Female":0})
                df["fbs"] = df["fbs"].replace({"Yes":1,"No":0})
                df["exang"] = df["exang"].replace({"Yes":1,"No":0})
                
                # Fill missing numeric values
                df = df.fillna(df.median(numeric_only=True))

                # Predict
                input_scaled = scaler.transform(df[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]])
                preds = model.predict(input_scaled)
                probs = model.predict_proba(input_scaled)[:,1]
                df["Prediction"] = ["Heart Disease" if p==1 else "No Heart Disease" for p in preds]
                df["Risk Score"] = probs
                df["AI Advice"] = df["Risk Score"].apply(ai_advice)

                st.success("Bulk Prediction Completed ✅")
                st.dataframe(df)

                # Save all to DB first
                for idx, row in df.iterrows():
                    c.execute(
                        "INSERT INTO history (patient_id,patient_name,prediction,risk_score) VALUES (?,?,?,?)",
                        (row["patient_id"], row["patient_name"], row["Prediction"], row["Risk Score"])
                    )
                conn.commit()

                # Download CSV for all predictions
                csv_bytes = df.to_csv(index=False).encode()
                st.download_button("Download All Results CSV", csv_bytes, "bulk_results.csv", "text/csv")

                # PDF download buttons for each patient
                st.markdown("### Individual Patient Reports")
                for idx, row in df.iterrows():
                    pdf_buf = generate_pdf(row["patient_id"], row["patient_name"], row["age"], row["Prediction"], row["Risk Score"])
                    st.download_button(
                        f"{row['patient_name']} PDF",
                        pdf_buf,
                        file_name=f"{row['patient_name']}_report.pdf",
                        mime="application/pdf"
                    )
            else:
                st.error("CSV format incorrect. Make sure all required columns are included.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# ---------------- DOCTOR DASHBOARD ---------------- #
if st.session_state["page"] == "Doctor Dashboard":

    st.header("👨‍⚕️ Doctor Dashboard")

    history_df = pd.read_sql_query("SELECT * FROM history", conn)

    if history_df.empty:
        st.info("No patient prediction history available.")
    else:
        st.subheader("Patient Prediction History")
        st.dataframe(history_df)

        st.subheader("📊 Patient Risk Score Graph")

        fig, ax = plt.subplots()

        for pid in history_df["patient_id"].unique():
            patient_data = history_df[history_df["patient_id"] == pid]

            ax.plot(
                patient_data.index,
                patient_data["risk_score"],
                marker='o',
                label=f"Patient {pid}"
            )

        ax.set_xlabel("Record Number")
        ax.set_ylabel("Risk Score")
        ax.set_ylim(0,1)
        ax.set_title("Heart Disease Risk Trend")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

import google.generativeai as genai
import streamlit as st

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel("gemini-pro")

# ---------------- CHATBOT ---------------- #
if st.session_state["page"]=="Chatbot":

    st.header("💬 DD CardioBot")
    st.subheader("Your AI Heart Health Assistant ❤️")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.chat_input("Ask anything about heart health...")

    if user_input:

        st.session_state.messages.append(("user", user_input))

        prompt = f"""
        You are DD CardioBot, an AI assistant that helps users with heart health information.
        Give simple, short and helpful advice about heart disease, symptoms, prevention,
        exercise, diet and when to consult a doctor.

        User Question: {user_input}
        """

        response = model.generate_content(prompt)
        reply = response.text

        st.session_state.messages.append(("bot", reply))

    for role,msg in st.session_state.messages:
        if role=="user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)



# ---------------- LOGOUT ---------------- #
if st.session_state["page"]=="Logout":
    st.session_state["logged_in"] = False
    st.session_state["page"] = "Home"
    st.success("Logged out successfully ✅")
    st.stop()












