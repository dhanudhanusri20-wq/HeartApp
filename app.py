import streamlit as st
from google import genai   
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import io
import sqlite3
import hashlib

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------------- GEMINI API ---------------- #

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel("gemini-pro")
    ai_available = True
except:
    ai_available = False


# ---------------- BACKGROUND IMAGE ---------------- #

def set_bg(image_file):

    try:
        with open(image_file, "rb") as f:
            data = f.read()

        encoded = base64.b64encode(data).decode()

        st.markdown(
        f"""
        <style>
        .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

    except:
        st.warning("Background image not found")

set_bg("bg.jpg")

# ---------------- SESSION STATE ---------------- #

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

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

# ---------------- LOGIN PAGE ---------------- #

def login_page():

    st.title("🔐 Heart Disease Prediction System")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and hash_password(password) == hash_password("1234"):

            st.session_state.logged_in = True
            st.success("Login Successful ✅")

        else:

            st.error("Invalid Login")

# ---------------- LOGIN CHECK ---------------- #

if not st.session_state.logged_in:

    login_page()
    st.stop()

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("Navigation")

st.session_state.page = st.sidebar.radio(
"Go to",
[
"Home",
"Single Prediction",
"Bulk Prediction",
"Doctor Dashboard",
"Chatbot",
"Logout"
]
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- AI ADVICE ---------------- #

def ai_advice(score):

    if score < 0.3:
        return "Maintain healthy lifestyle and regular checkups."

    elif score < 0.7:
        return "Consult a doctor and monitor diet and exercise."

    else:
        return "High risk. Immediate medical attention recommended."

# ---------------- PDF FUNCTION ---------------- #

def generate_pdf(pid,name,age,result,score):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("Heart Disease Prediction Report", styles["Title"]))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Patient ID : {pid}", styles["Normal"]))
    elements.append(Paragraph(f"Patient Name : {name}", styles["Normal"]))
    elements.append(Paragraph(f"Age : {age}", styles["Normal"]))
    elements.append(Paragraph(f"Result : {result}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Score : {score:.2f}", styles["Normal"]))
    elements.append(Paragraph(f"Advice : {ai_advice(score)}", styles["Normal"]))

    doc.build(elements)

    buffer.seek(0)

    return buffer

# ---------------- HOME ---------------- #
st.write(st.secrets)

if st.session_state.page == "Home":

    st.title("❤️ Heart Disease Prediction System")
    st.write("Use sidebar to access prediction, dashboard and AI chatbot.")

# ---------------- SINGLE PREDICTION ---------------- #

if st.session_state.page == "Single Prediction":

    st.header("Single Patient Prediction")

    patient_id = st.text_input("Patient ID")
    patient_name = st.text_input("Patient Name")

    age = st.number_input("Age",20,100,50)
    sex = st.selectbox("Sex",["Male","Female"])
    cp = st.selectbox("Chest Pain Type",[0,1,2,3])
    trestbps = st.number_input("Resting BP",80,200,120)
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

        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])

        scaled = scaler.transform(data)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        result = "Heart Disease" if prediction==1 else "No Heart Disease"

        st.success(result)

        st.info(f"Risk Score : {probability:.2f}")
        st.info(ai_advice(probability))

        st.session_state.history.append(probability)

        c.execute(
        "INSERT INTO history (patient_id,patient_name,prediction,risk_score) VALUES (?,?,?,?)",
        (patient_id,patient_name,result,probability)
        )

        conn.commit()

        pdf = generate_pdf(patient_id,patient_name,age,result,probability)

        st.download_button(
        "Download Report",
        pdf,
        f"{patient_name}_report.pdf",
        "application/pdf"
        )

        fig, ax = plt.subplots()
        ax.plot(st.session_state.history, marker="o")
        ax.set_title("Risk Score Trend")
        ax.set_ylim(0,1)
        st.pyplot(fig)

# ---------------- BULK PREDICTION ---------------- #

if st.session_state.page == "Bulk Prediction":

    st.header("Bulk Prediction")

    file = st.file_uploader("Upload CSV", type="csv")

    if file:

        df = pd.read_csv(file)

        features = df.iloc[:,2:]

        scaled = scaler.transform(features)

        preds = model.predict(scaled)
        probs = model.predict_proba(scaled)[:,1]

        df["Prediction"] = preds
        df["Risk Score"] = probs

        st.dataframe(df)

        csv = df.to_csv(index=False).encode()

        st.download_button(
        "Download Results",
        csv,
        "results.csv",
        "text/csv"
        )

# ---------------- DOCTOR DASHBOARD ---------------- #

if st.session_state.page == "Doctor Dashboard":

    st.header("Doctor Dashboard")

    data = pd.read_sql_query("SELECT * FROM history", conn)

    st.dataframe(data)

    if not data.empty:

        fig, ax = plt.subplots()

        ax.plot(data["risk_score"], marker="o")

        ax.set_title("Risk Score Trend")
        ax.set_ylim(0,1)

        st.pyplot(fig)

# ---------------- CHATBOT ---------------- #

if st.session_state["page"] == "Chatbot":

    st.header("💬 DD CardioBot")

    question = st.text_input("Ask anything about heart health")

    if question:

        try:

            prompt = f"""
You are a heart health assistant.
Explain symptoms, prevention, diet, exercise and medical advice in simple words.

User Question: {question}
"""

           response = client.generate_text(
    model="gemini-1.5",  # supported model
    prompt=prompt
)
st.success(response.text)

            st.success(response.text)

        except Exception as e:

            st.error("Gemini AI error")
            st.write(e)



# ---------------- LOGOUT ---------------- #

if st.session_state.page == "Logout":

    st.session_state.logged_in = False
    st.success("Logged Out")
    st.stop()




























