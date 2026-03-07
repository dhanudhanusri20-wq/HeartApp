import streamlit as st
# ---------------- CUSTOM UI STYLE ---------------- #
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1, h2, h3 {
    color: #1f4e79;
}

.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.stButton>button:hover {
    background-color: #125a94;
    color: white;
}

.stTextInput>div>div>input {
    border-radius: 8px;
}

.stNumberInput>div>div>input {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

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

# ---------------- Hugging Face Local Chatbot ---------------- #
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

tokenizer, chatbot_model = load_model()

def ask_chatbot(question):
    today = datetime.now().strftime("%B %d, %Y")
    prompt = f"Today is {today}. Answer simply and accurately: {question}"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = chatbot_model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="logo.png",
    layout="wide"
)


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
        pass

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

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ---------------- SIDEBAR ---------------- #
st.sidebar.image("logo.png", width=120)
st.sidebar.markdown("## ❤️ Heart AI System")

st.session_state.page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🧑 Patient Prediction", "📂 Bulk Prediction", "🏥 Doctor Dashboard", "💬 Chatbot", "🚪 Logout"]
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
def generate_pdf(pid, name, age, result, score):
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
if st.session_state.page == "Home":
    st.title("❤️ Heart Disease Prediction System")
    st.write("Use the sidebar to access prediction, dashboard, and DD CardioBot.")

# ---------------- SINGLE PREDICTION ---------------- #
if st.session_state.page == "Single Prediction":

    st.header("Single Patient Prediction")

    patient_id = st.text_input("Patient ID")
    patient_name = st.text_input("Patient Name")

    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120", ["Yes", "No"])
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("Major Vessels", [0,1,2,3])
    thal = st.selectbox("Thal", [1,2,3])

    # Convert categorical values
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    if st.button("Predict"):

        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                          thalach,exang,oldpeak,slope,ca,thal]])

        scaled = scaler.transform(data)

        prediction = model.predict(scaled)[0]

        probability = model.predict_proba(scaled)[0][1]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        st.success(result)

        # ---------------- RISK METER ---------------- #
        st.subheader("Risk Level")

        st.progress(probability)

        st.write(f"Risk Score: {probability*100:.1f}%")

        if probability < 0.3:
            st.success("Low Risk")
        elif probability < 0.7:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")

        # ---------------- AI HEALTH ADVICE ---------------- #
        st.subheader("Health Advice")

        if probability < 0.3:
            advice = "Maintain a healthy lifestyle with balanced diet and regular exercise."
        elif probability < 0.7:
            advice = "Monitor your health, improve diet, and consult a doctor regularly."
        else:
            advice = "High risk detected. Please consult a cardiologist immediately."

        st.info(advice)

        # ---------------- PROBABILITY GRAPH ---------------- #
        st.subheader("Prediction Probability")

        labels = ["No Heart Disease", "Heart Disease"]
        values = model.predict_proba(scaled)[0]

        fig, ax = plt.subplots()

        ax.bar(labels, values)

        ax.set_ylabel("Probability")
        ax.set_title("Model Confidence")

        st.pyplot(fig)

        # ---------------- SAVE HISTORY ---------------- #
        st.session_state.history.append(probability)

        c.execute(
            "INSERT INTO history (patient_id, patient_name, prediction, risk_score) VALUES (?,?,?,?)",
            (patient_id, patient_name, result, probability)
        )

        conn.commit()

        # ---------------- RISK TREND GRAPH ---------------- #
        st.subheader("Risk Score Trend")

        fig2, ax2 = plt.subplots()

        ax2.plot(st.session_state.history, marker="o")

        ax2.set_ylim(0,1)

        ax2.set_xlabel("Predictions")
        ax2.set_ylabel("Risk Score")

        ax2.set_title("Patient Risk Trend")

        st.pyplot(fig2)

        # ---------------- PDF REPORT ---------------- #
        pdf = generate_pdf(patient_id, patient_name, age, result, probability)

        st.download_button(
            "Download Report",
            pdf,
            f"{patient_name}_report.pdf",
            "application/pdf"
        )


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
        st.download_button("Download Results", csv, "results.csv", "text/csv")
# ---------------- DOCTOR DASHBOARD ---------------- #
if st.session_state.page == "Doctor Dashboard":

    st.header("🏥 Doctor Dashboard")

    # Load patient history
    data = pd.read_sql_query("SELECT * FROM history", conn)

    # ---------------- DASHBOARD METRICS ---------------- #
    st.subheader("Hospital Overview")

    total_patients = len(data)

    high_risk = len(data[data["risk_score"] > 0.7])

    avg_risk = data["risk_score"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Patients", total_patients)

    col2.metric("High Risk Patients", high_risk)

    col3.metric("Average Risk Score", f"{avg_risk:.2f}")

    st.divider()

    # ---------------- SEARCH PATIENT ---------------- #
    st.subheader("Search Patient")

    search = st.text_input("Enter Patient ID or Name")

    if search:
        filtered = data[
            data["patient_id"].astype(str).str.contains(search, case=False) |
            data["patient_name"].astype(str).str.contains(search, case=False)
        ]
    else:
        filtered = data

    # ---------------- PATIENT TABLE ---------------- #
    st.subheader("Patient History")

    st.dataframe(filtered)

    # ---------------- DOWNLOAD CSV ---------------- #
    csv = data.to_csv(index=False).encode()

    st.download_button(
        "Download Patient History",
        csv,
        "patient_history.csv",
        "text/csv"
    )


# ---------------- CHATBOT ---------------- #
if st.session_state.page == "Chatbot":

    st.header("💬 DD CardioBot (Offline)")

    # Step 1: Create memory for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Step 2: Show previous messages (ADD YOUR CODE HERE)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Step 3: Chat input box
    prompt = st.chat_input("Ask something about heart health...")

    # Step 4: When user asks question
    if prompt:

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        answer = ask_chatbot(prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ---------------- LOGOUT ---------------- #
if st.session_state.page == "Logout":
    st.session_state.logged_in = False
    st.success("Logged Out")
    st.stop()






























