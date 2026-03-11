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

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Single Prediction", "Bulk Prediction", "Doctor Dashboard", "Chatbot", "Logout"]
)

st.session_state.page = page


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



# ---------------- HOME ---------------- #
if st.session_state.page == "Home":

    col1, col2 = st.columns([1,5])

    with col1:
        st.image("logo.png", width=120)

    with col2:
        st.title("Heart Disease Prediction System")

    st.write(
        """
        Welcome to the **AI-Powered Heart Disease Prediction System**.

        This system helps doctors and healthcare professionals predict the risk of heart disease
        using machine learning and provides health insights for better decision making.
        """
    )

    st.subheader("Features of the System")

    st.markdown("""
    - ❤️ **Single Patient Prediction** – Predict heart disease risk for one patient  
    - 📂 **Bulk Prediction** – Upload CSV and analyze multiple patients  
    - 🏥 **Doctor Dashboard** – View patient history and statistics  
    - 💬 **DD CardioBot** – Ask heart health related questions  
    - 📄 **PDF Report Generation** – Download patient prediction report
    """)

    st.subheader("❤️ Heart Health Tip")

    st.info("Exercise at least 30 minutes daily and maintain a healthy diet to reduce heart disease risk.")

# ---------------- SINGLE PREDICTION ---------------- #
if st.session_state.page == "Single Prediction":

    st.header("Single Patient Prediction")

    # -------- Patient Details -------- #
    st.subheader("Patient Details")

    patient_name = st.text_input("Patient Name", key="patient_name")
    patient_id = st.text_input("Patient ID", key="patient_id")
    age = st.number_input("Age", 20, 100, 50, key="age")

    # -------- Medical Parameters -------- #
    st.subheader("Medical Parameters")

    sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
    cp = st.selectbox("Chest Pain Type", [0,1,2,3], key="cp")
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120, key="bp")
    chol = st.number_input("Cholesterol", 100, 400, 200, key="chol")
    fbs = st.selectbox("Fasting Blood Sugar >120", ["Yes", "No"], key="fbs")
    restecg = st.selectbox("Rest ECG", [0,1,2], key="restecg")
    thalach = st.number_input("Max Heart Rate", 60, 220, 150, key="thalach")
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"], key="exang")
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, key="oldpeak")
    slope = st.selectbox("Slope", [0,1,2], key="slope")
    ca = st.selectbox("Major Vessels", [0,1,2,3], key="ca")
    thal = st.selectbox("Thal", [1,2,3], key="thal")

    # -------- Convert categorical values -------- #
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    # -------- Predict Button -------- #
    if st.button("Predict Heart Disease"):

        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,
                          thalach,exang,oldpeak,slope,ca,thal]])

        scaled = scaler.transform(data)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        result = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"

        st.success(result)

        # -------- Risk Meter -------- #
        st.subheader("Risk Level")

        st.progress(float(probability))
        st.write(f"Risk Score: {probability*100:.2f}%")

        if probability < 0.3:
            st.success("Low Risk")
        elif probability < 0.7:
            st.warning("Moderate Risk")
        else:
            st.error("High Risk")

        # -------- Health Advice -------- #
        st.subheader("Health Advice")

        if probability < 0.3:
            advice = "Maintain a healthy lifestyle with balanced diet and regular exercise."
        elif probability < 0.7:
            advice = "Monitor your health and improve lifestyle habits."
        else:
            advice = "High risk detected. Please consult a cardiologist."

        st.info(advice)

        # -------- Probability Graph -------- #
        st.subheader("Prediction Probability")

        labels = ["No Heart Disease", "Heart Disease"]
        values = model.predict_proba(scaled)[0]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_ylabel("Probability")
        ax.set_title("Model Confidence")

        st.pyplot(fig)

        # -------- Explainable AI -------- #
        st.subheader("Explainable AI - Feature Importance")

        features = [
            "Age","Sex","Chest Pain","Blood Pressure","Cholesterol",
            "Fasting Blood Sugar","Rest ECG","Max Heart Rate",
            "Exercise Angina","ST Depression","Slope","Major Vessels","Thal"
        ]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            importances = abs(model.coef_[0])

        else:
            importances = np.zeros(len(features))

        fig2, ax2 = plt.subplots()
        ax2.barh(features, importances)
        ax2.set_xlabel("Impact on Prediction")
        ax2.set_title("Feature Importance")

        st.pyplot(fig2)

        # -------- Save History -------- #
        st.session_state.history.append(probability)

        c.execute(
            "INSERT INTO history (patient_id, patient_name, prediction, risk_score) VALUES (?,?,?,?)",
            (patient_id, patient_name, result, probability)
        )

        conn.commit()

        # -------- Risk Trend Graph -------- #
        st.subheader("Risk Score Trend")

        fig3, ax3 = plt.subplots()
        ax3.plot(st.session_state.history, marker="o")
        ax3.set_ylim(0,1)
        ax3.set_xlabel("Predictions")
        ax3.set_ylabel("Risk Score")
        ax3.set_title("Patient Risk Trend")

        st.pyplot(fig3)

        # ---------------- PDF REPORT FUNCTION ---------------- #

       import io
       from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
       from reportlab.lib.styles import getSampleStyleSheet
       from reportlab.lib.pagesizes import A4


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
    # ---------------- FAQ ANSWERS ---------------- #

faq_answers = {

"What are the symptoms of heart disease?":
"Common symptoms include chest pain, shortness of breath, fatigue, dizziness, and pain in the arms, neck, or jaw.",

"What causes heart disease?":
"Heart disease is mainly caused by plaque buildup in the arteries, high blood pressure, smoking, diabetes, high cholesterol, obesity, and lack of physical activity.",

"How can I reduce heart disease risk?":
"You can reduce risk by exercising regularly, eating a balanced diet, maintaining a healthy weight, avoiding smoking, and controlling blood pressure and cholesterol.",

"What foods are good for heart health?":
"Fruits, vegetables, whole grains, nuts, fish rich in omega-3 fatty acids, and olive oil are good for heart health.",

"What foods should heart patients avoid?":
"Heart patients should avoid foods high in saturated fat, trans fat, salt, processed foods, sugary drinks, and fried foods.",

"How much exercise is good for the heart?":
"At least 30 minutes of moderate exercise such as walking, cycling, or swimming for 5 days a week is recommended.",

"What is normal blood pressure?":
"A normal blood pressure level is around 120/80 mmHg.",

"What is high cholesterol?":
"High cholesterol occurs when there is too much cholesterol in the blood, which can lead to plaque buildup in arteries.",

"How does smoking affect the heart?":
"Smoking damages blood vessels, raises blood pressure, and increases the risk of heart attacks and strokes.",

"What is a heart attack?":
"A heart attack occurs when blood flow to the heart muscle is blocked, usually due to a blood clot.",

"What are early signs of heart attack?":
"Early signs include chest discomfort, shortness of breath, sweating, nausea, dizziness, and pain in the arms or jaw.",

"What should I do during chest pain?":
"If chest pain lasts more than a few minutes, seek immediate medical help or call emergency services.",

"How does stress affect the heart?":
"Chronic stress can increase blood pressure and contribute to heart disease.",

"Is walking good for heart health?":
"Yes, regular walking improves circulation, lowers blood pressure, and strengthens the heart.",

"What is coronary artery disease?":
"It is a condition where coronary arteries become narrowed or blocked due to plaque buildup.",

"How can diabetes affect the heart?":
"Diabetes damages blood vessels and increases the risk of heart disease.",

"What are heart disease risk factors?":
"Major risk factors include high blood pressure, high cholesterol, smoking, diabetes, obesity, and family history.",

"What lifestyle changes reduce heart disease?":
"Healthy diet, regular exercise, quitting smoking, managing stress, and maintaining healthy weight.",

"How important is sleep for heart health?":
"Getting 7-9 hours of quality sleep helps regulate blood pressure and reduces heart disease risk.",

"Can obesity cause heart problems?":
"Yes, obesity increases the risk of high blood pressure, diabetes, and heart disease.",

"What are healthy heart habits?":
"Eating a balanced diet, staying active, avoiding smoking, managing stress, and regular checkups.",

"What is normal heart rate?":
"A normal resting heart rate for adults is between 60 and 100 beats per minute.",

"How does alcohol affect the heart?":
"Excessive alcohol can raise blood pressure and increase heart disease risk.",

"Can heart disease be prevented?":
"Many heart diseases can be prevented through healthy lifestyle choices.",

"What tests detect heart disease?":
"Common tests include ECG, stress test, echocardiogram, blood tests, and angiography.",

"What is ECG?":
"ECG (Electrocardiogram) records the electrical activity of the heart.",

"What is angiography?":
"Angiography is an imaging test used to detect blockages in blood vessels.",

"What is the best diet for heart patients?":
"A heart healthy diet includes fruits, vegetables, whole grains, lean proteins, and low salt foods.",

"How often should heart checkups be done?":
"Adults should have heart checkups at least once a year.",

"When should I see a doctor for chest pain?":
"You should see a doctor immediately if chest pain is severe, lasts more than a few minutes, or spreads to the arm, neck, or jaw."

}
# ---------------- CHATBOT ---------------- #
if st.session_state.page == "Chatbot":

    st.header("💬 DD CardioBot (Offline)")

    # Chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.subheader("Common Heart Health Questions")

    questions = list(faq_answers.keys())

    # Scrollable question list
    with st.container(height=300):

        for q in questions:
            if st.button(q):

                st.session_state.messages.append(
                    {"role": "user", "content": q}
                )

                with st.chat_message("user"):
                    st.markdown(q)

                answer = faq_answers.get(q, ask_chatbot(q))

                with st.chat_message("assistant"):
                    st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

    # Chat input
    prompt = st.chat_input("Ask something about heart health...")

    if prompt:

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        answer = faq_answers.get(prompt, ask_chatbot(prompt))

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


# ---------------- LOGOUT ---------------- #
if st.session_state.page == "Logout":
    st.session_state.logged_in = False
    st.success("Logged Out")
    st.stop()


















































