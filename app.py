import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import hashlib
import datetime
import base64

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="logo.png",
    layout="centered"
)

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

# Create users table
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password_hash TEXT,
    role TEXT,
    created_at TEXT
)
""")

# Create history table
c.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    age INTEGER,
    result TEXT,
    risk_score REAL,
    created_at TEXT
)
""")
conn.commit()

# ---------------- PASSWORD HASH ---------------- #
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ---------------- DEFAULT ADMIN ---------------- #
def create_admin():
    c.execute("SELECT * FROM users WHERE username=?", ("admin",))
    if not c.fetchone():
        c.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                  ("admin", hash_password("admin123"), "admin", str(datetime.datetime.now())))
        conn.commit()

create_admin()

# ---------------- SESSION ---------------- #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = None

# ---------------- REGISTER ---------------- #
def register():
    st.subheader("Doctor Registration")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")

    if st.button("Register"):
        try:
            c.execute("INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                      (new_user, hash_password(new_pass), "doctor", str(datetime.datetime.now())))
            conn.commit()
            st.success("Doctor Registered Successfully!")
        except:
            st.error("Username already exists!")

# ---------------- LOGIN ---------------- #
def login():
    st.title("🔐 Login")
    user = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        hashed = hash_password(password)
        c.execute("SELECT id, role FROM users WHERE username=? AND password_hash=?", (user, hashed))
        result = c.fetchone()

        if result:
            st.session_state.logged_in = True
            st.session_state.user_id = result[0]
            st.session_state.role = result[1]
            st.session_state.username = user
            st.rerun()
        else:
            st.error("Invalid Credentials")

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- MAIN ---------------- #
if not st.session_state.logged_in:
    login()
    st.markdown("---")
    register()
    st.stop()

# ---------------- SIDEBAR ---------------- #
st.sidebar.title(f"Welcome {st.session_state.username}")
menu = st.sidebar.radio("Menu", ["Predict", "Dashboard", "Logout"])

# ---------------- PREDICT ---------------- #
if menu == "Predict":
    st.title("🫀 Heart Disease Prediction")

    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar >120", ["Yes", "No"])
    restecg = st.selectbox("Rest ECG", [0,1,2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0,1,2])
    ca = st.selectbox("Major Vessels", [0,1,2,3])
    thal = st.selectbox("Thal", [1,2,3])

    sex = 1 if sex=="Male" else 0
    fbs = 1 if fbs=="Yes" else 0
    exang = 1 if exang=="Yes" else 0

    if st.button("Predict"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak,
                                slope, ca, thal]])

        scaled = scaler.transform(input_data)
        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][1]

        result_text = "Heart Disease" if prediction==1 else "No Heart Disease"

        st.success(result_text)
        st.info(f"Risk Score: {probability:.2f}")

        # Save to DB
        c.execute("INSERT INTO history (user_id, age, result, risk_score, created_at) VALUES (?, ?, ?, ?, ?)",
                  (st.session_state.user_id, age, result_text, float(probability), str(datetime.datetime.now())))
        conn.commit()

# ---------------- DASHBOARD ---------------- #
elif menu == "Dashboard":
    st.title("📊 Analytics Dashboard")

    if st.session_state.role == "admin":
        df = pd.read_sql_query("SELECT * FROM history", conn)
    else:
        df = pd.read_sql_query("SELECT * FROM history WHERE user_id=?",
                               conn, params=(st.session_state.user_id,))

    if not df.empty:
        st.dataframe(df)

        st.subheader("Risk Trend")
        fig, ax = plt.subplots()
        ax.plot(df["risk_score"])
        ax.set_ylim(0,1)
        st.pyplot(fig)

        st.metric("Total Predictions", len(df))
    else:
        st.info("No data available.")

# ---------------- LOGOUT ---------------- #
elif menu == "Logout":
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.role = None
    st.session_state.username = None
    st.rerun()
