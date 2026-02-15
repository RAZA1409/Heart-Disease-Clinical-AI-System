# ============================================================
# FLASK WEB APPLICATION
# Heart Disease Clinical Dashboard
# ============================================================

import pandas as pd
import numpy as np
import joblib
import os
import datetime
import webbrowser
import sqlite3
from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.security import check_password_hash
from auth import init_db

app = Flask(__name__)
app.secret_key = "supersecretkey"

init_db()

# ------------------------------------------------------------
# Load trained model files
# ------------------------------------------------------------
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ------------------------------------------------------------
# Home Page
# ------------------------------------------------------------
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# ------------------------------------------------------------
# Login Route
# ------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()

        if result and check_password_hash(result[0], password):
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid Credentials")

    return render_template("login.html")

# ------------------------------------------------------------
# Logout
# ------------------------------------------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ------------------------------------------------------------
# Change Password
# ------------------------------------------------------------
@app.route("/change_password", methods=["GET", "POST"])
def change_password():

    if "user" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":

        current_password = request.form["current_password"]
        new_password = request.form["new_password"]

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute("SELECT password FROM users WHERE username = ?", (session["user"],))
        result = cursor.fetchone()

        if result and check_password_hash(result[0], current_password):

            # Strong password validation
            if len(new_password) < 8:
                conn.close()
                return render_template("change_password.html", error="Password must be at least 8 characters long")

            from werkzeug.security import generate_password_hash
            hashed_password = generate_password_hash(new_password)

            cursor.execute("UPDATE users SET password = ? WHERE username = ?",
                           (hashed_password, session["user"]))

            conn.commit()
            conn.close()

            return render_template("change_password.html", success="Password updated successfully")

        conn.close()
        return render_template("change_password.html", error="Current password incorrect")

    return render_template("change_password.html")
# ------------------------------------------------------------
# Dashboard (Protected)
# ------------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    record_file = "records/patient_records.csv"

    if not os.path.exists(record_file):
        return render_template("dashboard.html",
                               total=0,
                               low=0,
                               moderate=0,
                               high=0)

    df = pd.read_csv(record_file)

    total = len(df)
    low = len(df[df["Risk Level"] == "LOW"])
    moderate = len(df[df["Risk Level"] == "MODERATE"])
    high = len(df[df["Risk Level"] == "HIGH"])

    return render_template("dashboard.html",
                           total=total,
                           low=low,
                           moderate=moderate,
                           high=high)

# ------------------------------------------------------------
# Prediction Route (Protected)
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect(url_for("login"))

    patient_name = request.form["patient_name"].strip().title()
    if patient_name == "":
        patient_name = "Unknown"

    age = float(request.form["age"])
    sex = float(request.form["sex"])
    cp = float(request.form["cp"])
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    fbs = float(request.form["fbs"])
    restecg = float(request.form["restecg"])
    thalach = float(request.form["thalach"])
    exang = float(request.form["exang"])
    oldpeak = float(request.form["oldpeak"])
    slope = float(request.form["slope"])
    ca = float(request.form["ca"])
    thal = float(request.form["thal"])

    user_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Chest pain type": cp,
        "BP": trestbps,
        "Cholesterol": chol,
        "FBS over 120": fbs,
        "EKG results": restecg,
        "Max HR": thalach,
        "Exercise angina": exang,
        "ST depression": oldpeak,
        "Slope of ST": slope,
        "Number of vessels fluro": ca,
        "Thallium": thal
    }])

    user_scaled = scaler.transform(user_data)

    probability = model.predict_proba(user_scaled)[0][1]
    prediction = 1 if probability >= 0.5 else 0

    if probability < 0.3:
        risk_level = "LOW"
        color = "green"
    elif probability < 0.7:
        risk_level = "MODERATE"
        color = "orange"
    else:
        risk_level = "HIGH"
        color = "red"

    patient_id = "PID" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if not os.path.exists("records"):
        os.makedirs("records")

    record_file = "records/patient_records.csv"

    patient_record = {
        "Patient ID": patient_id,
        "Patient Name": patient_name,
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "Sex": sex,
        "Prediction": "Present" if prediction == 1 else "Absent",
        "Probability (%)": round(probability * 100, 2),
        "Risk Level": risk_level
    }

    record_df = pd.DataFrame([patient_record])

    if os.path.exists(record_file):
        record_df.to_csv(record_file, mode="a", header=False, index=False)
    else:
        record_df.to_csv(record_file, index=False)

    return render_template(
        "result.html",
        patient_name=patient_name,
        patient_id=patient_id,
        probability=round(probability * 100, 2),
        prediction="Present" if prediction == 1 else "Absent",
        risk_level=risk_level,
        color=color
    )

# ------------------------------------------------------------
# History Page (Protected)
# ------------------------------------------------------------
@app.route("/history")
def history():

    if "user" not in session:
        return redirect(url_for("login"))

    record_file = "records/patient_records.csv"

    if os.path.exists(record_file):
        df = pd.read_csv(record_file)
        records = df.to_dict(orient="records")
    else:
        records = []

    return render_template("history.html", records=records)

# ------------------------------------------------------------
# Delete Patient Record
# ------------------------------------------------------------
@app.route("/delete/<patient_id>")
def delete_record(patient_id):

    if "user" not in session:
        return redirect(url_for("login"))

    record_file = "records/patient_records.csv"

    if os.path.exists(record_file):
        df = pd.read_csv(record_file)

        # Remove record by Patient ID
        df = df[df["Patient ID"] != patient_id]

        df.to_csv(record_file, index=False)

    return redirect(url_for("history"))
# ------------------------------------------------------------
# Run App
# ------------------------------------------------------------
if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000/login")
    app.run(debug=True)