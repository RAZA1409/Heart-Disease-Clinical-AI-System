# ============================================================
# FLASK WEB APPLICATION
# Heart Disease Clinical Dashboard (FULL EXTENDED VERSION)
# ============================================================

import pandas as pd
import joblib
import os
import datetime
import webbrowser
import sqlite3
from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from auth import init_db

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ============================================================
# FEATURE MAPPING (UI → MODEL)
# ============================================================

FEATURE_MAP = {

    "cp": {
        "Typical Angina": 1,
        "Atypical Angina": 2,
        "Non-Anginal Pain": 3,
        "Asymptomatic": 4
    },

    "fbs": {
        "Yes": 1,
        "No": 0
    },

    "exang": {
        "Yes": 1,
        "No": 0
    },

    "restecg": {
        "Normal": 0,
        "ST-T Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    },

    "slope": {
        "Upsloping": 1,
        "Flat": 2,
        "Downsloping": 3
    },

    "thal": {
        "Normal": 3,
        "Fixed Defect": 6,
        "Reversible Defect": 7
    },

    "sex": {
        "Male": 1,
        "Female": 0
    }
}

# ============================================================
# REVERSE FEATURE MAP (MODEL → UI TEXT)
# ============================================================

REVERSE_FEATURE_MAP = {
    key: {v: k for k, v in value.items()}
    for key, value in FEATURE_MAP.items()
}

init_db()

# ------------------------------------------------------------
# Load Model Files
# ------------------------------------------------------------

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def safe_float(value):
    try:
        return float(value)
    except:
        return 0.0

def check_range(value, min_val=None, max_val=None):
    value = safe_float(value)
    if min_val is not None and value < min_val:
        return "LOW"
    if max_val is not None and value > max_val:
        return "HIGH"
    return "NORMAL"

# ------------------------------------------------------------
# Home
# ------------------------------------------------------------

@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# ------------------------------------------------------------
# Login
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

            if len(new_password) < 8:
                conn.close()
                return render_template("change_password.html",
                                       error="Password must be at least 8 characters long")

            hashed_password = generate_password_hash(new_password)

            cursor.execute("UPDATE users SET password = ? WHERE username = ?",
                           (hashed_password, session["user"]))

            conn.commit()
            conn.close()

            return render_template("change_password.html",
                                   success="Password updated successfully")

        conn.close()
        return render_template("change_password.html",
                               error="Current password incorrect")

    return render_template("change_password.html")

# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------

@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    record_file = "records/patient_records.csv"

    if not os.path.exists(record_file):
        return render_template("dashboard.html",
                               total=0, low=0, moderate=0, high=0,
                               low_percent=0, moderate_percent=0, high_percent=0,
                               insight="No patient data available.",
                               recent=[])

    df = pd.read_csv(record_file)
    # Risk trend data (last 10 records)
    trend_df = df.tail(10)
    trend_labels = trend_df["Timestamp"].tolist()
    trend_values = trend_df["Probability (%)"].tolist()

    # Convert to JSON-safe format
    trend_labels = [str(x) for x in trend_labels]
    trend_values = [float(x) for x in trend_values]
    # Recent 5 records
    recent = df.tail(5).iloc[::-1].to_dict(orient="records")

    # Average Probability
    avg_probability = round(df["Probability (%)"].mean(), 2) if not df.empty else 0

    # Highest Risk Patient
    high_risk_patients = df[df["Risk Level"] == "HIGH"]
    if not high_risk_patients.empty:
        top_patient = high_risk_patients.iloc[-1]["Patient Name"]
    else:
        top_patient = "None"

# Last Assessment Time
    last_timestamp = df.iloc[-1]["Timestamp"] if not df.empty else "N/A"
    total = len(df)
    low = len(df[df["Risk Level"] == "LOW"])
    moderate = len(df[df["Risk Level"] == "MODERATE"])
    high = len(df[df["Risk Level"] == "HIGH"])

    low_percent = round((low / total) * 100, 1) if total else 0
    moderate_percent = round((moderate / total) * 100, 1) if total else 0
    high_percent = round((high / total) * 100, 1) if total else 0

    # AI Insight Generator
    if high_percent >= 50:
        insight = "⚠ Majority of patients fall under High Risk. Immediate clinical attention advised."
    elif moderate_percent >= 40:
        insight = "⚠ Significant moderate risk detected. Monitoring recommended."
    else:
        insight = "✓ Majority patients fall under Low Risk category."

    # Last 3 patients
    recent = df.tail(3).iloc[::-1].to_dict(orient="records")

    return render_template("dashboard.html",
                       total=total,
                       low=low,
                       moderate=moderate,
                       high=high,
                       low_percent=low_percent,
                       moderate_percent=moderate_percent,
                       high_percent=high_percent,
                       recent=recent,
                       avg_probability=avg_probability,
                       top_patient=top_patient,
                       last_timestamp=last_timestamp,
                       trend_labels=trend_labels,
                       trend_values=trend_values)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect(url_for("login"))

    patient_name = request.form["patient_name"].strip().title() or "Unknown"

    # UI → Numeric
    sex = FEATURE_MAP["sex"][request.form["sex"]]
    cp = FEATURE_MAP["cp"][request.form["cp"]]
    fbs = FEATURE_MAP["fbs"][request.form["fbs"]]
    restecg = FEATURE_MAP["restecg"][request.form["restecg"]]
    exang = FEATURE_MAP["exang"][request.form["exang"]]
    slope = FEATURE_MAP["slope"][request.form["slope"]]
    thal = FEATURE_MAP["thal"][request.form["thal"]]

    age = float(request.form["age"])
    trestbps = float(request.form["trestbps"])
    chol = float(request.form["chol"])
    thalach = float(request.form["thalach"])
    oldpeak = float(request.form["oldpeak"])
    ca = float(request.form["ca"])

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
    probability = float(model.predict_proba(user_scaled)[0][1])

    prediction = 1 if probability >= 0.5 else 0

    if probability < 0.5:
        risk_level = "LOW"
    elif probability < 0.75:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    patient_id = "PID" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if not os.path.exists("records"):
        os.makedirs("records")

    record_file = "records/patient_records.csv"

    patient_record = {
        "Patient ID": patient_id,
        "Patient Name": patient_name,
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "Resting BP": trestbps,
        "Cholesterol": chol,
        "Max Heart Rate": thalach,
        "ST Depression": oldpeak,
        "Chest Pain Type": cp,
        "Prediction": "Present" if prediction == 1 else "Absent",
        "Probability (%)": round(probability * 100, 2),
        "Risk Level": risk_level
    }

    record_df = pd.DataFrame([patient_record])

    if os.path.exists(record_file):
        record_df.to_csv(record_file, mode="a", header=False, index=False)
    else:
        record_df.to_csv(record_file, index=False)

    return render_template("result.html",
                           patient_name=patient_name,
                           patient_id=patient_id,
                           probability=round(probability * 100, 2),
                           prediction="Present" if prediction == 1 else "Absent",
                           risk_level=risk_level)

# ------------------------------------------------------------
# History
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
# Patient Detail
# ------------------------------------------------------------

@app.route("/patient/<patient_id>")
def patient_detail(patient_id):

    if "user" not in session:
        return redirect(url_for("login"))

    record_file = "records/patient_records.csv"

    if not os.path.exists(record_file):
        return redirect(url_for("history"))

    df = pd.read_csv(record_file)
    patient_row = df[df["Patient ID"] == patient_id]

    if patient_row.empty:
        return redirect(url_for("history"))

    patient = patient_row.iloc[0].to_dict()

    # Numeric → Text
    try:
        patient["Chest Pain Type"] = REVERSE_FEATURE_MAP["cp"].get(
            patient.get("Chest Pain Type"),
            patient.get("Chest Pain Type")
        )
    except:
        pass

    # Clinical Ranges
    patient["Cholesterol Status"] = check_range(patient.get("Cholesterol"), max_val=200)
    patient["Resting BP Status"] = check_range(patient.get("Resting BP"), min_val=90, max_val=120)
    patient["Max Heart Rate Status"] = check_range(patient.get("Max Heart Rate"), min_val=60, max_val=100)
    patient["ST Depression Status"] = check_range(patient.get("ST Depression"), max_val=1)

    patient["Probability (%)"] = round(safe_float(patient.get("Probability (%)")), 2)

    return render_template("patient_detail.html", patient=patient)

# ------------------------------------------------------------
# Delete
# ------------------------------------------------------------

@app.route("/delete/<patient_id>")
def delete_record(patient_id):

    if "user" not in session:
        return redirect(url_for("login"))

    record_file = "records/patient_records.csv"

    if os.path.exists(record_file):
        df = pd.read_csv(record_file)
        df = df[df["Patient ID"] != patient_id]
        df.to_csv(record_file, index=False)

    return redirect(url_for("history"))

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000/login")
    app.run(debug=True)