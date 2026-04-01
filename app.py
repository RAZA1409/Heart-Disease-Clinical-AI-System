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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle
from database import get_db_connection
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from flask import send_file
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

from flask import make_response

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        remember = request.form.get("remember")

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()

        if result and check_password_hash(result[0], password):
            session["user"] = username

            response = make_response(redirect(url_for("dashboard")))

            if remember:
                response.set_cookie("remember_user", username, max_age=60*60*24*30)  # 30 days
            else:
                response.set_cookie("remember_user", "", expires=0)

            return response

        else:
            return render_template("login.html", error="Invalid Credentials")

    remembered_user = request.cookies.get("remember_user")
    return render_template("login.html", remembered_user=remembered_user)

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


@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "POST":
        username = request.form["username"]
        new_password = request.form["new_password"]

        if len(new_password) < 8:
            return render_template("forgot_password.html",
                                   error="Password must be at least 8 characters long")

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()

        if not user:
            conn.close()
            return render_template("forgot_password.html",
                                   error="Username not found")

        hashed_password = generate_password_hash(new_password)

        cursor.execute("UPDATE users SET password = ? WHERE username = ?",
                       (hashed_password, username))

        conn.commit()
        conn.close()

        return render_template("forgot_password.html",
                               success="Password reset successful. You can now login.")

    return render_template("forgot_password.html")


# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    patients = conn.execute("SELECT * FROM patients").fetchall()
    conn.close()

    total = len(patients)

    low = len([p for p in patients if p["risk_level"] == "LOW"])
    moderate = len([p for p in patients if p["risk_level"] == "MODERATE"])
    high = len([p for p in patients if p["risk_level"] == "HIGH"])

    low_percent = round((low / total) * 100, 1) if total else 0
    moderate_percent = round((moderate / total) * 100, 1) if total else 0
    high_percent = round((high / total) * 100, 1) if total else 0

    recent = patients[::-1][:5] if patients else []

    avg_probability = round(
        sum([p["probability"] for p in patients]) / total, 2
    ) if total else 0

    # 🔥 Highest Risk Patient
    high_patients = [p for p in patients if p["risk_level"] == "HIGH"]
    top_patient = high_patients[-1]["patient_name"] if high_patients else "None"

    last_timestamp = patients[-1]["timestamp"] if patients else "N/A"

    # 🔥 Trend Graph
    trend_data = patients[-10:]
    trend_labels = [p["timestamp"] for p in trend_data]
    trend_values = [p["probability"] for p in trend_data]

    # System Status
    if high_percent >= 50:
        system_status = "Critical"
    elif moderate_percent >= 40:
        system_status = "Warning"
    else:
        system_status = "Stable"

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
        trend_values=trend_values,
        system_status=system_status
    )
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

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
    INSERT INTO patients (
        patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak, slope, ca, thal,
        result, probability, risk_level
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
        restecg, thalach, exang, oldpeak, slope, ca, thal,
        "Present" if prediction == 1 else "Absent",
        round(probability * 100, 2),
        risk_level
    ))

    conn.commit()
    conn.close()


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

    conn = get_db_connection()
    patients = conn.execute(
        "SELECT * FROM patients ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()

    return render_template("history.html", patients=patients)



# ------------------------------------------------------------
# Patient Detail Page (FIXED)
# ------------------------------------------------------------

@app.route("/patient/<patient_id>")
def patient_detail(patient_id):

    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    patient_row = conn.execute(
        "SELECT * FROM patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()
    conn.close()

    if not patient_row:
        return redirect(url_for("history"))

    patient = dict(patient_row)

    # ----------------------------
    # Utility functions
    # ----------------------------
    def safe_float(val):
        try:
            return float(val)
        except:
            return 0

    def check_range(value, min_val=None, max_val=None):
        value = safe_float(value)
        if min_val is not None and value < min_val:
            return "Low"
        if max_val is not None and value > max_val:
            return "High"
        return "Normal"

    # ----------------------------
    # Clinical Analysis
    # ----------------------------
    patient["Cholesterol Status"] = check_range(patient.get("chol"), max_val=200)
    patient["Resting BP Status"] = check_range(patient.get("trestbps"), min_val=90, max_val=120)
    patient["Max Heart Rate Status"] = check_range(patient.get("thalach"), min_val=60, max_val=100)
    patient["ST Depression Status"] = check_range(patient.get("oldpeak"), max_val=1)

    patient["Probability (%)"] = round(safe_float(patient.get("probability")), 2)
    patient["explanation"] = generate_explanation(patient)
    patient["recommendations"] = generate_recommendation(patient)
    patient["alerts"] = generate_alerts(patient)
    # ----------------------------
    # Reverse Mapping (Text)
    # ----------------------------
    try:
        patient["cp_text"] = REVERSE_FEATURE_MAP["cp"].get(
            patient.get("cp"),
            patient.get("cp")
        )
    except:
        patient["cp_text"] = patient.get("cp")

    return render_template("patient_detail.html", patient=patient)
    

def generate_explanation(patient):

    explanations = []

    if patient.get("chol", 0) > 200:
        explanations.append("Elevated cholesterol levels may increase cardiovascular risk.")

    if patient.get("trestbps", 0) > 130:
        explanations.append("Blood pressure is higher than normal range.")

    if patient.get("thalach", 0) < 60:
        explanations.append("Heart rate is below normal limits.")

    if patient.get("oldpeak", 0) > 1:
        explanations.append("ST depression indicates possible cardiac stress.")

    if patient.get("ca", 0) > 1:
        explanations.append("Multiple vessels show signs of blockage.")

    if not explanations:
        return "All clinical parameters are within normal range."

    return " ".join(explanations)

def generate_recommendation(patient):

    recommendations = []

    # Cholesterol
    if patient.get("chol", 0) > 200:
        recommendations.append("Reduce fatty food intake and monitor cholesterol levels.")

    # Blood Pressure
    if patient.get("trestbps", 0) > 130:
        recommendations.append("Reduce salt intake and manage blood pressure regularly.")

    # Heart Rate
    if patient.get("thalach", 0) < 60:
        recommendations.append("Monitor heart rate and consult cardiologist if symptoms persist.")

    # ST Depression
    if patient.get("oldpeak", 0) > 1:
        recommendations.append("Possible cardiac stress detected. Avoid heavy exertion.")

    # Blocked vessels
    if patient.get("ca", 0) > 1:
        recommendations.append("Blocked vessels detected. Immediate cardiology consultation recommended.")

    # FINAL fallback
    if not recommendations:
        return ["Maintain healthy lifestyle and regular checkups."]

    return recommendations

def generate_alerts(patient):

    alerts = []

    # Cholesterol
    if patient.get("chol", 0) > 200:
        alerts.append("⚠️ High Cholesterol detected")

    # Blood Pressure
    if patient.get("trestbps", 0) > 130:
        alerts.append("⚠️ Elevated Blood Pressure")

    # Heart Rate
    if patient.get("thalach", 0) < 60:
        alerts.append("⚠️ Low Heart Rate")

    # ST Depression
    if patient.get("oldpeak", 0) > 1:
        alerts.append("⚠️ Cardiac stress detected")

    # Vessels
    if patient.get("ca", 0) > 1:
        alerts.append("⚠️ Multiple blocked vessels")

    # 🔥 Summary Alert
    if len(alerts) >= 3:
        alerts.append("🚨 Multiple risk factors detected. Immediate attention required.")

    return alerts

# ------------------------------------------------------------
# Patient Detail
# ------------------------------------------------------------

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table
from reportlab.lib import colors
from flask import send_file

@app.route("/download_report/<patient_id>")
def download_report(patient_id):

    if "user" not in session:
        return redirect(url_for("login"))

    from database import get_db_connection

    conn = get_db_connection()
    patient = conn.execute(
        "SELECT * FROM patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()
    conn.close()

    if not patient:
        return redirect(url_for("history"))

    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer)

    data = [
        ["Field", "Value"],
        ["Patient ID", patient["patient_id"]],
        ["Patient Name", patient["patient_name"]],
        ["Age", str(patient["age"])],
        ["Prediction", patient["result"]],
        ["Risk Level", patient["risk_level"]],
        ["Probability (%)", str(patient["probability"])],
        ["Timestamp", patient["timestamp"]],
    ]

    table = Table(data)
    table.setStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ])

    pdf.build([table])
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{patient_id}_report.pdf",
        mimetype='application/pdf'
    )
# ------------------------------------------------------------
# Delete
# ------------------------------------------------------------

@app.route("/delete/<patient_id>")
def delete_record(patient_id):

    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    conn.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("history"))

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000/login")
    app.run(debug=True)