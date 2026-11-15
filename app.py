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
import hmac
import secrets
import tempfile
import io
import logging
from google import genai
from google.genai import types
from flask import Flask, render_template, request, session, redirect, url_for, render_template_string, abort, flash
from werkzeug.security import check_password_hash, generate_password_hash
from database import get_db_connection, init_db
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle
from database import get_db_connection
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import timedelta
from flask import send_file
from dotenv import load_dotenv
# from openai import OpenAI

load_dotenv()

is_production = os.getenv("RENDER") is not None or os.getenv("FLASK_ENV") == "production"

# Secret Key Handling: Require explicit SECRET_KEY in production, allow development fallback
secret_key = os.getenv("SECRET_KEY")
if not secret_key:
    if is_production:
        raise RuntimeError(
            "FATAL: SECRET_KEY environment variable is not configured. "
            "A cryptographically secure SECRET_KEY is mandatory for production deployment on Render."
        )
    secret_key = "dev-insecure-secret-key-change-in-production"

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
print("Gemini Key Found:", os.getenv("GEMINI_API_KEY") is not None)

# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )
app = Flask(__name__)
app.secret_key = secret_key

app.permanent_session_lifetime = timedelta(days=30)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB global request payload ceiling

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

def clean_ai_response(text):

    text = text.replace("###", "")
    text = text.replace("**", "")
    text = text.replace("---", "")
    return text.strip()


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
    return redirect(url_for("dashboard"))




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

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()

        if result and check_password_hash(result[0], password):
            session["user"] = username
            if remember:
                session.permanent = True

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
    session.clear()
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
        conn = get_db_connection()
        # conn = sqlite3.connect("database.db")
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
        recovery_code = request.form["recovery_code"]
        new_password = request.form["new_password"]

        stored_code = os.getenv("ADMIN_RECOVERY_CODE")

        if recovery_code != stored_code:
            return render_template(
                "forgot_password.html",
                error="Invalid Recovery Code"
            )

        if len(new_password) < 8:
            return render_template("forgot_password.html",
                                   error="Password must be at least 8 characters long")

        conn = get_db_connection()
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

    # Load model info
    model_info = joblib.load("models/model_info.pkl")
    prf_data = model_info.get("prf_data", {})
    # ==========================
    # OVERFITTING DETECTION
    # ==========================

    overfitting_status = {}
    all_models = model_info.get("all_models", {})
    for model_name, metrics in all_models.items():
        train_acc = metrics.get("train", 0)
        test_acc = metrics.get("test", 0)
        gap = train_acc - test_acc

        if gap > 0.1:
            overfitting_status[model_name] = "Overfitting"
        else:
            overfitting_status[model_name] = "Good"

    roc_data = model_info.get("roc_data", {})
    all_models = model_info.get("all_models", {})
    feature_importance = model_info.get("feature_importance", {})
    model_labels = list(all_models.keys())
    model_values = [round(all_models[m]["test"] * 100, 2) for m in model_labels]
    train_loss_values = [round(all_models[m]["train_loss"], 3) for m in model_labels]
    test_loss_values = [round(all_models[m]["test_loss"], 3) for m in model_labels]
    opt = joblib.load("models/optimization_comparison.pkl")
    opt_labels = list(opt.keys())
    opt_values = [round(v*100,2) for v in opt.values()]
    conf_matrix = model_info.get("confusion_matrix", [[0,0],[0,0]])
    # 🔥 NEW: TRAIN accuracy
    train_values = [round(v["train"] * 100, 2) for v in all_models.values()]

    model_name = model_info["model_name"]
    model_accuracy = round(model_info["accuracy"] * 100, 2)
    return render_template("dashboard.html",
        model_name=model_name,
        model_labels=model_labels,
        model_values=model_values,
        model_accuracy=model_accuracy,
        roc_data=roc_data,
        prf_data=prf_data,
        overfitting_status=overfitting_status,
        train_values=train_values,
        train_loss_values=train_loss_values,
        test_loss_values=test_loss_values,
        conf_matrix=conf_matrix,
        total=total,
        feature_importance=feature_importance,
        low=low,
        opt_labels=opt_labels,
        opt_values=opt_values,
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


@app.route("/predict_page")
def predict_page():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("predict.html")


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
    # 🔥 RISK INTELLIGENCE ENGINE
    # ----------------------------

    # Risk Level
    risk_level = patient.get("risk_level", "UNKNOWN")

    # Confidence (based on probability)
    prob = safe_float(patient.get("probability"))
    confidence = abs(prob - 50) * 2   # scale 0–100

    # Key Driver Detection
    drivers = []

    if patient.get("chol", 0) > 200:
        drivers.append("High Cholesterol")

    if patient.get("trestbps", 0) > 130:
        drivers.append("High Blood Pressure")

    if patient.get("oldpeak", 0) > 1:
        drivers.append("Cardiac Stress")

    if patient.get("ca", 0) > 1:
        drivers.append("Blocked Vessels")

    key_driver = drivers[0] if drivers else "No major risk detected"

    # Trend logic
    if prob > 70:
        trend = "Risk increasing"
    elif prob > 40:
        trend = "Moderate risk zone"
    else:
        trend = "Stable condition"

    # Attach to patient dict
    patient["confidence"] = round(confidence, 2)
    patient["key_driver"] = key_driver
    patient["trend"] = trend
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

@app.route("/delete/<patient_id>", methods=["POST"])
def delete(patient_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("history"))



# ------------------------------------------------------------
# 🤖 AI CLINICAL ASSISTANT
# ------------------------------------------------------------

@app.route("/ai_assistant", methods=["POST"])
def ai_assistant():

    # --------------------------------------------------------
    # Check login
    # --------------------------------------------------------

    if "user" not in session:
        return {"error": "Unauthorized"}, 401

    # --------------------------------------------------------
    # Get request data
    # --------------------------------------------------------

    data = request.get_json() or {}

    user_message = data.get("message", "").strip()
    patient_id = data.get("patient_id", "").strip()

    if not user_message:
        return {"error": "Message is required"}, 400

    if not patient_id:
        return {"error": "Patient information is missing"}, 400

    try:

        # ----------------------------------------------------
        # Get patient from database
        # ----------------------------------------------------

        conn = get_db_connection()

        patient_row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?",
            (patient_id,)
        ).fetchone()

        conn.close()

        if not patient_row:
            return {"error": "Patient not found"}, 404

        patient = dict(patient_row)

        # ----------------------------------------------------
        # Convert encoded values into human-readable text
        # ----------------------------------------------------

        sex_text = REVERSE_FEATURE_MAP["sex"].get(
            patient.get("sex"),
            str(patient.get("sex"))
        )

        cp_text = REVERSE_FEATURE_MAP["cp"].get(
            patient.get("cp"),
            str(patient.get("cp"))
        )

        fbs_text = REVERSE_FEATURE_MAP["fbs"].get(
            patient.get("fbs"),
            str(patient.get("fbs"))
        )

        restecg_text = REVERSE_FEATURE_MAP["restecg"].get(
            patient.get("restecg"),
            str(patient.get("restecg"))
        )

        exang_text = REVERSE_FEATURE_MAP["exang"].get(
            patient.get("exang"),
            str(patient.get("exang"))
        )

        slope_text = REVERSE_FEATURE_MAP["slope"].get(
            patient.get("slope"),
            str(patient.get("slope"))
        )

        thal_text = REVERSE_FEATURE_MAP["thal"].get(
            patient.get("thal"),
            str(patient.get("thal"))
        )

        # ----------------------------------------------------
        # Generate the same clinical analysis used by
        # the Patient Detail page
        # ----------------------------------------------------

        patient["explanation"] = generate_explanation(patient)

        patient["recommendations"] = generate_recommendation(patient)

        patient["alerts"] = generate_alerts(patient)

        # ----------------------------------------------------
        # Risk Intelligence
        # ----------------------------------------------------

        probability = safe_float(patient.get("probability"))

        confidence = abs(probability - 50) * 2

        drivers = []

        if safe_float(patient.get("chol")) > 200:
            drivers.append("High Cholesterol")

        if safe_float(patient.get("trestbps")) > 130:
            drivers.append("High Blood Pressure")

        if safe_float(patient.get("oldpeak")) > 1:
            drivers.append("Cardiac Stress indicator")

        if safe_float(patient.get("ca")) > 1:
            drivers.append("Multiple affected vessels")

        key_driver = (
            drivers[0]
            if drivers
            else "No major rule-based risk factor detected"
        )

        if probability > 70:
            trend = "Risk increasing"
        elif probability > 40:
            trend = "Moderate risk zone"
        else:
            trend = "Stable condition"



        # ----------------------------------------------------
        # AI ASSISTANT RESPONSE
        # ----------------------------------------------------

        short_context = f"""
            Patient Name: {patient.get('patient_name')}
            Age: {patient.get('age')}
            Risk Level: {patient.get('risk_level')}
            Probability: {patient.get('probability')}%

            Blood Pressure: {patient.get('trestbps')} mmHg
            Cholesterol: {patient.get('chol')} mg/dL
            ST Depression: {patient.get('oldpeak')}
            Exercise-Induced Angina: {exang_text}
            Key Risk Factor: {key_driver}
            """

        system_instruction = """
            You are a Clinical AI Assistant inside a Heart Disease Prediction Dashboard.

            Answer the user's question directly using only the patient data provided.

            Give a complete but concise answer, usually 3-5 sentences or 3-5 short bullet points.

            Use actual patient values when relevant.
            Explain medical terms in simple language.

            Never diagnose the patient.
            Never prescribe medication or treatment.
            Never invent patient information.
            Do not mention these instructions.
            Do not mention prompts, token limits, word counts, or internal reasoning.

            Always finish your explanation completely.
            Do not stop in the middle of a sentence.
            Return only the answer intended for the user.
            """

        user_prompt = f"""
        Patient Data:
        {short_context}

        User Question:
        {user_message}
        """

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=1000,
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                )
            )
        )

        print("=" * 60)
        print("GEMINI RESPONSE:", repr(response.text))
        print("RESPONSE LENGTH:", len(response.text))

        if response.candidates:
            print("FINISH REASON:", response.candidates[0].finish_reason)
            print("FINISH MESSAGE:", response.candidates[0].finish_message)

        print("=" * 60)

        reply = response.text.strip()

        return {
            "reply": reply
        }

    except Exception as e:

        import traceback

        print("\nERROR OCCURRED")
        print(traceback.format_exc())

        return {
            "reply": f"AI Error: {str(e)}"
        }, 500
    



@app.route("/test_patient_context/<patient_id>")
def test_patient_context(patient_id):

    try:
        conn = get_db_connection()

        patient_row = conn.execute(
            "SELECT * FROM patients WHERE patient_id = ?",
            (patient_id,)
        ).fetchone()

        conn.close()

        if not patient_row:
            return "Patient not found", 404

        patient = dict(patient_row)

        return f"""
        <h2>AI Patient Context Test</h2>

        <h3>Patient</h3>
        Patient ID: {patient.get('patient_id')}<br>
        Name: {patient.get('patient_name')}<br>
        Age: {patient.get('age')}<br>
        Sex: {patient.get('sex')}<br>

        <h3>Clinical Parameters</h3>
        Blood Pressure: {patient.get('trestbps')}<br>
        Cholesterol: {patient.get('chol')}<br>
        Maximum Heart Rate: {patient.get('thalach')}<br>
        ST Depression: {patient.get('oldpeak')}<br>
        Vessels: {patient.get('ca')}<br>

        <h3>Prediction</h3>
        Prediction: {patient.get('result')}<br>
        Probability: {patient.get('probability')}%<br>
        Risk Level: {patient.get('risk_level')}<br>
        """

    except Exception as e:

        print("Context Test Error:", e)

        return f"Error: {str(e)}", 500




# @app.route("/test_ai")
# def test_ai():

#     try:

#         response = client.responses.create(
#             model="gpt-5-mini",
#             input="Say hello and confirm that the Clinical AI Assistant is connected."
#         )

#         return response.output_text

#     except Exception as e:

#         return f"AI Error: {str(e)}"


# ------------------------------------------------------------
# TEMPORARY: Secure Database Restoration Endpoint (Production Hotfix)
# ------------------------------------------------------------

RESTORE_FORM_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Production Database Restoration | Clinical AI</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #0f172a;
            color: #f8fafc;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 1rem;
            box-sizing: border-box;
        }
        .restore-card {
            background: #1e293b;
            padding: 2.25rem;
            border-radius: 0.75rem;
            border: 1px solid #334155;
            max-width: 480px;
            width: 100%;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
        }
        .header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1rem;
        }
        .header-icon {
            font-size: 1.75rem;
        }
        h2 {
            margin: 0;
            color: #38bdf8;
            font-size: 1.35rem;
            font-weight: 700;
        }
        p {
            color: #94a3b8;
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 0 0 1.5rem 0;
        }
        .warning-box {
            background: #0f172a;
            border-left: 3px solid #f59e0b;
            padding: 0.75rem 1rem;
            border-radius: 0.375rem;
            font-size: 0.825rem;
            color: #fbbf24;
            margin-bottom: 1.5rem;
            line-height: 1.4;
        }
        .error-box {
            background: rgba(239, 68, 68, 0.1);
            border-left: 3px solid #ef4444;
            padding: 0.75rem 1rem;
            border-radius: 0.375rem;
            font-size: 0.825rem;
            color: #f87171;
            margin-bottom: 1.5rem;
        }
        .form-group {
            margin-bottom: 1.25rem;
        }
        label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            font-size: 0.85rem;
            color: #e2e8f0;
        }
        input[type="file"] {
            width: 100%;
            padding: 0.65rem 0.75rem;
            border-radius: 0.375rem;
            border: 1px dashed #475569;
            background: #0f172a;
            color: #94a3b8;
            font-size: 0.85rem;
            box-sizing: border-box;
            cursor: pointer;
        }
        input[type="password"] {
            width: 100%;
            padding: 0.75rem 1rem;
            border-radius: 0.375rem;
            border: 1px solid #475569;
            background: #0f172a;
            color: #f8fafc;
            font-size: 0.95rem;
            box-sizing: border-box;
            transition: border-color 0.15s ease-in-out;
        }
        input[type="password"]:focus {
            outline: none;
            border-color: #38bdf8;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15);
        }
        .btn-submit {
            display: block;
            width: 100%;
            margin-top: 1.5rem;
            padding: 0.85rem 1rem;
            background: #0284c7;
            color: white;
            border: none;
            border-radius: 0.375rem;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s ease-in-out;
            text-align: center;
        }
        .btn-submit:hover {
            background: #0369a1;
        }
        .cancel-link {
            display: block;
            text-align: center;
            margin-top: 1.25rem;
            color: #64748b;
            text-decoration: none;
            font-size: 0.85rem;
            transition: color 0.15s ease-in-out;
        }
        .cancel-link:hover {
            color: #94a3b8;
        }
    </style>
</head>
<body>
    <div class="restore-card">
        <div class="header">
            <span class="header-icon">&#128260;</span>
            <h2>Database Restoration</h2>
        </div>
        <p>Restore the verified production database snapshot. Active database will be replaced atomically following deep integrity verification.</p>
        <div class="warning-box">
            Restricted to verified administrators. Confirm your administrator password and upload the SQLite backup file (.sqlite).
        </div>
        {% if error %}
        <div class="error-box">
            {{ error }}
        </div>
        {% endif %}
        <form method="POST" enctype="multipart/form-data">
            <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
            <div class="form-group">
                <label for="backup_file">Select Database Backup (.sqlite):</label>
                <input type="file" id="backup_file" name="backup_file" accept=".sqlite,.db" required>
            </div>
            <div class="form-group">
                <label for="admin_password">Confirm Administrator Password:</label>
                <input type="password" id="admin_password" name="admin_password" required autocomplete="current-password" placeholder="Enter administrator password">
            </div>
            <button type="submit" class="btn-submit">Verify &amp; Restore Database</button>
        </form>
        <a href="{{ url_for('dashboard') }}" class="cancel-link">&larr; Return to Dashboard</a>
    </div>
</body>
</html>"""

MAX_RESTORE_SIZE = 25 * 1024 * 1024  # 25 MB strict upload ceiling


def _perform_atomic_rollback(snapshot_path, target_path):
    """
    Safely rolls back the active database from the pre-restore snapshot.
    Returns (success: bool, error_msg: str or None).
    Verifies destination integrity after restoration.
    Logs errors securely without exposing patient records or passwords.
    """
    if not snapshot_path or not os.path.exists(snapshot_path):
        return False, "Pre-restoration snapshot file is missing"

    rb_source = None
    rb_dest = None
    try:
        rb_source = sqlite3.connect(snapshot_path)
        rb_dest = sqlite3.connect(target_path, timeout=30.0)
        rb_dest.execute("PRAGMA busy_timeout = 30000")
        rb_source.backup(rb_dest)

        # Verify integrity of rolled-back active database
        chk = rb_dest.execute("PRAGMA integrity_check").fetchall()
        if not chk or chk[0][0] != "ok":
            logging.getLogger("clinical_ai.restore").error("CRITICAL: Rollback destination failed integrity check.")
            return False, "Rollback destination failed integrity check"

        return True, None
    except Exception as rb_err:
        logging.getLogger("clinical_ai.restore").error(
            f"CRITICAL: Active database rollback failed with error: {type(rb_err).__name__}"
        )
        return False, f"Rollback failed with error: {type(rb_err).__name__}"
    finally:
        if rb_source:
            try:
                rb_source.close()
            except Exception:
                pass
        if rb_dest:
            try:
                rb_dest.close()
            except Exception:
                pass


@app.route("/admin/restore-database", methods=["GET", "POST"])
def admin_restore_database():
    # 1. Require session["user"] == "admin"
    if "user" not in session or session.get("user") != "admin":
        if request.method == "POST":
            abort(403)
        return redirect(url_for("login"))

    # 2. GET: Display form with fresh session CSRF token
    if request.method == "GET":
        token = secrets.token_hex(32)
        session["restore_csrf_token"] = token
        return render_template_string(RESTORE_FORM_HTML, csrf_token=token, error=None)

    # 3. POST Validation: CSRF Token Check using hmac.compare_digest()
    submitted_token = request.form.get("csrf_token", "")
    session_token = session.get("restore_csrf_token", "")
    session.pop("restore_csrf_token", None)  # Single-use: invalidate immediately

    if not session_token or not submitted_token or not hmac.compare_digest(submitted_token, session_token):
        return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Session security token expired or invalid. Please retry."), 400

    # 4. POST Validation: Admin Password Confirmation
    submitted_password = request.form.get("admin_password", "")
    if not submitted_password:
        return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Administrator password confirmation required."), 400

    conn = get_db_connection()
    try:
        user_row = conn.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()
        if not user_row or not check_password_hash(user_row["password"], submitted_password):
            return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Administrator password confirmation failed."), 403
    finally:
        conn.close()

    # 5. File Upload Inspection & Streaming Size Enforcement (25 MB limit)
    uploaded_file = request.files.get("backup_file")
    if not uploaded_file or not uploaded_file.filename:
        for upload in request.files.values():
            try:
                upload.close()
            except Exception:
                pass
        return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="No database backup file selected."), 400

    staging_path = None
    active_snapshot_path = None
    try:
        # Stream into temporary staging file outside repository
        temp_dir = tempfile.gettempdir()
        fd, staging_path = tempfile.mkstemp(prefix="restore_stage_", suffix=".sqlite", dir=temp_dir)
        total_bytes = 0
        with os.fdopen(fd, "wb") as f_out:
            while True:
                chunk = uploaded_file.stream.read(65536)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_RESTORE_SIZE:
                    return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Uploaded file exceeds maximum allowed size (25 MB)."), 413
                f_out.write(chunk)

        if total_bytes < 100:
            return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Uploaded file is empty or too small to be a valid SQLite database."), 400

        # Verify SQLite Magic Header (16 bytes)
        with open(staging_path, "rb") as f_check:
            header = f_check.read(16)
        if header != b"SQLite format 3\x00":
            return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Invalid file format: Not a recognized SQLite database."), 400

        # 6. Deep Schema & Integrity Verification (Read-Only)
        clean_staging = staging_path.replace("\\", "/")
        try:
            staging_conn = sqlite3.connect(f"file:{clean_staging}?mode=ro", uri=True)
            try:
                staging_conn.row_factory = sqlite3.Row
                # Integrity checks
                res = staging_conn.execute("PRAGMA integrity_check").fetchall()
                if not res or res[0][0] != "ok":
                    return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Database verification failed: SQLite integrity check failed."), 400

                qres = staging_conn.execute("PRAGMA quick_check").fetchall()
                if not qres or qres[0][0] != "ok":
                    return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Database verification failed: SQLite quick check failed."), 400

                # Table existence
                tables = [r[0] for r in staging_conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                if "users" not in tables or "patients" not in tables:
                    return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Database verification failed: Required tables (users, patients) are missing."), 400

                # Users table schema
                user_cols = {r["name"].lower() for r in staging_conn.execute("PRAGMA table_info(users)").fetchall()}
                if not {"id", "username", "password"}.issubset(user_cols):
                    return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Database verification failed: users table schema is invalid."), 400

                # Admin account check
                admin_user = staging_conn.execute("SELECT username, password FROM users WHERE username = 'admin'").fetchone()
                if not admin_user or not admin_user["password"]:
                    return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Database verification failed: Administrator account not found in uploaded backup."), 400

                # Patients table schema
                patient_cols = {r["name"].lower() for r in staging_conn.execute("PRAGMA table_info(patients)").fetchall()}
                required_patient_cols = {
                    "id", "patient_id", "patient_name", "age", "sex", "cp", "trestbps", "chol", "fbs",
                    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "result",
                    "probability", "risk_level", "timestamp"
                }
                if not required_patient_cols.issubset(patient_cols):
                    return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Database verification failed: patients table is missing required clinical columns."), 400

                # At least 1 patient record
                patient_count = staging_conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
                if patient_count < 1:
                    return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Database verification failed: Uploaded backup contains no patient records."), 400
            finally:
                staging_conn.close()
        except sqlite3.DatabaseError:
            return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error="Database verification failed: SQLite database image is malformed or corrupted."), 400

        # 7. Atomic Pre-Restore Snapshot (Rollback Mechanism)
        import database
        target_db_path = getattr(database, "DATABASE_PATH", None) or os.path.join(database.BASE_DIR, "database.db")

        if os.path.exists(target_db_path):
            active_conn = sqlite3.connect(target_db_path, timeout=30.0)
            try:
                active_conn.execute("PRAGMA busy_timeout = 30000")
                fd_snap, active_snapshot_path = tempfile.mkstemp(prefix="active_snap_", suffix=".sqlite", dir=temp_dir)
                os.close(fd_snap)
                snap_conn = sqlite3.connect(active_snapshot_path)
                try:
                    active_conn.backup(snap_conn)
                finally:
                    snap_conn.close()
            finally:
                active_conn.close()

        # 8. Atomic Database Restoration via Connection.backup()
        source_conn = sqlite3.connect(f"file:{clean_staging}?mode=ro", uri=True)
        dest_conn = sqlite3.connect(target_db_path, timeout=30.0)
        try:
            dest_conn.execute("PRAGMA busy_timeout = 30000")
            source_conn.backup(dest_conn)
        finally:
            source_conn.close()
            dest_conn.close()

        # 9. Post-Restoration Additive Schema Verification
        try:
            init_db()
        except Exception as post_err:
            logging.getLogger("clinical_ai.restore").error(
                f"Post-restoration initialization failed: {type(post_err).__name__}. Initiating rollback."
            )
            rb_ok, _ = _perform_atomic_rollback(active_snapshot_path, target_db_path)
            if rb_ok:
                error_msg = "Database restoration failed during post-restore initialization. Active database was successfully rolled back and preserved."
            else:
                error_msg = "CRITICAL: Database restoration failed and active database rollback failed. Please contact system administrator immediately."
            return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error=error_msg), 500

        flash("Database restored successfully.")
        return redirect(url_for("dashboard"))

    except Exception as e:
        # If any unexpected exception occurs, execute verified rollback to active snapshot
        logging.getLogger("clinical_ai.restore").error(
            f"Restoration exception encountered: {type(e).__name__}. Initiating rollback."
        )
        rollback_attempted = False
        rollback_success = False

        if active_snapshot_path and os.path.exists(active_snapshot_path):
            rollback_attempted = True
            import database
            target_db_path = getattr(database, "DATABASE_PATH", None) or os.path.join(database.BASE_DIR, "database.db")
            rollback_success, _ = _perform_atomic_rollback(active_snapshot_path, target_db_path)

        if rollback_attempted:
            if rollback_success:
                error_msg = "Database restoration failed. Active database was successfully rolled back and preserved."
            else:
                error_msg = "CRITICAL: Database restoration failed and active database rollback failed. Please contact system administrator immediately."
        else:
            error_msg = "Database restoration failed before database modification. Active database was preserved."

        return render_template_string(RESTORE_FORM_HTML, csrf_token=secrets.token_hex(32), error=error_msg), 500

    finally:
        # Close uploaded file streams to release underlying FileIO temporary files
        for upload in request.files.values():
            try:
                upload.close()
            except Exception:
                pass
        # Clean up all temporary files outside repository
        if staging_path and os.path.exists(staging_path):
            try:
                os.remove(staging_path)
            except Exception:
                pass
        if active_snapshot_path and os.path.exists(active_snapshot_path):
            try:
                os.remove(active_snapshot_path)
            except Exception:
                pass


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    

# if __name__ == "__main__":
#     webbrowser.open("http://127.0.0.1:5000/login")
#     app.run(debug=True)