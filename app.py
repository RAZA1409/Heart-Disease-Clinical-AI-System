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
import secrets
from google import genai
from google.genai import types
from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from database import get_db_connection, init_db
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import timedelta
from flask import send_file
from dotenv import load_dotenv
from risk_config import classify_risk, LOW_RISK_THRESHOLD, MODERATE_RISK_THRESHOLD
# from openai import OpenAI

load_dotenv()

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
print("Gemini Key Found:", os.getenv("GEMINI_API_KEY") is not None)

# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "e83a42f9b8c6451e9f1a23847b2c9e54d68201fa3829471abefc918237465abc")

app.permanent_session_lifetime = timedelta(days=30)

def generate_csrf_token():
    if "csrf_token" not in session:
        session["csrf_token"] = secrets.token_hex(32)
    return session["csrf_token"]

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=generate_csrf_token)

@app.before_request
def csrf_protect():
    generate_csrf_token()
    if request.method in ("POST", "PUT", "DELETE", "PATCH"):
        if request.is_json:
            token = request.headers.get("X-CSRFToken") or (request.get_json(silent=True) or {}).get("csrf_token")
        else:
            token = request.form.get("csrf_token")

        if not token or token != session.get("csrf_token"):
            if request.is_json:
                return {"error": "Invalid or missing CSRF token"}, 400
            if request.endpoint == "login":
                return render_template("login.html", error="Session security token expired. Please try again."), 400
            return "Security token validation failed (CSRF protection). Please refresh the page.", 400

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
# Load Model & Evaluation Metadata Artifacts
# ------------------------------------------------------------

model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

try:
    CACHED_MODEL_INFO = joblib.load("models/model_info.pkl")
except Exception:
    CACHED_MODEL_INFO = {}

try:
    CACHED_OPT_COMPARISON = joblib.load("models/optimization_comparison.pkl")
except Exception:
    CACHED_OPT_COMPARISON = {}

# ------------------------------------------------------------
# Load Diabetes Pipeline & Model Metadata Artifacts
# ------------------------------------------------------------
try:
    diabetes_pipeline = joblib.load("models/diabetes_pipeline.pkl")
    diabetes_feature_columns = joblib.load("models/diabetes_feature_columns.pkl")
    diabetes_model_info = joblib.load("models/diabetes_model_info.pkl")
except Exception as e:
    print(f"Warning: Diabetes artifacts not fully loaded: {e}")
    diabetes_pipeline = None
    diabetes_feature_columns = []
    diabetes_model_info = {}

# ------------------------------------------------------------
# Load Breast Cancer Pipeline & Model Metadata Artifacts
# ------------------------------------------------------------
try:
    breast_cancer_pipeline = joblib.load("models/breast_cancer_pipeline.pkl")
    breast_cancer_feature_columns = joblib.load("models/breast_cancer_feature_columns.pkl")
    breast_cancer_model_info = joblib.load("models/breast_cancer_model_info.pkl")
except Exception as e:
    print(f"Warning: Breast Cancer artifacts not fully loaded: {e}")
    breast_cancer_pipeline = None
    breast_cancer_feature_columns = []
    breast_cancer_model_info = {}

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

        if not stored_code or recovery_code != stored_code:
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
    try:
        diabetes_patients = conn.execute("SELECT * FROM diabetes_patients").fetchall()
    except Exception:
        diabetes_patients = []
    try:
        breast_cancer_patients = conn.execute("SELECT * FROM breast_cancer_patients").fetchall()
    except Exception:
        breast_cancer_patients = []
    conn.close()

    total = len(patients)
    total_diabetes = len(diabetes_patients)
    total_breast_cancer = len(breast_cancer_patients)
    breast_cancer_accuracy = round(breast_cancer_model_info.get("accuracy", 0.9825) * 100, 2)

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
    model_info = CACHED_MODEL_INFO or joblib.load("models/model_info.pkl")
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
    opt = CACHED_OPT_COMPARISON or joblib.load("models/optimization_comparison.pkl")
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
        system_status=system_status,
        total_diabetes=total_diabetes,
        diabetes_model_info=diabetes_model_info,
        total_breast_cancer=total_breast_cancer,
        breast_cancer_model_info=breast_cancer_model_info,
        breast_cancer_accuracy=breast_cancer_accuracy
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

    try:
        patient_name = request.form.get("patient_name", "").strip().title() or "Unknown"

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
    except (KeyError, ValueError, TypeError) as e:
        return render_template("predict.html", error=f"Invalid clinical input: {str(e)}"), 400

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

    prob_percentage = round(probability * 100, 2)
    risk_level = classify_risk(prob_percentage)

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
    if prob >= MODERATE_RISK_THRESHOLD:
        trend = "Risk increasing"
    elif prob >= LOW_RISK_THRESHOLD:
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
    patient_row = conn.execute(
        "SELECT * FROM patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()
    conn.close()

    if not patient_row:
        return redirect(url_for("history"))

    patient = dict(patient_row)

    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    sex_text = REVERSE_FEATURE_MAP["sex"].get(patient.get("sex"), str(patient.get("sex")))
    cp_text = REVERSE_FEATURE_MAP["cp"].get(patient.get("cp"), str(patient.get("cp")))
    fbs_text = REVERSE_FEATURE_MAP["fbs"].get(patient.get("fbs"), str(patient.get("fbs")))
    exang_text = REVERSE_FEATURE_MAP["exang"].get(patient.get("exang"), str(patient.get("exang")))

    elements = []
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1e3c4f'),
        alignment=1
    )
    elements.append(Paragraph("HEART DISEASE CLINICAL AI ASSESSMENT REPORT", title_style))
    elements.append(Spacer(1, 14))

    data = [
        ["Category", "Parameter", "Clinical Value"],
        ["Demographics", "Patient ID", str(patient.get("patient_id", ""))],
        ["", "Patient Name", str(patient.get("patient_name", ""))],
        ["", "Age", str(patient.get("age", "")) + " years"],
        ["", "Sex", sex_text],
        ["Vitals & Labs", "Resting Blood Pressure", str(patient.get("trestbps", "")) + " mmHg"],
        ["", "Serum Cholesterol", str(patient.get("chol", "")) + " mg/dL"],
        ["", "Max Heart Rate Achieved", str(patient.get("thalach", "")) + " bpm"],
        ["", "Fasting Blood Sugar > 120", fbs_text],
        ["Cardiac Tests", "Chest Pain Type", cp_text],
        ["", "ST Depression", str(patient.get("oldpeak", "")) + " mm"],
        ["", "Exercise Induced Angina", exang_text],
        ["", "Major Vessels Fluro (0-3)", str(patient.get("ca", ""))],
        ["AI Diagnostic Result", "Heart Disease Prediction", str(patient.get("result", ""))],
        ["", "Risk Probability", str(patient.get("probability", "")) + "%"],
        ["", "Clinical Risk Level", str(patient.get("risk_level", ""))],
        ["", "Assessment Date & Time", str(patient.get("timestamp", ""))[:19]],
    ]

    table = Table(data, colWidths=[130, 180, 160])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3c4f')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
    ]))

    elements.append(table)
    pdf.build(elements)
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
    if "user" not in session:
        return redirect(url_for("login"))
    conn = get_db_connection()
    conn.execute("DELETE FROM patients WHERE patient_id = ?", (patient_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("history"))


# ============================================================
# DIABETES CLINICAL AI MODULE
# ============================================================

def generate_diabetes_explanation(patient):
    explanations = []
    glucose = safe_float(patient.get("glucose"))
    bmi = safe_float(patient.get("bmi"))
    bp = safe_float(patient.get("blood_pressure"))
    insulin = safe_float(patient.get("insulin"))
    dpf = safe_float(patient.get("dpf"))

    if glucose >= 200:
        explanations.append("Markedly elevated plasma glucose concentration (≥200 mg/dL) indicates high risk of overt diabetes mellitus.")
    elif glucose >= 140:
        explanations.append("Elevated 2-hour plasma glucose (140-199 mg/dL) reflects impaired glucose tolerance (prediabetic metabolic state).")

    if bmi >= 30:
        explanations.append(f"Body Mass Index of {bmi:.1f} kg/m² indicates clinical obesity, which is strongly associated with peripheral insulin resistance.")
    elif bmi >= 25:
        explanations.append(f"Body Mass Index of {bmi:.1f} kg/m² reflects an overweight classification, exerting additional metabolic workload.")

    if bp >= 90:
        explanations.append("Elevated diastolic blood pressure indicates concomitant vascular and hypertensive strain.")

    if insulin > 166:
        explanations.append("Elevated postprandial serum insulin indicates compensatory hyperinsulinemia.")

    if dpf >= 0.6:
        explanations.append(f"High Diabetes Pedigree Function score ({dpf:.2f}) indicates substantial familial genetic predisposition.")

    if not explanations:
        return "All assessed clinical metabolic parameters are currently within normal baseline limits."

    return " ".join(explanations)


def generate_diabetes_recommendation(patient):
    recommendations = []
    glucose = safe_float(patient.get("glucose"))
    bmi = safe_float(patient.get("bmi"))
    bp = safe_float(patient.get("blood_pressure"))

    if glucose >= 140:
        recommendations.append("Schedule formal diagnostic testing (Fasting Plasma Glucose & Glycated Hemoglobin HbA1c) with an endocrinologist.")
        recommendations.append("Adopt medical nutrition therapy: prioritize low glycemic index complex carbs, high fiber, and avoid refined sugars.")

    if bmi >= 25:
        recommendations.append("Target a structured 5-10% body weight reduction through 150 minutes of moderate aerobic physical activity weekly.")

    if bp >= 85:
        recommendations.append("Regularly monitor blood pressure and implement dietary sodium restriction (<2,300 mg/day).")

    if not recommendations:
        recommendations.append("Maintain a balanced nutrient-dense diet, consistent daily physical activity, and routine annual wellness checkups.")

    return recommendations


def generate_diabetes_alerts(patient):
    alerts = []
    glucose = safe_float(patient.get("glucose"))
    bmi = safe_float(patient.get("bmi"))
    bp = safe_float(patient.get("blood_pressure"))

    if glucose >= 200:
        alerts.append("🚨 Critical Blood Glucose Elevation (≥200 mg/dL)")
    elif glucose >= 140:
        alerts.append("⚠️ Impaired Glucose Tolerance (140-199 mg/dL)")

    if bmi >= 30:
        alerts.append("⚠️ Elevated BMI: Class I+ Obesity (≥30 kg/m²)")

    if bp >= 90:
        alerts.append("⚠️ Diastolic Hypertension (≥90 mmHg)")

    return alerts


@app.route("/diabetes_predict_page")
@app.route("/diabetes/predict", methods=["GET"])
def diabetes_predict_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("diabetes_predict.html")


@app.route("/diabetes/predict", methods=["POST"])
def diabetes_predict():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        patient_name = request.form.get("patient_name", "").strip().title() or "Unknown"
        age = int(float(request.form["age"]))
        pregnancies = int(float(request.form.get("pregnancies", 0)))
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])
        skin_thickness = float(request.form.get("skin_thickness", 0))
        insulin = float(request.form.get("insulin", 0))
        bmi = float(request.form["bmi"])
        dpf = float(request.form["dpf"])

        if age < 1 or age > 120:
            raise ValueError("Age must be between 1 and 120 years.")
        if pregnancies < 0 or pregnancies > 25:
            raise ValueError("Pregnancies must be between 0 and 25.")
        if glucose < 0 or glucose > 500:
            raise ValueError("Glucose must be between 0 and 500 mg/dL.")
        if blood_pressure < 0 or blood_pressure > 300:
            raise ValueError("Blood pressure must be between 0 and 300 mmHg.")
        if bmi < 5 or bmi > 100:
            raise ValueError("BMI must be between 5 and 100.")
        if dpf < 0.0 or dpf > 5.0:
            raise ValueError("DPF must be between 0.0 and 5.0.")

    except (KeyError, ValueError, TypeError) as e:
        return render_template("diabetes_predict.html", error=f"Invalid clinical input: {str(e)}"), 400

    user_df = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    probability = float(diabetes_pipeline.predict_proba(user_df)[0][1])
    prob_percentage = round(probability * 100, 2)
    prediction = "Present" if probability >= 0.5 else "Absent"
    risk_level = classify_risk(prob_percentage)

    # Collision-resistant unique Patient ID with human-readable timestamp and 8-hex random entropy
    patient_id = "DPID" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + secrets.token_hex(4).upper()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO diabetes_patients (
        patient_id, patient_name, age, pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, result, probability, risk_level
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        patient_id, patient_name, age, pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi, dpf, prediction, prob_percentage, risk_level
    ))
    conn.commit()
    conn.close()

    return render_template(
        "diabetes_result.html",
        patient_name=patient_name,
        patient_id=patient_id,
        probability=prob_percentage,
        prediction=prediction,
        risk_level=risk_level
    )


@app.route("/diabetes/history")
def diabetes_history():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    patients = conn.execute(
        "SELECT * FROM diabetes_patients ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()

    return render_template("diabetes_history.html", patients=patients)


@app.route("/diabetes/patient/<patient_id>")
def diabetes_patient_detail(patient_id):
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    patient_row = conn.execute(
        "SELECT * FROM diabetes_patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()
    conn.close()

    if not patient_row:
        return redirect(url_for("diabetes_history"))

    patient = dict(patient_row)

    glucose = safe_float(patient.get("glucose"))
    bmi = safe_float(patient.get("bmi"))
    bp = safe_float(patient.get("blood_pressure"))
    insulin = safe_float(patient.get("insulin"))
    skin = safe_float(patient.get("skin_thickness"))

    # Glucose status
    if glucose >= 200:
        patient["glucose_status"] = "Diabetic Range (High)"
        patient["glucose_status_class"] = "danger"
    elif glucose >= 140:
        patient["glucose_status"] = "Impaired / Prediabetic"
        patient["glucose_status_class"] = "warning"
    else:
        patient["glucose_status"] = "Normal Range"
        patient["glucose_status_class"] = "normal"

    # BMI status
    if bmi >= 30:
        patient["bmi_status"] = "Obese Range (High Risk)"
        patient["bmi_status_class"] = "danger"
    elif bmi >= 25:
        patient["bmi_status"] = "Overweight"
        patient["bmi_status_class"] = "warning"
    else:
        patient["bmi_status"] = "Normal Healthy Range"
        patient["bmi_status_class"] = "normal"

    # BP status
    if bp >= 90:
        patient["bp_status"] = "Hypertension"
        patient["bp_status_class"] = "danger"
    elif bp >= 80:
        patient["bp_status"] = "Prehypertension"
        patient["bp_status_class"] = "warning"
    else:
        patient["bp_status"] = "Normal Range"
        patient["bp_status_class"] = "normal"

    # Insulin status
    if insulin > 166:
        patient["insulin_status"] = "Elevated"
        patient["insulin_status_class"] = "warning"
    elif insulin > 0:
        patient["insulin_status"] = "Normal Range"
        patient["insulin_status_class"] = "normal"
    else:
        patient["insulin_status"] = "Not Recorded / Zero"
        patient["insulin_status_class"] = "normal"

    # Skin thickness
    if skin > 40:
        patient["skin_status"] = "Elevated"
        patient["skin_status_class"] = "warning"
    else:
        patient["skin_status"] = "Normal Range"
        patient["skin_status_class"] = "normal"

    # Clinical Analysis
    patient["explanation"] = generate_diabetes_explanation(patient)
    patient["recommendations"] = generate_diabetes_recommendation(patient)
    patient["alerts"] = generate_diabetes_alerts(patient)

    # Risk Intelligence
    prob = safe_float(patient.get("probability"))
    confidence = abs(prob - 50) * 2
    patient["confidence"] = round(confidence, 1)

    drivers = []
    if glucose >= 140:
        drivers.append("Elevated Plasma Glucose")
    if bmi >= 28:
        drivers.append("Elevated Body Mass Index (BMI)")
    if safe_float(patient.get("dpf")) >= 0.5:
        drivers.append("Elevated Genetic Pedigree Score")
    if bp >= 85:
        drivers.append("Elevated Blood Pressure")

    patient["key_driver"] = drivers[0] if drivers else "Metabolic parameters within typical limits"

    if prob >= MODERATE_RISK_THRESHOLD:
        patient["trend"] = "High metabolic vigilance recommended"
    elif prob >= LOW_RISK_THRESHOLD:
        patient["trend"] = "Moderate risk zone — monitor lifestyle"
    else:
        patient["trend"] = "Stable metabolic indicators"

    return render_template("diabetes_patient_detail.html", patient=patient)


@app.route("/diabetes/delete/<patient_id>", methods=["POST"])
def diabetes_delete(patient_id):
    if "user" not in session:
        return redirect(url_for("login"))
    conn = get_db_connection()
    conn.execute("DELETE FROM diabetes_patients WHERE patient_id = ?", (patient_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("diabetes_history"))


@app.route("/diabetes/download_report/<patient_id>")
def diabetes_download_report(patient_id):
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    patient_row = conn.execute(
        "SELECT * FROM diabetes_patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()
    conn.close()

    if not patient_row:
        return redirect(url_for("diabetes_history"))

    patient = dict(patient_row)

    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = []
    title_style = ParagraphStyle(
        'DiabetesReportTitle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0284c7'),
        alignment=1
    )
    elements.append(Paragraph("DIABETES CLINICAL AI ASSESSMENT REPORT", title_style))
    elements.append(Spacer(1, 8))

    disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#64748b'),
        alignment=1
    )
    elements.append(Paragraph("Clinical Decision Support Estimate • Not an Official Medical Diagnosis", disclaimer_style))
    elements.append(Spacer(1, 14))

    data = [
        ["Category", "Parameter", "Clinical Value"],
        ["Demographics", "Patient ID", str(patient.get("patient_id", ""))],
        ["", "Patient Name", str(patient.get("patient_name", ""))],
        ["", "Age", str(patient.get("age", "")) + " years"],
        ["", "Pregnancies", str(patient.get("pregnancies", "0"))],
        ["Metabolic Parameters", "Plasma Glucose (2-hr)", str(patient.get("glucose", "")) + " mg/dL"],
        ["", "Diastolic Blood Pressure", str(patient.get("blood_pressure", "")) + " mmHg"],
        ["", "Body Mass Index (BMI)", str(patient.get("bmi", "")) + " kg/m²"],
        ["", "Triceps Skin Thickness", str(patient.get("skin_thickness", "")) + " mm"],
        ["", "2-Hour Serum Insulin", str(patient.get("insulin", "")) + " mu U/ml"],
        ["", "Diabetes Pedigree Function", str(patient.get("dpf", ""))],
        ["AI Decision Support", "Diabetes Model Prediction", "Risk Indicated" if str(patient.get("result", "")) == "Present" else "Low Risk / Absent"],
        ["", "Estimated Risk Probability", str(patient.get("probability", "")) + "%"],
        ["", "Statistical Risk Tier", str(patient.get("risk_level", "")) + " (Non-Diagnostic)"],
        ["", "Assessment Date & Time", str(patient.get("timestamp", ""))[:19]],
    ]

    table = Table(data, colWidths=[130, 180, 160])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
    ]))

    elements.append(table)
    pdf.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{patient_id}_diabetes_report.pdf",
        mimetype='application/pdf'
    )


# ============================================================
# 🎗️ BREAST CANCER CYTOLOGY MODULE (WDBC)
# ============================================================

def generate_breast_cancer_explanation(patient):
    explanations = []
    mean_radius = safe_float(patient.get("mean_radius"))
    mean_concavity = safe_float(patient.get("mean_concavity"))
    mean_concave_points = safe_float(patient.get("mean_concave_points"))
    mean_area = safe_float(patient.get("mean_area"))
    worst_concave_points = safe_float(patient.get("worst_concave_points"))
    worst_area = safe_float(patient.get("worst_area"))
    prob = safe_float(patient.get("probability"))

    if prob >= 50.0:
        explanations.append(f"SVM model inference indicates a high statistical probability of cytological malignancy ({prob:.1f}%).")
    else:
        explanations.append(f"SVM model inference indicates predominantly benign cytological characteristics (estimated malignancy probability {prob:.1f}%).")

    if mean_concave_points >= 0.05 or worst_concave_points >= 0.12:
        explanations.append(f"Elevated nuclear concave points ({mean_concave_points:.4f}) denote structural membrane indentations strongly correlated with malignant cellular morphology.")

    if mean_concavity >= 0.10:
        explanations.append(f"Elevated nuclear concavity ({mean_concavity:.4f}) reflects marked contour infolding in the aspirate cell nuclei.")

    if mean_area >= 700.0 or worst_area >= 1000.0:
        explanations.append(f"Nuclear hypertrophy detected with mean area of {mean_area:.1f} µm² (worst-case {worst_area:.1f} µm²), suggesting marked cellular pleomorphism.")

    if len(explanations) <= 1:
        explanations.append("Nuclear size, contour regularity, and chromatin texture distribution align closely with typical benign breast tissue aspirates.")

    return " ".join(explanations)


def generate_breast_cancer_recommendations(patient):
    recommendations = []
    prob = safe_float(patient.get("probability"))
    result = str(patient.get("result", ""))

    if prob >= 50.0 or result == "Malignant":
        recommendations.append("Urgent referral to a multidisciplinary breast surgical oncology clinic for comprehensive evaluation.")
        recommendations.append("Perform image-guided Core Needle Biopsy (CNB) for definitive histological grading, ER/PR, and HER2 receptor status.")
        recommendations.append("Schedule diagnostic bilateral mammography with digital tomosynthesis and targeted breast ultrasonography.")
        recommendations.append("Consider contrast-enhanced breast MRI if multifocal disease or dense fibroglandular tissue is suspected.")
    elif prob >= 25.0:
        recommendations.append("Schedule short-interval diagnostic follow-up imaging (BI-RADS 3 protocol) in 3 to 6 months.")
        recommendations.append("Conduct a comprehensive clinical breast examination by a qualified breast specialist.")
        recommendations.append("Consider histological core biopsy if palpable lesion or interval progression is noted.")
    else:
        recommendations.append("Continue routine annual screening mammography as recommended for age and clinical risk profile.")
        recommendations.append("Encourage monthly breast self-awareness and report any palpable changes, skin changes, or discharge promptly.")
        recommendations.append("Maintain periodic clinical breast examination during routine annual wellness visits.")

    return recommendations


def generate_breast_cancer_alerts(patient):
    alerts = []
    prob = safe_float(patient.get("probability"))
    result = str(patient.get("result", ""))
    mean_concave_points = safe_float(patient.get("mean_concave_points"))
    mean_concavity = safe_float(patient.get("mean_concavity"))
    mean_area = safe_float(patient.get("mean_area"))
    worst_perimeter = safe_float(patient.get("worst_perimeter"))

    if prob >= 70.0 or result == "Malignant":
        alerts.append("🚨 High Cytological Suspicion for Breast Carcinoma")

    if mean_concave_points >= 0.08:
        alerts.append(f"⚠️ Marked Nuclear Concave Point Severity ({mean_concave_points:.4f})")

    if mean_concavity >= 0.15:
        alerts.append(f"⚠️ Pronounced Nuclear Infolding / Concavity ({mean_concavity:.4f})")

    if mean_area >= 900.0:
        alerts.append(f"⚠️ Marked Nuclear Area Expansion ({mean_area:.1f} µm²)")

    if worst_perimeter >= 130.0:
        alerts.append(f"⚠️ Extreme Perimeter Dysmorphism in Worst-Region Analysis ({worst_perimeter:.1f})")

    return alerts


@app.route("/breast_cancer_predict_page")
@app.route("/breast_cancer/predict", methods=["GET"])
def breast_cancer_predict_page():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("breast_cancer_predict.html")


@app.route("/breast_cancer/predict", methods=["POST"])
def breast_cancer_predict():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        patient_name = request.form.get("patient_name", "").strip().title() or "Unknown"
        age = int(float(request.form["age"]))
        if age < 1 or age > 120:
            raise ValueError("Age must be between 1 and 120 years.")

        feature_values = {}
        for col in breast_cancer_feature_columns:
            val_str = request.form.get(col)
            if val_str is None or val_str.strip() == "":
                raise ValueError(f"Missing required cytology feature: {col}")
            val = float(val_str)
            if val < 0.0:
                raise ValueError(f"Cytology feature '{col}' cannot be negative.")
            feature_values[col] = val

    except (KeyError, ValueError, TypeError) as e:
        return render_template("breast_cancer_predict.html", error=f"Invalid clinical input: {str(e)}"), 400

    user_df = pd.DataFrame([feature_values])[breast_cancer_feature_columns]

    pos_idx = list(breast_cancer_pipeline.classes_).index(1)
    prob_malignant = float(breast_cancer_pipeline.predict_proba(user_df)[0][pos_idx])
    prob_percentage = round(prob_malignant * 100, 2)
    prediction = "Malignant" if prob_malignant >= 0.5 else "Benign"
    risk_level = classify_risk(prob_percentage)

    # Collision-resistant unique Patient ID with human-readable timestamp and 8-hex random entropy
    patient_id = "BCP" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + secrets.token_hex(4).upper()

    conn = get_db_connection()
    cursor = conn.cursor()

    col_names = ["patient_id", "patient_name", "age"] + breast_cancer_feature_columns + ["result", "probability", "risk_level"]
    placeholders = ", ".join(["?"] * len(col_names))
    sql_cols = ", ".join(col_names)
    val_list = [patient_id, patient_name, age] + [feature_values[c] for c in breast_cancer_feature_columns] + [prediction, prob_percentage, risk_level]

    cursor.execute(f"INSERT INTO breast_cancer_patients ({sql_cols}) VALUES ({placeholders})", val_list)
    conn.commit()
    conn.close()

    return render_template(
        "breast_cancer_result.html",
        patient_name=patient_name,
        patient_id=patient_id,
        probability=prob_percentage,
        prediction=prediction,
        risk_level=risk_level
    )


@app.route("/breast_cancer/history")
def breast_cancer_history():
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    patients = conn.execute(
        "SELECT * FROM breast_cancer_patients ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()

    return render_template("breast_cancer_history.html", patients=patients)


@app.route("/breast_cancer/patient/<patient_id>")
def breast_cancer_patient_detail(patient_id):
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    patient_row = conn.execute(
        "SELECT * FROM breast_cancer_patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()
    conn.close()

    if not patient_row:
        return redirect(url_for("breast_cancer_history"))

    patient = dict(patient_row)

    mean_radius = safe_float(patient.get("mean_radius"))
    mean_texture = safe_float(patient.get("mean_texture"))
    mean_perimeter = safe_float(patient.get("mean_perimeter"))
    mean_area = safe_float(patient.get("mean_area"))
    mean_concavity = safe_float(patient.get("mean_concavity"))
    mean_concave_points = safe_float(patient.get("mean_concave_points"))

    # Radius
    if mean_radius >= 17.0:
        patient["radius_status"] = "Marked Enlargement"
        patient["radius_status_class"] = "danger"
    elif mean_radius >= 14.5:
        patient["radius_status"] = "Moderate Atypia"
        patient["radius_status_class"] = "warning"
    else:
        patient["radius_status"] = "Normal Range"
        patient["radius_status_class"] = "normal"

    # Texture
    if mean_texture >= 22.0:
        patient["texture_status"] = "High Heterogeneity"
        patient["texture_status_class"] = "danger"
    elif mean_texture >= 19.0:
        patient["texture_status"] = "Borderline Coarse"
        patient["texture_status_class"] = "warning"
    else:
        patient["texture_status"] = "Homogeneous"
        patient["texture_status_class"] = "normal"

    # Perimeter
    if mean_perimeter >= 110.0:
        patient["perimeter_status"] = "Enlarged Perimeter"
        patient["perimeter_status_class"] = "danger"
    elif mean_perimeter >= 95.0:
        patient["perimeter_status"] = "Atypical Expansion"
        patient["perimeter_status_class"] = "warning"
    else:
        patient["perimeter_status"] = "Normal Perimeter"
        patient["perimeter_status_class"] = "normal"

    # Area
    if mean_area >= 900.0:
        patient["area_status"] = "Marked Expansion"
        patient["area_status_class"] = "danger"
    elif mean_area >= 650.0:
        patient["area_status"] = "Moderate Expansion"
        patient["area_status_class"] = "warning"
    else:
        patient["area_status"] = "Normal Area"
        patient["area_status_class"] = "normal"

    # Concavity
    if mean_concavity >= 0.15:
        patient["concavity_status"] = "Severe Infolding"
        patient["concavity_status_class"] = "danger"
    elif mean_concavity >= 0.08:
        patient["concavity_status"] = "Moderate Indentation"
        patient["concavity_status_class"] = "warning"
    else:
        patient["concavity_status"] = "Smooth Contour"
        patient["concavity_status_class"] = "normal"

    # Concave Points
    if mean_concave_points >= 0.08:
        patient["concave_points_status"] = "High Concave Severity"
        patient["concave_points_status_class"] = "danger"
    elif mean_concave_points >= 0.04:
        patient["concave_points_status"] = "Moderate Concavities"
        patient["concave_points_status_class"] = "warning"
    else:
        patient["concave_points_status"] = "Normal Contour"
        patient["concave_points_status_class"] = "normal"

    # Explanations, Recommendations, Alerts
    patient["explanation"] = generate_breast_cancer_explanation(patient)
    patient["recommendations"] = generate_breast_cancer_recommendations(patient)
    patient["alerts"] = generate_breast_cancer_alerts(patient)

    # Risk Intelligence
    prob = safe_float(patient.get("probability"))
    confidence = abs(prob - 50.0) * 2.0
    patient["confidence"] = round(confidence, 1)

    drivers = []
    if mean_concave_points >= 0.05:
        drivers.append("Nuclear Contour Indentation (Concave Points)")
    if mean_concavity >= 0.10:
        drivers.append("Nuclear Infolding Severity (Concavity)")
    if mean_area >= 700.0:
        drivers.append("Nuclear Area Hypertrophy")
    if mean_radius >= 15.0:
        drivers.append("Nuclear Radius Pleomorphism")
    if mean_texture >= 20.0:
        drivers.append("Chromatin Texture Heterogeneity")

    patient["key_driver"] = drivers[0] if drivers else "Nuclear morphometric measurements within benign baseline"

    if prob >= MODERATE_RISK_THRESHOLD:
        patient["trend"] = "High cytopathological vigilance; urgent biopsy indicated"
    elif prob >= LOW_RISK_THRESHOLD:
        patient["trend"] = "Intermediate atypia; short-interval follow-up recommended"
    else:
        patient["trend"] = "Consistent with stable benign cytological morphometry"

    return render_template("breast_cancer_patient_detail.html", patient=patient)


@app.route("/breast_cancer/delete/<patient_id>", methods=["POST"])
def breast_cancer_delete(patient_id):
    if "user" not in session:
        return redirect(url_for("login"))
    conn = get_db_connection()
    conn.execute("DELETE FROM breast_cancer_patients WHERE patient_id = ?", (patient_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("breast_cancer_history"))


@app.route("/breast_cancer/download_report/<patient_id>")
def breast_cancer_download_report(patient_id):
    if "user" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    patient_row = conn.execute(
        "SELECT * FROM breast_cancer_patients WHERE patient_id = ?",
        (patient_id,)
    ).fetchone()
    conn.close()

    if not patient_row:
        return redirect(url_for("breast_cancer_history"))

    patient = dict(patient_row)

    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = []
    title_style = ParagraphStyle(
        'CancerReportTitle',
        parent=styles['Heading1'],
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#ec4899'),
        alignment=1
    )
    elements.append(Paragraph("BREAST CANCER CYTOLOGY AI ASSESSMENT REPORT", title_style))
    elements.append(Spacer(1, 8))

    disclaimer_style = ParagraphStyle(
        'CancerDisclaimerStyle',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#64748b'),
        alignment=1
    )
    elements.append(Paragraph("Clinical Decision Support Estimate • Not an Official Medical Diagnosis", disclaimer_style))
    elements.append(Spacer(1, 14))

    data = [
        ["Category", "Parameter / Cytology Feature", "Clinical Value"],
        ["Demographics", "Patient ID", str(patient.get("patient_id", ""))],
        ["", "Patient Name", str(patient.get("patient_name", ""))],
        ["", "Age", str(patient.get("age", "")) + " years"],
        ["Mean Morphometry", "Mean Radius", f"{safe_float(patient.get('mean_radius')):.3f}"],
        ["", "Mean Texture", f"{safe_float(patient.get('mean_texture')):.3f}"],
        ["", "Mean Perimeter", f"{safe_float(patient.get('mean_perimeter')):.2f}"],
        ["", "Mean Area", f"{safe_float(patient.get('mean_area')):.1f} µm²"],
        ["", "Mean Smoothness", f"{safe_float(patient.get('mean_smoothness')):.5f}"],
        ["", "Mean Compactness", f"{safe_float(patient.get('mean_compactness')):.5f}"],
        ["", "Mean Concavity", f"{safe_float(patient.get('mean_concavity')):.5f}"],
        ["", "Mean Concave Points", f"{safe_float(patient.get('mean_concave_points')):.5f}"],
        ["", "Mean Symmetry", f"{safe_float(patient.get('mean_symmetry')):.4f}"],
        ["Extreme (Worst)", "Worst Radius", f"{safe_float(patient.get('worst_radius')):.3f}"],
        ["", "Worst Perimeter", f"{safe_float(patient.get('worst_perimeter')):.2f}"],
        ["", "Worst Area", f"{safe_float(patient.get('worst_area')):.1f} µm²"],
        ["", "Worst Concave Points", f"{safe_float(patient.get('worst_concave_points')):.5f}"],
        ["AI Decision Support", "Cytological Classification", "Malignancy Indicated" if str(patient.get("result", "")) == "Malignant" else "Benign Cytology Indicated"],
        ["", "Estimated Malignancy Probability", f"{patient.get('probability', '')}%"],
        ["", "Statistical Risk Tier", f"{patient.get('risk_level', '')} (Non-Diagnostic)"],
        ["", "Assessment Date & Time", str(patient.get("timestamp", ""))[:19]],
    ]

    table = Table(data, colWidths=[130, 180, 160])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ec4899')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#fdf2f8')])
    ]))

    elements.append(table)
    pdf.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{patient_id}_breast_cancer_report.pdf",
        mimetype='application/pdf'
    )



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

        if probability >= MODERATE_RISK_THRESHOLD:
            trend = "Risk increasing"
        elif probability >= LOW_RISK_THRESHOLD:
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
    # Endpoint disabled for patient data security and privacy compliance
    return "Endpoint disabled for security reasons", 404




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
# Run
# ------------------------------------------------------------


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


# if __name__ == "__main__":
#     webbrowser.open("http://127.0.0.1:5000/login")
#     app.run(debug=True)
