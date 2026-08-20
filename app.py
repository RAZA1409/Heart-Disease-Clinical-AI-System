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
from database import get_db_connection
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from datetime import timedelta
from flask import send_file
from dotenv import load_dotenv
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
app.secret_key = "supersecretkey"

app.permanent_session_lifetime = timedelta(days=30)

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
# Run
# ------------------------------------------------------------


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    

# if __name__ == "__main__":
#     webbrowser.open("http://127.0.0.1:5000/login")
#     app.run(debug=True)