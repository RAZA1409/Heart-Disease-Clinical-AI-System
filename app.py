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
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
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

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None
print("Gemini Key Found:", gemini_client is not None)

# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY")
# )
app = Flask(__name__)
app.secret_key = secret_key

app.permanent_session_lifetime = timedelta(days=30)

# Secure Cookie Settings for Production HTTPS
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

if is_production:
    # In production, cookies MUST be secure over HTTPS; cannot be overridden by environment variable
    app.config["SESSION_COOKIE_SECURE"] = True
else:
    # In local development, allow configurable cookie security (defaulting to False for local HTTP)
    app.config["SESSION_COOKIE_SECURE"] = os.getenv("SESSION_COOKIE_SECURE", "false").lower() in ("true", "1")

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
# Required Production Model Artifacts
# ------------------------------------------------------------

def _load_required_artifact(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FATAL: Required model artifact for {description} was not found at '{path}'. "
            f"Please verify all model artifacts are committed and deployed."
        )
    return joblib.load(path)

# Heart Disease Model Artifacts
model = _load_required_artifact("models/best_model.pkl", "Heart Disease Model")
scaler = _load_required_artifact("models/scaler.pkl", "Heart Disease Scaler")
feature_columns = _load_required_artifact("models/feature_columns.pkl", "Heart Disease Feature Columns")

try:
    CACHED_MODEL_INFO = joblib.load("models/model_info.pkl")
except Exception:
    CACHED_MODEL_INFO = {}

try:
    CACHED_OPT_COMPARISON = joblib.load("models/optimization_comparison.pkl")
except Exception:
    CACHED_OPT_COMPARISON = {}

# Diabetes Model Artifacts
diabetes_pipeline = _load_required_artifact("models/diabetes_pipeline.pkl", "Diabetes Pipeline")
diabetes_feature_columns = _load_required_artifact("models/diabetes_feature_columns.pkl", "Diabetes Feature Columns")
diabetes_model_info = _load_required_artifact("models/diabetes_model_info.pkl", "Diabetes Model Metadata")

# Breast Cancer Model Artifacts
breast_cancer_pipeline = _load_required_artifact("models/breast_cancer_pipeline.pkl", "Breast Cancer Pipeline")
breast_cancer_feature_columns = _load_required_artifact("models/breast_cancer_feature_columns.pkl", "Breast Cancer Feature Columns")
breast_cancer_model_info = _load_required_artifact("models/breast_cancer_model_info.pkl", "Breast Cancer Model Metadata")

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
                response.set_cookie(
                    "remember_user",
                    username,
                    max_age=60*60*24*30,
                    httponly=True,
                    samesite="Lax",
                    secure=app.config["SESSION_COOKIE_SECURE"]
                )
            else:
                response.set_cookie(
                    "remember_user",
                    "",
                    expires=0,
                    httponly=True,
                    samesite="Lax",
                    secure=app.config["SESSION_COOKIE_SECURE"]
                )

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
# Dashboard (Multi-Disease Platform Aggregation)
# ------------------------------------------------------------
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        if request.args.get("format") == "json" or request.is_json:
            return {"error": "Unauthorized"}, 401
        return redirect(url_for("login"))

    conn = get_db_connection()
    try:
        patients = conn.execute("SELECT * FROM patients").fetchall()
    except Exception:
        patients = []

    try:
        diabetes_patients = conn.execute("SELECT * FROM diabetes_patients").fetchall()
    except Exception:
        diabetes_patients = []

    try:
        breast_cancer_patients = conn.execute("SELECT * FROM breast_cancer_patients").fetchall()
    except Exception:
        breast_cancer_patients = []
    conn.close()

    # --------------------------------------------------------
    # 1. Multi-Disease Platform-Wide Patient Counts
    # --------------------------------------------------------
    count_heart = len(patients)
    count_diabetes = len(diabetes_patients)
    count_cancer = len(breast_cancer_patients)
    total_platform_patients = count_heart + count_diabetes + count_cancer

    platform_counts = {
        "heart": count_heart,
        "diabetes": count_diabetes,
        "breast_cancer": count_cancer
    }

    platform_percentages = {
        "heart": round((count_heart / total_platform_patients) * 100, 1) if total_platform_patients > 0 else 0.0,
        "diabetes": round((count_diabetes / total_platform_patients) * 100, 1) if total_platform_patients > 0 else 0.0,
        "breast_cancer": round((count_cancer / total_platform_patients) * 100, 1) if total_platform_patients > 0 else 0.0
    }

    # --------------------------------------------------------
    # 2. Disease-Specific & Platform-Wide Risk Distributions
    # --------------------------------------------------------
    def _extract_risk_counts(patient_rows):
        res = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
        for row in patient_rows:
            lvl = str(row["risk_level"] or "").strip().upper()
            if lvl in res:
                res[lvl] += 1
        return res

    heart_risk = _extract_risk_counts(patients)
    diabetes_risk = _extract_risk_counts(diabetes_patients)
    cancer_risk = _extract_risk_counts(breast_cancer_patients)

    disease_risk_distributions = {
        "heart": heart_risk,
        "diabetes": diabetes_risk,
        "breast_cancer": cancer_risk
    }

    platform_low = heart_risk["LOW"] + diabetes_risk["LOW"] + cancer_risk["LOW"]
    platform_moderate = heart_risk["MODERATE"] + diabetes_risk["MODERATE"] + cancer_risk["MODERATE"]
    platform_high = heart_risk["HIGH"] + diabetes_risk["HIGH"] + cancer_risk["HIGH"]

    platform_risk = {
        "LOW": platform_low,
        "MODERATE": platform_moderate,
        "HIGH": platform_high
    }

    platform_low_percent = round((platform_low / total_platform_patients) * 100, 1) if total_platform_patients > 0 else 0.0
    platform_moderate_percent = round((platform_moderate / total_platform_patients) * 100, 1) if total_platform_patients > 0 else 0.0
    platform_high_percent = round((platform_high / total_platform_patients) * 100, 1) if total_platform_patients > 0 else 0.0

    platform_risk_percentages = {
        "LOW": platform_low_percent,
        "MODERATE": platform_moderate_percent,
        "HIGH": platform_high_percent
    }

    platform_high_proportion = platform_high_percent

    if platform_high_percent >= 50.0:
        platform_status = "Critical"
    elif platform_moderate_percent >= 40.0:
        platform_status = "Warning"
    else:
        platform_status = "Stable"

    # --------------------------------------------------------
    # 3. Platform-Wide Average Probability (0-100% scale)
    # --------------------------------------------------------
    all_probabilities = []
    for rows in [patients, diabetes_patients, breast_cancer_patients]:
        for r in rows:
            val = r["probability"]
            if val is not None:
                try:
                    all_probabilities.append(float(val))
                except (ValueError, TypeError):
                    pass

    platform_avg_probability = round(sum(all_probabilities) / len(all_probabilities), 2) if all_probabilities else None

    # --------------------------------------------------------
    # 4. Multi-Disease Recent Assessments Stream (Up to 8 items)
    # --------------------------------------------------------
    recent_assessments = []

    for p in patients:
        recent_assessments.append({
            "patient_id": p["patient_id"],
            "patient_name": p["patient_name"],
            "disease_type": "heart",
            "disease_name": "Cardiovascular",
            "disease_icon": "🫀",
            "risk_level": str(p["risk_level"] or "UNKNOWN").upper(),
            "probability": safe_float(p["probability"]),
            "result": p["result"],
            "timestamp": str(p["timestamp"] or ""),
            "detail_url": url_for("patient_detail", patient_id=p["patient_id"])
        })

    for p in diabetes_patients:
        recent_assessments.append({
            "patient_id": p["patient_id"],
            "patient_name": p["patient_name"],
            "disease_type": "diabetes",
            "disease_name": "Diabetes Mellitus",
            "disease_icon": "🩸",
            "risk_level": str(p["risk_level"] or "UNKNOWN").upper(),
            "probability": safe_float(p["probability"]),
            "result": p["result"],
            "timestamp": str(p["timestamp"] or ""),
            "detail_url": url_for("diabetes_patient_detail", patient_id=p["patient_id"])
        })

    for p in breast_cancer_patients:
        recent_assessments.append({
            "patient_id": p["patient_id"],
            "patient_name": p["patient_name"],
            "disease_type": "breast_cancer",
            "disease_name": "Breast Cytology",
            "disease_icon": "🎗️",
            "risk_level": str(p["risk_level"] or "UNKNOWN").upper(),
            "probability": safe_float(p["probability"]),
            "result": p["result"],
            "timestamp": str(p["timestamp"] or ""),
            "detail_url": url_for("breast_cancer_patient_detail", patient_id=p["patient_id"])
        })

    recent_assessments.sort(key=lambda x: x["timestamp"], reverse=True)
    recent_assessments = recent_assessments[:8]

    # --------------------------------------------------------
    # 5. Legacy Heart-Specific Variables (Preserved for dashboard.html)
    # --------------------------------------------------------
    total = count_heart
    total_diabetes = count_diabetes
    total_breast_cancer = count_cancer
    breast_cancer_accuracy = round(breast_cancer_model_info.get("accuracy", 0.9825) * 100, 2)

    low = heart_risk["LOW"]
    moderate = heart_risk["MODERATE"]
    high = heart_risk["HIGH"]

    low_percent = round((low / total) * 100, 1) if total else 0
    moderate_percent = round((moderate / total) * 100, 1) if total else 0
    high_percent = round((high / total) * 100, 1) if total else 0

    recent = patients[::-1][:5] if patients else []
    avg_probability = round(
        sum([safe_float(p["probability"]) for p in patients]) / total, 2
    ) if total else 0

    high_patients = [p for p in patients if str(p["risk_level"] or "").upper() == "HIGH"]
    top_patient = high_patients[-1]["patient_name"] if high_patients else "None"
    last_timestamp = patients[-1]["timestamp"] if patients else "N/A"

    trend_data = patients[-10:]
    trend_labels = [p["timestamp"] for p in trend_data]
    trend_values = [safe_float(p["probability"]) for p in trend_data]

    if high_percent >= 50:
        system_status = "Critical"
    elif moderate_percent >= 40:
        system_status = "Warning"
    else:
        system_status = "Stable"

    # Heart Model Metadata
    model_info = CACHED_MODEL_INFO or joblib.load("models/model_info.pkl")
    prf_data = model_info.get("prf_data", {})
    all_models = model_info.get("all_models", {})

    overfitting_status = {}
    for m_name, metrics in all_models.items():
        train_acc = metrics.get("train", 0)
        test_acc = metrics.get("test", 0)
        gap = train_acc - test_acc
        if gap > 0.1:
            overfitting_status[m_name] = "Overfitting"
        else:
            overfitting_status[m_name] = "Good"

    roc_data = model_info.get("roc_data", {})
    feature_importance = model_info.get("feature_importance", {})
    model_labels = list(all_models.keys())
    model_values = [round(all_models[m]["test"] * 100, 2) for m in model_labels]
    train_loss_values = [round(all_models[m]["train_loss"], 3) for m in model_labels]
    test_loss_values = [round(all_models[m]["test_loss"], 3) for m in model_labels]
    opt = CACHED_OPT_COMPARISON or joblib.load("models/optimization_comparison.pkl")
    opt_labels = list(opt.keys())
    opt_values = [round(v * 100, 2) for v in opt.values()]
    conf_matrix = model_info.get("confusion_matrix", [[0, 0], [0, 0]])
    train_values = [round(v["train"] * 100, 2) for v in all_models.values()]

    model_name = model_info.get("model_name", "Logistic Regression")
    model_accuracy = round(model_info.get("accuracy", 0.9074) * 100, 2)

    # Return JSON if requested
    if request.args.get("format") == "json" or (request.is_json and request.method == "GET"):
        return jsonify({
            "total_platform_patients": total_platform_patients,
            "platform_counts": platform_counts,
            "platform_percentages": platform_percentages,
            "platform_risk": platform_risk,
            "platform_risk_percentages": platform_risk_percentages,
            "disease_risk_distributions": disease_risk_distributions,
            "platform_high_proportion": platform_high_proportion,
            "platform_avg_probability": platform_avg_probability,
            "platform_status": platform_status,
            "recent_assessments": recent_assessments
        })

    return render_template("dashboard.html",
        # New Platform-Wide Aggregations
        total_platform_patients=total_platform_patients,
        platform_counts=platform_counts,
        platform_percentages=platform_percentages,
        platform_risk=platform_risk,
        platform_risk_percentages=platform_risk_percentages,
        disease_risk_distributions=disease_risk_distributions,
        platform_high_proportion=platform_high_proportion,
        platform_avg_probability=platform_avg_probability,
        platform_status=platform_status,
        recent_assessments=recent_assessments,
        # Preserved Legacy Heart Variables
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


# ============================================================
# Disease Analytics Routes
# ============================================================

@app.route("/analytics/heart")
def heart_analytics():
    if "user" not in session:
        if request.args.get("format") == "json" or request.is_json:
            return {"error": "Unauthorized"}, 401
        return redirect(url_for("login"))

    conn = get_db_connection()
    try:
        rows = conn.execute("SELECT * FROM patients ORDER BY timestamp DESC").fetchall()
    except Exception:
        rows = []
    conn.close()

    total_patients = len(rows)
    risk_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    probabilities = []
    ages = []

    for r in rows:
        lvl = str(r["risk_level"] or "").strip().upper()
        if lvl in risk_counts:
            risk_counts[lvl] += 1
        p_val = safe_float(r["probability"])
        probabilities.append(p_val)
        if r["age"] is not None:
            try:
                ages.append(int(r["age"]))
            except (ValueError, TypeError):
                pass

    risk_percentages = {
        "LOW": round((risk_counts["LOW"] / total_patients) * 100, 1) if total_patients > 0 else 0.0,
        "MODERATE": round((risk_counts["MODERATE"] / total_patients) * 100, 1) if total_patients > 0 else 0.0,
        "HIGH": round((risk_counts["HIGH"] / total_patients) * 100, 1) if total_patients > 0 else 0.0
    }

    avg_probability = round(sum(probabilities) / len(probabilities), 2) if probabilities else None
    avg_age = round(sum(ages) / len(ages), 1) if ages else None
    recent_patients = [dict(r) for r in rows[:6]]

    # Heart Model Metadata
    try:
        model_meta = joblib.load("models/model_info.pkl")
    except Exception:
        model_meta = CACHED_MODEL_INFO or {}

    try:
        opt_meta = joblib.load("models/optimization_comparison.pkl")
    except Exception:
        opt_meta = CACHED_OPT_COMPARISON or {}

    lr_prf = model_meta.get("prf_data", {}).get("Logistic Regression", {})
    lr_roc = model_meta.get("roc_data", {}).get("Logistic Regression", {})
    lr_precision = round(lr_prf["precision"] * 100, 2) if "precision" in lr_prf else None
    lr_recall = round(lr_prf["recall"] * 100, 2) if "recall" in lr_prf else None
    lr_f1 = round(lr_prf["f1"] * 100, 2) if "f1" in lr_prf else None
    lr_auc = round(lr_roc["auc"], 4) if "auc" in lr_roc else 0.90

    model_performance = {
        "status": "Available" if model_meta else "Unavailable",
        "model_name": model_meta.get("model_name", "Logistic Regression"),
        "accuracy": round(model_meta["accuracy"] * 100, 2) if "accuracy" in model_meta else None,
        "precision": lr_precision,
        "recall": lr_recall,
        "f1": lr_f1,
        "roc_auc": lr_auc,
        "confusion_matrix": model_meta.get("confusion_matrix"),
        "feature_importance": model_meta.get("feature_importance", {}),
        "roc_data": model_meta.get("roc_data", {}),
        "prf_data": model_meta.get("prf_data", {}),
        "all_models": model_meta.get("all_models", {}),
        "optimization_comparison": opt_meta
    }

    analytics_data = {
        "disease_type": "heart",
        "disease_name": "Cardiovascular Disease",
        "disease_icon": "🫀",
        "patient_stats": {
            "total_patients": total_patients,
            "risk_distribution": risk_counts,
            "risk_percentages": risk_percentages,
            "avg_probability": avg_probability,
            "avg_age": avg_age,
            "recent_patients": recent_patients
        },
        "model_performance": model_performance
    }

    if request.args.get("format") == "json" or (request.is_json and request.method == "GET"):
        return jsonify(analytics_data)

    return render_template("heart_analytics.html", **analytics_data)


@app.route("/analytics/diabetes")
def diabetes_analytics():
    if "user" not in session:
        if request.args.get("format") == "json" or request.is_json:
            return {"error": "Unauthorized"}, 401
        return redirect(url_for("login"))

    conn = get_db_connection()
    try:
        rows = conn.execute("SELECT * FROM diabetes_patients ORDER BY timestamp DESC").fetchall()
    except Exception:
        rows = []
    conn.close()

    total_patients = len(rows)
    risk_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    probabilities = []
    glucoses = []
    bmis = []

    for r in rows:
        row_dict = dict(r)
        lvl = str(row_dict.get("risk_level") or "").strip().upper()
        if lvl in risk_counts:
            risk_counts[lvl] += 1
        p_val = safe_float(row_dict.get("probability"))
        probabilities.append(p_val)
        g_val = safe_float(row_dict.get("glucose"))
        if g_val > 0:
            glucoses.append(g_val)
        b_val = safe_float(row_dict.get("bmi"))
        if b_val > 0:
            bmis.append(b_val)

    risk_percentages = {
        "LOW": round((risk_counts["LOW"] / total_patients) * 100, 1) if total_patients > 0 else 0.0,
        "MODERATE": round((risk_counts["MODERATE"] / total_patients) * 100, 1) if total_patients > 0 else 0.0,
        "HIGH": round((risk_counts["HIGH"] / total_patients) * 100, 1) if total_patients > 0 else 0.0
    }

    avg_probability = round(sum(probabilities) / len(probabilities), 2) if probabilities else None
    avg_glucose = round(sum(glucoses) / len(glucoses), 1) if glucoses else None
    avg_bmi = round(sum(bmis) / len(bmis), 1) if bmis else None
    recent_patients = [dict(r) for r in rows[:6]]

    # Diabetes Model Metadata
    try:
        model_meta = joblib.load("models/diabetes_model_info.pkl")
    except Exception:
        model_meta = diabetes_model_info or {}

    model_performance = {
        "status": "Available" if model_meta else "Unavailable",
        "model_name": model_meta.get("model_name", "Random Forest Classifier (Balanced)"),
        "dataset": model_meta.get("dataset", "Pima Indians Diabetes Dataset"),
        "accuracy": round(model_meta["accuracy"] * 100, 2) if "accuracy" in model_meta else None,
        "precision": round(model_meta["precision"] * 100, 2) if "precision" in model_meta else None,
        "recall": round(model_meta["recall"] * 100, 2) if "recall" in model_meta else None,
        "f1": round(model_meta["f1"] * 100, 2) if "f1" in model_meta else None,
        "roc_auc": round(model_meta["roc_auc"], 4) if "roc_auc" in model_meta else None,
        "confusion_matrix": model_meta.get("confusion_matrix"),
        "feature_importance": model_meta.get("feature_importance", {}),
        "roc_curve": model_meta.get("roc_curve", {})
    }

    analytics_data = {
        "disease_type": "diabetes",
        "disease_name": "Diabetes Mellitus",
        "disease_icon": "🩸",
        "patient_stats": {
            "total_patients": total_patients,
            "risk_distribution": risk_counts,
            "risk_percentages": risk_percentages,
            "avg_probability": avg_probability,
            "avg_glucose": avg_glucose,
            "avg_bmi": avg_bmi,
            "recent_patients": recent_patients
        },
        "model_performance": model_performance
    }

    if request.args.get("format") == "json" or (request.is_json and request.method == "GET"):
        return jsonify(analytics_data)

    return render_template("diabetes_analytics.html", **analytics_data)


@app.route("/analytics/breast_cancer")
def breast_cancer_analytics():
    if "user" not in session:
        if request.args.get("format") == "json" or request.is_json:
            return {"error": "Unauthorized"}, 401
        return redirect(url_for("login"))

    conn = get_db_connection()
    try:
        rows = conn.execute("SELECT * FROM breast_cancer_patients ORDER BY timestamp DESC").fetchall()
    except Exception:
        rows = []
    conn.close()

    total_patients = len(rows)
    risk_counts = {"LOW": 0, "MODERATE": 0, "HIGH": 0}
    probabilities = []
    malignant_count = 0
    benign_count = 0

    for r in rows:
        lvl = str(r["risk_level"] or "").strip().upper()
        if lvl in risk_counts:
            risk_counts[lvl] += 1
        p_val = safe_float(r["probability"])
        probabilities.append(p_val)
        res = str(r["result"] or "").strip().capitalize()
        if res == "Malignant":
            malignant_count += 1
        elif res == "Benign":
            benign_count += 1

    risk_percentages = {
        "LOW": round((risk_counts["LOW"] / total_patients) * 100, 1) if total_patients > 0 else 0.0,
        "MODERATE": round((risk_counts["MODERATE"] / total_patients) * 100, 1) if total_patients > 0 else 0.0,
        "HIGH": round((risk_counts["HIGH"] / total_patients) * 100, 1) if total_patients > 0 else 0.0
    }

    avg_probability = round(sum(probabilities) / len(probabilities), 2) if probabilities else None
    recent_patients = [dict(r) for r in rows[:6]]

    # Breast Cancer Model Metadata
    try:
        model_meta = joblib.load("models/breast_cancer_model_info.pkl")
    except Exception:
        model_meta = breast_cancer_model_info or {}

    model_performance = {
        "status": "Available" if model_meta else "Unavailable",
        "model_name": model_meta.get("model_name", "Support Vector Machine (RBF Balanced)"),
        "dataset": model_meta.get("dataset", "Wisconsin Diagnostic Breast Cancer (WDBC)"),
        "accuracy": round(model_meta["accuracy"] * 100, 2) if "accuracy" in model_meta else None,
        "precision": round(model_meta["precision"] * 100, 2) if "precision" in model_meta else None,
        "recall": round(model_meta["recall"] * 100, 2) if "recall" in model_meta else None,
        "specificity": round(model_meta["specificity"] * 100, 2) if "specificity" in model_meta else None,
        "f1": round(model_meta["f1"] * 100, 2) if "f1" in model_meta else None,
        "roc_auc": round(model_meta["roc_auc"], 4) if "roc_auc" in model_meta else None,
        "confusion_matrix": model_meta.get("confusion_matrix"),
        "feature_columns": model_meta.get("feature_columns", []),
        "roc_curve": model_meta.get("roc_curve", {})
    }

    analytics_data = {
        "disease_type": "breast_cancer",
        "disease_name": "Breast Cytology (WDBC)",
        "disease_icon": "🎗️",
        "patient_stats": {
            "total_patients": total_patients,
            "risk_distribution": risk_counts,
            "risk_percentages": risk_percentages,
            "avg_probability": avg_probability,
            "malignant_count": malignant_count,
            "benign_count": benign_count,
            "recent_patients": recent_patients
        },
        "model_performance": model_performance
    }

    if request.args.get("format") == "json" or (request.is_json and request.method == "GET"):
        return jsonify(analytics_data)

    return render_template("breast_cancer_analytics.html", **analytics_data)


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

        # Physiological range validation matching frontend constraints and clinical bounds
        if age < 1 or age > 120:
            raise ValueError("Age must be between 1 and 120 years.")
        if trestbps < 50 or trestbps > 300:
            raise ValueError("Resting blood pressure must be between 50 and 300 mmHg.")
        if chol < 50 or chol > 700:
            raise ValueError("Serum cholesterol must be between 50 and 700 mg/dL.")
        if thalach < 40 or thalach > 250:
            raise ValueError("Maximum heart rate must be between 40 and 250 bpm.")
        if oldpeak < 0.0 or oldpeak > 10.0:
            raise ValueError("ST depression must be between 0.0 and 10.0 mm.")
        if ca < 0 or ca > 3:
            raise ValueError("Number of major vessels must be between 0 and 3.")

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
    pos_idx = list(model.classes_).index(1)
    probability = float(model.predict_proba(user_scaled)[0][pos_idx])

    prediction = 1 if probability >= 0.5 else 0

    prob_percentage = round(probability * 100, 2)
    risk_level = classify_risk(prob_percentage)

    # Collision-resistant unique Patient ID with human-readable timestamp and 8-hex random entropy
    patient_id = "PID" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + secrets.token_hex(4).upper()

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
    restecg_text = REVERSE_FEATURE_MAP["restecg"].get(patient.get("restecg"), str(patient.get("restecg")))
    exang_text = REVERSE_FEATURE_MAP["exang"].get(patient.get("exang"), str(patient.get("exang")))
    slope_text = REVERSE_FEATURE_MAP["slope"].get(patient.get("slope"), str(patient.get("slope")))
    thal_text = REVERSE_FEATURE_MAP["thal"].get(patient.get("thal"), str(patient.get("thal")))

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
    elements.append(Spacer(1, 6))

    disclaimer_style = ParagraphStyle(
        'HeartDisclaimerStyle',
        parent=styles['Normal'],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#64748b'),
        alignment=1
    )
    elements.append(Paragraph("Clinical Decision Support Estimate — Not an Official Medical Diagnosis", disclaimer_style))
    elements.append(Spacer(1, 12))

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
        ["", "Resting ECG Results", restecg_text],
        ["", "ST Depression", str(patient.get("oldpeak", "")) + " mm"],
        ["", "Slope of ST", slope_text],
        ["", "Exercise Induced Angina", exang_text],
        ["", "Major Vessels Fluro (0-3)", str(patient.get("ca", ""))],
        ["", "Thallium", thal_text],
        ["AI Decision Support", "Heart Disease Prediction", str(patient.get("result", ""))],
        ["", "Risk Probability", str(patient.get("probability", "")) + "%"],
        ["", "Statistical Risk Tier", str(patient.get("risk_level", "")) + " (Non-Diagnostic)"],
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

    pos_idx = list(diabetes_pipeline.classes_).index(1)
    probability = float(diabetes_pipeline.predict_proba(user_df)[0][pos_idx])
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

        conn = get_db_connection()
        patient = None
        resolved_disease = None

        # Prioritize table lookup based on disease_type hint or ID prefix
        requested_disease = (data.get("disease_type") or "").strip().lower()

        search_order = []
        if requested_disease == "heart" or patient_id.startswith("PID"):
            search_order = [("heart", "patients"), ("diabetes", "diabetes_patients"), ("breast_cancer", "breast_cancer_patients")]
        elif requested_disease == "diabetes" or patient_id.startswith("DPID"):
            search_order = [("diabetes", "diabetes_patients"), ("heart", "patients"), ("breast_cancer", "breast_cancer_patients")]
        elif requested_disease in ("breast_cancer", "cancer") or patient_id.startswith("BCP"):
            search_order = [("breast_cancer", "breast_cancer_patients"), ("heart", "patients"), ("diabetes", "diabetes_patients")]
        else:
            search_order = [("heart", "patients"), ("diabetes", "diabetes_patients"), ("breast_cancer", "breast_cancer_patients")]

        for dtype, table in search_order:
            row = conn.execute(f"SELECT * FROM {table} WHERE patient_id = ?", (patient_id,)).fetchone()
            if row:
                patient = dict(row)
                resolved_disease = dtype
                break

        conn.close()

        if not patient:
            return {"error": "Patient not found"}, 404

        # ----------------------------------------------------
        # Build disease-specific clinical context (De-identified)
        # ----------------------------------------------------
        probability = safe_float(patient.get("probability"))

        if resolved_disease == "heart":
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

            patient["explanation"] = generate_explanation(patient)
            patient["recommendations"] = generate_recommendation(patient)
            patient["alerts"] = generate_alerts(patient)

            drivers = []
            if safe_float(patient.get("chol")) > 200:
                drivers.append("High Cholesterol (>200 mg/dL)")
            if safe_float(patient.get("trestbps")) > 130:
                drivers.append("High Blood Pressure (>130 mmHg)")
            if safe_float(patient.get("oldpeak")) > 1:
                drivers.append("Cardiac Stress Indicator (ST depression >1.0)")
            if safe_float(patient.get("ca")) > 1:
                drivers.append("Multiple Fluoroscopy Vessels Colored (ca > 1)")

            key_driver = drivers[0] if drivers else "No major rule-based risk factor detected"

            short_context = f"""
Clinical Module: Cardiovascular Risk Prediction (Logistic Regression)
Patient Reference: [De-identified] ({patient_id})
Age: {patient.get('age')} years | Sex: {sex_text}
Prediction Result: {patient.get('result', 'N/A')}
Estimated Probability: {probability:.1f}%
Risk Tier: {patient.get('risk_level', 'N/A')}
Resting Blood Pressure: {patient.get('trestbps')} mmHg
Serum Cholesterol: {patient.get('chol')} mg/dL
Chest Pain Type: {cp_text}
Fasting Blood Sugar > 120 mg/dl: {fbs_text}
Resting ECG: {restecg_text}
Max Heart Rate Achieved: {patient.get('thalach')} bpm
Exercise-Induced Angina: {exang_text}
ST Depression (oldpeak): {patient.get('oldpeak')}
Slope of Peak Exercise ST: {slope_text}
Major Vessels Colored (ca): {patient.get('ca')}
Thalassemia: {thal_text}
Primary Risk Driver: {key_driver}
Clinical Alerts: {'; '.join(patient.get('alerts', [])) or 'None'}
Recommended Next Steps: {'; '.join(patient.get('recommendations', [])) or 'Routine monitoring'}
"""

        elif resolved_disease == "diabetes":
            glucose = safe_float(patient.get("glucose"))
            bmi = safe_float(patient.get("bmi"))
            bp = safe_float(patient.get("blood_pressure"))
            insulin = safe_float(patient.get("insulin"))
            skin = safe_float(patient.get("skin_thickness"))
            dpf = safe_float(patient.get("dpf"))
            pregnancies = patient.get("pregnancies", 0)

            patient["explanation"] = generate_diabetes_explanation(patient)
            patient["recommendations"] = generate_diabetes_recommendation(patient)
            patient["alerts"] = generate_diabetes_alerts(patient)

            drivers = []
            if glucose >= 140:
                drivers.append("Elevated Plasma Glucose (>=140 mg/dL)")
            if bmi >= 28:
                drivers.append("Elevated Body Mass Index (BMI >=28)")
            if dpf >= 0.5:
                drivers.append("Elevated Genetic Pedigree Score (DPF >=0.5)")
            if bp >= 85:
                drivers.append("Elevated Diastolic Blood Pressure (>=85 mmHg)")

            key_driver = drivers[0] if drivers else "Metabolic parameters within typical reference ranges"

            short_context = f"""
Clinical Module: Diabetes Mellitus Risk Prediction (Random Forest)
Patient Reference: [De-identified] ({patient_id})
Age: {patient.get('age')} years
Prediction Result: {patient.get('result', 'N/A')}
Estimated Probability: {probability:.1f}%
Risk Tier: {patient.get('risk_level', 'N/A')}
Plasma Glucose (2-hr oral glucose test): {glucose} mg/dL
Body Mass Index (BMI): {bmi} kg/m²
Diastolic Blood Pressure: {bp} mmHg
Serum Insulin: {insulin} mu U/ml
Triceps Skin Fold Thickness: {skin} mm
Diabetes Pedigree Function (DPF): {dpf}
Pregnancies: {pregnancies}
Primary Risk Driver: {key_driver}
Clinical Alerts: {'; '.join(patient.get('alerts', [])) or 'None'}
Recommended Next Steps: {'; '.join(patient.get('recommendations', [])) or 'Routine metabolic monitoring'}
"""

        else:  # breast_cancer
            mean_radius = safe_float(patient.get("mean_radius"))
            mean_texture = safe_float(patient.get("mean_texture"))
            mean_perimeter = safe_float(patient.get("mean_perimeter"))
            mean_area = safe_float(patient.get("mean_area"))
            mean_concavity = safe_float(patient.get("mean_concavity"))
            mean_concave_points = safe_float(patient.get("mean_concave_points"))
            worst_radius = safe_float(patient.get("worst_radius"))
            worst_texture = safe_float(patient.get("worst_texture"))
            worst_perimeter = safe_float(patient.get("worst_perimeter"))
            worst_area = safe_float(patient.get("worst_area"))
            worst_concavity = safe_float(patient.get("worst_concavity"))
            worst_concave_points = safe_float(patient.get("worst_concave_points"))

            patient["explanation"] = generate_breast_cancer_explanation(patient)
            patient["recommendations"] = generate_breast_cancer_recommendations(patient)
            patient["alerts"] = generate_breast_cancer_alerts(patient)

            drivers = []
            if mean_concave_points >= 0.05:
                drivers.append("Nuclear Contour Indentation (Concave Points >=0.05)")
            if mean_concavity >= 0.10:
                drivers.append("Nuclear Infolding Severity (Concavity >=0.10)")
            if mean_area >= 700.0:
                drivers.append("Nuclear Area Hypertrophy (>=700 µm²)")
            if mean_radius >= 15.0:
                drivers.append("Nuclear Radius Pleomorphism (>=15.0 µm)")
            if mean_texture >= 20.0:
                drivers.append("Chromatin Texture Heterogeneity (>=20.0)")

            key_driver = drivers[0] if drivers else "Nuclear morphometric measurements within benign baseline"

            short_context = f"""
Clinical Module: Breast Cytology Malignancy Prediction (Support Vector Machine / RBF Kernel)
Patient Reference: [De-identified] ({patient_id})
Age: {patient.get('age', 'N/A')} years
Prediction Result: {patient.get('result', 'N/A')}
Estimated Probability: {probability:.1f}%
Risk Tier: {patient.get('risk_level', 'N/A')}
Nuclear Mean Radius: {mean_radius} µm (Worst: {worst_radius} µm)
Nuclear Mean Texture: {mean_texture} (Worst: {worst_texture})
Nuclear Mean Perimeter: {mean_perimeter} mm (Worst: {worst_perimeter} mm)
Nuclear Mean Area: {mean_area} µm² (Worst: {worst_area} µm²)
Nuclear Mean Concavity: {mean_concavity} (Worst: {worst_concavity})
Nuclear Mean Concave Points: {mean_concave_points} (Worst: {worst_concave_points})
Primary Risk Driver: {key_driver}
Clinical Alerts: {'; '.join(patient.get('alerts', [])) or 'None'}
Recommended Next Steps: {'; '.join(patient.get('recommendations', [])) or 'Routine cytological follow-up'}
"""

        # ----------------------------------------------------
        # AI Assistant Client Validation
        # ----------------------------------------------------
        if not gemini_client:
            return {
                "reply": "AI Clinical Assistant is currently unavailable (GEMINI_API_KEY environment variable is not configured on this server)."
            }, 503

        module_label = {
            "heart": "Cardiovascular Disease",
            "diabetes": "Diabetes Mellitus",
            "breast_cancer": "Breast Cytology (WDBC)"
        }.get(resolved_disease, "Clinical AI")

        system_instruction = f"""You are an expert Clinical AI Assistant inside a Multi-Disease Clinical AI Prediction Platform, currently supporting the {module_label} module.

Your purpose is to explain statistical model inferences, risk levels, and clinical parameters to healthcare providers and authorized clinical staff.

Guidelines:
1. Answer the user's question directly, clearly, and concisely (typically 3-5 complete sentences or bullet points).
2. Rely strictly on the provided de-identified patient data and clinical parameters.
3. Explain medical concepts, abbreviations, and risk drivers in accessible clinical context.
4. Maintain a professional, objective clinical decision-support tone.
5. Strictly adhere to clinical safety boundaries:
   - Do NOT provide an official diagnosis or state that the patient definitively has or does not have the condition.
   - Do NOT prescribe medications, specific dosages, or therapeutic regimens.
   - Do NOT invent or infer unrecorded patient parameters.
   - Emphasize that all model outputs are statistical probabilities designed to augment clinical judgment.
6. Provide a complete answer without trailing off mid-sentence.
"""

        user_prompt = f"""Patient Clinical Record:
{short_context.strip()}

User Question:
{user_message}
"""

        # Cascading fallback model cascade for optimal reliability
        fallback_models = [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite"
        ]

        reply = None
        last_error = None

        for model_name in fallback_models:
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        max_output_tokens=1000,
                        thinking_config=types.ThinkingConfig(
                            thinking_level="low"
                        )
                    )
                )

                # Safe text extraction from response
                if hasattr(response, "text") and response.text:
                    t = response.text.strip()
                    if t:
                        reply = t
                        break

                if hasattr(response, "candidates") and response.candidates:
                    for cand in response.candidates:
                        if hasattr(cand, "content") and cand.content:
                            parts_text = []
                            for part in getattr(cand.content, "parts", []):
                                pt = getattr(part, "text", "")
                                if pt:
                                    parts_text.append(pt)
                            joined = "".join(parts_text).strip()
                            if joined:
                                reply = joined
                                break
                    if reply:
                        break

            except Exception as model_err:
                last_error = model_err
                print(f"Gemini model {model_name} invocation failed: {model_err}. Attempting fallback...")
                continue

        if reply:
            return {
                "reply": reply
            }

        print(f"All Gemini models exhausted. Final error: {last_error}")
        return {
            "reply": "⚠️ The Clinical AI service is currently experiencing high demand or upstream latency. Please try your question again in a moment."
        }, 503

    except Exception as e:

        import traceback

        print("\nERROR IN AI_ASSISTANT ROUTE:")
        print(traceback.format_exc())

        return {
            "reply": "⚠️ An internal error occurred while processing your clinical AI inquiry. Please retry."
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
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(host="0.0.0.0", port=port, debug=debug)


# if __name__ == "__main__":
#     webbrowser.open("http://127.0.0.1:5000/login")
#     app.run(debug=True)
