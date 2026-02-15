# ============================================================
# FLASK WEB APPLICATION
# Heart Disease Clinical Dashboard
# ============================================================

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import webbrowser
app = Flask(__name__)

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
    return render_template("index.html")


# ------------------------------------------------------------
# Prediction Route
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # ==============================
    # Get form data
    # ==============================
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

    # ==============================
    # Create dataframe
    # ==============================
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

    # Scale
    user_scaled = scaler.transform(user_data)

    # Predict
    probability = model.predict_proba(user_scaled)[0][1]
    prediction = 1 if probability >= 0.5 else 0

    # Risk Level
    if probability < 0.3:
        risk_level = "LOW"
        color = "green"
    elif probability < 0.7:
        risk_level = "MODERATE"
        color = "orange"
    else:
        risk_level = "HIGH"
        color = "red"

    # Generate Patient ID
    patient_id = "PID" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Save Record
    if not os.path.exists("records"):
        os.makedirs("records")

    record_file = "records/patient_records.csv"

    patient_record = {
        "Patient ID": patient_id,
        "Patient Name": patient_name,
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": age,
        "Sex": sex,
        "Chest Pain Type": cp,
        "Resting BP": trestbps,
        "Cholesterol": chol,
        "Fasting Blood Sugar": fbs,
        "Rest ECG": restecg,
        "Max Heart Rate": thalach,
        "Exercise Angina": exang,
        "ST Depression": oldpeak,
        "Slope": slope,
        "Vessels": ca,
        "Thalassemia": thal,
        "Prediction": "Present" if prediction == 1 else "Absent",
        "Probability (%)": round(probability * 100, 2),
        "Risk Level": risk_level
    }

    record_df = pd.DataFrame([patient_record])

    if os.path.exists(record_file):
        record_df.to_csv(record_file, mode="a", header=False, index=False)
    else:
        record_df.to_csv(record_file, index=False)

    # Return result page
    return render_template(
        "result.html",
        patient_name=patient_name,
        patient_id=patient_id,
        probability=round(probability * 100, 2),
        prediction="Present" if prediction == 1 else "Absent",
        risk_level=risk_level,
        color=color
    )

@app.route("/history")
def history():
    import os
    import pandas as pd

    record_file = "records/patient_records.csv"

    if os.path.exists(record_file):
        df = pd.read_csv(record_file)
        records = df.to_dict(orient="records")
    else:
        records = []

    return render_template("history.html", records=records)


@app.route('/dashboard')
def dashboard():

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
# Run App
# ------------------------------------------------------------
if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)