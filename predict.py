# ============================================================
# PREDICTION FILE
# This file loads trained model and makes predictions
# ============================================================

import numpy as np
import pandas as pd
import joblib
import os


# Load trained model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

print("\n==============================")
print("HEART DISEASE PREDICTION SYSTEM")
print("==============================")

print("\nEnter patient details carefully.\n")

# Collect inputs
age = float(input("Age: "))
sex = float(input("Sex (1=Male, 0=Female): "))
cp = float(input("Chest Pain Type (1-4): "))
trestbps = float(input("Resting BP: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting Blood Sugar >120 (1/0): "))
restecg = float(input("Rest ECG (0-2): "))
thalach = float(input("Max Heart Rate: "))
exang = float(input("Exercise Angina (1/0): "))
oldpeak = float(input("ST Depression: "))
slope = float(input("Slope (1-3): "))
ca = float(input("Number of Vessels (0-3): "))
thal = float(input("Thalassemia (3,6,7): "))

# Create DataFrame with correct column order
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

# Scale input
user_scaled = scaler.transform(user_data)

# Predict probability
probability = model.predict_proba(user_scaled)[0][1]
prediction = 1 if probability >= 0.5 else 0

if probability < 0.3:
    risk_level = "LOW"
elif probability < 0.7:
    risk_level = "MODERATE"
else:
    risk_level = "HIGH"

print("\n==============================")
print("PREDICTION RESULT")
print("==============================")

if probability >= 0.5:
    print("⚠ Heart Disease: PRESENT")
else:
    print("✓ Heart Disease: ABSENT")

print(f"Risk Probability: {probability*100:.2f}%")
print(f"Risk Level: {risk_level}")
print("==============================")

# ============================================================
#  SAVE PATIENT RECORD
# ============================================================

import datetime

# Create records folder if not exists
if not os.path.exists("records"):
    os.makedirs("records")

record_file = "records/patient_records.csv"

# Create dictionary of patient data
patient_record = {
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

# Append if file exists, else create new
if os.path.exists(record_file):
    record_df.to_csv(record_file, mode='a', header=False, index=False)
else:
    record_df.to_csv(record_file, index=False)

print("Patient record saved successfully!")