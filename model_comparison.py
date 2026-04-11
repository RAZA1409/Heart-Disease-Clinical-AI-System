# ============================================================
# MODEL COMPARISON & BEST MODEL SELECTION
# ============================================================

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv("Heart_Disease_Cleaned.csv")

print("Dataset loaded successfully!\n")


# ============================================================
# FEATURES & TARGET
# ============================================================

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}


# ============================================================
# TRAINING & EVALUATION
# ============================================================

print("Training models...\n")

results = {}

for name, model in models.items():
    print(f"Training {name}...")

    model.fit(X_train, y_train)

    # Predictions
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    # Error (Loss equivalent)
    error = 1 - test_acc

    # Store detailed results
    results[name] = {
        "train": train_acc,
        "test": test_acc,
        "error": error
    }

    print(f"{name} Train Accuracy: {train_acc:.4f}")
    print(f"{name} Test Accuracy: {test_acc:.4f}")
    print(f"{name} Error: {error:.4f}\n")


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n==============================")
print("MODEL COMPARISON RESULTS")
print("==============================\n")

for name, metrics in results.items():
    print(f"{name}:")
    print(f"  Train Accuracy: {metrics['train']:.4f}")
    print(f"  Test Accuracy : {metrics['test']:.4f}")
    print(f"  Error         : {metrics['error']:.4f}\n")

# ============================================================
# BEST MODEL SELECTION
# ============================================================

best_model_name = max(results, key=lambda x: results[x]["test"])
best_model = models[best_model_name]

print(f"\nBest Model Selected: {best_model_name}")
print(f"Best Accuracy: {results[best_model_name]['test']*100:.2f}%\n")

# ============================================================
# SAVE MODEL FILES
# ============================================================

if not os.path.exists("models"):
    os.makedirs("models")

# Save model
joblib.dump(best_model, "models/best_model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Save feature columns
joblib.dump(list(X.columns), "models/feature_columns.pkl")

# Save model info (🔥 VERY IMPORTANT FOR DASHBOARD)
model_info = {
    "model_name": best_model_name,
    "accuracy": results[best_model_name]["test"],
    "all_models": results
}

joblib.dump(model_info, "models/model_info.pkl")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("✅ Best model and related files saved successfully!")
print("📁 Check 'models/' folder")
print("\n🎯 Project upgraded to MULTI-MODEL SYSTEM 🚀")
