# ===============================
# HEART DISEASE OPTIMIZATION PROJECT
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from experiment_logger import log_experiment
import joblib
import os
# ===============================
# 1. LOAD DATASET
# ===============================

df = pd.read_csv("Heart_Disease_Cleaned.csv")

print("Dataset Shape:", df.shape)
print("Target unique values:", df["Heart Disease"].unique())

# Remove any missing values (safety)
df = df.dropna()

# ===============================
# 2. PREPROCESSING
# ===============================

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling (Very Important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Preprocessing Done!")

# ===============================
# 3. BASELINE SVM
# ===============================

baseline_model = SVC()
baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print("\nBaseline Accuracy:",
      round(baseline_accuracy * 100, 2), "%")

log_experiment(
    method="Baseline SVM",
    accuracy=baseline_accuracy,
    params={}
)
# ===============================
# 4. GRID SEARCH
# ===============================

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)

best_model = grid.best_estimator_

y_pred_grid = best_model.predict(X_test)

grid_accuracy = accuracy_score(y_test, y_pred_grid)

print("Grid Search Accuracy:",
      round(grid_accuracy * 100, 2), "%")


log_experiment(
    method="Grid Search SVM",
    accuracy=grid_accuracy,
    params=grid.best_params_
)

if not os.path.exists("models"):
    os.makedirs("models")

joblib.dump(best_model, "models/best_model.pkl")
# ===============================
# 5. COMPARISON
# ===============================

print("\n---- FINAL COMPARISON ----")
print("Baseline Accuracy:",
      round(baseline_accuracy * 100, 2), "%")

print("Grid Search Accuracy:",
      round(grid_accuracy * 100, 2), "%")