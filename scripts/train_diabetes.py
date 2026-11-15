"""
Diabetes Prediction Model Training Pipeline.
Fits a scikit-learn Pipeline with median imputation, standardization,
and a balanced Random Forest classifier on the Pima Indians Diabetes dataset.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "diabetes.csv") if os.path.exists(os.path.join(PROJECT_ROOT, "data", "diabetes.csv")) else os.path.join(PROJECT_ROOT, "diabetes.csv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")

def train_diabetes_model(data_path=None, output_dir=None):
    if data_path is None:
        data_path = DEFAULT_DATA_PATH
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    print(f"Loading diabetes dataset from {data_path}...")
    df = pd.read_csv(data_path)

    feature_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure',
        'SkinThickness', 'Insulin', 'BMI',
        'DiabetesPedigreeFunction', 'Age'
    ]
    target_col = 'Outcome'

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Features where 0 indicates missing measurement (standard clinical trait of Pima dataset)
    zero_as_missing_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    regular_cols = ['Pregnancies', 'DiabetesPedigreeFunction', 'Age']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ColumnTransformer pipeline: impute zeros with median, standard scale all features
    preprocessor = ColumnTransformer(
        transformers=[
            ('impute_and_scale', Pipeline([
                ('imputer', SimpleImputer(missing_values=0, strategy='median')),
                ('scaler', StandardScaler())
            ]), zero_as_missing_cols),
            ('scale_regular', StandardScaler(), regular_cols)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight='balanced',
            random_state=42
        ))
    ])

    print("Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    # Independent test evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Feature importances from the classifier
    clf = pipeline.named_steps['classifier']
    importances = clf.feature_importances_
    # Feature order transformed: zero_as_missing_cols + regular_cols
    transformed_features = zero_as_missing_cols + regular_cols
    feature_importance_dict = dict(zip(transformed_features, [float(v) for v in importances]))
    feature_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

    metadata = {
        "model_name": "Random Forest Classifier (Balanced)",
        "dataset": "Pima Indians Diabetes Dataset",
        "sample_count": len(df),
        "test_size": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "feature_columns": feature_cols,
        "feature_importance": feature_importance_dict,
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "auc": round(auc, 4)
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    pipeline_path = os.path.join(output_dir, "diabetes_pipeline.pkl")
    cols_path = os.path.join(output_dir, "diabetes_feature_columns.pkl")
    meta_path = os.path.join(output_dir, "diabetes_model_info.pkl")

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(feature_cols, cols_path)
    joblib.dump(metadata, meta_path)

    print("\n=== DIABETES MODEL EVALUATION METRICS ===")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"\nSaved artifacts to {output_dir}/:")
    print(f"  - {pipeline_path}")
    print(f"  - {cols_path}")
    print(f"  - {meta_path}")

    return metadata

if __name__ == "__main__":
    train_diabetes_model()
