"""
Breast Cancer (WDBC) Clinical AI Model Training Pipeline.
Fits a scikit-learn Pipeline with StandardScaler and Support Vector Machine (RBF kernel)
on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
Malignant is treated as the positive class (1), Benign as the negative class (0).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    balanced_accuracy_score, classification_report
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")

def train_breast_cancer_model(output_dir=None, export_dataset=False):
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    print("Loading official Wisconsin Diagnostic Breast Cancer (WDBC) dataset...")
    raw_data = load_breast_cancer()

    # In scikit-learn load_breast_cancer, target 0 = malignant, 1 = benign.
    # We strictly enforce Malignant as positive class (1) and Benign as negative (0).
    y = np.where(raw_data.target == 0, 1, 0)

    # Standardize feature names to clean snake_case
    feature_cols = [c.replace(' ', '_') for c in raw_data.feature_names]
    X = pd.DataFrame(raw_data.data, columns=feature_cols)

    # Optional reference dataset export (opt-in only, never overwrites by default)
    if export_dataset:
        data_dir = os.path.join(PROJECT_ROOT, "data")
        export_path = os.path.join(data_dir, "breast_cancer.csv") if os.path.exists(data_dir) else os.path.join(PROJECT_ROOT, "breast_cancer.csv")
        if not os.path.exists(export_path):
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            export_df = X.copy()
            export_df['target'] = y
            export_df.to_csv(export_path, index=False)
            print(f"Exported reference dataset to {export_path}")
        else:
            print(f"Reference dataset already exists at {export_path}; preserving existing file.")

    print(f"Dataset Shape: {X.shape[0]} records, {X.shape[1]} diagnostic numerical features")
    print(f"Missing Values: {X.isnull().sum().sum()}")
    print(f"Duplicate Rows: {X.duplicated().sum()}")
    print(f"Target Distribution: Benign (0) = {sum(y == 0)}, Malignant (1) = {sum(y == 1)}")

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Cross-validation on training data only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', C=1.0, probability=True, class_weight='balanced', random_state=42))
    ])

    print("\nRunning 5-Fold Stratified Cross-Validation on Training Data...")
    cv_scores = cross_validate(
        pipeline, X_train, y_train, cv=cv,
        scoring=['accuracy', 'roc_auc', 'recall', 'precision', 'f1']
    )

    print(f"  CV Accuracy : {cv_scores['test_accuracy'].mean() * 100:.2f}% (+/- {cv_scores['test_accuracy'].std() * 100:.2f}%)")
    print(f"  CV Recall   : {cv_scores['test_recall'].mean() * 100:.2f}% (+/- {cv_scores['test_recall'].std() * 100:.2f}%)")
    print(f"  CV ROC-AUC  : {cv_scores['test_roc_auc'].mean():.4f}")
    print(f"  CV F1-Score : {cv_scores['test_f1'].mean():.4f}")

    # Train final pipeline on X_train
    print("\nFitting pipeline on X_train...")
    pipeline.fit(X_train, y_train)

    # Independent test evaluation on held-out test set
    y_pred = pipeline.predict(X_test)
    pos_idx = list(pipeline.classes_).index(1)
    y_prob = pipeline.predict_proba(X_test)[:, pos_idx]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = float(tn / (tn + fp))

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metadata = {
        "model_name": "Support Vector Machine (RBF Balanced)",
        "dataset": "Wisconsin Diagnostic Breast Cancer (WDBC)",
        "sample_count": len(X),
        "test_size": len(X_test),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(spec, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "feature_columns": feature_cols,
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "auc": round(auc, 4)
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    pipeline_path = os.path.join(output_dir, "breast_cancer_pipeline.pkl")
    cols_path = os.path.join(output_dir, "breast_cancer_feature_columns.pkl")
    meta_path = os.path.join(output_dir, "breast_cancer_model_info.pkl")

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(feature_cols, cols_path)
    joblib.dump(metadata, meta_path)

    print("\n=== BREAST CANCER HELD-OUT TEST EVALUATION METRICS ===")
    print(f"Accuracy         : {acc * 100:.2f}%")
    print(f"Precision        : {prec * 100:.2f}%")
    print(f"Recall (Sens)    : {rec * 100:.2f}% (Malignant)")
    print(f"Specificity      : {spec * 100:.2f}% (Benign)")
    print(f"Balanced Accuracy: {bal_acc * 100:.2f}%")
    print(f"F1-Score         : {f1 * 100:.2f}%")
    print(f"ROC-AUC          : {auc:.4f}")
    print(f"Confusion Matrix (TN={tn}, FP={fp}, FN={fn}, TP={tp}):")
    print(np.array(cm))
    print(f"\nSaved artifacts to {output_dir}/:")
    print(f"  - {pipeline_path}")
    print(f"  - {cols_path}")
    print(f"  - {meta_path}")

    return metadata

if __name__ == "__main__":
    train_breast_cancer_model()
