# Heart Disease Model Training and Pipeline Architecture

**Module:** Cardiovascular Disease Clinical AI  
**Authoritative Training Pipeline:** `model_comparison.py`  
**Target Artifact:** `models/best_model.pkl`  

---

## 1. Authoritative Training Command

To train, evaluate, and select the production Heart Disease model artifacts, execute:

```bash
python model_comparison.py
```

This command runs the deterministic multi-model evaluation pipeline, selects the best performing classifier on held-out test data, and writes the authoritative production artifacts to `models/`.

---

## 2. Dataset and Preprocessing Split Strategy

- **Dataset File:** `Heart_Disease_Cleaned.csv`
- **Total Records:** 270 clinical observations
- **Target Variable:** `Heart Disease` (binary: 0 = Absence, 1 = Presence)
- **Input Features (13):**
  1. `Age`
  2. `Sex` (1 = Male, 0 = Female)
  3. `Chest pain type` (1 = Typical Angina, 2 = Atypical Angina, 3 = Non-Anginal Pain, 4 = Asymptomatic)
  4. `BP` (Resting blood pressure in mmHg)
  5. `Cholesterol` (Serum cholesterol in mg/dL)
  6. `FBS over 120` (Fasting blood sugar > 120 mg/dL: 1 = Yes, 0 = No)
  7. `EKG results` (0 = Normal, 1 = ST-T Abnormality, 2 = LVH)
  8. `Max HR` (Maximum heart rate achieved)
  9. `Exercise angina` (1 = Yes, 0 = No)
  10. `ST depression` (ST depression induced by exercise relative to rest)
  11. `Slope of ST` (1 = Upsloping, 2 = Flat, 3 = Downsloping)
  12. `Number of vessels fluro` (0–3 fluoroscopy-colored major vessels)
  13. `Thallium` (3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect)

### Split Strategy
- **Train / Test Ratio:** 80% train (216 samples) / 20% test (54 samples)
- **Random State:** `random_state=42`
- **Stratification:** None (unstratified split matching baseline production distribution)
- **Feature Scaling:** `sklearn.preprocessing.StandardScaler` fitted strictly on `X_train` and applied to `X_test` to prevent data leakage.

---

## 3. Model Selection Procedure

`model_comparison.py` trains and evaluates three candidate classifiers under identical training data and scaling:

1. **Logistic Regression:** `LogisticRegression(max_iter=1000)`
2. **Support Vector Classifier (RBF):** `SVC(probability=True)`
3. **Random Forest Classifier:** `RandomForestClassifier(n_estimators=100)`

### Selection Rule
The pipeline programmatically evaluates held-out test accuracy across all candidates:
```python
best_model_name = max(results, key=lambda x: results[x]["test"])
best_model = models[best_model_name]
```

### Baseline Performance
- **Selected Model:** Logistic Regression
- **Test Accuracy:** **90.74%** (49 / 54 correct predictions)
- **Confusion Matrix:** TN = 31, FP = 2, FN = 3, TP = 18
- **Precision:** 90.00% (18 / 20)
- **Recall (Sensitivity):** 85.71% (18 / 21)
- **Specificity:** 93.94% (31 / 33)
- **Log Loss:** Train Loss = 0.3497, Test Loss = 0.3167

---

## 4. Production Artifact Ownership

The following four artifacts are strictly owned by `model_comparison.py` and are loaded by the production web application (`app.py`):

| File Path | Description | Required By Production |
|---|---|---|
| `models/best_model.pkl` | Fitted Logistic Regression model | `app.py` (`_load_required_artifact`) |
| `models/scaler.pkl` | `StandardScaler` fitted on `X_train` (216 samples) | `app.py` (`_load_required_artifact`) |
| `models/feature_columns.pkl` | List of 13 feature names in exact order | `app.py` (`_load_required_artifact`) |
| `models/model_info.pkl` | Evaluation metrics, ROC/PRF coordinates, confusion matrix | `app.py` (Dashboard analytics) |

---

## 5. Experimental Artifact Ownership (GA Study)

`train_model.py` is an offline research script comparing a Genetic Algorithm (GA) against Grid Search for SVM hyperparameter optimization. Its artifacts are completely separated from production:

| File Path | Description | Role |
|---|---|---|
| `models/ga_model.pkl` | GA-optimized Support Vector Classifier | Research artifact |
| `models/ga_scaler.pkl` | Scaler fitted on stratified split | Research artifact |
| `models/ga_feature_columns.pkl` | Feature list for GA study | Research artifact |
| `models/ga_convergence.png` | Plot of GA fitness across 25 generations | Research visualization |
| `models/optimization_comparison.pkl` | GA vs Grid Search test accuracy comparison | Dashboard `optChart` display |

---

## 6. Safety Warning: Script Separation

> [!WARNING]
> **DO NOT** configure experimental scripts (such as `train_model.py`) to write to `models/scaler.pkl`, `models/feature_columns.pkl`, or `models/best_model.pkl`.
>
> `train_model.py` uses a **stratified** split (`stratify=y`), whereas the production Logistic Regression model in `best_model.pkl` requires the scaler fitted on the unstratified split in `model_comparison.py`.
>
> Overwriting `models/scaler.pkl` with a scaler from a different train/test split will cause a **silent feature distribution skew** at inference time, distorting clinical risk predictions.

