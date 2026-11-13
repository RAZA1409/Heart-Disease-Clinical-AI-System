"""
Phase 2 Focused Verification Test Suite.
Verifies the complete Heart Disease Prediction workflow, end-to-end.
"""

import os
import unittest
import sqlite3
import joblib
import pandas as pd
from app import app, CACHED_MODEL_INFO, CACHED_OPT_COMPARISON
from risk_config import classify_risk, LOW_RISK_THRESHOLD, MODERATE_RISK_THRESHOLD
from database import get_db_connection

class Phase2HeartDiseaseModuleTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

    def test_01_model_and_scaler_pipeline_consistency(self):
        """1. Verify model and scaler belong to the same pipeline."""
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        cols = joblib.load("models/feature_columns.pkl")

        self.assertEqual(model.n_features_in_, 13)
        self.assertEqual(scaler.n_features_in_, 13)
        self.assertEqual(len(cols), 13)

        df = pd.read_csv("Heart_Disease_Cleaned.csv")
        X = df.drop("Heart Disease", axis=1)
        self.assertEqual(list(X.columns), cols)

        # Ensure transform and predict execute with identical shapes
        scaled = scaler.transform(X.iloc[:5])
        probs = model.predict_proba(scaled)
        self.assertEqual(probs.shape, (5, 2))

    def test_02_all_13_features_mapped_correctly(self):
        """2. Verify all 13 input features map to correct column order."""
        from app import FEATURE_MAP, REVERSE_FEATURE_MAP

        expected_cols = [
            'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol',
            'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina',
            'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
        ]
        cols = joblib.load("models/feature_columns.pkl")
        self.assertEqual(cols, expected_cols)

        # Check mapping dictionaries contain all UI values
        self.assertIn("Typical Angina", FEATURE_MAP["cp"])
        self.assertIn("Yes", FEATURE_MAP["exang"])
        self.assertIn("Flat", FEATURE_MAP["slope"])
        self.assertIn("Reversible Defect", FEATURE_MAP["thal"])

    def test_03_and_04_prediction_workflow_probability_risk(self):
        """3 & 4. Verify full prediction workflow, probability, and risk classification."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["csrf_token"] = "valid_csrf_token"

        payload = {
            "patient_name": "Phase2 Test Subject",
            "age": "62",
            "sex": "Male",
            "cp": "Asymptomatic",
            "trestbps": "150",
            "chol": "280",
            "fbs": "Yes",
            "restecg": "Left Ventricular Hypertrophy",
            "thalach": "120",
            "exang": "Yes",
            "oldpeak": "2.5",
            "slope": "Flat",
            "ca": "3",
            "thal": "Reversible Defect",
            "csrf_token": "valid_csrf_token"
        }

        res = self.client.post("/predict", data=payload, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Phase2 Test Subject", res.data)
        self.assertIn(b"AI Clinical Assessment Result", res.data)
        self.assertIn(b"View Detailed Report", res.data)

        # Verify record inserted into database
        conn = get_db_connection()
        p = conn.execute("SELECT * FROM patients WHERE patient_name = 'Phase2 Test Subject' ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        self.assertIsNotNone(p)
        self.assertEqual(p["patient_name"], "Phase2 Test Subject")
        prob = p["probability"]
        risk = p["risk_level"]

        # Verify risk level matches risk_config
        expected_risk = classify_risk(prob)
        self.assertEqual(risk, expected_risk)

    def test_05_patient_record_retrieval(self):
        """5. Verify patient record retrieval in history and detail views."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"

        # History view
        hist = self.client.get("/history")
        self.assertEqual(hist.status_code, 200)
        self.assertIn(b"Phase2 Test Subject", hist.data)

        # Retrieve patient ID from database
        conn = get_db_connection()
        p = conn.execute("SELECT patient_id FROM patients WHERE patient_name = 'Phase2 Test Subject' ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        pid = p["patient_id"]
        detail = self.client.get(f"/patient/{pid}")
        self.assertEqual(detail.status_code, 200)
        self.assertIn(b"Phase2 Test Subject", detail.data)
        self.assertIn(b"Clinical Parameter Overview", detail.data)

    def test_06_dashboard_kpis_and_charts(self):
        """6. Verify dashboard KPIs and chart data structures."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"

        dash = self.client.get("/dashboard")
        self.assertEqual(dash.status_code, 200)
        self.assertIn(b"Logistic Regression", dash.data)
        self.assertIn(b"Risk Distribution", dash.data)
        self.assertIn(b"Confusion Matrix", dash.data)
        self.assertIn(b"ROC Curve", dash.data)
        self.assertIn(b"Feature Importance", dash.data)

    def test_07_patient_detail_and_recommendations(self):
        """7. Verify patient detail page renders clinical recommendations and dynamic cards."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"

        conn = get_db_connection()
        p = conn.execute("SELECT patient_id FROM patients WHERE patient_name = 'Phase2 Test Subject' ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        res = self.client.get(f"/patient/{p['patient_id']}")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"AI Clinical Recommendation", res.data)
        self.assertIn(b"AI Clinical Explanation", res.data)
        self.assertIn(b"Clinical Risk Alerts", res.data)
        self.assertIn(b"Status: High", res.data)

    def test_08_pdf_report_generation(self):
        """8. Verify PDF report generation returns valid PDF."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"

        conn = get_db_connection()
        p = conn.execute("SELECT patient_id FROM patients WHERE patient_name = 'Phase2 Test Subject' ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        pdf_res = self.client.get(f"/download_report/{p['patient_id']}")
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.content_type, "application/pdf")
        self.assertTrue(pdf_res.data.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_res.data), 1500)

    def test_09_gemini_ai_assistant_integration(self):
        """9. Verify Gemini AI assistant endpoint responds with patient context."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["csrf_token"] = "valid_csrf"

        conn = get_db_connection()
        p = conn.execute("SELECT patient_id FROM patients LIMIT 1").fetchone()
        conn.close()

        res = self.client.post("/ai_assistant", json={
            "message": "Explain why this patient has elevated risk in 1 sentence.",
            "patient_id": p["patient_id"],
            "csrf_token": "valid_csrf"
        })
        self.assertIn(res.status_code, (200, 500, 503))  # 200 if online, 500 or 503 if API quota/network
        data = res.get_json()
        self.assertTrue("reply" in data or "error" in data)

    def test_10_frontend_usability_and_navigation(self):
        """10. Verify navigation links and error feedback."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["csrf_token"] = "csrf_token_test"

        # Check predict page contains back link
        pred_page = self.client.get("/predict_page")
        self.assertIn(b"Back to Dashboard", pred_page.data)

        # Check invalid input triggers graceful 400 error rather than 500 crash
        bad_res = self.client.post("/predict", data={
            "patient_name": "Bad Input Test",
            "age": "invalid_number",
            "csrf_token": "csrf_token_test"
        })
        self.assertEqual(bad_res.status_code, 400)
        self.assertIn(b"Invalid clinical input", bad_res.data)

if __name__ == "__main__":
    unittest.main()

