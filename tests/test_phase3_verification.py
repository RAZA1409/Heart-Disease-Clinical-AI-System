"""
Phase 3 Verification Test Suite
Multi-Disease Clinical AI Platform - Diabetes Prediction Module
"""

import unittest
import os
import io
import shutil
import tempfile
import joblib
import pandas as pd
from werkzeug.security import generate_password_hash
from app import app
from database import get_db_connection, init_db
from risk_config import classify_risk, LOW_RISK_THRESHOLD, MODERATE_RISK_THRESHOLD

class Phase3DiabetesModuleVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.orig_testing = app.config.get("TESTING")
        cls.orig_csrf = app.config.get("WTF_CSRF_ENABLED")
        cls.orig_db_path = os.environ.get("DATABASE_PATH")
        cls.temp_dir = None

        try:
            app.config['TESTING'] = True
            app.config['WTF_CSRF_ENABLED'] = False

            # Create isolated temporary directory & SQLite database
            cls.temp_dir = tempfile.mkdtemp(prefix="phase3_isolated_test_")
            cls.test_db = os.path.join(cls.temp_dir, "test_phase3.db")
            os.environ["DATABASE_PATH"] = cls.test_db

            # Initialize complete application schema in the isolated database
            init_db()

            # Seed test-only admin user with known hashed credentials
            conn = get_db_connection()
            try:
                conn.execute("DELETE FROM users")
                test_admin_hash = generate_password_hash("Admin@1234")
                conn.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    ("admin", test_admin_hash)
                )

                # Seed synthetic heart disease patient fixtures (satisfies test_03 heart_count >= 1)
                conn.execute("""
                    INSERT INTO patients (
                        patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal,
                        result, probability, risk_level, timestamp
                    ) VALUES (
                        'PID_PHASE3_SYNTH_01', 'Synthetic Heart Subject', 50, 1, 1, 130, 220, 0,
                        0, 150, 0, 1.0, 1, 0, 1, 'Absent', 22.5, 'LOW', '2026-11-14 10:00:00'
                    )
                """)
                conn.commit()
            finally:
                conn.close()

            cls.client = app.test_client()
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        # 1. Restore DATABASE_PATH environment variable
        if hasattr(cls, "orig_db_path"):
            if cls.orig_db_path is not None:
                os.environ["DATABASE_PATH"] = cls.orig_db_path
            else:
                os.environ.pop("DATABASE_PATH", None)

        # 2. Restore application configuration
        if hasattr(cls, "orig_testing"):
            app.config["TESTING"] = cls.orig_testing
        if hasattr(cls, "orig_csrf"):
            app.config["WTF_CSRF_ENABLED"] = cls.orig_csrf

        # 3. Safely delete the temporary test directory
        if getattr(cls, "temp_dir", None) and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def get_authenticated_session(self):
        client = app.test_client()
        login_page = client.get("/login")
        html = login_page.get_data(as_text=True)
        token = ""
        for line in html.splitlines():
            if 'name="csrf_token"' in line:
                token = line.split('value="')[1].split('"')[0]
                break

        client.post("/login", data={
            "username": "admin",
            "password": "Admin@1234",
            "csrf_token": token
        }, follow_redirects=True)
        return client

    def get_csrf_token(self, client, url):
        res = client.get(url)
        html = res.get_data(as_text=True)
        for line in html.splitlines():
            if 'name="csrf_token"' in line:
                return line.split('value="')[1].split('"')[0]
        return ""

    def test_01_diabetes_artifacts_exist_and_load(self):
        """Verify diabetes model, feature columns, and metadata artifacts exist and load correctly."""
        self.assertTrue(os.path.exists("models/diabetes_pipeline.pkl"), "diabetes_pipeline.pkl missing")
        self.assertTrue(os.path.exists("models/diabetes_feature_columns.pkl"), "diabetes_feature_columns.pkl missing")
        self.assertTrue(os.path.exists("models/diabetes_model_info.pkl"), "diabetes_model_info.pkl missing")

        pipe = joblib.load("models/diabetes_pipeline.pkl")
        cols = joblib.load("models/diabetes_feature_columns.pkl")
        info = joblib.load("models/diabetes_model_info.pkl")

        self.assertIsNotNone(pipe, "Pipeline failed to load")
        self.assertEqual(len(cols), 8, f"Expected 8 features, got {len(cols)}")
        self.assertIn("accuracy", info)
        self.assertIn("roc_auc", info)
        self.assertIn("model_name", info)
        print(f"Loaded Diabetes Model: {info['model_name']}, Accuracy: {info['accuracy']*100:.2f}%, ROC-AUC: {info['roc_auc']:.4f}")

    def test_02_pipeline_inference(self):
        """Verify pipeline inference handles standard and zero/missing values correctly."""
        pipe = joblib.load("models/diabetes_pipeline.pkl")

        # Low risk input
        low_risk_df = pd.DataFrame([{
            'Pregnancies': 1,
            'Glucose': 85.0,
            'BloodPressure': 66.0,
            'SkinThickness': 20.0,
            'Insulin': 75.0,
            'BMI': 22.0,
            'DiabetesPedigreeFunction': 0.25,
            'Age': 24
        }])
        prob_low = pipe.predict_proba(low_risk_df)[0][1]
        self.assertTrue(0.0 <= prob_low <= 1.0, f"Invalid probability: {prob_low}")

        # High risk input
        high_risk_df = pd.DataFrame([{
            'Pregnancies': 6,
            'Glucose': 185.0,
            'BloodPressure': 92.0,
            'SkinThickness': 35.0,
            'Insulin': 240.0,
            'BMI': 38.5,
            'DiabetesPedigreeFunction': 0.95,
            'Age': 54
        }])
        prob_high = pipe.predict_proba(high_risk_df)[0][1]
        self.assertTrue(0.0 <= prob_high <= 1.0, f"Invalid probability: {prob_high}")
        self.assertGreater(prob_high, prob_low, "High risk input must yield higher probability than low risk")

        # Zero value imputation test (clinically missing values should not crash)
        zero_impute_df = pd.DataFrame([{
            'Pregnancies': 0,
            'Glucose': 110.0,
            'BloodPressure': 0.0, # missing BP
            'SkinThickness': 0.0, # missing skin thickness
            'Insulin': 0.0,       # missing insulin
            'BMI': 0.0,           # missing BMI
            'DiabetesPedigreeFunction': 0.3,
            'Age': 30
        }])
        prob_imputed = pipe.predict_proba(zero_impute_df)[0][1]
        self.assertTrue(0.0 <= prob_imputed <= 1.0, "Imputation pipeline failed on 0 values")

    def test_03_database_table_and_schema(self):
        """Verify diabetes_patients database table exists with all required columns and preserves heart disease records."""
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check diabetes_patients table
        cursor.execute("PRAGMA table_info(diabetes_patients)")
        columns = [row[1] for row in cursor.fetchall()]
        expected_cols = [
            'id', 'patient_id', 'patient_name', 'age', 'pregnancies',
            'glucose', 'blood_pressure', 'skin_thickness', 'insulin',
            'bmi', 'dpf', 'result', 'probability', 'risk_level', 'timestamp'
        ]
        for col in expected_cols:
            self.assertIn(col, columns, f"Column {col} missing in diabetes_patients table")

        # Check heart disease patients table preserved
        cursor.execute("SELECT COUNT(*) FROM patients")
        heart_count = cursor.fetchone()[0]
        self.assertGreaterEqual(heart_count, 1, "Existing heart disease records must be preserved")
        conn.close()

    def test_04_routes_require_authentication(self):
        """Verify unauthenticated requests to diabetes routes are redirected to login."""
        client = app.test_client()

        routes = [
            "/diabetes/predict",
            "/diabetes_predict_page",
            "/diabetes/history",
            "/diabetes/patient/DPID_TEST",
            "/diabetes/download_report/DPID_TEST"
        ]
        for route in routes:
            res = client.get(route)
            self.assertEqual(res.status_code, 302, f"Route {route} did not redirect unauthenticated user")
            self.assertIn("/login", res.headers.get("Location", ""))

        # Check unauthenticated POST deletion
        res = client.post("/diabetes/delete/DPID_TEST")
        # May fail CSRF or redirect to login (both protect the record)
        self.assertTrue(res.status_code in (302, 400), "Unauthenticated delete was not blocked")

    def test_05_prediction_workflow_and_persistence(self):
        """Test full prediction workflow, probability computation, risk classification, and database storage."""
        client = self.get_authenticated_session()
        token = self.get_csrf_token(client, "/diabetes/predict")

        test_patient_name = "Phase3 Test Patient"
        form_data = {
            "csrf_token": token,
            "patient_name": test_patient_name,
            "age": "48",
            "pregnancies": "3",
            "glucose": "165.0",
            "blood_pressure": "88.0",
            "skin_thickness": "32.0",
            "insulin": "180.0",
            "bmi": "33.4",
            "dpf": "0.68"
        }

        res = client.post("/diabetes/predict", data=form_data)
        self.assertEqual(res.status_code, 200, f"Prediction submission failed with status {res.status_code}")
        html = res.get_data(as_text=True)

        self.assertIn(test_patient_name, html)
        self.assertIn("DPID", html)
        self.assertIn("Decision Support Estimate", html)
        self.assertIn("Statistical Risk Tier", html)

        # Extract generated DPID from page
        dpid = ""
        for line in html.splitlines():
            if "DPID" in line:
                parts = line.split("DPID")
                candidate = "DPID" + parts[1].split("<")[0].split('"')[0].strip()
                if len(candidate) >= 18:
                    dpid = candidate
                    break

        self.assertTrue(dpid.startswith("DPID"), f"Failed to extract valid DPID, got: {dpid}")
        # Verify collision resistance (DPID prefix 4 + 14 digits timestamp + 8 hex chars = 26 characters)
        self.assertEqual(len(dpid), 26, f"Expected 26-character collision-resistant ID (DPID + 14-digit timestamp + 8-hex), got: {dpid}")

        # Rapid consecutive assessment test to ensure distinct collision-resistant IDs
        token2 = self.get_csrf_token(client, "/diabetes/predict")
        form_data2 = dict(form_data, patient_name="Concurrent Patient 2", csrf_token=token2)
        res2 = client.post("/diabetes/predict", data=form_data2)
        self.assertEqual(res2.status_code, 200)
        html2 = res2.get_data(as_text=True)
        dpid2 = ""
        for line in html2.splitlines():
            if "DPID" in line:
                parts = line.split("DPID")
                candidate = "DPID" + parts[1].split("<")[0].split('"')[0].strip()
                if len(candidate) >= 18:
                    dpid2 = candidate
                    break
        self.assertNotEqual(dpid, dpid2, f"Collision detected! Both IDs are: {dpid}")

        # Verify record in database
        conn = get_db_connection()
        record = conn.execute("SELECT * FROM diabetes_patients WHERE patient_id = ?", (dpid,)).fetchone()
        conn.close()

        self.assertIsNotNone(record, f"Record for {dpid} not found in diabetes_patients table")
        self.assertEqual(record["patient_name"], test_patient_name)
        self.assertEqual(record["age"], 48)
        self.assertAlmostEqual(record["glucose"], 165.0, places=1)
        self.assertAlmostEqual(record["bmi"], 33.4, places=1)
        self.assertTrue(record["risk_level"] in ("LOW", "MODERATE", "HIGH"))

        # Test patient detail page
        detail_res = client.get(f"/diabetes/patient/{dpid}")
        self.assertEqual(detail_res.status_code, 200)
        detail_html = detail_res.get_data(as_text=True)
        self.assertIn(test_patient_name, detail_html)
        self.assertIn("Plasma Glucose", detail_html)
        self.assertIn("Body Mass Index", detail_html)
        self.assertIn("Clinical Decision Support", detail_html)

        # Test PDF download
        pdf_res = client.get(f"/diabetes/download_report/{dpid}")
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.mimetype, "application/pdf")
        self.assertTrue(pdf_res.data.startswith(b"%PDF"), "Generated report is not a valid PDF")

        # Clean up both test patients
        del_token = self.get_csrf_token(client, "/diabetes/history")
        client.post(f"/diabetes/delete/{dpid}", data={"csrf_token": del_token}, follow_redirects=True)
        del_token2 = self.get_csrf_token(client, "/diabetes/history")
        client.post(f"/diabetes/delete/{dpid2}", data={"csrf_token": del_token2}, follow_redirects=True)

        # Verify deletion from DB
        conn = get_db_connection()
        record_after = conn.execute("SELECT * FROM diabetes_patients WHERE patient_id = ?", (dpid,)).fetchone()
        record2_after = conn.execute("SELECT * FROM diabetes_patients WHERE patient_id = ?", (dpid2,)).fetchone()
        conn.close()
        self.assertIsNone(record_after, "Patient record was not deleted from diabetes_patients table")
        self.assertIsNone(record2_after, "Concurrent patient record was not deleted from diabetes_patients table")

    def test_06_input_validation(self):
        """Test input validation and error rejection for invalid clinical values."""
        client = self.get_authenticated_session()
        token = self.get_csrf_token(client, "/diabetes/predict")

        # Test negative glucose
        bad_data = {
            "csrf_token": token,
            "patient_name": "Invalid Test",
            "age": "30",
            "pregnancies": "0",
            "glucose": "-50",
            "blood_pressure": "80",
            "skin_thickness": "20",
            "insulin": "50",
            "bmi": "22",
            "dpf": "0.3"
        }
        res = client.post("/diabetes/predict", data=bad_data)
        self.assertEqual(res.status_code, 400, "Negative glucose should be rejected with status 400")

        # Test impossible age
        bad_data["glucose"] = "100"
        bad_data["age"] = "150"
        res = client.post("/diabetes/predict", data=bad_data)
        self.assertEqual(res.status_code, 400, "Impossible age should be rejected with status 400")

    def test_07_dashboard_navigation_and_counts(self):
        """Test dashboard renders multi-disease navigation and displays correct counts."""
        client = self.get_authenticated_session()
        res = client.get("/dashboard")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertIn("Multi-Disease Clinical AI Platform", html)
        self.assertIn("Cardiovascular Disease Module", html)
        self.assertIn("Diabetes Mellitus Module", html)
        self.assertIn("/diabetes/predict", html)
        self.assertIn("/diabetes/history", html)

if __name__ == "__main__":
    unittest.main()
