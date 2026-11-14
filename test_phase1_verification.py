"""
Targeted Verification Suite for Phase 1 Stabilization.
Tests all 8 essential checks requested by user.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from werkzeug.security import generate_password_hash
from app import app
from database import init_db, get_db_connection
from risk_config import classify_risk, LOW_RISK_THRESHOLD, MODERATE_RISK_THRESHOLD


class Phase1VerificationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. Capture original environment & app configuration
        cls.orig_testing = app.config.get("TESTING")
        cls.orig_csrf = app.config.get("WTF_CSRF_ENABLED")
        cls.orig_db_path = os.environ.get("DATABASE_PATH")
        cls.temp_dir = None

        try:
            app.config['TESTING'] = True
            app.config['WTF_CSRF_ENABLED'] = False

            # 2. Create isolated temporary test directory and SQLite database
            cls.temp_dir = tempfile.mkdtemp(prefix="phase1_isolated_test_")
            cls.test_db = os.path.join(cls.temp_dir, "test_phase1.db")
            os.environ["DATABASE_PATH"] = cls.test_db

            # 3. Initialize complete application schema in the isolated database
            init_db()

            # 4. Seed test-only admin user with known hashed credentials
            conn = get_db_connection()
            conn.execute("DELETE FROM users")  # Remove default seed from init_db()
            test_admin_hash = generate_password_hash("Admin@1234")
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                ("admin", test_admin_hash)
            )

            # 5. Seed synthetic patient fixtures to test data preservation (test_08) and history/dashboard (test_07)
            synthetic_patients = [
                ("PID_TEST_001", "Shivansh", 45, 1, 1, 130, 220, 0, 0, 150, 0, 1.2, 1, 0, 1, "Absent", 20.0, "LOW", "2026-11-14 10:00:00"),
                ("PID_TEST_002", "Sanjana", 52, 0, 2, 125, 210, 0, 1, 140, 0, 0.8, 1, 0, 1, "Absent", 25.0, "LOW", "2026-11-14 10:05:00"),
                ("PID_TEST_003", "Kohli", 38, 1, 0, 140, 240, 1, 0, 160, 1, 2.0, 2, 1, 2, "Present", 75.0, "HIGH", "2026-11-14 10:10:00"),
            ]
            for p in synthetic_patients:
                conn.execute("""
                    INSERT INTO patients (
                        patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal,
                        result, probability, risk_level, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, p)
            conn.commit()
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

    def test_01_app_starts_successfully(self):
        """1. Application starts successfully."""
        self.assertIsNotNone(app)
        response = self.client.get("/")
        # Unauthenticated request should redirect to login
        self.assertIn(response.status_code, (302, 200))

    def test_02_login_and_logout_work(self):
        """2. Login and logout work."""
        # GET login page to obtain CSRF token
        get_res = self.client.get("/login")
        self.assertEqual(get_res.status_code, 200)

        with self.client.session_transaction() as sess:
            csrf_tok = sess.get("csrf_token")

        # POST login credentials with valid CSRF token
        login_res = self.client.post("/login", data={
            "username": "admin",
            "password": "Admin@1234",
            "csrf_token": csrf_tok
        }, follow_redirects=True)
        self.assertEqual(login_res.status_code, 200)
        self.assertIn(b"Clinical AI Dashboard", login_res.data)

        # Logout
        logout_res = self.client.get("/logout", follow_redirects=True)
        self.assertEqual(logout_res.status_code, 200)
        self.assertIn(b"Doctor Authentication Portal", logout_res.data)

    def test_03_unauthenticated_user_cannot_delete(self):
        """3. Unauthenticated users cannot delete patient records."""
        # Unauthenticated client attempting to POST delete
        unauth_client = app.test_client()
        res = unauth_client.post("/delete/PID20260401232937", data={
            "csrf_token": "dummy"
        })
        # Should be blocked either by CSRF (400) or session redirect (302)
        self.assertIn(res.status_code, (302, 400))

        # Even with CSRF token in session, unauthenticated delete redirects to login
        with unauth_client.session_transaction() as sess:
            sess["csrf_token"] = "valid_tok"
        res2 = unauth_client.post("/delete/PID20260401232937", data={
            "csrf_token": "valid_tok"
        })
        self.assertEqual(res2.status_code, 302)
        self.assertTrue(res2.headers["Location"].endswith("/login") or "/login" in res2.headers["Location"])

    def test_04_patient_debug_endpoint_disabled(self):
        """4. Patient debug endpoint does not expose clinical data."""
        res = self.client.get("/test_patient_context/PID20260401232937")
        self.assertEqual(res.status_code, 404)
        self.assertNotIn(b"Shivansh", res.data)
        self.assertNotIn(b"Blood Pressure", res.data)

    def test_05_prediction_works_with_compatible_model(self):
        """5. Existing heart disease prediction works with compatible model artifacts."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["csrf_token"] = "test_token"

        payload = {
            "patient_name": "Test Patient",
            "age": "58",
            "sex": "Male",
            "cp": "Typical Angina",
            "trestbps": "140",
            "chol": "260",
            "fbs": "No",
            "restecg": "Normal",
            "thalach": "130",
            "exang": "Yes",
            "oldpeak": "1.8",
            "slope": "Flat",
            "ca": "2",
            "thal": "Reversible Defect",
            "csrf_token": "test_token"
        }
        res = self.client.post("/predict", data=payload, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Test Patient", res.data)
        self.assertIn(b"AI Clinical Assessment Result", res.data)

    def test_06_risk_classification_consistency(self):
        """6. Risk classification is consistent."""
        self.assertEqual(classify_risk(15.0), "LOW")
        self.assertEqual(classify_risk(29.9), "LOW")
        self.assertEqual(classify_risk(30.0), "MODERATE")
        self.assertEqual(classify_risk(50.0), "MODERATE")
        self.assertEqual(classify_risk(69.9), "MODERATE")
        self.assertEqual(classify_risk(70.0), "HIGH")
        self.assertEqual(classify_risk(95.5), "HIGH")

    def test_07_patient_history_and_dashboard_work(self):
        """7. Patient history and dashboard still work."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"

        dash_res = self.client.get("/dashboard")
        self.assertEqual(dash_res.status_code, 200)
        self.assertIn(b"Clinical AI Dashboard", dash_res.data)
        self.assertIn(b"Logistic Regression", dash_res.data)

        hist_res = self.client.get("/history")
        self.assertEqual(hist_res.status_code, 200)
        self.assertIn(b"Patient History Records", hist_res.data)

    def test_08_database_data_preserved(self):
        """8. Database data is preserved (verifying synthetic patient fixtures remain intact)."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT patient_name FROM patients")
        names = [r["patient_name"] for r in cursor.fetchall()]
        conn.close()

        # Verify existing synthetic patient fixtures from setup are preserved
        self.assertIn("Shivansh", names)
        self.assertIn("Sanjana", names)
        self.assertIn("Kohli", names)

if __name__ == "__main__":
    unittest.main()
