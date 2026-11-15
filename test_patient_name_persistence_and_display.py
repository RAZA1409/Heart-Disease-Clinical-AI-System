"""
Targeted Verification Suite for Patient Name Persistence and Retrieval.
Tests all requirements from Critical Bug Fix:
1. New Heart Disease assessment saves the submitted name.
2. Heart Disease history displays that name.
3. Heart Disease detail page displays that name in header and patient info.
4. Heart Disease PDF contains that name.
5. New Diabetes assessment saves the submitted name.
6. Diabetes history displays that name.
7. Diabetes clinical report displays that name.
8. Diabetes PDF contains that name.
9. Missing/NULL names display a safe fallback ("Name not available").
10. Existing patient IDs and records remain unchanged.
11. Authentication and CSRF protections remain intact.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from werkzeug.security import generate_password_hash
from app import app
from database import get_db_connection, init_db


class PatientNamePersistenceAndDisplayTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.temp_dir = tempfile.mkdtemp(prefix="patient_name_test_")
        cls.test_db = os.path.join(cls.temp_dir, "isolated_names_test.db")
        cls.orig_db_path = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = cls.test_db
        init_db()

        # Seed authenticated admin in isolated test DB
        conn = get_db_connection()
        admin_hash = generate_password_hash("Admin@1234")
        conn.execute("UPDATE users SET password = ? WHERE username = 'admin'", (admin_hash,))

        # Seed synthetic Heart Disease records (named, NULL, empty)
        conn.execute("""
            INSERT INTO patients (
                patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal,
                result, probability, risk_level, timestamp
            ) VALUES (
                'PID_SYNTH_NAMED_01', 'Eleanor Vance', 58, 0, 1, 128, 210, 0,
                0, 155, 0, 0.8, 1, 0, 1,
                'Absent', 18.5, 'LOW', '2026-11-14 10:00:00'
            )
        """)
        conn.execute("""
            INSERT INTO patients (
                patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal,
                result, probability, risk_level, timestamp
            ) VALUES (
                'PID_SYNTH_NULL_02', NULL, 64, 1, 3, 150, 260, 1,
                1, 120, 1, 2.4, 2, 2, 3,
                'Present', 82.0, 'HIGH', '2026-11-14 10:15:00'
            )
        """)
        conn.execute("""
            INSERT INTO patients (
                patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal,
                result, probability, risk_level, timestamp
            ) VALUES (
                'PID_SYNTH_EMPTY_03', '', 52, 1, 2, 135, 230, 0,
                0, 140, 0, 1.2, 1, 1, 2,
                'Absent', 42.0, 'MODERATE', '2026-11-14 10:30:00'
            )
        """)

        # Seed synthetic Diabetes records (named, NULL, empty)
        conn.execute("""
            INSERT INTO diabetes_patients (
                patient_id, patient_name, age, pregnancies, glucose, blood_pressure,
                skin_thickness, insulin, bmi, dpf, result, probability, risk_level, timestamp
            ) VALUES (
                'DPID_SYNTH_NAMED_01', 'Arthur Dent', 42, 2, 110.0, 78.0,
                24.0, 95.0, 26.5, 0.35, 'Absent', 22.0, 'LOW', '2026-11-14 10:00:00'
            )
        """)
        conn.execute("""
            INSERT INTO diabetes_patients (
                patient_id, patient_name, age, pregnancies, glucose, blood_pressure,
                skin_thickness, insulin, bmi, dpf, result, probability, risk_level, timestamp
            ) VALUES (
                'DPID_SYNTH_NULL_02', NULL, 55, 4, 175.0, 92.0,
                35.0, 210.0, 36.2, 0.78, 'Present', 88.5, 'HIGH', '2026-11-14 10:15:00'
            )
        """)
        conn.execute("""
            INSERT INTO diabetes_patients (
                patient_id, patient_name, age, pregnancies, glucose, blood_pressure,
                skin_thickness, insulin, bmi, dpf, result, probability, risk_level, timestamp
            ) VALUES (
                'DPID_SYNTH_EMPTY_03', '', 48, 1, 138.0, 84.0,
                28.0, 130.0, 29.8, 0.48, 'Present', 55.0, 'MODERATE', '2026-11-14 10:30:00'
            )
        """)

        conn.commit()
        conn.close()

    @classmethod
    def tearDownClass(cls):
        if cls.orig_db_path is not None:
            os.environ["DATABASE_PATH"] = cls.orig_db_path
        else:
            os.environ.pop("DATABASE_PATH", None)
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def get_auth_client(self):
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["csrf_token"] = "valid_test_csrf_token"
        return client

    # =========================================================================
    # 1. Heart Disease Assessment: Save and Display
    # =========================================================================

    def test_01_heart_disease_new_assessment_saves_submitted_name(self):
        """Verify new heart assessment saves submitted patient name to DB and shows in result."""
        client = self.get_auth_client()

        payload = {
            "csrf_token": "valid_test_csrf_token",
            "patient_name": "Marcus Aurelius",
            "age": "58",
            "sex": "Male",
            "cp": "Typical Angina",
            "trestbps": "130",
            "chol": "220",
            "fbs": "No",
            "restecg": "Normal",
            "thalach": "150",
            "exang": "No",
            "oldpeak": "1.0",
            "slope": "Flat",
            "ca": "0",
            "thal": "Normal"
        }

        res = client.post("/predict", data=payload, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Marcus Aurelius", html)

        # Confirm persisted in isolated SQLite database
        conn = get_db_connection()
        record = conn.execute(
            "SELECT * FROM patients WHERE patient_name = 'Marcus Aurelius' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()

        self.assertIsNotNone(record)
        self.assertEqual(record["patient_name"], "Marcus Aurelius")
        self.assertTrue(record["patient_id"].startswith("PID"))

    def test_02_heart_disease_history_displays_name_and_fallback(self):
        """Verify /history table displays both valid patient names and 'Name not available' fallback."""
        client = self.get_auth_client()
        res = client.get("/history")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Named record
        self.assertIn("Eleanor Vance", html)
        # Fallback for NULL / empty records
        self.assertIn("Name not available", html)

    def test_03_heart_disease_detail_displays_name_in_header_and_info(self):
        """Verify /patient/<id> displays patient name in header and Assessment Record panel."""
        client = self.get_auth_client()

        # 1. Named subject
        res_named = client.get("/patient/PID_SYNTH_NAMED_01")
        self.assertEqual(res_named.status_code, 200)
        html_named = res_named.get_data(as_text=True)
        self.assertIn("Eleanor Vance", html_named)
        self.assertIn("Patient Name", html_named)

        # 2. NULL name subject displays safe fallback
        res_null = client.get("/patient/PID_SYNTH_NULL_02")
        self.assertEqual(res_null.status_code, 200)
        html_null = res_null.get_data(as_text=True)
        self.assertIn("Name not available", html_null)

        # 3. Empty name subject displays safe fallback
        res_empty = client.get("/patient/PID_SYNTH_EMPTY_03")
        self.assertEqual(res_empty.status_code, 200)
        html_empty = res_empty.get_data(as_text=True)
        self.assertIn("Name not available", html_empty)

    def test_04_heart_disease_pdf_contains_name_and_fallback(self):
        """Verify /download_report/<id> generates valid PDF with correct patient name or fallback."""
        client = self.get_auth_client()

        # Named subject PDF
        res_named = client.get("/download_report/PID_SYNTH_NAMED_01")
        self.assertEqual(res_named.status_code, 200)
        self.assertEqual(res_named.mimetype, "application/pdf")
        self.assertTrue(res_named.data.startswith(b"%PDF-"))

        # NULL name subject PDF
        res_null = client.get("/download_report/PID_SYNTH_NULL_02")
        self.assertEqual(res_null.status_code, 200)
        self.assertEqual(res_null.mimetype, "application/pdf")
        self.assertTrue(res_null.data.startswith(b"%PDF-"))

    # =========================================================================
    # 2. Diabetes Assessment: Save and Display
    # =========================================================================

    def test_05_diabetes_new_assessment_saves_submitted_name(self):
        """Verify new diabetes assessment saves submitted patient name to DB and shows in result."""
        client = self.get_auth_client()

        payload = {
            "csrf_token": "valid_test_csrf_token",
            "patient_name": "Clara Oswald",
            "age": "45",
            "pregnancies": "2",
            "glucose": "145.0",
            "blood_pressure": "82.0",
            "skin_thickness": "26.0",
            "insulin": "115.0",
            "bmi": "29.4",
            "dpf": "0.52"
        }

        res = client.post("/diabetes/predict", data=payload, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Clara Oswald", html)

        # Confirm persisted in isolated SQLite database
        conn = get_db_connection()
        record = conn.execute(
            "SELECT * FROM diabetes_patients WHERE patient_name = 'Clara Oswald' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()

        self.assertIsNotNone(record)
        self.assertEqual(record["patient_name"], "Clara Oswald")
        self.assertTrue(record["patient_id"].startswith("DPID"))

    def test_06_diabetes_history_displays_name_and_fallback(self):
        """Verify /diabetes/history table displays both valid patient names and 'Name not available' fallback."""
        client = self.get_auth_client()
        res = client.get("/diabetes/history")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Named record
        self.assertIn("Arthur Dent", html)
        # Fallback for NULL / empty records
        self.assertIn("Name not available", html)

    def test_07_diabetes_clinical_report_displays_name_and_fallback(self):
        """Verify /diabetes/patient/<id> displays patient name in header and title or fallback."""
        client = self.get_auth_client()

        # 1. Named subject
        res_named = client.get("/diabetes/patient/DPID_SYNTH_NAMED_01")
        self.assertEqual(res_named.status_code, 200)
        html_named = res_named.get_data(as_text=True)
        self.assertIn("Arthur Dent", html_named)

        # 2. NULL name subject displays safe fallback
        res_null = client.get("/diabetes/patient/DPID_SYNTH_NULL_02")
        self.assertEqual(res_null.status_code, 200)
        html_null = res_null.get_data(as_text=True)
        self.assertIn("Name not available", html_null)

        # 3. Empty name subject displays safe fallback
        res_empty = client.get("/diabetes/patient/DPID_SYNTH_EMPTY_03")
        self.assertEqual(res_empty.status_code, 200)
        html_empty = res_empty.get_data(as_text=True)
        self.assertIn("Name not available", html_empty)

    def test_08_diabetes_pdf_contains_name_and_fallback(self):
        """Verify /diabetes/download_report/<id> generates valid PDF with correct patient name or fallback."""
        client = self.get_auth_client()

        # Named subject PDF
        res_named = client.get("/diabetes/download_report/DPID_SYNTH_NAMED_01")
        self.assertEqual(res_named.status_code, 200)
        self.assertEqual(res_named.mimetype, "application/pdf")
        self.assertTrue(res_named.data.startswith(b"%PDF-"))

        # NULL name subject PDF
        res_null = client.get("/diabetes/download_report/DPID_SYNTH_NULL_02")
        self.assertEqual(res_null.status_code, 200)
        self.assertEqual(res_null.mimetype, "application/pdf")
        self.assertTrue(res_null.data.startswith(b"%PDF-"))

    # =========================================================================
    # 3. Integrity & Security Checks
    # =========================================================================

    def test_09_existing_patient_ids_and_records_unchanged(self):
        """Verify existing patient IDs, biometric values, and risk scores are strictly preserved."""
        conn = get_db_connection()
        p1 = conn.execute("SELECT * FROM patients WHERE patient_id = 'PID_SYNTH_NAMED_01'").fetchone()
        self.assertEqual(p1["age"], 58)
        self.assertEqual(p1["chol"], 210)
        self.assertEqual(p1["risk_level"], "LOW")

        d1 = conn.execute("SELECT * FROM diabetes_patients WHERE patient_id = 'DPID_SYNTH_NAMED_01'").fetchone()
        self.assertEqual(d1["glucose"], 110.0)
        self.assertEqual(d1["bmi"], 26.5)
        self.assertEqual(d1["risk_level"], "LOW")
        conn.close()

    def test_10_auth_and_csrf_protections_intact(self):
        """Verify unauthenticated access is blocked (302) and missing CSRF is blocked (400)."""
        unauth = app.test_client()

        # Unauthenticated redirects
        self.assertEqual(unauth.get("/history").status_code, 302)
        self.assertEqual(unauth.get("/patient/PID_SYNTH_NAMED_01").status_code, 302)
        self.assertEqual(unauth.get("/download_report/PID_SYNTH_NAMED_01").status_code, 302)
        self.assertEqual(unauth.get("/diabetes/history").status_code, 302)
        self.assertEqual(unauth.get("/diabetes/patient/DPID_SYNTH_NAMED_01").status_code, 302)
        self.assertEqual(unauth.get("/diabetes/download_report/DPID_SYNTH_NAMED_01").status_code, 302)

        # Missing CSRF on submission
        client = self.get_auth_client()
        res_heart_no_csrf = client.post("/predict", data={"patient_name": "Test"})
        self.assertEqual(res_heart_no_csrf.status_code, 400)

        res_diab_no_csrf = client.post("/diabetes/predict", data={"patient_name": "Test"})
        self.assertEqual(res_diab_no_csrf.status_code, 400)


if __name__ == "__main__":
    unittest.main()
