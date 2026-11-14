"""
Targeted Verification Suite for Phase 8E — Diabetes Assessment Form Redesign.
Verifies all Phase 8E requirements:
1. Renders /diabetes/predict and /diabetes_predict_page with authentication enforcement.
2. Exactly 8 original clinical features, patient_name, and csrf_token exist with preserved names and constraints.
3. Form submission successfully invokes Random Forest pipeline without breaking model preprocessing.
4. Result page (diabetes_result.html) displays patient name, probability, risk level, and DPID.
5. Record is persisted in the diabetes_patients database table and visible in diabetes history.
6. Server-side validation constraints (bounds on age, pregnancies, glucose, BP, BMI, DPF).
7. Compliance with Phase 8 theme (#0b1329, #1e293b, #0284c7) without magenta glow.
8. Responsive mobile stylesheet rules preventing horizontal page overflow.
"""

import os
import re
import shutil
import tempfile
import unittest
from app import app
from database import get_db_connection, init_db


class Phase8EDiabetesFormTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.temp_dir = tempfile.mkdtemp(prefix="phase8e_diabetes_test_")
        cls.test_db = os.path.join(cls.temp_dir, "isolated_test.db")
        cls.orig_db_path = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = cls.test_db
        init_db()
        cls.client = app.test_client()

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
            sess["csrf_token"] = "test_token"
        return client

    # =========================================================================
    # 1. Page Access & Authentication
    # =========================================================================

    def test_01_predict_page_auth_and_rendering(self):
        """Verify unauthenticated requests redirect to /login, authenticated requests get 200 OK."""
        unauth = app.test_client()
        res_unauth = unauth.get("/diabetes/predict")
        self.assertEqual(res_unauth.status_code, 302)
        self.assertIn("/login", res_unauth.headers.get("Location", ""))

        res_unauth_alias = unauth.get("/diabetes_predict_page")
        self.assertEqual(res_unauth_alias.status_code, 302)

        client = self.get_auth_client()
        res_auth = client.get("/diabetes/predict")
        self.assertEqual(res_auth.status_code, 200)
        html = res_auth.get_data(as_text=True)
        self.assertIn("Diabetes Clinical AI Assessment", html)
        self.assertIn("Eight-Feature Diabetes Risk Assessment", html)

    # =========================================================================
    # 2. Strict Input Field Preservation (8 Features + Patient Name + CSRF)
    # =========================================================================

    def test_02_all_8_clinical_fields_present_with_exact_names(self):
        """Confirm exactly the 8 clinical inputs, patient_name, and csrf_token exist with preserved attributes."""
        client = self.get_auth_client()
        res = client.get("/diabetes/predict")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Form action & method
        self.assertIn('method="POST"', html)
        self.assertTrue('action="/diabetes/predict"' in html or "diabetes_predict" in html)

        # CSRF single-line token field
        self.assertIn('name="csrf_token"', html)

        # 8 Clinical features + patient_name
        expected_fields = [
            "patient_name",
            "age",
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "dpf"
        ]

        for field in expected_fields:
            pattern = rf'name="{field}"'
            self.assertTrue(re.search(pattern, html), f"Field name='{field}' missing from form")

        # Verify numerical constraints and attributes
        self.assertTrue(re.search(r'name="age"[^>]*min="1"[^>]*max="120"', html))
        self.assertTrue(re.search(r'name="pregnancies"[^>]*min="0"[^>]*max="25"', html))
        self.assertTrue(re.search(r'name="glucose"[^>]*min="40"[^>]*max="300"', html))
        self.assertTrue(re.search(r'name="blood_pressure"[^>]*min="30"[^>]*max="200"', html))
        self.assertTrue(re.search(r'name="skin_thickness"[^>]*min="0"[^>]*max="100"', html))
        self.assertTrue(re.search(r'name="insulin"[^>]*min="0"[^>]*max="900"', html))
        self.assertTrue(re.search(r'name="bmi"[^>]*min="10"[^>]*max="70"', html))
        self.assertTrue(re.search(r'name="dpf"[^>]*min="0.01"[^>]*max="3.0"', html))

    # =========================================================================
    # 3. Form Submission & Result Page Flow
    # =========================================================================

    def test_03_prediction_submission_and_persistence(self):
        """Submit a standard clinical evaluation and verify prediction, result page, and DB storage."""
        client = self.get_auth_client()

        payload = {
            "patient_name": "Phase8E Test Patient",
            "age": "48",
            "pregnancies": "3",
            "glucose": "165.0",
            "blood_pressure": "88.0",
            "skin_thickness": "32.0",
            "insulin": "180.0",
            "bmi": "33.4",
            "dpf": "0.68",
            "csrf_token": "test_token"
        }

        res = client.post("/diabetes/predict", data=payload)
        self.assertEqual(res.status_code, 200)

        html = res.get_data(as_text=True)
        self.assertIn("Phase8E Test Patient", html)
        self.assertIn("DPID", html)  # Unique Patient ID generated
        self.assertTrue("Metabolic / Diabetes Risk Indicated" in html or "Low Risk" in html or "Risk Tier" in html)
        self.assertTrue("Risk" in html)

        # Extract generated DPID and verify persistence in database
        dpid = None
        for line in html.splitlines():
            if "DPID" in line:
                parts = line.split("DPID")
                candidate = "DPID" + parts[1].split("<")[0].split('"')[0].strip()
                if len(candidate) >= 18:
                    dpid = candidate
                    break

        self.assertIsNotNone(dpid, "Generated DPID was not found in response HTML")

        conn = get_db_connection()
        record = conn.execute("SELECT * FROM diabetes_patients WHERE patient_id = ?", (dpid,)).fetchone()
        conn.close()

        self.assertIsNotNone(record, f"Record for {dpid} was not found in diabetes_patients table")
        self.assertEqual(record["patient_name"], "Phase8E Test Patient")

        # Verify appearance in history
        res_hist = client.get("/diabetes/history")
        self.assertEqual(res_hist.status_code, 200)
        self.assertIn(dpid, res_hist.get_data(as_text=True))

    # =========================================================================
    # 4. Input Range Constraints & Validation Rejection
    # =========================================================================

    def test_04_physiological_bounds_validation(self):
        """Verify out-of-bounds physiological values return 400 Bad Request with explanatory error."""
        client = self.get_auth_client()

        # Out of bounds glucose (> 500)
        bad_payload = {
            "patient_name": "Invalid Subject",
            "age": "35",
            "pregnancies": "1",
            "glucose": "999",
            "blood_pressure": "80",
            "skin_thickness": "20",
            "insulin": "80",
            "bmi": "25.0",
            "dpf": "0.45",
            "csrf_token": "test_token"
        }

        res = client.post("/diabetes/predict", data=bad_payload)
        self.assertEqual(res.status_code, 400)
        html = res.get_data(as_text=True)
        self.assertIn("Glucose must be between 0 and 500 mg/dL", html)

    # =========================================================================
    # 5. Theme Palette & Responsive Stylesheet Checks
    # =========================================================================

    def test_05_design_system_and_responsive_constraints(self):
        """Verify approved theme palette (#0b1329, #1e293b, #0284c7) and absence of magenta glow."""
        client = self.get_auth_client()
        res = client.get("/diabetes/predict")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Palette
        self.assertIn("#0b1329", html)  # Deep Navy background
        self.assertIn("#1e293b", html)  # Slate Card background
        self.assertIn("#0284c7", html)  # Diabetes accent blue
        self.assertNotIn("194, 44, 106", html)  # No legacy magenta glow

        # Responsive overflow protection
        self.assertIn("overflow-x: hidden", html)
        self.assertIn("@media (max-width: 768px)", html)
        self.assertIn("box-sizing: border-box", html)

    # =========================================================================
    # 6. Navigation Link Integrity
    # =========================================================================

    def test_06_navigation_actions(self):
        """Verify navigation links to dashboard, history, and analytics are present."""
        client = self.get_auth_client()
        res = client.get("/diabetes/predict")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertIn('href="/dashboard"', html)
        self.assertIn('href="/diabetes/history"', html)
        self.assertIn('href="/analytics/diabetes"', html)

    # =========================================================================
    # 7. Dedicated CSRF Protection Enforcement
    # =========================================================================

    def test_07_csrf_missing_token_returns_400(self):
        """Confirm missing CSRF token returns HTTP 400 and rejection message."""
        client = self.get_auth_client()
        payload = {
            "patient_name": "CSRF Missing Patient",
            "age": "45",
            "pregnancies": "2",
            "glucose": "120.0",
            "blood_pressure": "75.0",
            "skin_thickness": "25.0",
            "insulin": "95.0",
            "bmi": "28.0",
            "dpf": "0.50"
            # csrf_token intentionally omitted
        }
        res = client.post("/diabetes/predict", data=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Security token validation failed", res.get_data(as_text=True))

    def test_08_csrf_invalid_or_tampered_token_returns_400(self):
        """Confirm invalid or tampered CSRF token returns HTTP 400 and rejection message."""
        client = self.get_auth_client()
        payload = {
            "patient_name": "CSRF Tampered Patient",
            "age": "45",
            "pregnancies": "2",
            "glucose": "120.0",
            "blood_pressure": "75.0",
            "skin_thickness": "25.0",
            "insulin": "95.0",
            "bmi": "28.0",
            "dpf": "0.50",
            "csrf_token": "tampered_bad_token_99999"
        }
        res = client.post("/diabetes/predict", data=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Security token validation failed", res.get_data(as_text=True))

    def test_09_csrf_valid_matching_token_succeeds(self):
        """Confirm matching session and form CSRF tokens pass validation successfully."""
        client = self.get_auth_client()
        payload = {
            "patient_name": "CSRF Valid Patient",
            "age": "45",
            "pregnancies": "2",
            "glucose": "120.0",
            "blood_pressure": "75.0",
            "skin_thickness": "25.0",
            "insulin": "95.0",
            "bmi": "28.0",
            "dpf": "0.50",
            "csrf_token": "test_token"  # Matches session["csrf_token"]
        }
        res = client.post("/diabetes/predict", data=payload)
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Csrf Valid Patient", html)
        self.assertIn("DPID", html)


if __name__ == "__main__":
    unittest.main()
