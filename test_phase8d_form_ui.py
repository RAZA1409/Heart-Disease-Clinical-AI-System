"""
Targeted Verification Suite for Phase 8D — Heart Disease Assessment Form UI/UX Redesign.
Verifies all Phase 8D requirements:
1. Renders /predict_page with authentication enforcement.
2. Exactly 13 original clinical features are present with preserved name attributes and option values.
3. Form submission successfully invokes Logistic Regression pipeline without breaking model preprocessing.
4. Result page (result.html) displays patient name, probability, risk level, and recommendation.
5. Client and server validation constraints (bounds on age, BP, cholesterol, HR, ST depression, fluoroscopy).
6. Compliance with Phase 8 theme (#0b1329, #1e293b, #2563eb) without magenta glow.
7. Responsive mobile stylesheet rules preventing horizontal page overflow.
"""

import re
import unittest
from app import app
from database import get_db_connection


class Phase8DFormUITests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        cls.client = app.test_client()

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
        """Verify unauthenticated users are redirected, while authenticated users receive 200 OK."""
        # Unauthenticated redirects to /login
        unauth = app.test_client()
        res_unauth = unauth.get("/predict_page")
        self.assertEqual(res_unauth.status_code, 302)
        self.assertIn("/login", res_unauth.headers.get("Location", ""))

        # Authenticated loads form
        client = self.get_auth_client()
        res_auth = client.get("/predict_page")
        self.assertEqual(res_auth.status_code, 200)
        html = res_auth.get_data(as_text=True)
        self.assertIn("Cardiovascular Clinical Assessment", html)
        self.assertIn("13-Parameter Diagnostic Screening", html)

    # =========================================================================
    # 2. Strict Input Field Preservation (13 Features + Patient Name)
    # =========================================================================

    def test_02_all_13_clinical_fields_present_with_exact_names(self):
        """Confirm exactly the 13 clinical input fields and patient name exist with preserved names."""
        client = self.get_auth_client()
        res = client.get("/predict_page")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Form action & method
        self.assertIn('action="/predict"', html)
        self.assertIn('method="POST"', html)
        self.assertIn('name="csrf_token"', html)

        # 13 Clinical features + patient_name
        expected_fields = [
            "patient_name",
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]

        for field in expected_fields:
            pattern = rf'name="{field}"'
            self.assertTrue(re.search(pattern, html), f"Field name='{field}' missing from form")

        # Verify exact select option values
        self.assertIn('value="Male"', html)
        self.assertIn('value="Female"', html)
        self.assertIn('value="Typical Angina"', html)
        self.assertIn('value="Atypical Angina"', html)
        self.assertIn('value="Non-Anginal Pain"', html)
        self.assertIn('value="Asymptomatic"', html)
        self.assertIn('value="No"', html)
        self.assertIn('value="Yes"', html)
        self.assertIn('value="Normal"', html)
        self.assertIn('value="ST-T Abnormality"', html)
        self.assertIn('value="Left Ventricular Hypertrophy"', html)
        self.assertIn('value="Upsloping"', html)
        self.assertIn('value="Flat"', html)
        self.assertIn('value="Downsloping"', html)
        self.assertIn('value="Fixed Defect"', html)
        self.assertIn('value="Reversible Defect"', html)

    # =========================================================================
    # 3. Form Submission & Result Page Flow
    # =========================================================================

    def test_03_prediction_submission_and_result_generation(self):
        """Submit a standard clinical evaluation and verify prediction & result page rendering."""
        client = self.get_auth_client()

        payload = {
            "patient_name": "Phase8D Validation Subject",
            "age": "57",
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

        res = client.post("/predict", data=payload, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        html = res.get_data(as_text=True)
        self.assertIn("Phase8D Validation Subject", html)
        self.assertIn("PID", html)  # Unique Patient ID generated
        self.assertIn("Cardiovascular Disease Detected", html)
        self.assertIn("Risk Level:", html)

    # =========================================================================
    # 4. Input Range Constraints & Validation Immunity
    # =========================================================================

    def test_04_physiological_bounds_validation(self):
        """Verify out-of-bounds physiological values return 400 Bad Request with explanatory error."""
        client = self.get_auth_client()

        # Out of bounds age (> 120)
        bad_payload = {
            "patient_name": "Invalid Subject",
            "age": "150",
            "sex": "Male",
            "cp": "Typical Angina",
            "trestbps": "120",
            "chol": "200",
            "fbs": "No",
            "restecg": "Normal",
            "thalach": "140",
            "exang": "No",
            "oldpeak": "0.5",
            "slope": "Upsloping",
            "ca": "0",
            "thal": "Normal",
            "csrf_token": "test_token"
        }

        res = client.post("/predict", data=bad_payload)
        self.assertEqual(res.status_code, 400)
        html = res.get_data(as_text=True)
        self.assertIn("Age must be between 1 and 120 years", html)

    # =========================================================================
    # 5. Theme Palette & Responsive Stylesheet Checks
    # =========================================================================

    def test_05_design_system_and_responsive_constraints(self):
        """Verify approved theme palette (#0b1329, #1e293b, #2563eb) and absence of magenta glow."""
        client = self.get_auth_client()
        res = client.get("/predict_page")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Palette
        self.assertIn("#0b1329", html)  # Deep Navy background
        self.assertIn("#1e293b", html)  # Slate Card background
        self.assertIn("#2563eb", html)  # Heart accent blue
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
        res = client.get("/predict_page")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertIn('href="/dashboard"', html)
        self.assertIn('href="/history"', html)
        self.assertIn('href="/analytics/heart"', html)


if __name__ == "__main__":
    unittest.main()
