"""
Multi-Disease Clinical AI Platform - Chatbot Verification Suite
Verifies:
1. Reusable AI Chatbot component availability across all 3 modules:
   - Heart Disease (patient_detail.html)
   - Diabetes (diabetes_patient_detail.html)
   - Breast Cancer (breast_cancer_patient_detail.html)
2. Safe DOM rendering (Zero innerHTML in rendered templates).
3. Authentication and CSRF security on /ai_assistant endpoint.
4. Input validation (missing message, missing patient_id, nonexistent patient).
5. Disease-specific clinical context extraction and patient de-identification.
6. Cascading model fallback across Gemini models.
7. Graceful 503 error handling without stack trace leakage when upstream fails.
8. Safe text extraction when response.text is None.
"""

import unittest
from unittest.mock import patch, MagicMock
import datetime
import secrets
import sqlite3

from app import app
from database import get_db_connection


class MultiDiseaseChatbotIntegrationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        cls.client = app.test_client()

        # Seed synthetic test records for each module to verify detail views
        cls.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        cls.heart_pid = "PID" + cls.timestamp + secrets.token_hex(4).upper()
        cls.diabetes_pid = "DPID" + cls.timestamp + secrets.token_hex(4).upper()
        cls.cancer_pid = "BCP" + cls.timestamp + secrets.token_hex(4).upper()

        conn = get_db_connection()

        # 1. Heart Disease Patient
        conn.execute("""
            INSERT INTO patients (
                patient_id, patient_name, age, sex, cp, trestbps, chol,
                fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,
                result, probability, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cls.heart_pid, "Confidential Heart Patient", 55, 1, 0, 135, 245,
            0, 0, 142, 1, 1.8, 1, 1, 2, "Present", 78.5, "HIGH"
        ))

        # 2. Diabetes Patient
        conn.execute("""
            INSERT INTO diabetes_patients (
                patient_id, patient_name, age, pregnancies, glucose,
                blood_pressure, skin_thickness, insulin, bmi, dpf,
                result, probability, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cls.diabetes_pid, "Confidential Diabetes Patient", 48, 3, 155.0,
            82.0, 28.0, 110.0, 31.4, 0.65, "Present", 72.0, "HIGH"
        ))

        # 3. Breast Cancer Patient
        conn.execute("""
            INSERT INTO breast_cancer_patients (
                patient_id, patient_name, age, mean_radius, mean_texture,
                mean_perimeter, mean_area, mean_smoothness, mean_compactness,
                mean_concavity, mean_concave_points, mean_symmetry,
                mean_fractal_dimension, radius_error, texture_error,
                perimeter_error, area_error, smoothness_error, compactness_error,
                concavity_error, concave_points_error, symmetry_error,
                fractal_dimension_error, worst_radius, worst_texture,
                worst_perimeter, worst_area, worst_smoothness, worst_compactness,
                worst_concavity, worst_concave_points, worst_symmetry,
                worst_fractal_dimension, result, probability, risk_level
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cls.cancer_pid, "Confidential Cancer Patient", 52, 16.5, 21.0,
            108.0, 850.0, 0.10, 0.15, 0.14, 0.08, 0.19,
            0.06, 0.45, 1.2, 3.2, 45.0, 0.007, 0.025,
            0.035, 0.015, 0.02, 0.004, 20.2, 28.5,
            135.0, 1250.0, 0.14, 0.35, 0.40, 0.18, 0.32,
            0.09, "Malignant", 88.0, "HIGH"
        ))

        conn.commit()
        conn.close()

    @classmethod
    def tearDownClass(cls):
        # Clean up test rows
        conn = get_db_connection()
        conn.execute("DELETE FROM patients WHERE patient_id = ?", (cls.heart_pid,))
        conn.execute("DELETE FROM diabetes_patients WHERE patient_id = ?", (cls.diabetes_pid,))
        conn.execute("DELETE FROM breast_cancer_patients WHERE patient_id = ?", (cls.cancer_pid,))
        conn.commit()
        conn.close()

    def get_auth_client(self):
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["csrf_token"] = "valid_csrf_token"
        return client

    # =========================================================================
    # TEST 1: Chatbot UI Presence Across All 3 Modules
    # =========================================================================

    def test_01_chatbot_present_on_all_three_detail_views(self):
        """Verify the AI Chatbot component renders on Heart, Diabetes, and Breast Cancer views."""
        client = self.get_auth_client()

        routes = [
            (f"/patient/{self.heart_pid}", "Cardiovascular AI Assistant", "🫀"),
            (f"/diabetes/patient/{self.diabetes_pid}", "Diabetes AI Assistant", "🩸"),
            (f"/breast_cancer/patient/{self.cancer_pid}", "Breast Cytology AI Assistant", "🎗️")
        ]

        for url, expected_title, expected_icon in routes:
            resp = client.get(url)
            self.assertEqual(resp.status_code, 200, f"Failed loading {url}")
            html = resp.get_data(as_text=True)

            # Reusable chatbot DOM markers
            self.assertIn("aiChatButton", html, f"aiChatButton missing from {url}")
            self.assertIn("aiChatBox", html, f"aiChatBox missing from {url}")
            self.assertIn("aiChatMessages", html, f"aiChatMessages missing from {url}")
            self.assertIn("addAIMessage", html, f"addAIMessage missing from {url}")
            self.assertIn(expected_title, html, f"Expected title '{expected_title}' missing from {url}")
            self.assertIn(expected_icon, html, f"Expected icon '{expected_icon}' missing from {url}")

            # Zero innerHTML in rendered document
            self.assertNotIn("innerHTML", html, f"Unsafe 'innerHTML' found in {url}!")

    # =========================================================================
    # TEST 2: Security & Authentication
    # =========================================================================

    def test_02_unauthenticated_request_rejected(self):
        """Verify unauthenticated requests to /ai_assistant return HTTP 401."""
        client = app.test_client()
        with client.session_transaction() as sess:
            # Session exists with CSRF token, but user is not authenticated
            sess["csrf_token"] = "valid_csrf_token"

        resp = client.post("/ai_assistant", json={
            "patient_id": self.heart_pid,
            "message": "Explain this result",
            "csrf_token": "valid_csrf_token"
        })
        self.assertEqual(resp.status_code, 401)
        data = resp.get_json()
        self.assertIn("error", data)
        self.assertEqual(data["error"], "Unauthorized")

    def test_03_csrf_protection_on_ai_assistant(self):
        """Verify request with invalid/missing CSRF token returns HTTP 400."""
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["csrf_token"] = "valid_csrf_token"

        resp = client.post("/ai_assistant", json={
            "patient_id": self.heart_pid,
            "message": "Explain this result",
            "csrf_token": "wrong_token"
        })
        self.assertEqual(resp.status_code, 400)

    # =========================================================================
    # TEST 3: Input Validation
    # =========================================================================

    def test_04_missing_message_returns_400(self):
        """Verify empty message payload returns HTTP 400."""
        client = self.get_auth_client()
        resp = client.post("/ai_assistant", json={
            "patient_id": self.heart_pid,
            "message": "   ",
            "csrf_token": "valid_csrf_token"
        })
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "Message is required")

    def test_05_missing_patient_id_returns_400(self):
        """Verify missing patient_id returns HTTP 400."""
        client = self.get_auth_client()
        resp = client.post("/ai_assistant", json={
            "patient_id": "",
            "message": "Explain the risk",
            "csrf_token": "valid_csrf_token"
        })
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "Patient information is missing")

    def test_06_nonexistent_patient_returns_404(self):
        """Verify querying non-existent patient returns HTTP 404."""
        client = self.get_auth_client()
        resp = client.post("/ai_assistant", json={
            "patient_id": "PID_NONEXISTENT_99999",
            "message": "Explain the risk",
            "csrf_token": "valid_csrf_token"
        })
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["error"], "Patient not found")

    # =========================================================================
    # TEST 4: Heart Disease Assistant Flow & De-identification
    # =========================================================================

    @patch("app.gemini_client")
    def test_07_heart_disease_assistant_flow(self, mock_gemini):
        """Verify Heart Disease assistant builds correct context and de-identifies PII."""
        mock_response = MagicMock()
        mock_response.text = "Cardiovascular analysis: Elevated cholesterol and ST depression indicate high risk."
        mock_response.candidates = []
        mock_gemini.models.generate_content.return_value = mock_response

        client = self.get_auth_client()
        resp = client.post("/ai_assistant", json={
            "patient_id": self.heart_pid,
            "disease_type": "heart",
            "message": "Why is the patient at high risk?",
            "csrf_token": "valid_csrf_token"
        })

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["reply"], mock_response.text)

        # Inspect prompt passed to model
        self.assertTrue(mock_gemini.models.generate_content.called)
        call_kwargs = mock_gemini.models.generate_content.call_args[1]
        contents = call_kwargs.get("contents", "")

        # PII de-identification check
        self.assertNotIn("Confidential Heart Patient", contents)
        self.assertIn(f"[De-identified] ({self.heart_pid})", contents)

        # Cardiovascular metrics check
        self.assertIn("Cardiovascular Risk Prediction", contents)
        self.assertIn("Serum Cholesterol: 245 mg/dL", contents)
        self.assertIn("ST Depression (oldpeak): 1.8", contents)

    # =========================================================================
    # TEST 5: Diabetes Assistant Flow & De-identification
    # =========================================================================

    @patch("app.gemini_client")
    def test_08_diabetes_assistant_flow(self, mock_gemini):
        """Verify Diabetes assistant builds correct context and de-identifies PII."""
        mock_response = MagicMock()
        mock_response.text = "Metabolic analysis: Elevated 2-hour glucose (155 mg/dL) and BMI (31.4) are primary drivers."
        mock_response.candidates = []
        mock_gemini.models.generate_content.return_value = mock_response

        client = self.get_auth_client()
        resp = client.post("/ai_assistant", json={
            "patient_id": self.diabetes_pid,
            "disease_type": "diabetes",
            "message": "What clinical factors contributed to this score?",
            "csrf_token": "valid_csrf_token"
        })

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["reply"], mock_response.text)

        call_kwargs = mock_gemini.models.generate_content.call_args[1]
        contents = call_kwargs.get("contents", "")

        # PII de-identification check
        self.assertNotIn("Confidential Diabetes Patient", contents)
        self.assertIn(f"[De-identified] ({self.diabetes_pid})", contents)

        # Diabetes metrics check
        self.assertIn("Diabetes Mellitus Risk Prediction", contents)
        self.assertIn("Plasma Glucose (2-hr oral glucose test): 155.0 mg/dL", contents)
        self.assertIn("Body Mass Index (BMI): 31.4 kg/m²", contents)

    # =========================================================================
    # TEST 6: Breast Cancer Assistant Flow & De-identification
    # =========================================================================

    @patch("app.gemini_client")
    def test_09_breast_cancer_assistant_flow(self, mock_gemini):
        """Verify Breast Cancer assistant builds correct context and de-identifies PII."""
        mock_response = MagicMock()
        mock_response.text = "Cytology analysis: Nuclear concave points (0.08) and mean area (850 µm²) indicate high atypia."
        mock_response.candidates = []
        mock_gemini.models.generate_content.return_value = mock_response

        client = self.get_auth_client()
        resp = client.post("/ai_assistant", json={
            "patient_id": self.cancer_pid,
            "disease_type": "breast_cancer",
            "message": "Explain the nuclear morphology features.",
            "csrf_token": "valid_csrf_token"
        })

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["reply"], mock_response.text)

        call_kwargs = mock_gemini.models.generate_content.call_args[1]
        contents = call_kwargs.get("contents", "")

        # PII de-identification check
        self.assertNotIn("Confidential Cancer Patient", contents)
        self.assertIn(f"[De-identified] ({self.cancer_pid})", contents)

        # Breast Cancer metrics check
        self.assertIn("Breast Cytology Malignancy Prediction", contents)
        self.assertIn("Nuclear Mean Radius: 16.5 µm", contents)
        self.assertIn("Nuclear Mean Concave Points: 0.08", contents)

    # =========================================================================
    # TEST 7: Cascading Model Fallback on Error
    # =========================================================================

    @patch("app.gemini_client")
    def test_10_cascading_model_fallback(self, mock_gemini):
        """Verify that when primary model fails, system falls back to secondary model."""
        successful_response = MagicMock()
        successful_response.text = "Fallback model response succeeded."
        successful_response.candidates = []

        # Primary model raises exception, fallback succeeds
        mock_gemini.models.generate_content.side_effect = [
            Exception("503 Model Unavailable (Primary)"),
            successful_response
        ]

        client = self.get_auth_client()
        resp = client.post("/ai_assistant", json={
            "patient_id": self.heart_pid,
            "message": "Test fallback",
            "csrf_token": "valid_csrf_token"
        })

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["reply"], "Fallback model response succeeded.")
        # Verified that generate_content was invoked twice (cascaded)
        self.assertEqual(mock_gemini.models.generate_content.call_count, 2)

    # =========================================================================
    # TEST 8: All Models Failing Returns Clean 503 Without Stack Trace
    # =========================================================================

    @patch("app.gemini_client")
    def test_11_all_models_failing_returns_clean_503(self, mock_gemini):
        """Verify exhausted fallback models return user-friendly 503 without stack traces."""
        mock_gemini.models.generate_content.side_effect = Exception("Internal Upstream Failure")

        client = self.get_auth_client()
        resp = client.post("/ai_assistant", json={
            "patient_id": self.heart_pid,
            "message": "Test all failing",
            "csrf_token": "valid_csrf_token"
        })

        self.assertEqual(resp.status_code, 503)
        reply = resp.get_json()["reply"]
        self.assertIn("currently experiencing high demand", reply)
        # Confirm no raw stack trace leaked
        self.assertNotIn("Traceback", reply)
        self.assertNotIn("Internal Upstream Failure", reply)

    # =========================================================================
    # TEST 9: Missing Gemini Client Returns 503
    # =========================================================================

    @patch("app.gemini_client", None)
    def test_12_unconfigured_gemini_client_returns_503(self):
        """Verify unconfigured GEMINI_API_KEY returns clear 503 notification."""
        client = self.get_auth_client()
        resp = client.post("/ai_assistant", json={
            "patient_id": self.heart_pid,
            "message": "Test missing client",
            "csrf_token": "valid_csrf_token"
        })

        self.assertEqual(resp.status_code, 503)
        reply = resp.get_json()["reply"]
        self.assertIn("currently unavailable", reply)

    # =========================================================================
    # TEST 10: Safe Handling When response.text is None
    # =========================================================================

    @patch("app.gemini_client")
    def test_13_none_response_text_extracted_safely_from_parts(self, mock_gemini):
        """Verify that when response.text is None, text is extracted safely from candidate parts."""
        mock_part = MagicMock()
        mock_part.text = "Extracted safely from candidate parts."

        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]

        mock_response = MagicMock()
        mock_response.text = None  # Simulates None text
        mock_response.candidates = [mock_candidate]

        mock_gemini.models.generate_content.return_value = mock_response

        client = self.get_auth_client()
        resp = client.post("/ai_assistant", json={
            "patient_id": self.heart_pid,
            "message": "Test candidate extraction",
            "csrf_token": "valid_csrf_token"
        })

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["reply"], "Extracted safely from candidate parts.")


if __name__ == "__main__":
    unittest.main()
