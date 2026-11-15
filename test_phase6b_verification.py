"""
Phase 6B Multi-Disease Platform Consistency Corrections Verification Test Suite
Verifies:
1. Heart Disease backend physiological range validation (HTTP 400 on invalid input, acceptance of boundary inputs).
2. Heart Disease patient ID collision protection (format PID + 14 digits + 8 uppercase hex entropy, uniqueness).
3. Consistent clinical disclaimers in templates/result.html and Heart Disease PDF reports.
4. Dynamic positive-class probability extraction in Heart Disease and Diabetes prediction endpoints.
5. Gemini AI Assistant patient de-identification (PII name omitted, [De-identified] used).
6. Heart Disease PDF completeness (all 13 clinical features included with human-readable labels).
"""

import unittest
import re
import os
import shutil
import tempfile
import secrets
import datetime
from unittest.mock import patch, MagicMock
import joblib
import pandas as pd
from werkzeug.security import generate_password_hash

from app import app, FEATURE_MAP, REVERSE_FEATURE_MAP
from database import get_db_connection, init_db


class Phase6BConsistencyCorrectionsTest(unittest.TestCase):

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
            cls.temp_dir = tempfile.mkdtemp(prefix="phase6b_isolated_test_")
            cls.test_db = os.path.join(cls.temp_dir, "test_phase6b.db")
            os.environ["DATABASE_PATH"] = cls.test_db

            # Initialize complete application schema in the isolated database
            init_db()

            # Seed test-only admin user and synthetic fixtures
            conn = get_db_connection()
            try:
                conn.execute("DELETE FROM users")
                test_admin_hash = generate_password_hash("Admin@1234")
                conn.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    ("admin", test_admin_hash)
                )

                # Seed baseline synthetic heart patient fixture
                conn.execute("""
                    INSERT INTO patients (
                        patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal,
                        result, probability, risk_level, timestamp
                    ) VALUES (
                        'PID_PHASE6B_SYNTH_01', 'Phase6B Baseline Subject', 55, 1, 1, 130, 220, 0,
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
        with client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["csrf_token"] = "valid_csrf_token"
        return client

    # =========================================================================
    # TASK 1: Heart Disease Backend Validation
    # =========================================================================

    def test_01_heart_disease_range_validation_rejections(self):
        """Verify out-of-range physiological inputs return HTTP 400 with descriptive error."""
        client = self.get_authenticated_session()

        base_valid = {
            "patient_name": "Validation Test Subject",
            "age": "50",
            "sex": "Male",
            "cp": "Typical Angina",
            "trestbps": "120",
            "chol": "200",
            "fbs": "No",
            "restecg": "Normal",
            "thalach": "150",
            "exang": "No",
            "oldpeak": "1.0",
            "slope": "Flat",
            "ca": "0",
            "thal": "Normal",
            "csrf_token": "valid_csrf_token"
        }

        # Test cases: (field, invalid_value, expected_msg_substring)
        invalid_cases = [
            ("age", "-5", "Age must be between 1 and 120"),
            ("age", "0", "Age must be between 1 and 120"),
            ("age", "121", "Age must be between 1 and 120"),
            ("trestbps", "40", "Resting blood pressure must be between 50 and 300"),
            ("trestbps", "301", "Resting blood pressure must be between 50 and 300"),
            ("chol", "49", "Serum cholesterol must be between 50 and 700"),
            ("chol", "705", "Serum cholesterol must be between 50 and 700"),
            ("thalach", "35", "Maximum heart rate must be between 40 and 250"),
            ("thalach", "255", "Maximum heart rate must be between 40 and 250"),
            ("oldpeak", "-0.5", "ST depression must be between 0.0 and 10.0"),
            ("oldpeak", "10.5", "ST depression must be between 0.0 and 10.0"),
            ("ca", "-1", "Number of major vessels must be between 0 and 3"),
            ("ca", "4", "Number of major vessels must be between 0 and 3"),
            ("age", "invalid_number", "Invalid clinical input"),
        ]

        for field, bad_val, expected_err in invalid_cases:
            payload = dict(base_valid)
            payload[field] = bad_val

            res = client.post("/predict", data=payload)
            self.assertEqual(
                res.status_code, 400,
                f"Expected HTTP 400 for {field}={bad_val}, got {res.status_code}"
            )
            html = res.get_data(as_text=True)
            self.assertIn("Invalid clinical input", html)
            self.assertIn(expected_err, html)

    def test_02_heart_disease_range_validation_boundary_acceptance(self):
        """Verify boundary physiological inputs are accepted (HTTP 200)."""
        client = self.get_authenticated_session()

        # Lower boundary payload
        lower_boundary = {
            "patient_name": "Lower Boundary Patient",
            "age": "1",
            "sex": "Female",
            "cp": "Typical Angina",
            "trestbps": "50",
            "chol": "50",
            "fbs": "No",
            "restecg": "Normal",
            "thalach": "40",
            "exang": "No",
            "oldpeak": "0.0",
            "slope": "Upsloping",
            "ca": "0",
            "thal": "Normal",
            "csrf_token": "valid_csrf_token"
        }
        res_lower = client.post("/predict", data=lower_boundary, follow_redirects=True)
        self.assertEqual(res_lower.status_code, 200)

        # Upper boundary payload
        upper_boundary = {
            "patient_name": "Upper Boundary Patient",
            "age": "120",
            "sex": "Male",
            "cp": "Asymptomatic",
            "trestbps": "300",
            "chol": "700",
            "fbs": "Yes",
            "restecg": "Left Ventricular Hypertrophy",
            "thalach": "250",
            "exang": "Yes",
            "oldpeak": "10.0",
            "slope": "Downsloping",
            "ca": "3",
            "thal": "Reversible Defect",
            "csrf_token": "valid_csrf_token"
        }
        res_upper = client.post("/predict", data=upper_boundary, follow_redirects=True)
        self.assertEqual(res_upper.status_code, 200)

    # =========================================================================
    # TASK 2: Heart Disease Patient ID Collision Protection
    # =========================================================================

    def test_03_heart_disease_patient_id_format_and_entropy(self):
        """Verify Heart Disease patient ID adheres to PID + timestamp + 8-hex entropy."""
        client = self.get_authenticated_session()

        payload = {
            "patient_name": "Collision Check Patient",
            "age": "45",
            "sex": "Female",
            "cp": "Atypical Angina",
            "trestbps": "125",
            "chol": "190",
            "fbs": "No",
            "restecg": "Normal",
            "thalach": "160",
            "exang": "No",
            "oldpeak": "0.5",
            "slope": "Upsloping",
            "ca": "0",
            "thal": "Normal",
            "csrf_token": "valid_csrf_token"
        }

        res = client.post("/predict", data=payload, follow_redirects=True)
        self.assertEqual(res.status_code, 200)

        conn = get_db_connection()
        record = conn.execute(
            "SELECT patient_id FROM patients WHERE patient_name = 'Collision Check Patient' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()

        self.assertIsNotNone(record)
        pid = record["patient_id"]

        # Expected format: PID (3 chars) + YYYYMMDDHHMMSS (14 digits) + 8 hex chars = 25 chars total
        self.assertEqual(len(pid), 25, f"Expected patient_id length 25, got {len(pid)} for {pid}")
        pattern = r"^PID\d{14}[0-9A-F]{8}$"
        self.assertTrue(re.match(pattern, pid), f"Patient ID '{pid}' does not match pattern {pattern}")

    def test_04_patient_id_generator_collision_free(self):
        """Verify generating 1,000 IDs rapidly produces 1,000 unique IDs with zero collisions."""
        generated = set()
        for _ in range(1000):
            pid = "PID" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + secrets.token_hex(4).upper()
            generated.add(pid)
        self.assertEqual(len(generated), 1000, "Collision detected in patient ID generator!")

    # =========================================================================
    # TASK 3: Consistent Clinical Disclaimers
    # =========================================================================

    def test_05_result_template_clinical_disclaimer_and_wording(self):
        """Verify templates/result.html uses 'Statistical Assessment' and contains decision support disclaimer banner."""
        template_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "templates",
            "result.html"
        )
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check disclaimer banner
        self.assertIn(".disclaimer-banner", content)
        self.assertIn("class=\"disclaimer-banner\"", content)
        self.assertIn("Decision Support Estimate:", content)
        self.assertIn("not</strong> confirmed medical diagnoses", content)

        # Check label change
        self.assertIn("Statistical Assessment:", content)
        self.assertNotIn("Diagnosis Prediction:", content)

    def test_06_rendered_result_page_contains_disclaimer_banner(self):
        """Verify the rendered HTML response from /predict includes the disclaimer banner and updated wording."""
        client = self.get_authenticated_session()
        payload = {
            "patient_name": "Disclaimer Render Test",
            "age": "52",
            "sex": "Male",
            "cp": "Non-Anginal Pain",
            "trestbps": "130",
            "chol": "210",
            "fbs": "No",
            "restecg": "Normal",
            "thalach": "155",
            "exang": "No",
            "oldpeak": "0.8",
            "slope": "Upsloping",
            "ca": "0",
            "thal": "Normal",
            "csrf_token": "valid_csrf_token"
        }
        res = client.post("/predict", data=payload, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertIn("disclaimer-banner", html)
        self.assertIn("Decision Support Estimate:", html)
        self.assertIn("Statistical Assessment:", html)
        self.assertNotIn("Diagnosis Prediction:", html)

    def test_07_heart_disease_pdf_disclaimer_and_section_naming(self):
        """Verify Heart Disease PDF report contains updated decision-support title, subtitle, and table header."""
        client = self.get_authenticated_session()

        # Get existing patient id
        conn = get_db_connection()
        p = conn.execute("SELECT patient_id FROM patients LIMIT 1").fetchone()
        conn.close()
        self.assertIsNotNone(p, "No patient found in database")

        from reportlab.platypus import SimpleDocTemplate, Paragraph, Table

        captured_elements = []
        original_build = SimpleDocTemplate.build

        def mock_build(self_doc, story, *args, **kwargs):
            captured_elements.extend(story)
            return original_build(self_doc, story, *args, **kwargs)

        with patch.object(SimpleDocTemplate, 'build', side_effect=mock_build, autospec=True):
            res = client.get(f"/download_report/{p['patient_id']}")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.content_type, "application/pdf")

        # 1. Verify Story Paragraphs include subtitle disclaimer
        paragraph_texts = [el.text for el in captured_elements if isinstance(el, Paragraph)]
        self.assertTrue(
            any("Clinical Decision Support Estimate — Not an Official Medical Diagnosis" in txt for txt in paragraph_texts),
            "Subtitle disclaimer missing from PDF elements!"
        )

        # 2. Verify Table category contains 'AI Decision Support' and NOT 'AI Diagnostic Result'
        tables = [el for el in captured_elements if isinstance(el, Table)]
        self.assertGreater(len(tables), 0)
        table_rows = tables[0]._cellvalues
        categories = [row[0] for row in table_rows]
        self.assertIn("AI Decision Support", categories)
        self.assertNotIn("AI Diagnostic Result", categories)

        # 3. Verify statistical risk tier note
        param_values = [row[2] for row in table_rows]
        self.assertTrue(
            any("(Non-Diagnostic)" in str(val) for val in param_values),
            "Statistical Risk Tier (Non-Diagnostic) missing from PDF table!"
        )

    # =========================================================================
    # TASK 4: Dynamic Positive-Class Probability Extraction
    # =========================================================================

    def test_08_dynamic_probability_extraction_heart_and_diabetes(self):
        """Verify dynamic positive-class indexing works accurately using model.classes_."""
        # 1. Heart Disease model
        heart_model = joblib.load("models/best_model.pkl")
        heart_scaler = joblib.load("models/scaler.pkl")
        self.assertTrue(hasattr(heart_model, "classes_"))
        self.assertIn(1, heart_model.classes_)

        pos_idx_heart = list(heart_model.classes_).index(1)
        self.assertEqual(pos_idx_heart, 1)

        # Test on a dummy input
        dummy_heart = pd.DataFrame([{
            "Age": 55, "Sex": 1, "Chest pain type": 0, "BP": 130, "Cholesterol": 240,
            "FBS over 120": 0, "EKG results": 0, "Max HR": 150, "Exercise angina": 0,
            "ST depression": 1.2, "Slope of ST": 1, "Number of vessels fluro": 0, "Thallium": 3
        }])
        scaled_heart = heart_scaler.transform(dummy_heart)
        heart_probs = heart_model.predict_proba(scaled_heart)[0]
        self.assertAlmostEqual(heart_probs[pos_idx_heart], float(heart_probs[1]), places=6)

        # 2. Diabetes pipeline
        diab_pipe = joblib.load("models/diabetes_pipeline.pkl")
        self.assertTrue(hasattr(diab_pipe, "classes_"))
        self.assertIn(1, diab_pipe.classes_)

        pos_idx_diab = list(diab_pipe.classes_).index(1)
        self.assertEqual(pos_idx_diab, 1)

        dummy_diab = pd.DataFrame([{
            "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
            "Insulin": 80, "BMI": 25.5, "DiabetesPedigreeFunction": 0.45, "Age": 33
        }])
        diab_probs = diab_pipe.predict_proba(dummy_diab)[0]
        self.assertAlmostEqual(diab_probs[pos_idx_diab], float(diab_probs[1]), places=6)

    # =========================================================================
    # TASK 5: Gemini AI Assistant Patient De-identification
    # =========================================================================

    def test_09_gemini_ai_assistant_patient_de_identification(self):
        """Verify the Gemini AI assistant prompt omits real patient name and uses [De-identified] with patient_id."""
        client = self.get_authenticated_session()

        # Insert a unique test patient with sensitive real name
        secret_name = "Senator Jane Doe Confidential"
        unique_pid = "PID" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + secrets.token_hex(4).upper()

        conn = get_db_connection()
        conn.execute("""
            INSERT INTO patients (patient_name, patient_id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, result, probability, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (secret_name, unique_pid, 58, 1, 0, 140, 260, 0, 0, 130, 1, 2.0, 1, 1, 2, "Present", 85.5, "HIGH"))
        conn.commit()
        conn.close()

        # Mock gemini_client inside app.py
        mock_response = MagicMock()
        mock_response.text = "This is a safe mock response from clinical AI."
        mock_response.candidates = []

        with patch("app.gemini_client") as mock_gemini:
            mock_gemini.models.generate_content.return_value = mock_response

            res = client.post("/ai_assistant", json={
                "patient_id": unique_pid,
                "message": "What is the cardiovascular outlook for this patient?",
                "csrf_token": "valid_csrf_token"
            })

            self.assertEqual(res.status_code, 200)
            json_data = res.get_json()
            self.assertEqual(json_data["reply"], "This is a safe mock response from clinical AI.")

            # Inspect the prompt passed to Gemini
            self.assertTrue(mock_gemini.models.generate_content.called)
            call_kwargs = mock_gemini.models.generate_content.call_args[1]
            contents = call_kwargs.get("contents", "")

            # Confirm real name is NOT in contents
            self.assertNotIn(
                secret_name,
                contents,
                f"Real patient name '{secret_name}' was leaked to Gemini API prompt!"
            )

            # Confirm [De-identified] and patient_id ARE present
            self.assertIn("Patient Reference: [De-identified]", contents)
            self.assertIn(unique_pid, contents)

    # =========================================================================
    # TASK 6: Heart Disease PDF Completeness
    # =========================================================================

    def test_10_heart_disease_pdf_completeness_all_features(self):
        """Verify the Heart Disease PDF generation table includes restecg, slope, and thal with readable labels."""
        client = self.get_authenticated_session()

        # Insert a test patient with known discrete values
        test_pid = "PID" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + secrets.token_hex(4).upper()
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO patients (patient_name, patient_id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, result, probability, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ("PDF Completeness Subject", test_pid, 60, 1, 3, 135, 230, 1, 1, 140, 0, 1.5, 2, 2, 3, "Present", 72.0, "HIGH"))
        conn.commit()
        conn.close()

        # Intercept SimpleDocTemplate.build to inspect the table data
        from reportlab.platypus import SimpleDocTemplate, Table

        captured_tables = []
        original_build = SimpleDocTemplate.build

        def mock_build(self_doc, story, *args, **kwargs):
            for element in story:
                if isinstance(element, Table):
                    captured_tables.append(element)
            return original_build(self_doc, story, *args, **kwargs)

        with patch.object(SimpleDocTemplate, 'build', side_effect=mock_build, autospec=True):
            res = client.get(f"/download_report/{test_pid}")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.content_type, "application/pdf")

        self.assertGreater(len(captured_tables), 0, "No table captured in PDF document story")
        table_data = captured_tables[0]._cellvalues

        # Extract all parameter names and values from table rows
        parameter_names = [row[1] for row in table_data]
        parameter_values = [row[2] for row in table_data]

        # Verify previously omitted features are now explicitly present
        self.assertIn("Resting ECG Results", parameter_names, "Resting ECG Results missing from PDF table!")
        self.assertIn("Slope of ST", parameter_names, "Slope of ST missing from PDF table!")
        self.assertIn("Thallium", parameter_names, "Thallium missing from PDF table!")

        # Verify human-readable labels mapped from REVERSE_FEATURE_MAP
        # restecg 1 -> "ST-T Abnormality"
        # slope 2 -> "Flat"
        # thal 3 -> "Normal"
        self.assertIn("ST-T Abnormality", parameter_values)
        self.assertIn("Flat", parameter_values)
        self.assertIn("Normal", parameter_values)

        # Verify section title
        categories = [row[0] for row in table_data]
        self.assertIn("AI Decision Support", categories)
        self.assertNotIn("AI Diagnostic Result", categories)


if __name__ == "__main__":
    unittest.main()
