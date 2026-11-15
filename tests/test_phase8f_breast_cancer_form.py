"""
Targeted Verification Suite for Phase 8F — Breast Cancer Assessment Form Redesign.
Verifies all Phase 8F requirements:
1. Authentication enforcement and page rendering.
2. All 30 exact feature names, IDs, types, step, and required attributes.
3. Strict preservation of feature ordering in HTML.
4. Patient name and age bounds validation.
5. Single-line CSRF token field preservation.
6. Benign and Malignant sample presets in JavaScript and UI buttons.
7. Form action and method integrity.
8. Multi-disease and module navigation links.
9. Theme palette (#0b1329, #1e293b, #ec4899) and responsive CSS rules.
10. CSRF protection enforcement (missing token -> 400, tampered token -> 400, valid token -> 200).
11. End-to-end model inference and persistence in an isolated temporary SQLite database.
12. Validation error handling for negative values and missing required features.
"""

import os
import re
import shutil
import sqlite3
import tempfile
import unittest
from app import app
from database import get_db_connection, init_db


class Phase8FBreastCancerFormTests(unittest.TestCase):

    EXPECTED_30_FEATURES = [
        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area",
        "mean_smoothness",
        "mean_compactness",
        "mean_concavity",
        "mean_concave_points",
        "mean_symmetry",
        "mean_fractal_dimension",
        "radius_error",
        "texture_error",
        "perimeter_error",
        "area_error",
        "smoothness_error",
        "compactness_error",
        "concavity_error",
        "concave_points_error",
        "symmetry_error",
        "fractal_dimension_error",
        "worst_radius",
        "worst_texture",
        "worst_perimeter",
        "worst_area",
        "worst_smoothness",
        "worst_compactness",
        "worst_concavity",
        "worst_concave_points",
        "worst_symmetry",
        "worst_fractal_dimension"
    ]

    BENIGN_SAMPLE = {
        "patient_name": "Eleanor Vance",
        "age": "44",
        "mean_radius": "13.54",
        "mean_texture": "14.36",
        "mean_perimeter": "87.46",
        "mean_area": "566.3",
        "mean_smoothness": "0.09779",
        "mean_compactness": "0.08129",
        "mean_concavity": "0.06664",
        "mean_concave_points": "0.04781",
        "mean_symmetry": "0.1885",
        "mean_fractal_dimension": "0.05766",
        "radius_error": "0.2699",
        "texture_error": "0.7886",
        "perimeter_error": "2.058",
        "area_error": "23.56",
        "smoothness_error": "0.008462",
        "compactness_error": "0.0146",
        "concavity_error": "0.02387",
        "concave_points_error": "0.01315",
        "symmetry_error": "0.0198",
        "fractal_dimension_error": "0.0023",
        "worst_radius": "15.11",
        "worst_texture": "19.26",
        "worst_perimeter": "99.7",
        "worst_area": "711.2",
        "worst_smoothness": "0.144",
        "worst_compactness": "0.1773",
        "worst_concavity": "0.239",
        "worst_concave_points": "0.1288",
        "worst_symmetry": "0.2977",
        "worst_fractal_dimension": "0.07259",
        "csrf_token": "test_token"
    }

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.temp_dir = tempfile.mkdtemp(prefix="phase8f_breast_cancer_test_")
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
        res_unauth = unauth.get("/breast_cancer/predict")
        self.assertEqual(res_unauth.status_code, 302)
        self.assertIn("/login", res_unauth.headers.get("Location", ""))

        res_alias = unauth.get("/breast_cancer_predict_page")
        self.assertEqual(res_alias.status_code, 302)

        client = self.get_auth_client()
        res_auth = client.get("/breast_cancer/predict")
        self.assertEqual(res_auth.status_code, 200)
        html = res_auth.get_data(as_text=True)
        self.assertIn("Breast Cancer Cytology AI Assessment", html)
        self.assertIn("Thirty-Feature Fine Needle Aspirate (FNA) Morphometry", html)

    # =========================================================================
    # 2. Strict Input Field Preservation (30 Features + Name + Age + CSRF)
    # =========================================================================

    def test_02_all_30_features_present_with_exact_names_and_ids(self):
        """Confirm exactly all 30 cytological features exist with matching name and id attributes."""
        client = self.get_auth_client()
        res = client.get("/breast_cancer/predict")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        for feature in self.EXPECTED_30_FEATURES:
            name_pat = rf'name="{feature}"'
            id_pat = rf'id="{feature}"'
            self.assertTrue(re.search(name_pat, html), f"Feature name='{feature}' missing from HTML")
            self.assertTrue(re.search(id_pat, html), f"Feature id='{feature}' missing from HTML")

    # =========================================================================
    # 3. Exact Feature Ordering in HTML
    # =========================================================================

    def test_03_exact_feature_ordering_in_html(self):
        """Verify that all 30 cytological inputs appear in the HTML in the exact expected ML sequence."""
        client = self.get_auth_client()
        res = client.get("/breast_cancer/predict")
        html = res.get_data(as_text=True)

        positions = []
        for feature in self.EXPECTED_30_FEATURES:
            match = re.search(rf'name="{feature}"', html)
            self.assertIsNotNone(match, f"Feature {feature} not found in HTML")
            positions.append((feature, match.start()))

        for i in range(len(positions) - 1):
            curr_feat, curr_pos = positions[i]
            next_feat, next_pos = positions[i + 1]
            self.assertLess(
                curr_pos,
                next_pos,
                f"Feature ordering mismatch: {curr_feat} (pos {curr_pos}) should appear before {next_feat} (pos {next_pos})"
            )

    # =========================================================================
    # 4. Input Type, Step, and Required Attributes
    # =========================================================================

    def test_04_input_attributes_type_step_required(self):
        """Confirm all 30 feature inputs have type='number', step='any', and required."""
        client = self.get_auth_client()
        res = client.get("/breast_cancer/predict")
        html = res.get_data(as_text=True)

        for feature in self.EXPECTED_30_FEATURES:
            # Match the whole input tag for this feature
            pattern = rf'<input[^>]*name="{feature}"[^>]*>'
            match = re.search(pattern, html)
            self.assertIsNotNone(match, f"Input tag for {feature} not found")
            tag = match.group(0)
            self.assertIn('type="number"', tag, f"Input {feature} missing type='number'")
            self.assertIn('step="any"', tag, f"Input {feature} missing step='any'")
            self.assertIn('required', tag, f"Input {feature} missing required")

    # =========================================================================
    # 5. Patient Name and Age Constraints
    # =========================================================================

    def test_05_patient_name_and_age_constraints(self):
        """Verify patient_name is required text, and age has min='1', max='120', required."""
        client = self.get_auth_client()
        res = client.get("/breast_cancer/predict")
        html = res.get_data(as_text=True)

        # patient_name
        self.assertTrue(re.search(r'<input[^>]*name="patient_name"[^>]*type="text"[^>]*required', html) or
                        re.search(r'<input[^>]*type="text"[^>]*name="patient_name"[^>]*required', html))

        # age
        age_match = re.search(r'<input[^>]*name="age"[^>]*>', html)
        self.assertIsNotNone(age_match)
        age_tag = age_match.group(0)
        self.assertIn('type="number"', age_tag)
        self.assertIn('min="1"', age_tag)
        self.assertIn('max="120"', age_tag)
        self.assertIn('required', age_tag)

    # =========================================================================
    # 6. Single-Line CSRF Token Field Preservation
    # =========================================================================

    def test_06_single_line_csrf_token_field(self):
        """Confirm CSRF hidden input is present on a single contiguous line for test scraper compatibility."""
        client = self.get_auth_client()
        res = client.get("/breast_cancer/predict")
        html = res.get_data(as_text=True)

        found_single_line = False
        for line in html.splitlines():
            if 'name="csrf_token"' in line and 'value="' in line:
                found_single_line = True
                break

        self.assertTrue(
            found_single_line,
            "CSRF token input field must exist on a single line containing both name='csrf_token' and value='...'"
        )

    # =========================================================================
    # 7. Benign & Malignant Sample Presets
    # =========================================================================

    def test_07_sample_presets_present_and_functional(self):
        """Verify sample preset buttons and loadSample javascript data for Benign and Malignant cases."""
        client = self.get_auth_client()
        res = client.get("/breast_cancer/predict")
        html = res.get_data(as_text=True)

        # Buttons
        self.assertIn("loadSample('benign')", html)
        self.assertIn("loadSample('malignant')", html)
        self.assertIn("Load Sample: Benign", html)
        self.assertIn("Load Sample: Malignant", html)

        # JavaScript data object
        self.assertIn("const samples =", html)
        self.assertIn("benign:", html)
        self.assertIn("malignant:", html)
        self.assertIn("function loadSample(type)", html)

        # Verify key feature values in JS
        self.assertIn("Eleanor Vance", html)
        self.assertIn("Clara Oswald", html)
        self.assertIn("17.99", html)
        self.assertIn("13.54", html)

    # =========================================================================
    # 8. Form Method & Endpoint Target
    # =========================================================================

    def test_08_form_action_and_method(self):
        """Verify form targets /breast_cancer/predict with method='POST'."""
        client = self.get_auth_client()
        res = client.get("/breast_cancer/predict")
        html = res.get_data(as_text=True)

        self.assertIn('method="POST"', html)
        self.assertTrue('action="/breast_cancer/predict"' in html or 'breast_cancer_predict' in html)

    # =========================================================================
    # 9. Navigation Links
    # =========================================================================

    def test_09_navigation_links(self):
        """Verify navigation links to dashboard, history, and analytics."""
        client = self.get_auth_client()
        res = client.get("/breast_cancer/predict")
        html = res.get_data(as_text=True)

        self.assertIn('href="/dashboard"', html)
        self.assertIn('href="/breast_cancer/history"', html)
        self.assertIn('href="/analytics/breast_cancer"', html)

    # =========================================================================
    # 10. Design System & Responsive CSS Constraints
    # =========================================================================

    def test_10_theme_palette_and_responsive_rules(self):
        """Verify approved theme palette (#0b1329, #1e293b, #ec4899) and responsive rules."""
        client = self.get_auth_client()
        res = client.get("/breast_cancer/predict")
        html = res.get_data(as_text=True)

        self.assertIn("#0b1329", html)  # Deep Navy background
        self.assertIn("#1e293b", html)  # Elevated Slate Card
        self.assertIn("#ec4899", html)  # Pink disease accent
        self.assertNotIn("194, 44, 106", html)  # No legacy magenta glow

        # Responsive styling
        self.assertIn("overflow-x: hidden", html)
        self.assertIn("@media (max-width: 768px)", html)
        self.assertIn("box-sizing: border-box", html)

    # =========================================================================
    # 11. CSRF Protection: Missing Token
    # =========================================================================

    def test_11_csrf_missing_token_rejected(self):
        """Confirm missing CSRF token returns HTTP 400 and security error message."""
        client = self.get_auth_client()
        payload = dict(self.BENIGN_SAMPLE)
        payload.pop("csrf_token", None)

        res = client.post("/breast_cancer/predict", data=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Security token validation failed", res.get_data(as_text=True))

    # =========================================================================
    # 12. CSRF Protection: Invalid / Tampered Token
    # =========================================================================

    def test_12_csrf_invalid_token_rejected(self):
        """Confirm invalid or tampered CSRF token returns HTTP 400 and security error message."""
        client = self.get_auth_client()
        payload = dict(self.BENIGN_SAMPLE, csrf_token="tampered_fake_token_12345")

        res = client.post("/breast_cancer/predict", data=payload)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Security token validation failed", res.get_data(as_text=True))

    # =========================================================================
    # 13. CSRF Protection: Valid Token Accepted
    # =========================================================================

    def test_13_csrf_valid_token_accepted(self):
        """Confirm matching session and form CSRF tokens pass security validation."""
        client = self.get_auth_client()
        payload = dict(self.BENIGN_SAMPLE, patient_name="Csrf Success Patient")

        res = client.post("/breast_cancer/predict", data=payload)
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Csrf Success Patient", html)
        self.assertIn("BCP", html)

    # =========================================================================
    # 14. Valid Synthetic Prediction with Isolated DB Persistence
    # =========================================================================

    def test_14_synthetic_prediction_and_isolated_db_persistence(self):
        """Submit benign patient and verify inference, result display, and isolated DB persistence."""
        client = self.get_auth_client()
        payload = dict(self.BENIGN_SAMPLE, patient_name="Phase8F Isolated Patient")

        res = client.post("/breast_cancer/predict", data=payload)
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        self.assertIn("Phase8F Isolated Patient", html)
        self.assertIn("BCP", html)
        self.assertIn("Malignancy", html)

        # Extract generated BCP ID
        bcpid = None
        for line in html.splitlines():
            if "BCP" in line:
                parts = line.split("BCP")
                candidate = "BCP" + parts[1].split("<")[0].split('"')[0].strip()
                if len(candidate) >= 18:
                    bcpid = candidate
                    break

        self.assertIsNotNone(bcpid, "Generated BCP ID not found in result HTML")

        # 1. Verify record is present in ISOLATED test database
        conn = get_db_connection()
        record = conn.execute("SELECT * FROM breast_cancer_patients WHERE patient_id = ?", (bcpid,)).fetchone()
        conn.close()

        self.assertIsNotNone(record, f"Record {bcpid} not found in isolated test database")
        self.assertEqual(record["patient_name"], "Phase8F Isolated Patient")
        self.assertAlmostEqual(record["mean_radius"], 13.54, places=2)

        # 2. Confirm record is NOT in the real development database
        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        real_db_path = os.path.join(proj_root, "database.db")
        if not os.path.exists(real_db_path):
            real_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")
        if os.path.exists(real_db_path):
            real_conn = sqlite3.connect(real_db_path)
            real_record = real_conn.execute("SELECT * FROM breast_cancer_patients WHERE patient_id = ?", (bcpid,)).fetchone()
            real_conn.close()
            self.assertIsNone(real_record, "Synthetic test patient leaked into real development database!")

    # =========================================================================
    # 15. Input Validation: Missing & Negative Feature Rejection
    # =========================================================================

    def test_15_missing_and_negative_feature_validation(self):
        """Verify missing features and negative values return HTTP 400 with descriptive error."""
        client = self.get_auth_client()

        # A. Missing feature
        missing_payload = dict(self.BENIGN_SAMPLE)
        missing_payload.pop("mean_radius", None)
        res_missing = client.post("/breast_cancer/predict", data=missing_payload)
        self.assertEqual(res_missing.status_code, 400)
        self.assertIn("Missing required cytology feature", res_missing.get_data(as_text=True))

        # B. Negative feature value
        negative_payload = dict(self.BENIGN_SAMPLE, mean_radius="-5.0")
        res_neg = client.post("/breast_cancer/predict", data=negative_payload)
        self.assertEqual(res_neg.status_code, 400)
        self.assertIn("cannot be negative", res_neg.get_data(as_text=True))

        # C. Out-of-bounds age
        age_payload = dict(self.BENIGN_SAMPLE, age="150")
        res_age = client.post("/breast_cancer/predict", data=age_payload)
        self.assertEqual(res_age.status_code, 400)
        self.assertIn("Age must be between 1 and 120 years", res_age.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
