"""
Targeted Verification Suite for Phase 8C — Dedicated Multi-Disease Analytics Pages Redesign.
Verifies all Phase 8C requirements:
1. Heart Disease Analytics (/analytics/heart) renders with metrics, charts, confusion matrix, ROC curve, and navigation.
2. Diabetes Analytics (/analytics/diabetes) renders with Random Forest metrics, ROC-AUC, metabolic features, and navigation.
3. Breast Cancer Analytics (/analytics/breast_cancer) renders with SVM RBF metrics, WDBC nuclear features, ROC-AUC, and navigation.
4. Resilience against empty patient databases (zero records) without division-by-zero crashes.
5. Tolerance for missing or unreadable model metadata.
6. Authentication & session security enforcement (redirect to login / 401 for JSON).
7. Clean JSON payload preservation for API consumers.
8. Approved styling & absence of magenta hover glow.
"""

import unittest
from unittest.mock import patch, MagicMock
from app import app
from database import get_db_connection


class Phase8CAnalyticsUITests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        cls.client = app.test_client()

    def get_auth_client(self):
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["user"] = "admin"
        return client

    # =========================================================================
    # 1. Heart Disease Analytics Route & UI Verification
    # =========================================================================

    def test_01_heart_analytics_renders_and_displays_data(self):
        """Verify /analytics/heart renders with complete clinical metrics and visualizations."""
        client = self.get_auth_client()
        res = client.get("/analytics/heart")
        self.assertEqual(res.status_code, 200)

        html = res.get_data(as_text=True)

        # Titles and branding
        self.assertIn("Cardiovascular Disease Clinical Analytics", html)
        self.assertIn("Logistic Regression", html)

        # Navigation links
        self.assertIn('href="/dashboard"', html)
        self.assertIn('href="/history"', html)
        self.assertIn('href="/predict_page"', html)
        self.assertIn('href="/analytics/diabetes"', html)
        self.assertIn('href="/analytics/breast_cancer"', html)

        # Performance KPIs & metrics
        self.assertIn("90.74%", html)
        self.assertIn("ROC-AUC Score", html)
        self.assertIn("Precision / Recall", html)

        # Production status and verification wording
        self.assertIn("Benchmark Verified", html)
        self.assertNotIn("Validated & Certified", html)
        self.assertNotIn("Validated &amp; Certified", html)

        # Charts canvas elements
        self.assertIn('id="heartRiskChart"', html)
        self.assertIn('id="heartRocChart"', html)
        self.assertIn('id="heartFeatureChart"', html)
        self.assertIn('id="heartModelCompareChart"', html)

        # Confusion matrix
        self.assertIn("Confusion Matrix Breakdown", html)
        self.assertIn("True Negative (TN)", html)
        self.assertIn("True Positive (TP)", html)

    # =========================================================================
    # 2. Diabetes Analytics Route & UI Verification
    # =========================================================================

    def test_02_diabetes_analytics_renders_and_displays_data(self):
        """Verify /analytics/diabetes renders with Random Forest metrics and glycemic charts."""
        client = self.get_auth_client()
        res = client.get("/analytics/diabetes")
        self.assertEqual(res.status_code, 200)

        html = res.get_data(as_text=True)

        # Titles and branding
        self.assertIn("Diabetes Mellitus Clinical Analytics", html)
        self.assertIn("Random Forest Classifier", html)

        # Navigation links
        self.assertIn('href="/dashboard"', html)
        self.assertIn('href="/diabetes/history"', html)
        self.assertIn('href="/diabetes/predict"', html)

        # Metrics and metadata
        self.assertIn("75.32%", html)
        self.assertIn("0.8196", html)
        self.assertIn("Clinical Sensitivity", html)

        # Production status and refined clinical wording
        self.assertIn("Benchmark Verified", html)
        self.assertNotIn("Validated & Certified", html)
        self.assertNotIn("Validated &amp; Certified", html)
        self.assertNotIn("preventing dangerous false negatives", html)
        self.assertIn("16 positive cases were missed", html)

        # Charts canvas elements
        self.assertIn('id="diabetesRiskChart"', html)
        self.assertIn('id="diabetesRocChart"', html)
        self.assertIn('id="diabetesFeatureChart"', html)
        self.assertIn('id="diabetesMetricsChart"', html)

        # Confusion matrix
        self.assertIn("Confusion Matrix Breakdown", html)
        self.assertIn("Non-diabetic correctly ruled out", html)

    # =========================================================================
    # 3. Breast Cancer Analytics Route & UI Verification
    # =========================================================================

    def test_03_breast_cancer_analytics_renders_and_displays_data(self):
        """Verify /analytics/breast_cancer renders with SVM RBF metrics and WDBC diagnostics."""
        client = self.get_auth_client()
        res = client.get("/analytics/breast_cancer")
        self.assertEqual(res.status_code, 200)

        html = res.get_data(as_text=True)

        # Titles and branding
        self.assertIn("Breast Cytology", html)
        self.assertIn("Support Vector Machine", html)

        # Navigation links
        self.assertIn('href="/dashboard"', html)
        self.assertIn('href="/breast_cancer/history"', html)
        self.assertIn('href="/breast_cancer/predict"', html)

        # Metrics and metadata
        self.assertIn("98.25%", html)
        self.assertIn("0.9954", html)
        self.assertIn("Sensitivity / Specificity", html)

        # Audit corrections: Literature markers chart & refined clinical wording
        self.assertIn("Literature-Reported Diagnostic Markers (WDBC)", html)
        self.assertNotIn("Key Nuclear Morphology Feature Weights", html)
        self.assertNotIn("guarantees malignant lesions are not overlooked", html)
        self.assertNotIn("virtually eliminates false positives", html)
        self.assertIn("Benchmark Verified", html)
        self.assertNotIn("Validated & Certified", html)
        self.assertNotIn("Validated &amp; Certified", html)
        self.assertIn("not model-derived feature importance values", html)

        # Charts canvas elements
        self.assertIn('id="cancerRiskChart"', html)
        self.assertIn('id="cancerRocChart"', html)
        self.assertIn('id="cancerFeatureChart"', html)
        self.assertIn('id="cancerMetricsChart"', html)

        # Confusion matrix
        self.assertIn("Confusion Matrix Breakdown", html)
        self.assertIn("Benign accurately confirmed", html)

    # =========================================================================
    # 4. Resilience to Empty Databases (Zero Records)
    # =========================================================================

    def test_04_empty_database_aggregation_immunity(self):
        """Verify all three analytics pages render with 200 OK when patient tables are empty."""
        with patch("app.get_db_connection") as mock_conn:
            mock_db = MagicMock()
            mock_db.execute.return_value.fetchall.return_value = []
            mock_conn.return_value = mock_db

            client = self.get_auth_client()

            # Heart
            res_heart = client.get("/analytics/heart")
            self.assertEqual(res_heart.status_code, 200)
            self.assertIn("No cardiovascular assessments recorded yet", res_heart.get_data(as_text=True))

            # Diabetes
            res_diabetes = client.get("/analytics/diabetes")
            self.assertEqual(res_diabetes.status_code, 200)
            self.assertIn("No diabetes assessments recorded yet", res_diabetes.get_data(as_text=True))

            # Breast Cancer
            res_cancer = client.get("/analytics/breast_cancer")
            self.assertEqual(res_cancer.status_code, 200)
            self.assertIn("No breast cytology assessments recorded yet", res_cancer.get_data(as_text=True))

    # =========================================================================
    # 5. Missing / Unavailable Model Metadata Tolerance
    # =========================================================================

    def test_05_missing_model_metadata_graceful_fallback(self):
        """Verify pages load safely even if joblib model files fail to load."""
        with patch("joblib.load", side_effect=FileNotFoundError("Mock model file missing")):
            client = self.get_auth_client()

            for endpoint in ["/analytics/heart", "/analytics/diabetes", "/analytics/breast_cancer"]:
                res = client.get(endpoint)
                self.assertEqual(res.status_code, 200, f"{endpoint} failed with status {res.status_code}")

    # =========================================================================
    # 6. Authentication & Session Security Enforcement
    # =========================================================================

    def test_06_unauthenticated_requests_redirect_or_reject(self):
        """Verify unauthenticated users cannot access analytics pages."""
        unauth_client = app.test_client()

        for endpoint in ["/analytics/heart", "/analytics/diabetes", "/analytics/breast_cancer"]:
            # HTML request redirects to /login
            res_html = unauth_client.get(endpoint)
            self.assertEqual(res_html.status_code, 302)
            self.assertIn("/login", res_html.headers.get("Location", ""))

            # JSON request returns 401
            res_json = unauth_client.get(f"{endpoint}?format=json")
            self.assertEqual(res_json.status_code, 401)

    # =========================================================================
    # 7. JSON API Payload Preservation
    # =========================================================================

    def test_07_json_format_api_payloads(self):
        """Verify ?format=json returns valid structured analytics payloads."""
        client = self.get_auth_client()

        # Heart JSON
        res_heart = client.get("/analytics/heart?format=json")
        self.assertEqual(res_heart.status_code, 200)
        data_h = res_heart.get_json()
        self.assertEqual(data_h["disease_type"], "heart")
        self.assertIn("patient_stats", data_h)
        self.assertIn("model_performance", data_h)

        # Diabetes JSON
        res_diab = client.get("/analytics/diabetes?format=json")
        self.assertEqual(res_diab.status_code, 200)
        data_d = res_diab.get_json()
        self.assertEqual(data_d["disease_type"], "diabetes")
        self.assertIn("patient_stats", data_d)
        self.assertIn("model_performance", data_d)

        # Breast Cancer JSON
        res_cancer = client.get("/analytics/breast_cancer?format=json")
        self.assertEqual(res_cancer.status_code, 200)
        data_c = res_cancer.get_json()
        self.assertEqual(data_c["disease_type"], "breast_cancer")
        self.assertIn("patient_stats", data_c)
        self.assertIn("model_performance", data_c)

    # =========================================================================
    # 8. Unapproved Styling & Glow Cleanliness
    # =========================================================================

    def test_08_no_magenta_glow_and_consistent_palette(self):
        """Verify elimination of old magenta hover glow and presence of approved theme palette."""
        client = self.get_auth_client()

        for endpoint in ["/analytics/heart", "/analytics/diabetes", "/analytics/breast_cancer"]:
            res = client.get(endpoint)
            self.assertEqual(res.status_code, 200)
            html = res.get_data(as_text=True)

            # Ensure old magenta glow is absent
            self.assertNotIn("194, 44, 106", html, f"Magenta glow found in {endpoint}")
            # Ensure approved background palette #0b1329 is used
            self.assertIn("#0b1329", html)
            # Ensure approved card background #1e293b is used
            self.assertIn("#1e293b", html)


    # =========================================================================
    # 9. Audit Correction Verification: Missing Metadata Renders N/A
    # =========================================================================

    def test_09_missing_metadata_renders_na_without_hardcoded_fallbacks(self):
        """Verify templates render N/A instead of hardcoded numbers when model metadata is missing."""
        client = self.get_auth_client()

        with patch("joblib.load", side_effect=FileNotFoundError("Mock model file missing")), \
             patch("app.CACHED_MODEL_INFO", {}), \
             patch("app.diabetes_model_info", {}), \
             patch("app.breast_cancer_model_info", {}):

            # Heart KPI shows N/A
            res_heart = client.get("/analytics/heart")
            self.assertEqual(res_heart.status_code, 200)
            html_heart = res_heart.get_data(as_text=True)
            self.assertIn('<div class="kpi-value" style="color: #38bdf8;">N/A</div>', html_heart)
            self.assertNotIn('<div class="kpi-value" style="color: #38bdf8;">90.74%</div>', html_heart)

            # Diabetes KPI shows N/A
            res_diab = client.get("/analytics/diabetes")
            self.assertEqual(res_diab.status_code, 200)
            html_diab = res_diab.get_data(as_text=True)
            self.assertIn('<div class="kpi-value" style="color: #38bdf8;">N/A</div>', html_diab)
            self.assertNotIn('<div class="kpi-value" style="color: #38bdf8;">75.32%</div>', html_diab)

            # Breast Cancer KPI shows N/A
            res_cancer = client.get("/analytics/breast_cancer")
            self.assertEqual(res_cancer.status_code, 200)
            html_cancer = res_cancer.get_data(as_text=True)
            self.assertIn('<div class="kpi-value" style="color: #38bdf8;">N/A</div>', html_cancer)
            self.assertNotIn('<div class="kpi-value" style="color: #38bdf8;">98.25%</div>', html_cancer)


if __name__ == "__main__":
    unittest.main()
