"""
Phase 8A Backend Multi-Disease Dashboard Aggregation Test Suite
Verifies:
1. Multi-disease platform aggregation across patients, diabetes_patients, and breast_cancer_patients.
2. Platform-wide counts, percentages, risk distributions, and average probability.
3. Chronological sorting of recent cross-platform assessments (up to 8 items).
4. Safe handling of empty tables, missing probabilities, and unmapped risk levels (zero division immunity).
5. Authentication enforcement on all analytics routes (/analytics/heart, /analytics/diabetes, /analytics/breast_cancer).
6. Comprehensive disease-specific metrics and model performance returned by each analytics route.
7. Graceful degradation when model metadata artifacts are missing.
8. Preservation of legacy dashboard variables and existing prediction routes.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import datetime
import secrets
import sqlite3
from werkzeug.security import generate_password_hash

from app import app
from database import get_db_connection, init_db


class Phase8ABackendAggregationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.orig_testing = app.config.get("TESTING")
        cls.orig_csrf = app.config.get("WTF_CSRF_ENABLED")
        cls.orig_db_path = os.environ.get("DATABASE_PATH")
        cls.temp_dir = None

        try:
            app.config["TESTING"] = True
            app.config["WTF_CSRF_ENABLED"] = False

            # Create isolated temporary directory & SQLite database
            cls.temp_dir = tempfile.mkdtemp(prefix="phase8a_isolated_test_")
            cls.test_db = os.path.join(cls.temp_dir, "test_phase8a.db")
            os.environ["DATABASE_PATH"] = cls.test_db

            # Initialize complete application schema in the isolated database
            init_db()

            # Generate unique test patient IDs
            cls.ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            cls.heart_pid = "PID" + cls.ts + secrets.token_hex(4).upper()
            cls.diab_pid = "DPID" + cls.ts + secrets.token_hex(4).upper()
            cls.cancer_pid = "BCP" + cls.ts + secrets.token_hex(4).upper()

            conn = get_db_connection()
            try:
                # Seed test-only admin user
                conn.execute("DELETE FROM users")
                test_admin_hash = generate_password_hash("Admin@1234")
                conn.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    ("admin", test_admin_hash)
                )

                # Seed 1 Heart Patient
                conn.execute("""
                    INSERT INTO patients (
                        patient_id, patient_name, age, sex, cp, trestbps, chol,
                        fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,
                        result, probability, risk_level, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cls.heart_pid, "Synthetic Heart Patient", 60, 1, 0, 140, 250,
                    0, 0, 130, 1, 2.0, 1, 1, 2, "Present", 82.5, "HIGH", "2026-05-01 10:00:00"
                ))

                # Seed 1 Diabetes Patient
                conn.execute("""
                    INSERT INTO diabetes_patients (
                        patient_id, patient_name, age, pregnancies, glucose,
                        blood_pressure, skin_thickness, insulin, bmi, dpf,
                        result, probability, risk_level, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cls.diab_pid, "Synthetic Diabetes Patient", 45, 2, 145.0,
                    80.0, 25.0, 100.0, 29.5, 0.55, "Present", 65.0, "MODERATE", "2026-06-01 12:00:00"
                ))

                # Seed 1 Breast Cancer Patient
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
                        worst_fractal_dimension, result, probability, risk_level, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cls.cancer_pid, "Synthetic Cancer Patient", 50, 12.0, 18.0,
                    80.0, 480.0, 0.08, 0.09, 0.05, 0.03, 0.16,
                    0.05, 0.3, 0.9, 2.1, 25.0, 0.005, 0.015,
                    0.02, 0.01, 0.015, 0.003, 14.0, 22.0,
                    95.0, 620.0, 0.11, 0.20, 0.15, 0.08, 0.25,
                    0.07, "Benign", 22.0, "LOW", "2026-07-01 14:00:00"
                ))

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

    def get_auth_client(self):
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["csrf_token"] = "valid_token"
        return client

    # =========================================================================
    # 1. Main Dashboard Authentication & Access
    # =========================================================================

    def test_01_dashboard_unauthenticated_access_rejected(self):
        """Unauthenticated requests to /dashboard redirect or return 401."""
        unauth_client = app.test_client()
        # Browser request redirects to login
        res = unauth_client.get("/dashboard")
        self.assertEqual(res.status_code, 302)
        self.assertIn("/login", res.headers.get("Location", ""))

        # JSON request returns 401
        res_json = unauth_client.get("/dashboard?format=json")
        self.assertEqual(res_json.status_code, 401)

    # =========================================================================
    # 2. Main Dashboard Multi-Disease Aggregation
    # =========================================================================

    def test_02_dashboard_aggregation_metrics(self):
        """Dashboard calculates accurate platform-wide multi-disease statistics."""
        client = self.get_auth_client()
        res = client.get("/dashboard?format=json")
        self.assertEqual(res.status_code, 200)

        data = res.get_json()
        self.assertIn("total_platform_patients", data)
        self.assertIn("platform_counts", data)
        self.assertIn("platform_percentages", data)
        self.assertIn("platform_risk", data)
        self.assertIn("disease_risk_distributions", data)
        self.assertIn("recent_assessments", data)

        # Platform count should equal sum of disease counts
        expected_total = (
            data["platform_counts"]["heart"] +
            data["platform_counts"]["diabetes"] +
            data["platform_counts"]["breast_cancer"]
        )
        self.assertEqual(data["total_platform_patients"], expected_total)
        self.assertGreaterEqual(data["total_platform_patients"], 3)

        # Percentages sum to 100%
        pct_sum = (
            data["platform_percentages"]["heart"] +
            data["platform_percentages"]["diabetes"] +
            data["platform_percentages"]["breast_cancer"]
        )
        self.assertAlmostEqual(pct_sum, 100.0, delta=0.5)

        # Risk distributions exist for all 3 modules
        dist = data["disease_risk_distributions"]
        self.assertIn("heart", dist)
        self.assertIn("diabetes", dist)
        self.assertIn("breast_cancer", dist)
        for d in ["heart", "diabetes", "breast_cancer"]:
            self.assertIn("LOW", dist[d])
            self.assertIn("MODERATE", dist[d])
            self.assertIn("HIGH", dist[d])

        # Platform risk totals
        risk = data["platform_risk"]
        self.assertEqual(
            risk["LOW"],
            dist["heart"]["LOW"] + dist["diabetes"]["LOW"] + dist["breast_cancer"]["LOW"]
        )
        self.assertEqual(
            risk["MODERATE"],
            dist["heart"]["MODERATE"] + dist["diabetes"]["MODERATE"] + dist["breast_cancer"]["MODERATE"]
        )
        self.assertEqual(
            risk["HIGH"],
            dist["heart"]["HIGH"] + dist["diabetes"]["HIGH"] + dist["breast_cancer"]["HIGH"]
        )

        # Platform average probability
        self.assertIsNotNone(data["platform_avg_probability"])
        self.assertGreaterEqual(data["platform_avg_probability"], 0.0)
        self.assertLessEqual(data["platform_avg_probability"], 100.0)

    # =========================================================================
    # 3. Recent Assessments Multi-Disease Stream & Chronological Ordering
    # =========================================================================

    def test_03_recent_assessments_sorting_and_attributes(self):
        """Recent assessments stream includes items across diseases and sorts descending."""
        client = self.get_auth_client()
        res = client.get("/dashboard?format=json")
        data = res.get_json()

        stream = data["recent_assessments"]
        self.assertIsInstance(stream, list)
        self.assertLessEqual(len(stream), 8)
        self.assertGreaterEqual(len(stream), 3)

        # Verify each item has required metadata
        for item in stream:
            self.assertIn("patient_id", item)
            self.assertIn("disease_type", item)
            self.assertIn("disease_icon", item)
            self.assertIn("risk_level", item)
            self.assertIn("probability", item)
            self.assertIn("timestamp", item)
            self.assertIn("detail_url", item)

        # Verify descending order by timestamp
        timestamps = [item["timestamp"] for item in stream if item["timestamp"]]
        for i in range(len(timestamps) - 1):
            self.assertGreaterEqual(timestamps[i], timestamps[i + 1])

    # =========================================================================
    # 4. Template Rendering & Legacy Variable Preservation
    # =========================================================================

    def test_04_dashboard_html_renders_legacy_and_new_variables(self):
        """Dashboard HTML renders successfully without breaking existing template variables."""
        client = self.get_auth_client()
        res = client.get("/dashboard")
        self.assertEqual(res.status_code, 200)

        html = res.get_data(as_text=True)
        # Verify existing page structure
        self.assertIn("Clinical AI Dashboard", html)
        self.assertIn("riskChart", html)
        self.assertIn("modelChart", html)
        self.assertIn("Cardiovascular Disease Module", html)
        self.assertIn("Diabetes Mellitus Module", html)
        self.assertIn("Breast Cancer Cytology Module", html)

    # =========================================================================
    # 5. Empty Table & Missing Value Resilience
    # =========================================================================

    def test_05_empty_table_aggregation_immunity(self):
        """Dashboard handles zero records without division-by-zero or crash."""
        with patch("app.get_db_connection") as mock_conn:
            mock_db = MagicMock()
            # Simulate all three queries returning empty lists
            mock_db.execute.return_value.fetchall.return_value = []
            mock_conn.return_value = mock_db

            client = self.get_auth_client()
            res = client.get("/dashboard?format=json")
            self.assertEqual(res.status_code, 200)

            data = res.get_json()
            self.assertEqual(data["total_platform_patients"], 0)
            self.assertEqual(data["platform_percentages"]["heart"], 0.0)
            self.assertEqual(data["platform_percentages"]["diabetes"], 0.0)
            self.assertEqual(data["platform_percentages"]["breast_cancer"], 0.0)
            self.assertIsNone(data["platform_avg_probability"])
            self.assertEqual(data["recent_assessments"], [])
            self.assertEqual(data["platform_status"], "Stable")

    def test_06_unmapped_risk_and_null_probability_handled_safely(self):
        """Dashboard safely processes rows with None probabilities or unmapped risk values."""
        corrupt_rows = [
            {"patient_id": "P_CORRUPT_1", "patient_name": "Test", "probability": None, "risk_level": None, "result": "N/A", "timestamp": None},
            {"patient_id": "P_CORRUPT_2", "patient_name": "Test", "probability": "INVALID", "risk_level": "UNKNOWN", "result": "N/A", "timestamp": "2026-01-01 00:00:00"}
        ]

        with patch("app.get_db_connection") as mock_conn:
            mock_db = MagicMock()
            mock_db.execute.return_value.fetchall.return_value = corrupt_rows
            mock_conn.return_value = mock_db

            client = self.get_auth_client()
            res = client.get("/dashboard?format=json")
            self.assertEqual(res.status_code, 200)

            data = res.get_json()
            self.assertIsInstance(data["total_platform_patients"], int)
            self.assertEqual(data["platform_risk"]["LOW"], 0)
            self.assertEqual(data["platform_risk"]["HIGH"], 0)

    # =========================================================================
    # 6. Analytics Route: Cardiovascular (/analytics/heart)
    # =========================================================================

    def test_07_analytics_heart_authentication_and_data(self):
        """Verify /analytics/heart enforces auth and returns complete statistics and metadata."""
        unauth_client = app.test_client()
        res_unauth = unauth_client.get("/analytics/heart?format=json")
        self.assertEqual(res_unauth.status_code, 401)

        client = self.get_auth_client()
        # JSON format
        res = client.get("/analytics/heart?format=json")
        self.assertEqual(res.status_code, 200)

        data = res.get_json()
        self.assertEqual(data["disease_type"], "heart")
        self.assertEqual(data["disease_name"], "Cardiovascular Disease")

        stats = data["patient_stats"]
        self.assertGreaterEqual(stats["total_patients"], 1)
        self.assertIn("risk_distribution", stats)
        self.assertIn("risk_percentages", stats)

        perf = data["model_performance"]
        self.assertEqual(perf["status"], "Available")
        self.assertEqual(perf["model_name"], "Logistic Regression")
        self.assertGreaterEqual(perf["accuracy"], 80.0)
        self.assertIn("feature_importance", perf)
        self.assertIn("confusion_matrix", perf)

        # HTML format
        res_html = client.get("/analytics/heart")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("Cardiovascular Module Analytics", res_html.get_data(as_text=True))

    # =========================================================================
    # 7. Analytics Route: Diabetes (/analytics/diabetes)
    # =========================================================================

    def test_08_analytics_diabetes_authentication_and_data(self):
        """Verify /analytics/diabetes enforces auth and returns complete statistics and metadata."""
        unauth_client = app.test_client()
        res_unauth = unauth_client.get("/analytics/diabetes?format=json")
        self.assertEqual(res_unauth.status_code, 401)

        client = self.get_auth_client()
        # JSON format
        res = client.get("/analytics/diabetes?format=json")
        self.assertEqual(res.status_code, 200)

        data = res.get_json()
        self.assertEqual(data["disease_type"], "diabetes")
        self.assertEqual(data["disease_name"], "Diabetes Mellitus")

        stats = data["patient_stats"]
        self.assertGreaterEqual(stats["total_patients"], 1)
        self.assertIn("risk_distribution", stats)
        self.assertIn("avg_glucose", stats)
        self.assertIn("avg_bmi", stats)

        perf = data["model_performance"]
        self.assertEqual(perf["status"], "Available")
        self.assertEqual(perf["model_name"], "Random Forest Classifier (Balanced)")
        self.assertGreaterEqual(perf["accuracy"], 70.0)
        self.assertIn("roc_auc", perf)
        self.assertIn("feature_importance", perf)

        # HTML format
        res_html = client.get("/analytics/diabetes")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("Diabetes Mellitus Module Analytics", res_html.get_data(as_text=True))

    # =========================================================================
    # 8. Analytics Route: Breast Cancer (/analytics/breast_cancer)
    # =========================================================================

    def test_09_analytics_breast_cancer_authentication_and_data(self):
        """Verify /analytics/breast_cancer enforces auth and returns complete statistics and metadata."""
        unauth_client = app.test_client()
        res_unauth = unauth_client.get("/analytics/breast_cancer?format=json")
        self.assertEqual(res_unauth.status_code, 401)

        client = self.get_auth_client()
        # JSON format
        res = client.get("/analytics/breast_cancer?format=json")
        self.assertEqual(res.status_code, 200)

        data = res.get_json()
        self.assertEqual(data["disease_type"], "breast_cancer")
        self.assertEqual(data["disease_name"], "Breast Cytology (WDBC)")

        stats = data["patient_stats"]
        self.assertGreaterEqual(stats["total_patients"], 1)
        self.assertIn("malignant_count", stats)
        self.assertIn("benign_count", stats)

        perf = data["model_performance"]
        self.assertEqual(perf["status"], "Available")
        self.assertEqual(perf["model_name"], "Support Vector Machine (RBF Balanced)")
        self.assertGreaterEqual(perf["accuracy"], 95.0)
        self.assertIn("specificity", perf)
        self.assertIn("roc_auc", perf)

        # HTML format
        res_html = client.get("/analytics/breast_cancer")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("Breast Cytology Module Analytics", res_html.get_data(as_text=True))

    # =========================================================================
    # 9. Graceful Degradation on Missing Model Artifacts
    # =========================================================================

    def test_10_analytics_handles_missing_model_metadata_gracefully(self):
        """Analytics routes return Status: Unavailable instead of crashing when metadata is missing."""
        with patch("joblib.load", side_effect=FileNotFoundError("Mocked file missing")):
            with patch("app.CACHED_MODEL_INFO", {}):
                client = self.get_auth_client()
                res = client.get("/analytics/heart?format=json")
                self.assertEqual(res.status_code, 200)
                data = res.get_json()
                self.assertEqual(data["model_performance"]["status"], "Unavailable")
                self.assertIsNone(data["model_performance"]["accuracy"])

    # =========================================================================
    # 10. Existing Assessment & Prediction Form Routes Intact
    # =========================================================================

    def test_11_existing_routes_and_forms_remain_functional(self):
        """Confirm /predict_page, /diabetes_predict_page, /breast_cancer_predict_page are unaffected."""
        client = self.get_auth_client()
        self.assertEqual(client.get("/predict_page").status_code, 200)
        self.assertEqual(client.get("/diabetes_predict_page").status_code, 200)
        self.assertEqual(client.get("/breast_cancer_predict_page").status_code, 200)


if __name__ == "__main__":
    unittest.main()
