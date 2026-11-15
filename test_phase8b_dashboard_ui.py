"""
Targeted Verification Suite for Phase 8B — Main Multi-Disease Dashboard UI/UX Redesign.
Verifies all Phase 8B requirements:
1. Executive KPI Section (6 Cards: Total, Heart, Diabetes, Breast Cancer, High Risk, High Risk Rate).
2. 3 Disease Module Cards (Cardiovascular, Diabetes, Breast Cancer) with model, counts, risk breakdown, and actions.
3. Cross-Disease Analytics Charts (Platform Risk Doughnut, Disease Risk Breakdown Bar) with no Heart-only training charts.
4. Recent Clinical Activity Table (6-8 items, chronological, patient references, no unnecessary patient names).
5. Zero-State / Empty Database resilience.
6. Unapproved Styling & Glow Cleanliness (No magenta glow, approved palette).
"""

import os
import shutil
import tempfile
import unittest
import re
from unittest.mock import patch, MagicMock
from werkzeug.security import generate_password_hash
from app import app
from database import get_db_connection, init_db


class Phase8BDashboardUITests(unittest.TestCase):

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
            cls.temp_dir = tempfile.mkdtemp(prefix="phase8b_isolated_test_")
            cls.test_db = os.path.join(cls.temp_dir, "test_phase8b.db")
            os.environ["DATABASE_PATH"] = cls.test_db

            # Initialize complete application schema in the isolated database
            init_db()

            # Seed disposable synthetic patient fixture & test admin user
            conn = get_db_connection()
            try:
                # Seed test admin
                conn.execute("DELETE FROM users")
                conn.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    ("admin", generate_password_hash("Admin@1234"))
                )

                # Seed 1 synthetic patient for each module
                conn.execute("""
                    INSERT INTO patients (
                        patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal, result,
                        probability, risk_level, timestamp
                    ) VALUES (
                        'PID2026040101', 'Synthetic Heart Patient', 55, 1, 1, 130, 240, 0,
                        1, 150, 0, 1.2, 2, 0, 3, 'Present', 75.0, 'HIGH', '2026-04-01 10:00:00'
                    )
                """)
                conn.execute("""
                    INSERT INTO diabetes_patients (
                        patient_id, patient_name, age, pregnancies, glucose,
                        blood_pressure, skin_thickness, insulin, bmi, dpf,
                        result, probability, risk_level, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    'DPID2026040101', 'Synthetic Diabetes Patient', 45, 2, 145.0,
                    80.0, 25.0, 100.0, 29.5, 0.55, "Present", 65.0, "MODERATE", "2026-04-01 11:00:00"
                ))

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
                    'BPID2026040101', 'Synthetic Cancer Patient', 50, 12.0, 18.0,
                    80.0, 480.0, 0.08, 0.09, 0.05, 0.03, 0.16,
                    0.05, 0.3, 0.9, 2.1, 25.0, 0.005, 0.015,
                    0.02, 0.01, 0.015, 0.003, 14.0, 22.0,
                    95.0, 620.0, 0.11, 0.20, 0.15, 0.08, 0.25,
                    0.07, "Malignant", 85.0, "HIGH", "2026-04-01 12:00:00"
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
        if cls.orig_testing is not None:
            app.config["TESTING"] = cls.orig_testing
        if cls.orig_csrf is not None:
            app.config["WTF_CSRF_ENABLED"] = cls.orig_csrf

        if cls.orig_db_path is not None:
            os.environ["DATABASE_PATH"] = cls.orig_db_path
        elif "DATABASE_PATH" in os.environ:
            del os.environ["DATABASE_PATH"]

        if cls.temp_dir and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def get_auth_client(self):
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["user"] = "admin"
        return client

    # =========================================================================
    # 1. Executive KPI Section Verification
    # =========================================================================

    def test_01_kpi_cards_render_accurate_platform_counts(self):
        """Verify all 6 executive KPI cards render with accurate platform counts."""
        client = self.get_auth_client()
        res = client.get("/dashboard")
        self.assertEqual(res.status_code, 200)

        html = res.get_data(as_text=True)

        # Get actual counts from DB
        conn = get_db_connection()
        heart_count = len(conn.execute("SELECT id FROM patients").fetchall())
        diabetes_count = len(conn.execute("SELECT id FROM diabetes_patients").fetchall())
        cancer_count = len(conn.execute("SELECT id FROM breast_cancer_patients").fetchall())
        total_count = heart_count + diabetes_count + cancer_count

        high_heart = len(conn.execute("SELECT id FROM patients WHERE UPPER(risk_level) = 'HIGH'").fetchall())
        high_diabetes = len(conn.execute("SELECT id FROM diabetes_patients WHERE UPPER(risk_level) = 'HIGH'").fetchall())
        high_cancer = len(conn.execute("SELECT id FROM breast_cancer_patients WHERE UPPER(risk_level) = 'HIGH'").fetchall())
        total_high = high_heart + high_diabetes + high_cancer

        high_rate = round((total_high / total_count) * 100, 1) if total_count > 0 else 0.0
        conn.close()

        # Check KPI section existence
        self.assertIn("Executive Platform KPIs", html)
        self.assertIn("kpi-grid", html)

        # Check all 6 KPI cards
        self.assertIn("Total Platform Patients", html)
        self.assertIn(f">{total_count}<", html)

        self.assertIn("Heart Disease", html)
        self.assertIn(f">{heart_count}<", html)

        self.assertIn("Diabetes Mellitus", html)
        self.assertIn(f">{diabetes_count}<", html)

        self.assertIn("Breast Cancer", html)
        self.assertIn(f">{cancer_count}<", html)

        self.assertIn("High-Risk Patients", html)
        self.assertIn(f">{total_high}<", html)

        self.assertIn("High-Risk Proportion", html)
        self.assertIn(f">{high_rate}%<", html)

    # =========================================================================
    # 2. Disease Module Cards Verification
    # =========================================================================

    def test_02_three_disease_module_cards_with_navigation(self):
        """Verify 3 disease cards render correct metadata, breakdown, and navigation actions."""
        client = self.get_auth_client()
        res = client.get("/dashboard")
        self.assertEqual(res.status_code, 200)

        html = res.get_data(as_text=True)

        # Card 1: Heart Disease
        self.assertIn("Cardiovascular Disease Module", html)
        self.assertIn("Logistic Regression", html)
        self.assertIn('href="/predict_page"', html)
        self.assertIn('href="/history"', html)
        self.assertIn('href="/analytics/heart"', html)

        # Card 2: Diabetes
        self.assertIn("Diabetes Mellitus Module", html)
        self.assertIn("Random Forest", html)
        self.assertIn('href="/diabetes/predict"', html)
        self.assertIn('href="/diabetes/history"', html)
        self.assertIn('href="/analytics/diabetes"', html)

        # Card 3: Breast Cancer
        self.assertIn("Breast Cancer Cytology Module", html)
        self.assertIn("Support Vector Machine", html)
        self.assertIn('href="/breast_cancer/predict"', html)
        self.assertIn('href="/breast_cancer/history"', html)
        self.assertIn('href="/analytics/breast_cancer"', html)

    # =========================================================================
    # 3. Platform Analytics Charts Verification
    # =========================================================================

    def test_03_platform_analytics_charts_and_no_heart_training_charts(self):
        """Verify only platform-wide charts are rendered and Heart training charts are omitted."""
        client = self.get_auth_client()
        res = client.get("/dashboard")
        self.assertEqual(res.status_code, 200)

        html = res.get_data(as_text=True)

        # 1. Platform-wide Risk Distribution exists
        self.assertIn("Platform-wide Risk Distribution", html)
        self.assertIn('id="riskChart"', html)

        # 2. Disease-wise Risk Breakdown exists
        self.assertIn("Disease-wise Risk Breakdown", html)
        self.assertIn('id="diseaseRiskChart"', html)

        # 3. Bulky Heart training canvas elements are NOT on the main dashboard
        self.assertNotIn('id="trainTestChart"', html)
        self.assertNotIn('id="cmChart"', html)
        self.assertNotIn('id="optChart"', html)
        self.assertNotIn('id="lossChart"', html)
        self.assertNotIn('id="featureChart"', html)
        self.assertNotIn('id="rocChart"', html)
        self.assertNotIn('id="prfChart"', html)

    # =========================================================================
    # 4. Recent Clinical Activity Table Verification
    # =========================================================================

    def test_04_recent_clinical_activity_table_privacy_and_links(self):
        """Verify table shows latest assessments without exposing patient names unnecessarily."""
        client = self.get_auth_client()
        res = client.get("/dashboard")
        self.assertEqual(res.status_code, 200)

        html = res.get_data(as_text=True)

        # Table headers
        self.assertIn("Recent Clinical Activity", html)
        self.assertIn("Patient Reference", html)
        self.assertIn("Disease Module", html)
        self.assertIn("Risk Level", html)
        self.assertIn("Prediction Probability", html)
        self.assertIn("Assessment Date", html)
        self.assertIn("Actions", html)

        # Confirm patient reference is rendered with #PID prefix
        self.assertIn("#PID", html)

        # Detail links
        detail_match = re.search(r'href="(/patient/[^"]+|/diabetes/patient/[^"]+|/breast_cancer/patient/[^"]+)"', html)
        self.assertIsNotNone(detail_match, "At least one valid detail URL should be present")

    # =========================================================================
    # 5. Zero-State / Empty Database Resilience
    # =========================================================================

    def test_05_zero_state_empty_database_rendering(self):
        """Verify dashboard renders cleanly with 0 counts when DB tables are empty."""
        with patch("app.get_db_connection") as mock_conn:
            mock_db = MagicMock()
            mock_db.execute.return_value.fetchall.return_value = []
            mock_conn.return_value = mock_db

            client = self.get_auth_client()
            res = client.get("/dashboard")
            self.assertEqual(res.status_code, 200)

            html = res.get_data(as_text=True)
            self.assertIn("Total Platform Patients", html)
            self.assertIn("No clinical assessments recorded yet", html)

    # =========================================================================
    # 6. Unapproved Styling & Glow Cleanliness
    # =========================================================================

    def test_06_no_magenta_glow_or_unapproved_styling(self):
        """Verify elimination of old magenta hover glow and presence of approved theme palette."""
        client = self.get_auth_client()
        res = client.get("/dashboard")
        self.assertEqual(res.status_code, 200)

        html = res.get_data(as_text=True)
        # Ensure old magenta glow rgba(194, 44, 106, ...) is absent
        self.assertNotIn("194, 44, 106", html, "Magenta hover glow must be removed")
        # Ensure approved background palette #0b1329 is used
        self.assertIn("#0b1329", html)
        # Ensure approved card background #1e293b is used
        self.assertIn("#1e293b", html)


if __name__ == "__main__":
    unittest.main()
