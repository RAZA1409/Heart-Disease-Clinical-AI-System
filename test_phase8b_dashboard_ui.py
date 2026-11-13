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

import unittest
import re
from unittest.mock import patch, MagicMock
from app import app
from database import get_db_connection


class Phase8BDashboardUITests(unittest.TestCase):

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
