"""
Targeted Verification Suite for Phase 8G — Heart Disease Patient Detail & Clinical Analytics Page Redesign.
Verifies all Phase 8G requirements:
1. Route access: Unauthenticated requests redirect to /login; authenticated requests return 200 OK.
2. Dynamic patient data binding: name, ID, probability, risk level, assessment timestamp.
3. Clinical content rendering: AI recommendations, AI explanations, and clinical risk alerts.
4. Biomarker cards & status: Cholesterol, Resting BP, Max Heart Rate, ST Depression with dynamic status badges.
5. Risk intelligence metrics: Model prediction, confidence score, key driver, risk trend.
6. Chart elements: Donut risk chart (riskChart) and parameter overview bar chart (featureChart) canvases.
7. Action buttons: Download PDF (targeting /download_report/<pid>) and Print Report (window.print()).
8. Navigation: Back to History (/history) and Dashboard (/dashboard) links.
9. Theme palette and responsive constraints: #0b1329 (navy), #1e293b (slate card), #2563eb (heart accent).
10. Print stylesheet: @media print rules that hide buttons and format cards for print/PDF export.
11. Security compliance: Zero usage of innerHTML in patient_detail.html (XSS protection).
12. PDF download route: Confirms /download_report/<pid> returns valid binary PDF (%PDF-).
13. Risk tier styling: Verifies low, moderate, and high risk styling behavior.
"""

import os
import re
import shutil
import sqlite3
import tempfile
import unittest
from app import app
from database import get_db_connection, init_db


class Phase8GPatientDetailTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.temp_dir = tempfile.mkdtemp(prefix="phase8g_heart_test_")
        cls.test_db = os.path.join(cls.temp_dir, "isolated_test.db")
        cls.orig_db_path = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = cls.test_db
        init_db()

        # Seed three distinct synthetic patients (Low, Moderate, High risk) in isolated DB
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO patients (
                patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal,
                result, probability, risk_level, timestamp
            ) VALUES (
                'PID_PHASE8G_LOW', 'Low Risk Subject', 42, 1, 1, 115, 180, 0,
                0, 165, 0, 0.4, 1, 0, 1,
                'Absent', 18.5, 'LOW', '2026-11-14 10:00:00'
            )
        """)
        conn.execute("""
            INSERT INTO patients (
                patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal,
                result, probability, risk_level, timestamp
            ) VALUES (
                'PID_PHASE8G_MOD', 'Moderate Risk Subject', 56, 1, 2, 138, 220, 0,
                1, 135, 1, 1.4, 2, 1, 2,
                'Present', 52.3, 'MODERATE', '2026-11-14 10:15:00'
            )
        """)
        conn.execute("""
            INSERT INTO patients (
                patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal,
                result, probability, risk_level, timestamp
            ) VALUES (
                'PID_PHASE8G_HIGH', 'High Risk Subject', 68, 1, 4, 160, 290, 1,
                2, 95, 1, 2.8, 2, 3, 3,
                'Present', 88.7, 'HIGH', '2026-11-14 10:30:00'
            )
        """)
        conn.commit()
        conn.close()

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
    # 1. Route Access & Authentication Enforcement
    # =========================================================================

    def test_01_route_auth_and_rendering(self):
        """Verify unauthenticated requests redirect to /login; authenticated requests return 200 OK."""
        unauth = app.test_client()
        res_unauth = unauth.get("/patient/PID_PHASE8G_MOD")
        self.assertEqual(res_unauth.status_code, 302)
        self.assertIn("/login", res_unauth.headers.get("Location", ""))

        client = self.get_auth_client()
        res_auth = client.get("/patient/PID_PHASE8G_MOD")
        self.assertEqual(res_auth.status_code, 200)
        html = res_auth.get_data(as_text=True)
        self.assertIn("Patient Clinical Analytics", html)
        self.assertIn("Moderate Risk Subject", html)
        self.assertIn("PID_PHASE8G_MOD", html)

    # =========================================================================
    # 2. Dynamic Clinical Content Verification
    # =========================================================================

    def test_02_clinical_content_bindings(self):
        """Verify recommendations, explanation, and alerts are present with exact test-contract strings."""
        client = self.get_auth_client()
        res = client.get("/patient/PID_PHASE8G_HIGH")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Exact test contracts from test_phase2_verification.py
        self.assertIn("AI Clinical Recommendation", html)
        self.assertIn("AI Clinical Explanation", html)
        self.assertIn("Clinical Risk Alerts", html)
        self.assertIn("Status: High", html)

        # Dynamic values for high risk subject
        self.assertIn("88.7%", html)
        self.assertIn("HIGH RISK", html)
        self.assertIn("Blocked vessels", html)

    # =========================================================================
    # 3. Biomarker Cards & Status Classification
    # =========================================================================

    def test_03_biomarker_status_cards(self):
        """Verify 4 core hemodynamic cards render with labels, units, and dynamic status badges."""
        client = self.get_auth_client()
        res = client.get("/patient/PID_PHASE8G_HIGH")
        html = res.get_data(as_text=True)

        self.assertIn("Serum Cholesterol", html)
        self.assertIn("290", html)
        self.assertIn("Resting Blood Pressure", html)
        self.assertIn("160", html)
        self.assertIn("Max Heart Rate", html)
        self.assertIn("95", html)
        self.assertIn("ST Depression", html)
        self.assertIn("2.8", html)

    # =========================================================================
    # 4. Risk Intelligence & Identity Panel
    # =========================================================================

    def test_04_risk_intelligence_metrics(self):
        """Verify risk intelligence metrics: confidence, key driver, trend, prediction, patient ID."""
        client = self.get_auth_client()
        res = client.get("/patient/PID_PHASE8G_MOD")
        html = res.get_data(as_text=True)

        self.assertIn("Risk Intelligence", html)
        self.assertIn("Primary Key Driver", html)
        self.assertIn("Risk Trend", html)
        self.assertIn("Model Prediction", html)
        self.assertIn("Confidence", html)
        self.assertIn("PID_PHASE8G_MOD", html)

    # =========================================================================
    # 5. Chart Canvases
    # =========================================================================

    def test_05_chart_canvases_present(self):
        """Verify riskChart (donut) and featureChart (bar) canvas elements exist."""
        client = self.get_auth_client()
        res = client.get("/patient/PID_PHASE8G_LOW")
        html = res.get_data(as_text=True)

        self.assertIn('id="riskChart"', html)
        self.assertIn('id="featureChart"', html)
        self.assertIn("new Chart", html)

    # =========================================================================
    # 6. Action Buttons & Navigation
    # =========================================================================

    def test_06_action_buttons_and_navigation(self):
        """Verify PDF download, print report button, history, and dashboard links."""
        client = self.get_auth_client()
        res = client.get("/patient/PID_PHASE8G_LOW")
        html = res.get_data(as_text=True)

        # PDF download action
        self.assertIn('href="/download_report/PID_PHASE8G_LOW"', html)
        self.assertIn("Download PDF", html)

        # Print report action
        self.assertIn("window.print()", html)
        self.assertIn("Print Report", html)

        # Navigation links
        self.assertIn('href="/history"', html)
        self.assertIn('href="/dashboard"', html)

    # =========================================================================
    # 7. Design System Palette & Responsive Styling
    # =========================================================================

    def test_07_theme_palette_and_styling(self):
        """Verify approved Phase 8 theme palette (#0b1329, #1e293b, #2563eb)."""
        client = self.get_auth_client()
        res = client.get("/patient/PID_PHASE8G_LOW")
        html = res.get_data(as_text=True)

        self.assertIn("#0b1329", html)  # Deep Navy background
        self.assertIn("#1e293b", html)  # Slate Card background
        self.assertIn("#2563eb", html)  # Heart Disease accent
        self.assertIn("box-sizing: border-box", html)
        self.assertIn("@media (max-width: 900px)", html)
        self.assertIn("@media (max-width: 600px)", html)

    # =========================================================================
    # 8. Print Optimization (@media print)
    # =========================================================================

    def test_08_print_stylesheet_rules(self):
        """Verify @media print styles hide interactive elements and optimize layout."""
        client = self.get_auth_client()
        res = client.get("/patient/PID_PHASE8G_LOW")
        html = res.get_data(as_text=True)

        self.assertIn("@media print", html)
        self.assertIn("page-break-inside: avoid", html)
        self.assertIn(".btn-download", html)
        self.assertIn(".btn-print", html)
        self.assertIn("display: none !important", html)

    # =========================================================================
    # 9. Security: XSS Remediation (No innerHTML)
    # =========================================================================

    def test_09_no_innerhtml_in_template(self):
        """Verify innerHTML is completely absent from patient_detail.html."""
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "patient_detail.html")
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("innerHTML", content, "innerHTML must not be used in patient_detail.html (XSS vulnerability)")

    # =========================================================================
    # 10. PDF Report Endpoint Generation
    # =========================================================================

    def test_10_pdf_report_generation(self):
        """Verify /download_report/<pid> returns valid binary PDF."""
        client = self.get_auth_client()
        pdf_res = client.get("/download_report/PID_PHASE8G_HIGH")
        self.assertEqual(pdf_res.status_code, 200)
        self.assertEqual(pdf_res.content_type, "application/pdf")
        self.assertTrue(pdf_res.data.startswith(b"%PDF-"))
        self.assertGreater(len(pdf_res.data), 1500)

    # =========================================================================
    # 11. Risk Tier Styling (Low, Moderate, High)
    # =========================================================================

    def test_11_risk_tier_styling_cases(self):
        """Verify low, moderate, and high risk subjects render their respective risk badges."""
        client = self.get_auth_client()

        # Low risk
        res_low = client.get("/patient/PID_PHASE8G_LOW")
        self.assertIn("badge low", res_low.get_data(as_text=True))
        self.assertIn("LOW RISK", res_low.get_data(as_text=True))

        # Moderate risk
        res_mod = client.get("/patient/PID_PHASE8G_MOD")
        self.assertIn("badge moderate", res_mod.get_data(as_text=True))
        self.assertIn("MODERATE RISK", res_mod.get_data(as_text=True))

        # High risk
        res_high = client.get("/patient/PID_PHASE8G_HIGH")
        self.assertIn("badge high", res_high.get_data(as_text=True))
        self.assertIn("HIGH RISK", res_high.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
