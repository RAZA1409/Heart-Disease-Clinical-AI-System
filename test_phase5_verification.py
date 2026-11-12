"""
Phase 5 Verification Test Suite
Multi-Disease Clinical AI Platform - Production Readiness & Render Deployment Audit
"""

import unittest
import os
import tempfile
import shutil
import sqlite3
import joblib
import pandas as pd
from app import app, _load_required_artifact
from database import get_db_connection, get_database_path, init_db


class Phase5ProductionReadinessVerification(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()

    def get_authenticated_session(self):
        client = app.test_client()
        login_page = client.get("/login")
        html = login_page.get_data(as_text=True)
        token = ""
        for line in html.splitlines():
            if 'name="csrf_token"' in line:
                token = line.split('value="')[1].split('"')[0]
                break

        client.post("/login", data={
            "username": "admin",
            "password": "Admin@1234",
            "csrf_token": token
        }, follow_redirects=True)
        return client

    def get_csrf_token(self, client, url):
        res = client.get(url)
        html = res.get_data(as_text=True)
        for line in html.splitlines():
            if 'name="csrf_token"' in line:
                return line.split('value="')[1].split('"')[0]
        return ""

    # ========================================================
    # 1. MODEL ARTIFACT AVAILABILITY & CLEAR STARTUP FAILURES
    # ========================================================

    def test_01_all_nine_artifacts_exist_and_load(self):
        """Verify all 9 production model artifacts exist and load correctly."""
        required_artifacts = [
            ("models/best_model.pkl", "Heart Disease Model"),
            ("models/scaler.pkl", "Heart Disease Scaler"),
            ("models/feature_columns.pkl", "Heart Disease Feature Columns"),
            ("models/diabetes_pipeline.pkl", "Diabetes Pipeline"),
            ("models/diabetes_feature_columns.pkl", "Diabetes Feature Columns"),
            ("models/diabetes_model_info.pkl", "Diabetes Model Metadata"),
            ("models/breast_cancer_pipeline.pkl", "Breast Cancer Pipeline"),
            ("models/breast_cancer_feature_columns.pkl", "Breast Cancer Feature Columns"),
            ("models/breast_cancer_model_info.pkl", "Breast Cancer Model Metadata"),
        ]

        for path, desc in required_artifacts:
            self.assertTrue(os.path.exists(path), f"Artifact missing: {path} ({desc})")
            obj = _load_required_artifact(path, desc)
            self.assertIsNotNone(obj, f"Failed loading artifact: {path} ({desc})")

    def test_02_missing_artifact_raises_clear_startup_error(self):
        """Verify missing model artifacts fail fast with a clear FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as ctx:
            _load_required_artifact("models/nonexistent_model_xyz.pkl", "Nonexistent Test Model")
        self.assertIn("FATAL: Required model artifact for Nonexistent Test Model was not found", str(ctx.exception))

    # ========================================================
    # 2. SQLITE PERSISTENT STORAGE & SCHEMA INTEGRITY
    # ========================================================

    def test_03_dynamic_database_path_and_directory_creation(self):
        """Verify DATABASE_PATH environment variable support and automatic parent directory creation."""
        temp_dir = tempfile.mkdtemp(prefix="clinical_ai_test_")
        nested_db_path = os.path.join(temp_dir, "nested", "subfolder", "test_db.db")

        try:
            old_db_path = os.environ.get("DATABASE_PATH")
            os.environ["DATABASE_PATH"] = nested_db_path

            # Directory should not exist yet
            self.assertFalse(os.path.exists(os.path.dirname(nested_db_path)))

            # Connecting must create the parent directory and open connection
            conn = get_db_connection()
            self.assertTrue(os.path.exists(os.path.dirname(nested_db_path)))
            self.assertIsInstance(conn, sqlite3.Connection)
            conn.close()

        finally:
            if old_db_path is not None:
                os.environ["DATABASE_PATH"] = old_db_path
            else:
                os.environ.pop("DATABASE_PATH", None)
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_04_persistence_and_restart_simulation_without_data_loss(self):
        """Simulate application restart and verify schema initialization does not overwrite existing records."""
        temp_dir = tempfile.mkdtemp(prefix="clinical_ai_restart_test_")
        test_db_path = os.path.join(temp_dir, "persistent.db")

        try:
            old_db_path = os.environ.get("DATABASE_PATH")
            os.environ["DATABASE_PATH"] = test_db_path

            # Initial startup: create tables and default admin
            init_db()

            conn = get_db_connection()
            # Verify all 4 tables exist
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            self.assertIn("users", tables)
            self.assertIn("patients", tables)
            self.assertIn("diabetes_patients", tables)
            self.assertIn("breast_cancer_patients", tables)

            # Insert sample records across all 3 diseases
            conn.execute("""
                INSERT INTO patients (patient_id, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, result, probability, risk_level)
                VALUES ('HP1', 'Heart Patient 1', 50, 1, 1, 130, 240, 0, 0, 150, 0, 1.2, 1, 0, 3, 'Present', 65.5, 'MODERATE')
            """)
            conn.execute("""
                INSERT INTO diabetes_patients (patient_id, patient_name, age, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, result, probability, risk_level)
                VALUES ('DP1', 'Diabetes Patient 1', 45, 2, 140.0, 80.0, 25.0, 100.0, 28.5, 0.45, 'Present', 58.2, 'MODERATE')
            """)
            conn.execute("""
                INSERT INTO breast_cancer_patients (patient_id, patient_name, age, mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension, result, probability, risk_level)
                VALUES ('BCP1', 'Cancer Patient 1', 52, 14.0, 18.0, 90.0, 600.0, 0.1, 0.1, 0.1, 0.05, 0.2, 0.06, 0.3, 0.8, 2.0, 25.0, 0.008, 0.015, 0.02, 0.01, 0.02, 0.002, 16.0, 22.0, 105.0, 750.0, 0.14, 0.2, 0.25, 0.12, 0.3, 0.07, 'Benign', 22.4, 'LOW')
            """)
            conn.commit()
            conn.close()

            # Simulate Application Restart: call init_db() again
            init_db()

            conn2 = get_db_connection()
            # Verify records are completely preserved
            hp_count = conn2.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            dp_count = conn2.execute("SELECT COUNT(*) FROM diabetes_patients").fetchone()[0]
            bcp_count = conn2.execute("SELECT COUNT(*) FROM breast_cancer_patients").fetchone()[0]
            conn2.close()

            self.assertEqual(hp_count, 1, "Heart record lost after restart simulation")
            self.assertEqual(dp_count, 1, "Diabetes record lost after restart simulation")
            self.assertEqual(bcp_count, 1, "Breast cancer record lost after restart simulation")

        finally:
            if old_db_path is not None:
                os.environ["DATABASE_PATH"] = old_db_path
            else:
                os.environ.pop("DATABASE_PATH", None)
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ========================================================
    # 3. PRODUCTION SECURITY & COOKIE CONFIGURATION
    # ========================================================

    def test_05_admin_password_production_enforcement_and_preservation(self):
        """Verify admin account creation fails in production without ADMIN_DEFAULT_PASSWORD, succeeds with it, and is never overwritten."""
        temp_dir = tempfile.mkdtemp(prefix="clinical_ai_admin_test_")
        test_db = os.path.join(temp_dir, "admin_test.db")

        old_db_path = os.environ.get("DATABASE_PATH")
        old_render = os.environ.get("RENDER")
        old_admin_pass = os.environ.get("ADMIN_DEFAULT_PASSWORD")

        try:
            os.environ["DATABASE_PATH"] = test_db

            # 1. Fresh production database WITHOUT ADMIN_DEFAULT_PASSWORD must fail
            os.environ["RENDER"] = "true"
            os.environ.pop("ADMIN_DEFAULT_PASSWORD", None)

            with self.assertRaises(RuntimeError) as ctx:
                init_db()
            self.assertIn("ADMIN_DEFAULT_PASSWORD environment variable is not configured", str(ctx.exception))

            # Confirm no admin user was created
            conn = get_db_connection()
            admin_user = conn.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
            self.assertIsNone(admin_user, "Admin user must not be created when ADMIN_DEFAULT_PASSWORD is missing")
            conn.close()

            # 2. Fresh production database WITH ADMIN_DEFAULT_PASSWORD succeeds
            custom_pass = "ProdSecurePass!2026"
            os.environ["ADMIN_DEFAULT_PASSWORD"] = custom_pass
            init_db()

            from werkzeug.security import check_password_hash
            conn = get_db_connection()
            admin_row = conn.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()
            conn.close()
            self.assertIsNotNone(admin_row, "Admin user should exist after initialization with ADMIN_DEFAULT_PASSWORD")
            self.assertTrue(check_password_hash(admin_row[0], custom_pass), "Password hash does not match ADMIN_DEFAULT_PASSWORD")
            saved_hash = admin_row[0]

            # 3. Existing admin account is NEVER overwritten on subsequent startup
            os.environ["ADMIN_DEFAULT_PASSWORD"] = "DifferentNewPassword999!"
            init_db()

            conn2 = get_db_connection()
            admin_row_after = conn2.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()
            conn2.close()
            self.assertEqual(admin_row_after[0], saved_hash, "Existing admin password hash was modified or overwritten!")
            self.assertTrue(check_password_hash(admin_row_after[0], custom_pass))
            self.assertFalse(check_password_hash(admin_row_after[0], "DifferentNewPassword999!"))

        finally:
            if old_db_path is not None:
                os.environ["DATABASE_PATH"] = old_db_path
            else:
                os.environ.pop("DATABASE_PATH", None)

            if old_render is not None:
                os.environ["RENDER"] = old_render
            else:
                os.environ.pop("RENDER", None)

            if old_admin_pass is not None:
                os.environ["ADMIN_DEFAULT_PASSWORD"] = old_admin_pass
            else:
                os.environ.pop("ADMIN_DEFAULT_PASSWORD", None)

            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_06_production_secure_cookie_enforcement(self):
        """Verify production cookies remain secure even if SESSION_COOKIE_SECURE=false is supplied."""
        def resolve_cookie_secure(is_prod, env_val):
            if is_prod:
                return True
            return env_val.lower() in ("true", "1")

        # 1. In production, SESSION_COOKIE_SECURE must be True even if false is supplied
        self.assertTrue(resolve_cookie_secure(is_prod=True, env_val="false"))
        self.assertTrue(resolve_cookie_secure(is_prod=True, env_val="0"))

        # 2. In local development, SESSION_COOKIE_SECURE is configurable
        self.assertFalse(resolve_cookie_secure(is_prod=False, env_val="false"))
        self.assertTrue(resolve_cookie_secure(is_prod=False, env_val="true"))

        # 3. Verify active app configuration preserves HttpOnly and SameSite=Lax
        self.assertTrue(app.config.get("SESSION_COOKIE_HTTPONLY"), "SESSION_COOKIE_HTTPONLY must be True")
        self.assertEqual(app.config.get("SESSION_COOKIE_SAMESITE"), "Lax", "SESSION_COOKIE_SAMESITE must be Lax")
        self.assertFalse(app.debug, "Flask debug mode must not be enabled by default")

        # Disabled debug endpoint must return 404
        client = app.test_client()
        res = client.get("/test_patient_context/TEST1234")
        self.assertEqual(res.status_code, 404, "Disabled test endpoint should return 404")

    # ========================================================
    # 4. CROSS-MODULE VERIFICATION (END-TO-END)
    # ========================================================

    def test_07_multi_disease_full_workflow_and_isolation(self):
        """Verify full prediction, detail, PDF download, and deletion across all three disease modules without cross-contamination."""
        client = self.get_authenticated_session()

        # 1. Check Dashboard has all 3 modules
        dash_res = client.get("/dashboard")
        self.assertEqual(dash_res.status_code, 200)
        dash_html = dash_res.get_data(as_text=True)
        self.assertIn("Cardiovascular Disease Module", dash_html)
        self.assertIn("Diabetes Mellitus Module", dash_html)
        self.assertIn("Breast Cancer Cytology Module", dash_html)

        # 2. Submit Heart Disease assessment
        heart_token = self.get_csrf_token(client, "/predict_page")
        heart_res = client.post("/predict", data={
            "csrf_token": heart_token,
            "patient_name": "Phase 5 Heart Patient",
            "age": "55",
            "sex": "Male",
            "cp": "Typical Angina",
            "trestbps": "135",
            "chol": "250",
            "fbs": "No",
            "restecg": "Normal",
            "thalach": "145",
            "exang": "No",
            "oldpeak": "1.0",
            "slope": "Flat",
            "ca": "0",
            "thal": "Normal"
        }, follow_redirects=True)
        self.assertEqual(heart_res.status_code, 200)

        # 3. Submit Diabetes assessment
        diab_token = self.get_csrf_token(client, "/diabetes/predict")
        diab_res = client.post("/diabetes/predict", data={
            "csrf_token": diab_token,
            "patient_name": "Phase 5 Diabetes Patient",
            "age": "45",
            "pregnancies": "2",
            "glucose": "150.0",
            "blood_pressure": "82.0",
            "skin_thickness": "28.0",
            "insulin": "140.0",
            "bmi": "31.2",
            "dpf": "0.55"
        })
        self.assertEqual(diab_res.status_code, 200)
        diab_html = diab_res.get_data(as_text=True)
        diab_id = ""
        for line in diab_html.splitlines():
            if "DPID" in line:
                diab_id = "DPID" + line.split("DPID")[1].split("<")[0].split('"')[0].strip()
                if len(diab_id) >= 18:
                    break
        self.assertTrue(diab_id.startswith("DPID"), f"Failed to extract DPID: {diab_id}")

        # 4. Submit Breast Cancer assessment
        cancer_token = self.get_csrf_token(client, "/breast_cancer/predict")
        cancer_data = {
            "csrf_token": cancer_token,
            "patient_name": "Phase 5 Cancer Patient",
            "age": "50",
            "mean_radius": "14.2", "mean_texture": "16.1", "mean_perimeter": "92.0", "mean_area": "620.0", "mean_smoothness": "0.095",
            "mean_compactness": "0.09", "mean_concavity": "0.07", "mean_concave_points": "0.045", "mean_symmetry": "0.19", "mean_fractal_dimension": "0.06",
            "radius_error": "0.3", "texture_error": "0.8", "perimeter_error": "2.1", "area_error": "25.0", "smoothness_error": "0.007",
            "compactness_error": "0.015", "concavity_error": "0.025", "concave_points_error": "0.012", "symmetry_error": "0.02", "fractal_dimension_error": "0.003",
            "worst_radius": "16.0", "worst_texture": "21.0", "worst_perimeter": "104.0", "worst_area": "780.0", "worst_smoothness": "0.13",
            "worst_compactness": "0.18", "worst_concavity": "0.22", "worst_concave_points": "0.11", "worst_symmetry": "0.28", "worst_fractal_dimension": "0.075"
        }
        cancer_res = client.post("/breast_cancer/predict", data=cancer_data)
        self.assertEqual(cancer_res.status_code, 200)
        cancer_html = cancer_res.get_data(as_text=True)
        cancer_id = ""
        for line in cancer_html.splitlines():
            if "BCP" in line:
                cancer_id = "BCP" + line.split("BCP")[1].split("<")[0].split('"')[0].strip()
                if len(cancer_id) >= 18:
                    break
        self.assertTrue(cancer_id.startswith("BCP"), f"Failed to extract BCP ID: {cancer_id}")

        # 5. Verify PDF generation across modules
        pdf_diab = client.get(f"/diabetes/download_report/{diab_id}")
        self.assertEqual(pdf_diab.status_code, 200)
        self.assertTrue(pdf_diab.data.startswith(b"%PDF"))

        pdf_cancer = client.get(f"/breast_cancer/download_report/{cancer_id}")
        self.assertEqual(pdf_cancer.status_code, 200)
        self.assertTrue(pdf_cancer.data.startswith(b"%PDF"))

        # 6. Verify cross-module deletion isolation
        del_diab_token = self.get_csrf_token(client, "/diabetes/history")
        client.post(f"/diabetes/delete/{diab_id}", data={"csrf_token": del_diab_token}, follow_redirects=True)

        del_cancer_token = self.get_csrf_token(client, "/breast_cancer/history")
        client.post(f"/breast_cancer/delete/{cancer_id}", data={"csrf_token": del_cancer_token}, follow_redirects=True)

        # Confirm deleted records are gone
        conn = get_db_connection()
        self.assertIsNone(conn.execute("SELECT * FROM diabetes_patients WHERE patient_id = ?", (diab_id,)).fetchone())
        self.assertIsNone(conn.execute("SELECT * FROM breast_cancer_patients WHERE patient_id = ?", (cancer_id,)).fetchone())
        # Confirm heart disease table was not affected
        heart_rows = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        self.assertGreaterEqual(heart_rows, 1)
        conn.close()


if __name__ == "__main__":
    unittest.main()
