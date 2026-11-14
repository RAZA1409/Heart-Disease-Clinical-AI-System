"""
Targeted Verification Suite for Phase 8H — Authentication & Assessment Result UI Redesign.
Verifies all Phase 8H requirements:
1. Result page rendering & dynamic data bindings (patient_name, patient_id, probability, risk level, recommendations, disclaimer, actions).
2. Login page structure, 'Doctor Authentication Portal', single-line CSRF token, show/hide toggle.
3. Valid doctor login and remember-me cookie persistence.
4. Invalid login credential handling and error display.
5. Forgot password: invalid recovery code rejection without redirect.
6. Forgot password: short password (< 8 chars) rejection without redirect.
7. Forgot password: mismatched confirm password rejection without redirect.
8. Forgot password: valid reset updates database, redirects (302) to /login, flashes success message, and enables login with new password.
9. Change password authenticated flow: auth enforcement, current password verification, length check, and update.
10. Strict XSS security compliance: Zero innerHTML occurrences in all redesigned templates.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from flask import render_template
from werkzeug.security import generate_password_hash, check_password_hash

from app import app
from database import get_db_connection, init_db


class Phase8HAuthAndResultTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False  # Allows standard test client POSTs while custom CSRF hook handles token
        cls.temp_dir = tempfile.mkdtemp(prefix="phase8h_auth_test_")
        cls.test_db = os.path.join(cls.temp_dir, "isolated_test.db")
        cls.orig_db_path = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = cls.test_db
        init_db()

        # Ensure ADMIN_RECOVERY_CODE is set for tests
        cls.orig_recovery_code = os.environ.get("ADMIN_RECOVERY_CODE")
        os.environ["ADMIN_RECOVERY_CODE"] = "raza1409"

        # Seed initial admin user in isolated database
        conn = get_db_connection()
        conn.execute("DELETE FROM users")
        initial_hash = generate_password_hash("DoctorPass123!")
        conn.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            ("admin", initial_hash)
        )

        # Seed sample patient in isolated database
        conn.execute("""
            INSERT INTO patients (
                patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal,
                result, probability, risk_level, timestamp
            ) VALUES (
                'PID_PHASE8H_TEST', 'Sarah Connor', 54, 0, 2, 130, 210, 0,
                0, 150, 0, 1.0, 1, 0, 1,
                'Absent', 24.5, 'LOW', '2026-11-14 11:00:00'
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

        if cls.orig_recovery_code is not None:
            os.environ["ADMIN_RECOVERY_CODE"] = cls.orig_recovery_code
        else:
            os.environ.pop("ADMIN_RECOVERY_CODE", None)

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def get_auth_client(self):
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["csrf_token"] = "valid_csrf_token"
        return client

    # =========================================================================
    # 1. Result Page Rendering & Data Bindings
    # =========================================================================

    def test_01_result_page_rendering_and_data_bindings(self):
        """Verify result.html renders patient info, probability, risk tiers, recommendations, disclaimer, and action links."""
        with app.test_request_context():
            # Test 1A: LOW risk rendering
            html_low = render_template(
                "result.html",
                patient_name="Sarah Connor",
                patient_id="PID_PHASE8H_001",
                prediction="Absent",
                probability=18.5,
                risk_level="LOW",
                timestamp="2026-11-14 12:00:00"
            )
            self.assertIn("AI Clinical Assessment Result", html_low)
            self.assertIn("Sarah Connor", html_low)
            self.assertIn("PID_PHASE8H_001", html_low)
            self.assertIn("18.5%", html_low)
            self.assertIn("LOW", html_low)
            self.assertIn("risk-badge-low", html_low)
            self.assertIn("bar-low", html_low)
            self.assertIn("Low immediate cardiovascular risk", html_low)
            self.assertIn("Clinical Decision Support Estimate", html_low)
            self.assertIn("View Detailed Report", html_low)
            self.assertIn("New Assessment", html_low)
            self.assertIn("/predict", html_low)
            self.assertIn("/patient/PID_PHASE8H_001", html_low)
            self.assertIn("/dashboard", html_low)
            self.assertIn("/history", html_low)

            # Test 1B: MODERATE risk rendering
            html_mod = render_template(
                "result.html",
                patient_name="",
                patient_id="PID_PHASE8H_002",
                prediction="Present",
                probability=54.2,
                risk_level="MODERATE",
                timestamp="2026-11-14 12:05:00"
            )
            self.assertIn("Name not available", html_mod)
            self.assertIn("PID_PHASE8H_002", html_mod)
            self.assertIn("54.2%", html_mod)
            self.assertIn("MODERATE", html_mod)
            self.assertIn("risk-badge-moderate", html_mod)
            self.assertIn("bar-moderate", html_mod)
            self.assertIn("Moderate risk detected", html_mod)

            # Test 1C: HIGH risk rendering
            html_high = render_template(
                "result.html",
                patient_name="John Doe, MD & Patient",
                patient_id="PID_PHASE8H_003",
                prediction="Present",
                probability=88.9,
                risk_level="HIGH",
                timestamp="2026-11-14 12:10:00"
            )
            self.assertIn("John Doe, MD &amp; Patient", html_high)  # Properly escaped
            self.assertIn("PID_PHASE8H_003", html_high)
            self.assertIn("88.9%", html_high)
            self.assertIn("HIGH", html_high)
            self.assertIn("risk-badge-high", html_high)
            self.assertIn("bar-high", html_high)
            self.assertIn("High cardiovascular risk detected", html_high)

    # =========================================================================
    # 2. Login Page Rendering & Single-line CSRF Contract
    # =========================================================================

    def test_02_login_page_renders_and_csrf_intact(self):
        """Verify login.html renders branding, inputs, and exact single-line CSRF token."""
        res = self.client.get("/login")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)

        # Exact test contracts
        self.assertIn("Doctor Authentication Portal", html)
        self.assertIn("Clinical AI System", html)
        self.assertIn('name="username"', html)
        self.assertIn('name="password"', html)
        self.assertIn('name="remember"', html)
        self.assertIn('togglePassword()', html)
        self.assertIn('href="/forgot_password"', html)

        # Verify CSRF token is on a single line so line.split('value="')[1] works
        csrf_lines = [line.strip() for line in html.splitlines() if 'name="csrf_token"' in line]
        self.assertTrue(len(csrf_lines) >= 1, "CSRF input field must be present")
        csrf_line = csrf_lines[0]
        self.assertIn('value="', csrf_line)
        extracted_token = csrf_line.split('value="')[1].split('"')[0]
        self.assertTrue(len(extracted_token) > 0, "CSRF token must not be empty")

    # =========================================================================
    # 3. Valid Login & Remember Me Cookie
    # =========================================================================

    def test_03_valid_login_and_remember_me(self):
        """Verify valid doctor credentials redirect to /dashboard and sets remember_user cookie when requested."""
        # Test with remember=on
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["csrf_token"] = "valid_token"

        res = client.post("/login", data={
            "username": "admin",
            "password": "DoctorPass123!",
            "remember": "on",
            "csrf_token": "valid_token"
        }, follow_redirects=False)

        self.assertEqual(res.status_code, 302)
        self.assertIn("/dashboard", res.headers.get("Location", ""))

        # Check session user
        with client.session_transaction() as sess:
            self.assertEqual(sess.get("user"), "admin")

        # Check remember_user cookie is set
        set_cookie_header = res.headers.get("Set-Cookie", "")
        self.assertIn("remember_user=admin", set_cookie_header)

        # Test without remember: cookie should be expired/cleared
        client2 = app.test_client()
        with client2.session_transaction() as sess:
            sess["csrf_token"] = "valid_token"

        res2 = client2.post("/login", data={
            "username": "admin",
            "password": "DoctorPass123!",
            "csrf_token": "valid_token"
        }, follow_redirects=False)

        self.assertEqual(res2.status_code, 302)
        set_cookie2 = res2.headers.get("Set-Cookie", "")
        self.assertIn("remember_user=", set_cookie2)

    # =========================================================================
    # 4. Invalid Login Credentials
    # =========================================================================

    def test_04_invalid_login_shows_error(self):
        """Verify bad credentials render 'Invalid Credentials' without setting session."""
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["csrf_token"] = "valid_token"

        res = client.post("/login", data={
            "username": "admin",
            "password": "WrongPassword999!",
            "csrf_token": "valid_token"
        }, follow_redirects=True)

        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Invalid Credentials", html)

        with client.session_transaction() as sess:
            self.assertNotIn("user", sess)

    # =========================================================================
    # 5. Forgot Password: Invalid Recovery Code
    # =========================================================================

    def test_05_forgot_password_invalid_recovery_code_fails(self):
        """Verify submitting incorrect recovery code returns error and does NOT redirect."""
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["csrf_token"] = "valid_token"

        res = client.post("/forgot_password", data={
            "username": "admin",
            "recovery_code": "wrong_recovery_code",
            "new_password": "NewValidPassword123!",
            "confirm_password": "NewValidPassword123!",
            "csrf_token": "valid_token"
        }, follow_redirects=False)

        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Invalid Recovery Code", html)

    # =========================================================================
    # 6. Forgot Password: Short Password (< 8 Chars)
    # =========================================================================

    def test_06_forgot_password_short_password_fails(self):
        """Verify submitting new password shorter than 8 characters returns error without redirect."""
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["csrf_token"] = "valid_token"

        res = client.post("/forgot_password", data={
            "username": "admin",
            "recovery_code": "raza1409",
            "new_password": "short",
            "confirm_password": "short",
            "csrf_token": "valid_token"
        }, follow_redirects=False)

        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Password must be at least 8 characters long", html)

    # =========================================================================
    # 7. Forgot Password: Mismatched Passwords
    # =========================================================================

    def test_07_forgot_password_mismatched_password_fails(self):
        """Verify submitting mismatched new and confirm passwords returns error without redirect."""
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["csrf_token"] = "valid_token"

        res = client.post("/forgot_password", data={
            "username": "admin",
            "recovery_code": "raza1409",
            "new_password": "ValidPassword123!",
            "confirm_password": "DifferentPassword456!",
            "csrf_token": "valid_token"
        }, follow_redirects=False)

        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("Passwords do not match", html)

    # =========================================================================
    # 8. Forgot Password: Valid Reset Redirects to /login with Flash
    # =========================================================================

    def test_08_forgot_password_valid_reset_redirects_to_login_with_flash(self):
        """Verify valid recovery code & new password updates database, redirects (302) to /login, flashes success, and permits login."""
        client = app.test_client()
        with client.session_transaction() as sess:
            sess["csrf_token"] = "valid_token"

        new_pass = "BrandNewDoctorPass999!"
        res = client.post("/forgot_password", data={
            "username": "admin",
            "recovery_code": "raza1409",
            "new_password": new_pass,
            "confirm_password": new_pass,
            "csrf_token": "valid_token"
        }, follow_redirects=False)

        # 1. Confirms HTTP 302 redirect to /login
        self.assertEqual(res.status_code, 302)
        self.assertIn("/login", res.headers.get("Location", ""))

        # 2. Confirms database updated with new hashed password
        conn = get_db_connection()
        row = conn.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()
        conn.close()
        self.assertIsNotNone(row)
        self.assertTrue(check_password_hash(row[0], new_pass))

        # 3. Follow redirect and confirm flash message rendered
        res_login = client.get("/login")
        self.assertEqual(res_login.status_code, 200)
        login_html = res_login.get_data(as_text=True)
        self.assertIn("Password reset successfully. Please log in with your new password.", login_html)

        # 4. Verify logging in with new password succeeds
        with client.session_transaction() as sess:
            sess["csrf_token"] = "login_token"

        res_auth = client.post("/login", data={
            "username": "admin",
            "password": new_pass,
            "csrf_token": "login_token"
        }, follow_redirects=False)

        self.assertEqual(res_auth.status_code, 302)
        self.assertIn("/dashboard", res_auth.headers.get("Location", ""))

        with client.session_transaction() as sess:
            self.assertEqual(sess.get("user"), "admin")

    # =========================================================================
    # 9. Change Password Authenticated Flow
    # =========================================================================

    def test_09_change_password_authenticated_flow(self):
        """Verify /change_password requires authentication, validates current password, length, and updates successfully."""
        # Unauthenticated request redirects to /login
        unauth = app.test_client()
        res_unauth = unauth.get("/change_password")
        self.assertEqual(res_unauth.status_code, 302)
        self.assertIn("/login", res_unauth.headers.get("Location", ""))

        # Authenticated client
        auth_client = self.get_auth_client()

        # GET change password page
        res_get = auth_client.get("/change_password")
        self.assertEqual(res_get.status_code, 200)
        self.assertIn("Change Password", res_get.get_data(as_text=True))

        # Reset admin password in DB to known state
        conn = get_db_connection()
        known_pass = "CurrentSecurePass123!"
        conn.execute("UPDATE users SET password = ? WHERE username = 'admin'",
                     (generate_password_hash(known_pass),))
        conn.commit()
        conn.close()

        # POST with incorrect current password
        res_wrong = auth_client.post("/change_password", data={
            "current_password": "WrongPassword!",
            "new_password": "ValidNewPassword456!",
            "csrf_token": "valid_csrf_token"
        })
        self.assertEqual(res_wrong.status_code, 200)
        self.assertIn("Current password incorrect", res_wrong.get_data(as_text=True))

        # POST with new password < 8 characters
        res_short = auth_client.post("/change_password", data={
            "current_password": known_pass,
            "new_password": "short",
            "csrf_token": "valid_csrf_token"
        })
        self.assertEqual(res_short.status_code, 200)
        self.assertIn("Password must be at least 8 characters long", res_short.get_data(as_text=True))

        # POST with valid change
        newer_pass = "SuperUpdatedPass789!"
        res_valid = auth_client.post("/change_password", data={
            "current_password": known_pass,
            "new_password": newer_pass,
            "csrf_token": "valid_csrf_token"
        })
        self.assertEqual(res_valid.status_code, 200)
        self.assertIn("Password updated successfully", res_valid.get_data(as_text=True))

        # Verify DB updated
        conn = get_db_connection()
        row = conn.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()
        conn.close()
        self.assertTrue(check_password_hash(row[0], newer_pass))

    # =========================================================================
    # 10. Security Compliance: Zero innerHTML in Redesigned Templates
    # =========================================================================

    def test_10_no_innerhtml_in_redesigned_templates(self):
        """Verify strict XSS security compliance: zero occurrences of innerHTML in all redesigned Phase 8H templates."""
        target_templates = [
            "templates/result.html",
            "templates/login.html",
            "templates/forgot_password.html",
            "templates/change_password.html"
        ]

        for tmpl_path in target_templates:
            with self.subTest(template=tmpl_path):
                self.assertTrue(os.path.exists(tmpl_path), f"Template {tmpl_path} must exist")
                with open(tmpl_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.assertNotIn("innerHTML", content, f"Forbidden innerHTML found in {tmpl_path}")


if __name__ == "__main__":
    unittest.main()
