"""
Focused Test Suite: Temporary Production Database Backup Endpoint
Tests security, authentication, CSRF validation, password verification,
online backup consistency, and response headers on hotfix branch.
"""

import os
import io
import shutil
import sqlite3
import tempfile
import unittest
from werkzeug.security import generate_password_hash

import database
from database import get_db_connection, init_db


class TestProductionBackupEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create an isolated temporary directory and database
        cls.test_dir = tempfile.mkdtemp(prefix="test_backup_endpoint_")
        cls.test_db = os.path.join(cls.test_dir, "test_backup.db")

        # Snapshot and redirect DATABASE_PATH
        cls.orig_db_path = getattr(database, "DATABASE_PATH", None)
        database.DATABASE_PATH = cls.test_db
        os.environ["DATABASE_PATH"] = cls.test_db

        from app import app
        cls.app = app
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

        # Initialize isolated test database and seed known admin credentials
        init_db()
        cls.admin_pass = "TestSecureAdmin2026!"
        conn = get_db_connection()
        conn.execute("UPDATE users SET password = ? WHERE username = 'admin'",
                     (generate_password_hash(cls.admin_pass),))
        # Insert a sample patient record to verify data integrity in backup
        conn.execute("""
            INSERT INTO patients (patient_id, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, result, probability, risk_level)
            VALUES ('P_BACKUP_TEST', 'Backup Verification Patient', 55, 1, 2, 130, 240, 0, 1, 145, 0, 1.5, 1, 0, 3, 'Present', 70.0, 'HIGH')
        """)
        conn.commit()
        conn.close()

    @classmethod
    def tearDownClass(cls):
        # Restore original database path
        if cls.orig_db_path is not None:
            database.DATABASE_PATH = cls.orig_db_path
        os.environ.pop("DATABASE_PATH", None)

        # Cleanup isolated temporary directory
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_01_unauthenticated_get_redirects_to_login(self):
        """Unauthenticated GET request must redirect to /login."""
        response = self.client.get("/admin/backup-database", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))

    def test_02_unauthenticated_post_redirects_to_login(self):
        """Unauthenticated POST request must redirect to /login."""
        response = self.client.post("/admin/backup-database", data={"csrf_token": "foo", "admin_password": "bar"}, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))

    def test_03_non_admin_session_redirects_to_login(self):
        """Session with non-admin username must redirect to /login."""
        with self.client.session_transaction() as sess:
            sess["user"] = "nurse_joy"

        response = self.client.get("/admin/backup-database", follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers.get("Location", ""))

    def test_04_authenticated_get_renders_form_and_sets_csrf(self):
        """Authenticated admin GET must render form with explicit CSRF token."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"

        response = self.client.get("/admin/backup-database")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("Production Database Backup", html)
        self.assertIn('name="csrf_token"', html)
        self.assertIn('name="admin_password"', html)

        # Confirm token in session
        with self.client.session_transaction() as sess:
            token = sess.get("backup_csrf_token")
            self.assertIsNotNone(token)
            self.assertIn(f'value="{token}"', html)

    def test_05_post_missing_csrf_returns_400(self):
        """POST without CSRF token must return HTTP 400."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["backup_csrf_token"] = "valid_token_123"

        response = self.client.post("/admin/backup-database", data={"admin_password": self.admin_pass})
        self.assertEqual(response.status_code, 400)

    def test_06_post_invalid_csrf_returns_400(self):
        """POST with invalid/mismatched CSRF token must return HTTP 400."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["backup_csrf_token"] = "valid_token_123"

        response = self.client.post("/admin/backup-database", data={
            "csrf_token": "tampered_wrong_token",
            "admin_password": self.admin_pass
        })
        self.assertEqual(response.status_code, 400)

    def test_07_post_invalid_password_returns_403(self):
        """POST with valid CSRF token but wrong admin password must return HTTP 403."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["backup_csrf_token"] = "test_csrf_token_xyz"

        response = self.client.post("/admin/backup-database", data={
            "csrf_token": "test_csrf_token_xyz",
            "admin_password": "WrongPassword999!"
        })
        self.assertEqual(response.status_code, 403)

    def test_08_successful_backup_download_and_integrity(self):
        """Valid admin session, valid CSRF token, and correct password returns valid SQLite attachment."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"
            sess["backup_csrf_token"] = "valid_auth_token_777"

        response = self.client.post("/admin/backup-database", data={
            "csrf_token": "valid_auth_token_777",
            "admin_password": self.admin_pass
        })

        self.assertEqual(response.status_code, 200)

        # Check headers
        self.assertEqual(response.headers.get("Content-Type"), "application/x-sqlite3")
        self.assertIn("production_database_backup.sqlite", response.headers.get("Content-Disposition", ""))
        self.assertIn("no-store", response.headers.get("Cache-Control", ""))
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")

        # Verify SQLite binary data integrity
        backup_bytes = response.data
        self.assertTrue(len(backup_bytes) > 0)

        # Verify the backup bytes can be opened as a real SQLite database
        test_conn = sqlite3.connect(":memory:")
        test_conn.deserialize(backup_bytes)
        cursor = test_conn.cursor()

        # Verify table existence
        tables = [r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        self.assertIn("users", tables)
        self.assertIn("patients", tables)

        # Verify patient record was captured in the backup
        patient = cursor.execute("SELECT patient_name FROM patients WHERE patient_id = 'P_BACKUP_TEST'").fetchone()
        self.assertIsNotNone(patient)
        self.assertEqual(patient[0], "Backup Verification Patient")

        test_conn.close()

        # Verify CSRF token was invalidated after download
        with self.client.session_transaction() as sess:
            self.assertNotIn("backup_csrf_token", sess)

    def test_09_original_database_remains_unmodified(self):
        """Verify the source database was not modified or corrupted by backup operations."""
        conn = get_db_connection()
        patient_count = conn.execute("SELECT COUNT(*) FROM patients WHERE patient_id = 'P_BACKUP_TEST'").fetchone()[0]
        admin_user = conn.execute("SELECT username FROM users WHERE username = 'admin'").fetchone()
        conn.close()

        self.assertEqual(patient_count, 1)
        self.assertIsNotNone(admin_user)


if __name__ == "__main__":
    unittest.main()
