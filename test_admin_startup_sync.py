"""
Focused Test Suite: Database Startup, Admin Creation & Password Synchronization
Verifies:
1. Admin creation on fresh DB.
2. Existing password preservation when ADMIN_SYNC_ON_STARTUP=false.
3. Explicit password synchronization when ADMIN_SYNC_ON_STARTUP=true.
4. Initialization failure when ADMIN_DEFAULT_PASSWORD is missing in production.
5. Multi-worker concurrent safety and transaction resilience.
6. Preservation of exact password characters without silent trimming.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from werkzeug.security import check_password_hash, generate_password_hash

import database
from database import get_db_connection, init_db


class TestAdminStartupSync(unittest.TestCase):
    def setUp(self):
        # Create a completely isolated temporary directory and database for each test
        self.temp_dir = tempfile.mkdtemp(prefix="test_admin_startup_")
        self.test_db = os.path.join(self.temp_dir, "test_admin.db")

        # Snapshot current environment variables
        self.original_env = {
            "DATABASE_PATH": os.environ.get("DATABASE_PATH"),
            "RENDER": os.environ.get("RENDER"),
            "FLASK_ENV": os.environ.get("FLASK_ENV"),
            "ADMIN_DEFAULT_PASSWORD": os.environ.get("ADMIN_DEFAULT_PASSWORD"),
            "ADMIN_SYNC_ON_STARTUP": os.environ.get("ADMIN_SYNC_ON_STARTUP"),
        }
        self.original_db_path_attr = getattr(database, "DATABASE_PATH", None)

        # Point environment to test database
        os.environ["DATABASE_PATH"] = self.test_db
        database.DATABASE_PATH = self.test_db

    def tearDown(self):
        # Restore environment variables exactly
        for key, val in self.original_env.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

        if self.original_db_path_attr is not None:
            database.DATABASE_PATH = self.original_db_path_attr

        # Clean up temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_admin_creation_on_fresh_db(self):
        """Admin user and all 4 tables are created with ADMIN_DEFAULT_PASSWORD on a fresh DB."""
        test_pass = "FreshSecureAdminPass2026!"
        os.environ["ADMIN_DEFAULT_PASSWORD"] = test_pass
        os.environ["ADMIN_SYNC_ON_STARTUP"] = "false"

        init_db()

        conn = get_db_connection()
        try:
            # Check all 4 tables exist
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            self.assertIn("users", tables)
            self.assertIn("patients", tables)
            self.assertIn("diabetes_patients", tables)
            self.assertIn("breast_cancer_patients", tables)

            # Check admin row
            user_row = conn.execute("SELECT username, password FROM users WHERE username = 'admin'").fetchone()
            self.assertIsNotNone(user_row, "Admin user must exist")
            self.assertEqual(user_row["username"], "admin")
            self.assertTrue(check_password_hash(user_row["password"], test_pass))
        finally:
            conn.close()

    def test_02_existing_password_preservation_when_sync_false(self):
        """When ADMIN_SYNC_ON_STARTUP is false (default), existing credentials are never overwritten."""
        initial_pass = "InitialPass123"
        os.environ["ADMIN_DEFAULT_PASSWORD"] = initial_pass
        os.environ["ADMIN_SYNC_ON_STARTUP"] = "false"

        init_db()

        # Simulate user changing their password via Forgot Password / Change Password
        conn = get_db_connection()
        user_modified_pass = "UserChangedPassword456!@"
        conn.execute(
            "UPDATE users SET password = ? WHERE username = 'admin'",
            (generate_password_hash(user_modified_pass),)
        )
        conn.commit()
        modified_hash = conn.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()["password"]
        conn.close()

        # Restart application with ADMIN_DEFAULT_PASSWORD still set to initial_pass
        # and ADMIN_SYNC_ON_STARTUP unset/false
        os.environ.pop("ADMIN_SYNC_ON_STARTUP", None)
        init_db()

        conn2 = get_db_connection()
        try:
            current_row = conn2.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()
            self.assertEqual(current_row["password"], modified_hash, "Password hash must remain unchanged")
            self.assertTrue(check_password_hash(current_row["password"], user_modified_pass))
            self.assertFalse(check_password_hash(current_row["password"], initial_pass))
        finally:
            conn2.close()

    def test_03_explicit_password_synchronization_when_sync_true(self):
        """When ADMIN_SYNC_ON_STARTUP is true, admin password synchronizes with ADMIN_DEFAULT_PASSWORD."""
        initial_pass = "InitialPass123"
        os.environ["ADMIN_DEFAULT_PASSWORD"] = initial_pass
        os.environ["ADMIN_SYNC_ON_STARTUP"] = "false"

        init_db()

        # Simulate user changing password previously
        conn = get_db_connection()
        conn.execute(
            "UPDATE users SET password = ? WHERE username = 'admin'",
            (generate_password_hash("OldForgottenPass"),)
        )
        conn.commit()
        conn.close()

        # Now operator configures ADMIN_SYNC_ON_STARTUP=true with a new default password
        new_synced_pass = "NewlyConfiguredPassword2026#$"
        os.environ["ADMIN_DEFAULT_PASSWORD"] = new_synced_pass
        os.environ["ADMIN_SYNC_ON_STARTUP"] = "true"

        init_db()

        conn2 = get_db_connection()
        try:
            current_row = conn2.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()
            self.assertTrue(check_password_hash(current_row["password"], new_synced_pass),
                            "Password should be synchronized to the new default password")
            self.assertFalse(check_password_hash(current_row["password"], "OldForgottenPass"))
            self.assertFalse(check_password_hash(current_row["password"], initial_pass))
        finally:
            conn2.close()

    def test_04_initialization_failure_when_admin_default_password_missing_in_production(self):
        """init_db() must raise RuntimeError in production when ADMIN_DEFAULT_PASSWORD is not set."""
        os.environ["RENDER"] = "true"
        os.environ.pop("ADMIN_DEFAULT_PASSWORD", None)

        with self.assertRaises(RuntimeError) as ctx:
            init_db()
        self.assertIn("ADMIN_DEFAULT_PASSWORD environment variable is not configured", str(ctx.exception))

        # Also verify with FLASK_ENV=production
        os.environ.pop("RENDER", None)
        os.environ["FLASK_ENV"] = "production"

        with self.assertRaises(RuntimeError) as ctx2:
            init_db()
        self.assertIn("ADMIN_DEFAULT_PASSWORD environment variable is not configured", str(ctx2.exception))

        # Confirm no admin user was inserted
        conn = get_db_connection()
        try:
            admin_row = conn.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
            self.assertIsNone(admin_row, "Admin must not be created on missing production password")
        finally:
            conn.close()

    def test_05_multi_worker_concurrent_safety(self):
        """Simulate multiple concurrent workers calling init_db() simultaneously."""
        test_pass = "ConcurSafePass99!"
        os.environ["ADMIN_DEFAULT_PASSWORD"] = test_pass
        os.environ["ADMIN_SYNC_ON_STARTUP"] = "true"

        errors = []

        def worker_task(worker_id):
            try:
                init_db()
                return True
            except Exception as e:
                errors.append((worker_id, str(e)))
                return False

        # Run 8 concurrent worker initializations
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_task, i) for i in range(8)]
            results = [f.result() for f in futures]

        self.assertEqual(len(errors), 0, f"Concurrent workers encountered errors: {errors}")
        self.assertTrue(all(results))

        conn = get_db_connection()
        try:
            # Verify exactly 1 admin user exists
            admin_count = conn.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'").fetchone()[0]
            self.assertEqual(admin_count, 1, "There must be exactly 1 admin record despite concurrent workers")

            # Verify integrity
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            self.assertEqual(integrity, "ok")
        finally:
            conn.close()

    def test_06_exact_password_preservation_no_silent_trimming(self):
        """Ensure spaces and special characters are preserved without silent trimming."""
        pass_with_spaces = "   lead_trail_space_pass_!@#   "
        os.environ["ADMIN_DEFAULT_PASSWORD"] = pass_with_spaces

        init_db()

        conn = get_db_connection()
        try:
            row = conn.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()
            self.assertTrue(check_password_hash(row["password"], pass_with_spaces))
            # Trimmed password should fail
            self.assertFalse(check_password_hash(row["password"], pass_with_spaces.strip()))
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
