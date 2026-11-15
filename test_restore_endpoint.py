"""
Test Suite: Secure Production Database Restoration Endpoint
File: test_restore_endpoint.py

Verifies:
1. Authorization: Unauthenticated and non-admin requests are rejected.
2. CSRF Protection: Missing or invalid CSRF tokens are rejected.
3. Password Re-Authentication: Valid admin password required; incorrect password rejected.
4. Upload Streaming Size Limit: Files exceeding 25 MB are rejected.
5. Format Validation: Non-SQLite files rejected.
6. Integrity Validation: Corrupted SQLite files rejected.
7. Schema Validation: Incompatible schemas and missing tables/columns rejected.
8. Account Validation: Missing admin user or 0 patient records rejected.
9. Successful Atomic Restoration: Preserves exact admin hash, patient IDs, timestamps, clinical fields.
10. Concurrency Safety: Multi-threaded reader access during restoration.
11. Atomic Rollback: Preserves active database on failure.
12. Secure Cleanup: All temporary staging and snapshot files removed in success and failure paths.

SYNTHETIC DATA ONLY. NO PRODUCTION PATIENTS OR HASHES.
"""

import io
import os
import shutil
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from werkzeug.security import generate_password_hash, check_password_hash

import database
from database import init_db, get_db_connection
from app import app


class TestRestoreEndpoint(unittest.TestCase):

    def setUp(self):
        # 1. Create isolated temporary workspace for this test run
        self.temp_dir = tempfile.mkdtemp(prefix="test_restore_isolated_")
        self.active_db = os.path.join(self.temp_dir, "active.db")

        # Snapshot original database.DATABASE_PATH
        self.orig_db_path = getattr(database, "DATABASE_PATH", None)
        database.DATABASE_PATH = self.active_db

        # 2. Configure Flask test client
        app.config["TESTING"] = True
        app.config["SECRET_KEY"] = "test-secret-key-audit-restore-2026"
        self.client = app.test_client()

        # 3. Initialize active database
        init_db()

        # Seed pre-restore admin password and 1 synthetic pre-existing patient
        self.pre_restore_admin_pass = "PreRestoreAdminPass123!"
        conn = get_db_connection()
        conn.execute(
            "UPDATE users SET password = ? WHERE username = 'admin'",
            (generate_password_hash(self.pre_restore_admin_pass),)
        )
        conn.execute("""
            INSERT INTO patients (
                patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak, slope, ca, thal, result, probability, risk_level, timestamp
            ) VALUES (
                'PRE_PID_001', 'Pre Existing Synthetic Patient', 45, 1, 1, 120, 200, 0,
                0, 150, 0, 1.0, 1, 0, 3, 'Present', 65.0, 'MODERATE', '2026-01-01 10:00:00'
            )
        """)
        conn.commit()
        conn.close()

    def tearDown(self):
        # Restore original database.DATABASE_PATH
        if self.orig_db_path is not None:
            database.DATABASE_PATH = self.orig_db_path

        # Clean up isolated test directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_synthetic_backup(self, db_path, admin_pass="RestoredAdminPass2026!", patient_count=3, valid=True, missing_table=False, missing_col=False):
        """Helper to create a well-formed synthetic SQLite database file for testing."""
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        if not missing_table:
            cur.execute("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL
                )
            """)
            if admin_pass is not None:
                cur.execute(
                    "INSERT INTO users (username, password) VALUES (?, ?)",
                    ("admin", generate_password_hash(admin_pass))
                )

        if missing_col:
            # Table missing required clinical columns
            cur.execute("""
                CREATE TABLE patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    patient_name TEXT
                )
            """)
            for i in range(patient_count):
                cur.execute(
                    "INSERT INTO patients (patient_id, patient_name) VALUES (?, ?)",
                    (f"SYNTH_RESTORE_{i+1:03d}", f"Restored Synthetic Patient {i+1}")
                )
        else:
            cur.execute("""
                CREATE TABLE patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    patient_name TEXT,
                    age INTEGER,
                    sex INTEGER,
                    cp INTEGER,
                    trestbps INTEGER,
                    chol INTEGER,
                    fbs INTEGER,
                    restecg INTEGER,
                    thalach INTEGER,
                    exang INTEGER,
                    oldpeak REAL,
                    slope INTEGER,
                    ca INTEGER,
                    thal INTEGER,
                    result TEXT,
                    probability REAL,
                    risk_level TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            for i in range(patient_count):
                cur.execute("""
                    INSERT INTO patients (
                        patient_id, patient_name, age, sex, cp, trestbps, chol, fbs,
                        restecg, thalach, exang, oldpeak, slope, ca, thal, result, probability, risk_level, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"SYNTH_RESTORE_{i+1:03d}", f"Restored Synthetic Patient {i+1}", 50 + i, i % 2, 2, 130 + i, 210 + i, 0,
                    1, 140 + i, 0, 1.5, 2, 0, 3, "Present", 75.0 + i, "MODERATE", f"2026-03-0{i+1} 12:00:00"
                ))

        conn.commit()
        conn.close()

    def _login_as_admin(self):
        """Helper to set authenticated admin session."""
        with self.client.session_transaction() as sess:
            sess["user"] = "admin"

    def _get_csrf_token(self):
        """Helper to get a valid restore CSRF token."""
        self._login_as_admin()
        resp = self.client.get("/admin/restore-database")
        self.assertEqual(resp.status_code, 200)
        with self.client.session_transaction() as sess:
            return sess.get("restore_csrf_token")

    # =========================================================================
    # Authorization & Authentication Tests
    # =========================================================================

    def test_01_unauthenticated_access_rejected(self):
        """Unauthenticated requests must be redirected (GET) or aborted 403 (POST)."""
        get_resp = self.client.get("/admin/restore-database")
        self.assertEqual(get_resp.status_code, 302)
        self.assertIn("/login", get_resp.headers.get("Location", ""))

        post_resp = self.client.post("/admin/restore-database")
        self.assertEqual(post_resp.status_code, 403)

    def test_02_non_admin_session_rejected(self):
        """Non-admin authenticated sessions (e.g. standard doctor user) must be rejected."""
        with self.client.session_transaction() as sess:
            sess["user"] = "dr_regular"

        get_resp = self.client.get("/admin/restore-database")
        self.assertEqual(get_resp.status_code, 302)
        self.assertIn("/login", get_resp.headers.get("Location", ""))

        post_resp = self.client.post("/admin/restore-database")
        self.assertEqual(post_resp.status_code, 403)

    def test_03_csrf_validation_enforced(self):
        """POST without valid CSRF token must return HTTP 400 and leave DB untouched."""
        self._login_as_admin()

        # Missing CSRF
        resp = self.client.post(
            "/admin/restore-database",
            data={"admin_password": self.pre_restore_admin_pass},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn(b"Session security token expired or invalid", resp.data)

        # Invalid / tampered CSRF
        with self.client.session_transaction() as sess:
            sess["restore_csrf_token"] = "valid_token_32chars_long_12345"

        resp2 = self.client.post(
            "/admin/restore-database",
            data={
                "csrf_token": "tampered_token_value_xyz",
                "admin_password": self.pre_restore_admin_pass
            },
            content_type="multipart/form-data"
        )
        self.assertEqual(resp2.status_code, 400)

    def test_04_admin_password_reauthentication_enforced(self):
        """Incorrect admin password must return HTTP 403 and leave active DB untouched."""
        token = self._get_csrf_token()

        resp = self.client.post(
            "/admin/restore-database",
            data={
                "csrf_token": token,
                "admin_password": "WrongPassword999!"
            },
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 403)
        self.assertIn(b"Administrator password confirmation failed", resp.data)

        # Verify active DB untouched
        conn = get_db_connection()
        pcount = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        self.assertEqual(pcount, 1)
        patient_row = conn.execute("SELECT patient_id FROM patients").fetchone()
        self.assertEqual(patient_row["patient_id"], "PRE_PID_001")
        conn.close()

    # =========================================================================
    # Upload Validation Tests
    # =========================================================================

    def test_05_oversized_upload_rejected(self):
        """Uploads exceeding 25 MB must be rejected with HTTP 413 without exhausting memory."""
        token = self._get_csrf_token()

        # Simulate 26 MB stream of zeroes
        oversized_data = io.BytesIO(b"0" * (26 * 1024 * 1024))
        resp = None
        try:
            resp = self.client.post(
                "/admin/restore-database",
                data={
                    "csrf_token": token,
                    "admin_password": self.pre_restore_admin_pass,
                    "backup_file": (oversized_data, "oversized.sqlite")
                },
                content_type="multipart/form-data"
            )
            self.assertEqual(resp.status_code, 413)
            self.assertIn(b"exceeds maximum allowed size", resp.data)
        finally:
            oversized_data.close()
            if resp is not None:
                if hasattr(resp, "request") and hasattr(resp.request, "input_stream"):
                    try:
                        resp.request.input_stream.close()
                    except Exception:
                        pass
                resp.close()

    def test_06_reject_non_sqlite_file(self):
        """Non-SQLite files (e.g. text/html/malformed binary) must be rejected with HTTP 400."""
        token = self._get_csrf_token()

        fake_file = io.BytesIO(b"This is not a sqlite database file at all! Plaintext payload.\n" + b"A" * 500)
        try:
            resp = self.client.post(
                "/admin/restore-database",
                data={
                    "csrf_token": token,
                    "admin_password": self.pre_restore_admin_pass,
                    "backup_file": (fake_file, "fake.sqlite")
                },
                content_type="multipart/form-data"
            )
            self.assertEqual(resp.status_code, 400)
            self.assertIn(b"Not a recognized SQLite database", resp.data)
        finally:
            fake_file.close()

    def test_07_reject_corrupted_sqlite_file(self):
        """SQLite files failing PRAGMA integrity_check must be rejected with HTTP 400."""
        token = self._get_csrf_token()

        # Valid SQLite header followed by corrupted random bytes
        corrupt_bytes = b"SQLite format 3\x00" + b"\xFF" * 4096
        corrupt_stream = io.BytesIO(corrupt_bytes)
        try:
            resp = self.client.post(
                "/admin/restore-database",
                data={
                    "csrf_token": token,
                    "admin_password": self.pre_restore_admin_pass,
                    "backup_file": (corrupt_stream, "corrupt.sqlite")
                },
                content_type="multipart/form-data"
            )
            self.assertEqual(resp.status_code, 400)
            self.assertTrue(b"integrity check failed" in resp.data or b"malformed or corrupted" in resp.data)
        finally:
            corrupt_stream.close()

    def test_08_reject_schema_mismatch(self):
        """SQLite files missing required tables or required columns must be rejected with HTTP 400."""
        token = self._get_csrf_token()

        # Missing clinical columns
        bad_db_path = os.path.join(self.temp_dir, "bad_schema.sqlite")
        self._create_synthetic_backup(bad_db_path, missing_col=True)

        with open(bad_db_path, "rb") as f:
            resp = self.client.post(
                "/admin/restore-database",
                data={
                    "csrf_token": token,
                    "admin_password": self.pre_restore_admin_pass,
                    "backup_file": (f, "bad_schema.sqlite")
                },
                content_type="multipart/form-data"
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn(b"missing required clinical columns", resp.data)

    def test_09_reject_missing_admin_user(self):
        """Uploads without an admin user in the users table must be rejected with HTTP 400."""
        token = self._get_csrf_token()

        no_admin_path = os.path.join(self.temp_dir, "no_admin.sqlite")
        self._create_synthetic_backup(no_admin_path, admin_pass=None)

        with open(no_admin_path, "rb") as f:
            resp = self.client.post(
                "/admin/restore-database",
                data={
                    "csrf_token": token,
                    "admin_password": self.pre_restore_admin_pass,
                    "backup_file": (f, "no_admin.sqlite")
                },
                content_type="multipart/form-data"
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn(b"Administrator account not found", resp.data)

    def test_10_reject_empty_patient_records(self):
        """Uploads with zero patient records must be rejected to prevent accidental wiping."""
        token = self._get_csrf_token()

        empty_patients_path = os.path.join(self.temp_dir, "empty_patients.sqlite")
        self._create_synthetic_backup(empty_patients_path, patient_count=0)

        with open(empty_patients_path, "rb") as f:
            resp = self.client.post(
                "/admin/restore-database",
                data={
                    "csrf_token": token,
                    "admin_password": self.pre_restore_admin_pass,
                    "backup_file": (f, "empty_patients.sqlite")
                },
                content_type="multipart/form-data"
            )
        self.assertEqual(resp.status_code, 400)
        self.assertIn(b"contains no patient records", resp.data)

    # =========================================================================
    # Successful Restoration & Preservation Tests
    # =========================================================================

    def test_11_successful_atomic_restoration(self):
        """Verify successful end-to-end restoration with exact data preservation."""
        token = self._get_csrf_token()

        new_admin_pass = "CompletelyNewRestoredAdminPass2026#$"
        synth_backup_path = os.path.join(self.temp_dir, "valid_synth_backup.sqlite")
        self._create_synthetic_backup(synth_backup_path, admin_pass=new_admin_pass, patient_count=3)

        with open(synth_backup_path, "rb") as f:
            resp = self.client.post(
                "/admin/restore-database",
                data={
                    "csrf_token": token,
                    "admin_password": self.pre_restore_admin_pass,
                    "backup_file": (f, "valid_synth_backup.sqlite")
                },
                content_type="multipart/form-data"
            )

        self.assertEqual(resp.status_code, 302)
        self.assertIn("/dashboard", resp.headers.get("Location", ""))

        # Verify active database content after restore
        conn = get_db_connection()
        try:
            # 1. PRAGMA integrity check
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            self.assertEqual(integrity, "ok")

            # 2. Patients table has exactly the 3 restored synthetic records
            patients = conn.execute("SELECT patient_id, patient_name, age, timestamp FROM patients ORDER BY id").fetchall()
            self.assertEqual(len(patients), 3)
            self.assertEqual(patients[0]["patient_id"], "SYNTH_RESTORE_001")
            self.assertEqual(patients[0]["patient_name"], "Restored Synthetic Patient 1")
            self.assertEqual(patients[0]["timestamp"], "2026-03-01 12:00:00")
            self.assertEqual(patients[1]["patient_id"], "SYNTH_RESTORE_002")
            self.assertEqual(patients[2]["patient_id"], "SYNTH_RESTORE_003")

            # 3. Admin credentials match the restored backup
            admin_row = conn.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()
            self.assertTrue(check_password_hash(admin_row["password"], new_admin_pass))
            self.assertFalse(check_password_hash(admin_row["password"], self.pre_restore_admin_pass))
        finally:
            conn.close()

    # =========================================================================
    # Concurrency & Lock Handling Tests
    # =========================================================================

    def test_12_concurrent_access_during_restoration(self):
        """Verify that concurrent reader threads query the database safely without lock crashes."""
        token = self._get_csrf_token()
        synth_backup_path = os.path.join(self.temp_dir, "concurrent_synth_backup.sqlite")
        self._create_synthetic_backup(synth_backup_path, patient_count=5)

        reader_errors = []
        stop_readers = False

        def reader_task():
            while not stop_readers:
                try:
                    c = get_db_connection()
                    c.execute("PRAGMA busy_timeout = 30000")
                    c.execute("SELECT COUNT(*) FROM patients").fetchone()
                    c.close()
                except Exception as e:
                    reader_errors.append(str(e))

        # Launch concurrent background reader threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(reader_task) for _ in range(4)]

            # Perform restore while readers are querying
            with open(synth_backup_path, "rb") as f:
                resp = self.client.post(
                    "/admin/restore-database",
                    data={
                        "csrf_token": token,
                        "admin_password": self.pre_restore_admin_pass,
                        "backup_file": (f, "concurrent_synth_backup.sqlite")
                    },
                    content_type="multipart/form-data"
                )

            stop_readers = True

        self.assertEqual(resp.status_code, 302)
        self.assertEqual(len(reader_errors), 0, f"Concurrent readers failed: {reader_errors}")

        conn = get_db_connection()
        pcount = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        self.assertEqual(pcount, 5)
        conn.close()

    # =========================================================================
    # Failure Recovery & Rollback Tests
    # =========================================================================

    def test_13_atomic_rollback_on_failure(self):
        """Verify that when an error occurs during post-restoration initialization, active DB is rolled back."""
        token = self._get_csrf_token()

        # Capture pre-restore active records
        conn_pre = get_db_connection()
        pre_patients = [dict(r) for r in conn_pre.execute("SELECT patient_id, patient_name FROM patients").fetchall()]
        conn_pre.close()

        # Submit an upload that will trigger a failure
        synth_backup_path = os.path.join(self.temp_dir, "rollback_test.sqlite")
        self._create_synthetic_backup(synth_backup_path, patient_count=3)

        # Mock init_db to simulate an unexpected error during post-restore initialization
        import app as app_module
        orig_init_db = app_module.init_db

        def failing_init_db():
            raise RuntimeError("Simulated failure in post-restoration migration")

        app_module.init_db = failing_init_db

        try:
            with open(synth_backup_path, "rb") as f:
                resp = self.client.post(
                    "/admin/restore-database",
                    data={
                        "csrf_token": token,
                        "admin_password": self.pre_restore_admin_pass,
                        "backup_file": (f, "rollback_test.sqlite")
                    },
                    content_type="multipart/form-data"
                )

            # Route catches error, executes rollback, and returns 500
            self.assertEqual(resp.status_code, 500)
            self.assertIn(b"Active database was successfully rolled back and preserved", resp.data)

            # Verify active DB was rolled back completely
            conn_post = get_db_connection()
            post_patients = [dict(r) for r in conn_post.execute("SELECT patient_id, patient_name FROM patients").fetchall()]
            conn_post.close()

            self.assertEqual(pre_patients, post_patients, "Active database was not rolled back to original state!")
        finally:
            app_module.init_db = orig_init_db

    def test_14_temp_file_cleanup(self):
        """Verify that no temporary staging files remain in tempdir after success or failure."""
        temp_system_dir = tempfile.gettempdir()
        initial_stages = [f for f in os.listdir(temp_system_dir) if f.startswith("restore_stage_") or f.startswith("active_snap_")]

        token = self._get_csrf_token()
        synth_backup_path = os.path.join(self.temp_dir, "cleanup_test.sqlite")
        self._create_synthetic_backup(synth_backup_path, patient_count=2)

        with open(synth_backup_path, "rb") as f:
            self.client.post(
                "/admin/restore-database",
                data={
                    "csrf_token": token,
                    "admin_password": self.pre_restore_admin_pass,
                    "backup_file": (f, "cleanup_test.sqlite")
                },
                content_type="multipart/form-data"
            )

        post_stages = [f for f in os.listdir(temp_system_dir) if f.startswith("restore_stage_") or f.startswith("active_snap_")]
        new_leaked_files = set(post_stages) - set(initial_stages)
        self.assertEqual(len(new_leaked_files), 0, f"Leaked temporary files detected: {new_leaked_files}")

    def test_15_production_secret_key_enforcement(self):
        """In production (RENDER=true or FLASK_ENV=production), SECRET_KEY must be configured."""
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.abspath(__file__))

        # Subprocess without SECRET_KEY in production mode (RENDER=true)
        env_missing = os.environ.copy()
        env_missing["RENDER"] = "true"
        env_missing.pop("SECRET_KEY", None)
        env_missing["PYTHONPATH"] = repo_root

        proc = subprocess.run(
            [sys.executable, "-c", "import app"],
            cwd=self.temp_dir,
            env=env_missing,
            capture_output=True,
            text=True
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("SECRET_KEY environment variable is not configured", proc.stderr)

        # Subprocess without SECRET_KEY in production mode (FLASK_ENV=production)
        env_missing_flask = os.environ.copy()
        env_missing_flask["FLASK_ENV"] = "production"
        env_missing_flask.pop("SECRET_KEY", None)
        env_missing_flask.pop("RENDER", None)
        env_missing_flask["PYTHONPATH"] = repo_root

        proc_flask = subprocess.run(
            [sys.executable, "-c", "import app"],
            cwd=self.temp_dir,
            env=env_missing_flask,
            capture_output=True,
            text=True
        )
        self.assertNotEqual(proc_flask.returncode, 0)
        self.assertIn("SECRET_KEY environment variable is not configured", proc_flask.stderr)

        # Subprocess with SECRET_KEY in production mode succeeds
        env_valid = os.environ.copy()
        env_valid["RENDER"] = "true"
        env_valid["SECRET_KEY"] = "configured-production-secret-12345"
        env_valid["PYTHONPATH"] = repo_root

        proc_valid = subprocess.run(
            [sys.executable, "-c", "import app; assert app.app.secret_key == 'configured-production-secret-12345'"],
            cwd=repo_root,
            env=env_valid,
            capture_output=True,
            text=True
        )
        self.assertEqual(proc_valid.returncode, 0, f"Import failed: {proc_valid.stderr}")

    def test_16_rollback_failure_alert_and_distinction(self):
        """When rollback fails, system must alert admin critically and never claim DB was preserved."""
        token = self._get_csrf_token()

        synth_backup_path = os.path.join(self.temp_dir, "rollback_fail_test.sqlite")
        self._create_synthetic_backup(synth_backup_path, patient_count=2)

        import app as app_module
        orig_rollback = app_module._perform_atomic_rollback

        # Mock rollback failure
        def failing_rollback(snap, target):
            return False, "Simulated disk corruption during rollback"

        app_module._perform_atomic_rollback = failing_rollback

        # Force init_db to fail to trigger rollback
        orig_init_db = app_module.init_db

        def failing_init_db():
            raise RuntimeError("Trigger failure to initiate rollback")

        app_module.init_db = failing_init_db

        try:
            with open(synth_backup_path, "rb") as f:
                resp = self.client.post(
                    "/admin/restore-database",
                    data={
                        "csrf_token": token,
                        "admin_password": self.pre_restore_admin_pass,
                        "backup_file": (f, "rollback_fail_test.sqlite")
                    },
                    content_type="multipart/form-data"
                )

            self.assertEqual(resp.status_code, 500)
            # Must explicitly warn that rollback failed
            self.assertIn(b"CRITICAL: Database restoration failed and active database rollback failed", resp.data)
            # Must NEVER claim that database was preserved
            self.assertNotIn(b"successfully rolled back and preserved", resp.data)
        finally:
            app_module._perform_atomic_rollback = orig_rollback
            app_module.init_db = orig_init_db

    def test_17_multiprocess_concurrent_readers_during_actual_restoration(self):
        """Verify that concurrent reader OS processes query the database safely during actual HTTP restoration without lock crashes or corruption."""
        import subprocess
        import sys
        import time

        token = self._get_csrf_token()
        synth_backup_path = os.path.join(self.temp_dir, "mp_restore_backup.sqlite")
        self._create_synthetic_backup(synth_backup_path, patient_count=5)

        reader_script = f"""
import sqlite3
import time
db = r'{self.active_db}'
errors = []
reads = 0
start = time.time()
while time.time() - start < 3.0:
    try:
        c = sqlite3.connect(db, timeout=30.0)
        c.execute("PRAGMA busy_timeout = 30000")
        row = c.execute("SELECT COUNT(*) FROM patients").fetchone()
        c.close()
        reads += 1
        time.sleep(0.01)
    except Exception as e:
        errors.append(str(e))
print(f"READER_READS:{{reads}},ERRORS:{{len(errors)}}")
if errors:
    print("ERROR_DETAILS:", errors[:5])
"""

        p1 = subprocess.Popen([sys.executable, "-c", reader_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        p2 = subprocess.Popen([sys.executable, "-c", reader_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            # Allow reader processes to start querying the active database
            time.sleep(0.2)

            # Perform actual HTTP multipart restoration
            with open(synth_backup_path, "rb") as f:
                resp = self.client.post(
                    "/admin/restore-database",
                    data={
                        "csrf_token": token,
                        "admin_password": self.pre_restore_admin_pass,
                        "backup_file": (f, "mp_restore_backup.sqlite")
                    },
                    content_type="multipart/form-data"
                )

            self.assertEqual(resp.status_code, 302)

            out1, err1 = p1.communicate(timeout=10)
            out2, err2 = p2.communicate(timeout=10)

            self.assertEqual(p1.returncode, 0, f"Process 1 failed with stderr: {err1}")
            self.assertEqual(p2.returncode, 0, f"Process 2 failed with stderr: {err2}")
            self.assertIn("ERRORS:0", out1, f"Process 1 encountered query errors: {out1}")
            self.assertIn("ERRORS:0", out2, f"Process 2 encountered query errors: {out2}")

            # Verify active DB was cleanly restored and integrity check passes
            conn = get_db_connection()
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            self.assertEqual(integrity, "ok")
            pcount = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            self.assertEqual(pcount, 5)
            conn.close()
        finally:
            if p1.poll() is None:
                p1.kill()
            if p2.poll() is None:
                p2.kill()


if __name__ == "__main__":
    unittest.main()
