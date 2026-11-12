import sqlite3
import os
from werkzeug.security import generate_password_hash


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join(BASE_DIR, "database.db"))


def get_db_connection():
    db_dir = os.path.dirname(DATABASE_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    conn = get_db_connection()
    cursor = conn.cursor()

    # USERS TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)

    # PATIENTS TABLE (Heart Disease)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
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

    # DIABETES PATIENTS TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diabetes_patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            age INTEGER,
            pregnancies INTEGER,
            glucose REAL,
            blood_pressure REAL,
            skin_thickness REAL,
            insulin REAL,
            bmi REAL,
            dpf REAL,
            result TEXT,
            probability REAL,
            risk_level TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # BREAST CANCER PATIENTS TABLE (WDBC Diagnostic)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS breast_cancer_patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            patient_name TEXT,
            age INTEGER,
            mean_radius REAL,
            mean_texture REAL,
            mean_perimeter REAL,
            mean_area REAL,
            mean_smoothness REAL,
            mean_compactness REAL,
            mean_concavity REAL,
            mean_concave_points REAL,
            mean_symmetry REAL,
            mean_fractal_dimension REAL,
            radius_error REAL,
            texture_error REAL,
            perimeter_error REAL,
            area_error REAL,
            smoothness_error REAL,
            compactness_error REAL,
            concavity_error REAL,
            concave_points_error REAL,
            symmetry_error REAL,
            fractal_dimension_error REAL,
            worst_radius REAL,
            worst_texture REAL,
            worst_perimeter REAL,
            worst_area REAL,
            worst_smoothness REAL,
            worst_compactness REAL,
            worst_concavity REAL,
            worst_concave_points REAL,
            worst_symmetry REAL,
            worst_fractal_dimension REAL,
            result TEXT,
            probability REAL,
            risk_level TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create default admin ONLY if it doesn't already exist
    cursor.execute(
        "SELECT id FROM users WHERE username = ?",
        ("admin",)
    )

    existing_admin = cursor.fetchone()

    if not existing_admin:
        default_admin_pass = os.getenv("ADMIN_DEFAULT_PASSWORD", "admin123")
        hashed_password = generate_password_hash(default_admin_pass)

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            ("admin", hashed_password)
        )

    conn.commit()
    conn.close()