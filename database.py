import sqlite3
import os
from werkzeug.security import generate_password_hash


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database.db")


def get_db_connection():
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

    # PATIENTS TABLE
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

    # Create default admin ONLY if it doesn't already exist
    cursor.execute(
        "SELECT id FROM users WHERE username = ?",
        ("admin",)
    )

    existing_admin = cursor.fetchone()

    if not existing_admin:
        hashed_password = generate_password_hash("admin123")

        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            ("admin", hashed_password)
        )

    conn.commit()
    conn.close()