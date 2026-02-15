import sqlite3
from werkzeug.security import generate_password_hash

def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)

    # Create default admin user (only once)
    hashed_password = generate_password_hash("admin123")

    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                       ("admin", hashed_password))
    except:
        pass

    conn.commit()
    conn.close()