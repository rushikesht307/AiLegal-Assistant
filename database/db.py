import sqlite3
import os

DB_PATH = os.path.join("database", "legal_ai.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the documents table if it doesn't exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            file_id       TEXT PRIMARY KEY,
            file_name     TEXT,
            document_type TEXT,
            confidence    REAL,
            upload_time   TEXT,
            text_path     TEXT,
            status        TEXT
        )
    """)
    conn.commit()
    conn.close()
