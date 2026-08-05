import datetime
from .db import get_connection


def add_document(file_id, file_name, document_type, confidence, text_path, status="processed"):
    """Save all details of one uploaded document into the database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO documents
        (file_id, file_name, document_type, confidence, upload_time, text_path, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        file_id,
        file_name,
        document_type,
        confidence,
        datetime.datetime.now().strftime("%d %b %Y %H:%M"),
        text_path,
        status,
    ))
    conn.commit()
    conn.close()


def get_all_documents():
    """Return every stored document (for the reports list)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents ORDER BY upload_time DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_document(file_id):
    """Return one document by its id."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM documents WHERE file_id = ?", (file_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None
