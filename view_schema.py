import sqlite3

conn = sqlite3.connect("database/legal_ai.db")
cur = conn.cursor()
print("=== TABLE SCHEMA ===")
cur.execute("SELECT sql FROM sqlite_master WHERE type='table';")
for row in cur.fetchall():
    print(row[0])
print("\n=== COLUMNS ===")
cur.execute("PRAGMA table_info(documents);")
for col in cur.fetchall():
    print(f"{col[0]}. {col[1]} ({col[2]})")
print("\n=== STORED DATA ===")
cur.execute("SELECT * FROM documents;")
rows = cur.fetchall()
if rows:
    for r in rows:
        print(r)
else:
    print("No documents stored yet. Upload a file first!")
conn.close()