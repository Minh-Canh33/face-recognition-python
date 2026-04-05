import sqlite3
from pathlib import Path

DB_PATH = Path(r"D:\Face_recognizer\data\face.db")

def connect_db():
    conn = sqlite3.connect(DB_PATH)
    return conn

def create_table():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL ,
        encoding BLOB NOT NULL
        )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_table()