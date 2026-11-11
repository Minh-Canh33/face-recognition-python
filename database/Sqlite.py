
import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = BASE_DIR / "data" / "face.db"


def connect_db():
    """Tao ket noi SQLite."""
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=10
    )

    conn.execute("PRAGMA busy_timeout = 10000")

    return conn


def create_table():
    """Tao bang faces neu chua ton tai."""
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL
            )
        """)


if __name__ == "__main__":
    create_table()
    print(f"Database ready: {DB_PATH}")
