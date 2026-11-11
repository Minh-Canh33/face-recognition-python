
import pickle
import numpy as np

from database.Sqlite import (
    connect_db,
    create_table
)

from database.insert_faces import insert_faces
from database.load_faces import load_faces


class PeopleManager:

    def __init__(self):
        create_table()

    def get_all(self):
        """Lay danh sach ban ghi."""
        with connect_db() as conn:
            rows = conn.execute("""
                SELECT id, name
                FROM faces
                ORDER BY id DESC
            """).fetchall()

        return rows

    def count(self):
        with connect_db() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM faces"
            ).fetchone()

        return row[0]

    def register(self, name, encoding):
        name = str(name).strip()

        if not name:
            raise ValueError(
                "Name cannot be empty."
            )

        vector = np.asarray(
            encoding, dtype=np.float64
        )

        if vector.shape != (128,):
            raise ValueError(
                "Invalid face embedding."
            )

        return insert_faces(name, vector)

    def rename(self, person_id, new_name):
        new_name = str(new_name).strip()

        if not new_name:
            raise ValueError(
                "Name cannot be empty."
            )

        with connect_db() as conn:
            cursor = conn.execute(
                """
                UPDATE faces
                SET name = ?
                WHERE id = ?
                """,
                (new_name, int(person_id))
            )

            return cursor.rowcount > 0

    def delete(self, person_id):
        with connect_db() as conn:
            cursor = conn.execute(
                """
                DELETE FROM faces
                WHERE id = ?
                """,
                (int(person_id),)
            )

            return cursor.rowcount > 0

    def load_profiles(self):
        """Lay embeddings cho camera worker."""
        return load_faces()
