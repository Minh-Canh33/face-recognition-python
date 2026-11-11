
import pickle
import numpy as np

from .Sqlite import connect_db, create_table


def insert_faces(name, encoding):
    """
    Luu mot face embedding vao SQLite.

    name: ten nguoi dang ky
    encoding: vector 128 chieu
    """

    name = str(name).strip()

    if not name:
        raise ValueError(
            "Person name cannot be empty."
        )

    encoding = np.asarray(
        encoding,
        dtype=np.float64
    )

    if encoding.shape != (128,):
        raise ValueError(
            "Face encoding must have 128 values."
        )

    if not np.all(np.isfinite(encoding)):
        raise ValueError(
            "Face encoding contains invalid values."
        )

    create_table()

    encoding_blob = pickle.dumps(
        encoding,
        protocol=pickle.HIGHEST_PROTOCOL
    )

    with connect_db() as conn:
        cursor = conn.execute(
            """
            INSERT INTO faces (name, encoding)
            VALUES (?, ?)
            """,
            (name, encoding_blob)
        )

        return cursor.lastrowid
