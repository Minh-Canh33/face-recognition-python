
import pickle
import logging
import numpy as np

from .Sqlite import connect_db, create_table


logger = logging.getLogger(__name__)


def load_faces():
    """
    Doc embeddings tu SQLite.

    Returns:
        names: list[str]
        encodings: list[np.ndarray]
    """

    create_table()

    with connect_db() as conn:
        rows = conn.execute(
            """
            SELECT name, encoding
            FROM faces
            ORDER BY id
            """
        ).fetchall()

    names = []
    encodings = []

    for name, blob in rows:
        try:
            # Chi doc database cuc bo tin cay
            encoding = pickle.loads(blob)

            encoding = np.asarray(
                encoding,
                dtype=np.float64
            )

            if encoding.shape != (128,):
                raise ValueError(
                    "Invalid embedding shape."
                )

            if not np.all(np.isfinite(encoding)):
                raise ValueError(
                    "Embedding contains NaN or Inf."
                )

            names.append(name)
            encodings.append(encoding)

        except (
            ValueError,
            TypeError,
            pickle.UnpicklingError
        ) as exc:
            logger.warning(
                "Skipping invalid face record %s: %s",
                name,
                exc
            )

    return names, encodings
