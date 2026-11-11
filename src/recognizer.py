
import logging
import numpy as np
import face_recognition

from database.load_faces import load_faces


logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.50

known_names = []
known_encodings = np.empty(
    (0, 128),
    dtype=np.float64
)


def reload_known_faces():
    """
    Nap lai khuon mat tu SQLite.
    Goi sau khi them/xoa nguoi.
    """
    global known_names, known_encodings

    names, encodings = load_faces()

    known_names = list(names)

    if len(encodings) == 0:
        known_encodings = np.empty(
            (0, 128),
            dtype=np.float64
        )
    else:
        known_encodings = np.asarray(
            encodings,
            dtype=np.float64
        ).reshape(-1, 128)

    logger.info(
        "Loaded %d face records",
        len(known_names)
    )

    return len(known_names)


def match_face(
    encoding,
    threshold=DEFAULT_THRESHOLD
):
    """
    Tim khuon mat gan nhat trong DB.

    Returns:
        (name, distance)
    """

    if not 0 < threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if len(known_names) == 0:
        return "Unknown", None

    encoding = np.asarray(
        encoding,
        dtype=np.float64
    )

    if encoding.shape != (128,):
        raise ValueError(
            "Face encoding must have 128 values."
        )

    distances = face_recognition.face_distance(
        known_encodings,
        encoding
    )

    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])

    if best_distance <= threshold:
        return (
            known_names[best_index],
            best_distance
        )

    return "Unknown", best_distance


def recognize_faces(
    rgb_frame,
    face_locations,
    threshold=DEFAULT_THRESHOLD
):
    """
    Nhan dien tat ca khuon mat trong frame.

    rgb_frame: anh RGB
    face_locations: vi tri khuon mat
    """

    if rgb_frame is None:
        return []

    if not face_locations:
        return []

    encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    results = []

    for encoding in encodings:
        name, distance = match_face(
            encoding,
            threshold
        )

        results.append(name)

        logger.debug(
            "Result: %s | Distance: %s",
            name,
            distance
        )

    return results


# Nap du lieu lan dau khi khoi dong
reload_known_faces()
