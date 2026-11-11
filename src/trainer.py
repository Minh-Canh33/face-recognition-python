
import logging
from pathlib import Path

import cv2
import numpy as np
import face_recognition

from database.Sqlite import BASE_DIR
from database.insert_faces import insert_faces
from database.load_faces import load_faces


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

IMAGE_DIR = BASE_DIR / "data" / "Pic"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


def encode_to_db():
    """
    Doc anh, tao face embeddings va luu DB.

    Moi file anh can co dung 1 khuon mat.
    Ten file duoc su dung lam ten nguoi.
    """

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        logger.warning(
            "No images found in %s",
            IMAGE_DIR
        )
        return 0

    # Tranh them trung khi chay trainer lai
    existing_names, _ = load_faces()
    registered_names = set(existing_names)

    success_count = 0
    skipped_count = 0

    for image_path in image_paths:
        name = image_path.stem.strip()

        if not name:
            skipped_count += 1
            continue

        if name in registered_names:
            logger.info(
                "Already registered: %s",
                name
            )
            skipped_count += 1
            continue

        try:
            # Ho tro duong dan Unicode tren Windows
            image_bytes = np.frombuffer(
                image_path.read_bytes(),
                dtype=np.uint8
            )

            image = cv2.imdecode(
                image_bytes,
                cv2.IMREAD_COLOR
            )

            if image is None:
                raise ValueError(
                    "Cannot decode image."
                )

            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            locations = face_recognition.face_locations(
                rgb_image
            )

            if len(locations) != 1:
                logger.warning(
                    "%s: expected 1 face, found %d",
                    image_path.name,
                    len(locations)
                )
                skipped_count += 1
                continue

            encodings = face_recognition.face_encodings(
                rgb_image,
                locations
            )

            if len(encodings) != 1:
                logger.warning(
                    "%s: encoding failed",
                    image_path.name
                )
                skipped_count += 1
                continue

            record_id = insert_faces(
                name,
                encodings[0]
            )

            registered_names.add(name)
            success_count += 1

            logger.info(
                "Registered: %s | ID: %s",
                name,
                record_id
            )

        except Exception:
            logger.exception(
                "Failed to process %s",
                image_path.name
            )
            skipped_count += 1

    logger.info(
        "Finished | Added: %d | Skipped: %d",
        success_count,
        skipped_count
    )

    return success_count


if __name__ == "__main__":
    encode_to_db()
