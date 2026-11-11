
import numpy as np
import pytest

import src.recognizer as recognizer


@pytest.fixture
def mock_gallery(monkeypatch):
    """
    Tao du lieu khuon mat gia lap.
    Khong su dung anh khuon mat that.
    """
    names = ["Alice", "Bob"]

    encodings = np.array([
        np.zeros(128),
        np.ones(128)
    ], dtype=np.float64)

    monkeypatch.setattr(
        recognizer,
        "known_names",
        names
    )

    monkeypatch.setattr(
        recognizer,
        "known_encodings",
        encodings
    )

    return names, encodings


def test_exact_match(mock_gallery):
    """Embedding giong Alice phai nhan Alice."""
    embedding = np.zeros(128)

    name, distance = recognizer.match_face(
        embedding,
        threshold=0.5
    )

    assert name == "Alice"
    assert distance == pytest.approx(0.0)


def test_second_identity(mock_gallery):
    """Embedding giong Bob phai nhan Bob."""
    embedding = np.ones(128)

    name, distance = recognizer.match_face(
        embedding,
        threshold=0.5
    )

    assert name == "Bob"
    assert distance == pytest.approx(0.0)


def test_unknown_identity(mock_gallery):
    """Embedding xa ca hai nguoi phai Unknown."""
    embedding = np.full(128, 0.5)

    name, distance = recognizer.match_face(
        embedding,
        threshold=0.5
    )

    assert name == "Unknown"
    assert distance > 0.5


def test_empty_gallery(monkeypatch):
    """Database rong khong duoc crash."""
    monkeypatch.setattr(
        recognizer,
        "known_names",
        []
    )

    monkeypatch.setattr(
        recognizer,
        "known_encodings",
        np.empty((0, 128))
    )

    name, distance = recognizer.match_face(
        np.zeros(128)
    )

    assert name == "Unknown"
    assert distance is None


def test_invalid_embedding(mock_gallery):
    """Vector sai kich thuoc phai bi tu choi."""
    with pytest.raises(ValueError):
        recognizer.match_face(
            np.zeros(64)
        )


def test_invalid_threshold(mock_gallery):
    """Threshold ngoai pham vi phai bao loi."""
    with pytest.raises(ValueError):
        recognizer.match_face(
            np.zeros(128),
            threshold=1.5
        )
