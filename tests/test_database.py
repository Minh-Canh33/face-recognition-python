
import numpy as np
import pytest

import database.Sqlite as sqlite_module

from database.insert_faces import insert_faces
from database.load_faces import load_faces
from src.people_manager import PeopleManager


@pytest.fixture
def test_database(tmp_path, monkeypatch):
    """
    Thay duong dan database bang file tam.
    Tuyet doi khong sua database that.
    """
    temp_db = tmp_path / "test_faces.db"

    monkeypatch.setattr(
        sqlite_module,
        "DB_PATH",
        temp_db
    )

    sqlite_module.create_table()

    return temp_db


def test_insert_face(test_database):
    encoding = np.zeros(128)

    record_id = insert_faces(
        "Alice",
        encoding
    )

    assert record_id > 0


def test_load_face(test_database):
    encoding = np.full(128, 0.25)

    insert_faces("Alice", encoding)

    names, encodings = load_faces()

    assert names == ["Alice"]

    np.testing.assert_allclose(
        encodings[0],
        encoding
    )


def test_multiple_faces(test_database):
    insert_faces(
        "Alice", np.zeros(128)
    )
    insert_faces(
        "Bob", np.ones(128)
    )

    manager = PeopleManager()

    assert manager.count() == 2


def test_rename_person(test_database):
    manager = PeopleManager()

    person_id = manager.register(
        "Alice",
        np.zeros(128)
    )

    changed = manager.rename(
        person_id,
        "Alice Updated"
    )

    assert changed is True

    names, _ = load_faces()

    assert "Alice Updated" in names
    assert "Alice" not in names


def test_delete_person(test_database):
    manager = PeopleManager()

    person_id = manager.register(
        "Alice",
        np.zeros(128)
    )

    deleted = manager.delete(person_id)

    assert deleted is True
    assert manager.count() == 0


def test_reject_invalid_encoding(test_database):
    with pytest.raises(ValueError):
        insert_faces(
            "Invalid",
            np.zeros(64)
        )


def test_reject_empty_name(test_database):
    with pytest.raises(ValueError):
        insert_faces(
            "",
            np.zeros(128)
        )
