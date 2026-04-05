from .Sqlite import connect_db
import pickle

def load_faces():
    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute("""SELECT name, encoding FROM faces""")
    rows = cursor.fetchall() # trả về danh sách các tuple

    names = []
    encodings = []

    for name,blob in rows:
        names.append(name)
        encodings.append(pickle.loads(blob))

    conn.close()
    return names,encodings