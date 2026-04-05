from .Sqlite import connect_db
import pickle

def insert_faces(name,encoding):
    conn = connect_db()
    cursor = conn.cursor()

    encoding_blob = pickle.dumps(encoding)

    cursor.execute("""INSERT INTO faces (name, encoding) VALUES (?,?)""",(name,encoding_blob))

    conn.commit()
    conn.close()
