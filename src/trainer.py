import os
import face_recognition
import cv2
from database.insert_faces import insert_faces

path = r"D:\Face_recognizer\data\Pic"
def encode_to_db():
    my_list = os.listdir(path)

    for cl in my_list:
        img = cv2.imread(f"{path}\{cl}")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encoding = face_recognition.face_encodings(img)[0]

        name = os.path.splitext(cl)[0]
        if len(encoding) > 0:
            insert_faces(name,encoding)
            print("thêm " + name + " vào DB")
        
    print("Lưu mã hóa thành công")

if __name__ == "__main__":
    encode_to_db()
