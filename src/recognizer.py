import face_recognition
from database.load_faces import load_faces
import numpy as np

know_names,know_encodings = load_faces() #danh sách tên và vector ảnh trong db

def recognize_faces(rgb_frame,face_locations):
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []

    for encodeFace in face_encodings: #duyệt tất cả khuông mặt trong frame chụp được( các location)
        matches = face_recognition.compare_faces(know_encodings, encodeFace) # so sánh khuôn mặt -> true false
        faceDis = face_recognition.face_distance(know_encodings, encodeFace) # hiển thị khoảng cách euclidean giữa các vector -> 1d numpy array

        name = "Unknown"

        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex] and faceDis[matchIndex] < 0.5: 
                name = know_names[matchIndex]

        face_names.append(name)
    return face_names
