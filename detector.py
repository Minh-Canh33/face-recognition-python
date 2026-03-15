import dlib
import face_recognition 
import cv2 
import numpy as np
from datetime import datetime
from pathlib import Path
import os

path = "D:\Face_recognizer\data\Pic"
images = []
classname = []
mylist = os.listdir(path) #['Donal Trump.jpg', 'elon musk .jpg', 'Joker.jpg', 'Me.jpg']

for cl in mylist:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg) # lưu ảnh dưới dạng ma trận 

    #name = cl.split(',')
    #classname.append(name[0])
    classname.append(os.path.splitext(cl)[0]) # lưu tên người dùng 
# print(len(images))
# print(len(classname))

# step encoding
def encode(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #chuyển từ BGR (blue-grey-red) -> RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = encode(images)
print("Mã hóa thành công")
print(len(encodeListKnow))

#start webcam
cap = cv2.VideoCapture(0) #mở 1 webcap 

while True:
    ret, frame = cap.read()
    framS = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    framS = cv2.cvtColor(framS,cv2.COLOR_BGR2RGB)

    #xác định vị trí khuôn mặt và mã hóa
    facecurFrame = face_recognition.face_locations(framS)# lấy từng khuôn mặt và vị trí khuôn mặt hiện tại
    encodecurFrame = face_recognition.face_encodings(framS) # mã hóa từng khuôn mặt

    for encodeFace, faceLoc in zip(encodecurFrame,facecurFrame):
        matches = face_recognition.compare_faces(encodeListKnow,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if faceDis[matchIndex] < 0.5:
            name = classname[matchIndex].upper()
        else:
            name = "unknow"

        #in tên lên màn hình Frame
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*2,x2*2,y2*2,x1*2
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,name,(x2,y2),cv2.Formatter_FMT_DEFAULT,1,(255,255,255),2)


    cv2.imshow("Face - recognizer",frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


