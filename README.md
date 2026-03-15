# Face Recognition with Python

## 📌 Introduction

This project uses Python and computer vision techniques to perform real-time face recognition.
The system encodes known faces from images and compares them with faces detected from a webcam or screen capture.

## ⚙️ Technologies Used

* Python
* OpenCV
* face_recognition
* dlib
* NumPy

## 📂 Project Structure

```
Face_recognizer
│
├── data
│   └── Pic
│       ├── person1.jpg
│       ├── person2.jpg
│       └── person3.jpg
│
├── detector.py
├── requirements.txt
└── README.md
```

## 🚀 Features

* Detect faces in real time
* Encode faces from images
* Compare detected faces with known faces
* Display the recognized person's name

## 🛠 Installation

Clone the repository:

```
git clone https://github.com/yourusername/face-recognition-python.git
```

Install dependencies:

```
pip install -r requirements.txt
```

## ▶️ Run the project

```
python detector.py
```

Press **Q** to exit the program.

## 📸 Dataset

Images used for training should be placed in the `data/Pic` folder.
Each image filename represents the person's name.

Example:

```
ElonMusk.jpg
DonaldTrump.jpg
MyPhoto.jpg
```

## 📖 How it works

1. Load images from the dataset folder
2. Encode each face using the face_recognition library
3. Capture frames from the webcam
4. Detect faces in each frame
5. Compare with known encodings
6. Display the matched name on the screen

## 👤 Author

Minh Canh
