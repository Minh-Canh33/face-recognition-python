# 📦 Face Recognition Python  

A simple **Face Recognition System** built with Python using computer vision and machine learning techniques.  

Project supports face detection, recognition, training, and database storage.  

---

## 🚀 Features  

- 🎯 Real-time face detection (camera)  
- 🧠 Face recognition with encoding  
- 💾 Store and load face data using SQLite  
- 🏗️ Modular structure (easy to extend)  
- 🖥️ Simple GUI with Tkinter  

---

## 📁 Project Structure  
ace_recognizer/
│
├── src/ # Core logic
│ ├── detector.py
│ ├── recognizer.py
│ ├── trainer.py
│ ├── utils.py
│ └── main.py
│
├── database/ # Database handling
│ ├── Sqlite.py
│ ├── insert_faces.py
│ ├── load_faces.py
│ └── init.py
│
---

## ⚙️ Installation  

### 1. Clone repository  

```bash
git clone https://github.com/Minh-Canh33/face-recognition-python.git
cd face-recognition-python
├── data/ # Database + raw data (ignored)
├── testtk.py # Tkinter UI demo
├── requirements.txt
├── .gitignore
└── README.md

### 2. Create virtual environment
python -m venv env-face
    Windows:
env-face\Scripts\activate
    Mac/Linux:
source env-face/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

🧠 How it works
    📷 Capture frame from camera
    🔍 Detect faces (detector.py)
    🧬 Encode & compare (recognizer.py)
    🗂️ Load known faces from database
    🏷️ Display name on screen
💾 Database
    Uses SQLite
    Stored in: data/face.db
⚠️ File is ignored in Git
    🛠️ Future Improvements
    🚀 Improve recognition accuracy
    ⚡ Optimize performance (reduce lag)
    🧑‍💻 Add user management UI
    ☁️ Deploy as web/app
📌 Notes
    Make sure camera is available
    Good lighting improves accuracy
    Large datasets improve recognition
👨‍💻 Author

Minh Canh

⭐ Support

If you like this project, give it a ⭐ on GitHub!
