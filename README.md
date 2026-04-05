📦 Face Recognition Python

A simple Face Recognition System built with Python using computer vision and machine learning techniques.
Project supports face detection, recognition, training, and database storage.

🚀 Features
🎯 Real-time face detection (camera)
🧠 Face recognition with encoding
💾 Store and load face data using SQLite
🏗️ Modular structure (easy to extend)
🖥️ Simple GUI with Tkinter
📁 Project Structure
Face_recognizer/
│
├── src/                # Core logic
│   ├── detector.py     # Detect faces from image/frame
│   ├── recognizer.py   # Recognize faces
│   ├── trainer.py      # Train face encodings
│   ├── utils.py        # Helper functions
│   └── main.py         # Main pipeline (camera + processing)
│
├── database/           # Database handling
│   ├── Sqlite.py
│   ├── insert_faces.py
│   ├── load_faces.py
│   └── __init__.py
│
├── data/               # (ignored) database + raw data
│
├── testtk.py           # Tkinter UI demo
│
├── .gitignore
└── README.md
⚙️ Installation
1. Clone repo
git clone https://github.com/Minh-Canh33/face-recognition-python.git
cd face-recognition-python
2. Create virtual environment
python -m venv env-face
source env-face/Scripts/activate   # Windows
3. Install dependencies
pip install -r requirements.txt
▶️ Usage
🔹 Run main face recognition
python src/main.py
🔹 Run UI (Tkinter)
python testtk.py
🧠 How it works
📷 Capture frame from camera
🔍 Detect faces (detector.py)
🧬 Encode & compare (recognizer.py)
🗂️ Load known faces from database
🏷️ Display name on screen
💾 Database
Uses SQLite
Stored in:
data/face.db

⚠️ File is ignored in Git (.gitignore)

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
⭐ If you like this project

Give it a ⭐ on GitHub!
