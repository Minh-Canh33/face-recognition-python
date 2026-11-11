

# Face Recognition — Real-Time Desktop Recognition & Evaluation

A Python desktop application for real-time face detection, face recognition, and identity management, built with PySide6, OpenCV, face_recognition/dlib, and SQLite.

The project combines a webcam-based recognition pipeline, a desktop interface for managing registered faces, automated tests, and an offline benchmark that reports identification accuracy, false acceptance/rejection rates, and processing latency.
Project scope: A learning and portfolio project demonstrating Python development, computer vision integration, desktop UI design, data persistence, and evaluation. It is not designed for security-critical authentication or surveillance.

Features

- Live recognition: Capture frames from a webcam, detect faces, compute 128-dimensional face embeddings, and display matched identities or Unknown.
- Responsive desktop UI: PySide6 interface with Live Recognition and People Management pages.
- Camera lifecycle management: Run capture and recognition in a QThread, with Start/Stop controls and proper camera release.
- People Management: Register a face from the camera, view stored records, rename entries, and delete selected records.
- Local storage: Store names and embeddings in SQLite (data/face.db).
- Configurable matching threshold: Adjust the face-distance threshold from the interface (default: 0.50).
- Live performance indicators: Display FPS, Recognition FPS, and the latest recognition-pipeline latency in milliseconds.
- Offline evaluation: Measure identification accuracy, FAR, FRR, wrong-identity rate, mean latency, and P95 latency on a labeled image dataset.
- Reports and tests: Export benchmark CSV/JSON reports and run automated tests using pytest.
  Tech Stack
  Component	Technology	Purpose
  Language	Python 3.11	Application logic
  Desktop UI	PySide6 (Qt for Python)	GUI and event handling
  Image and camera processing	OpenCV	Webcam frames and image conversion
  Face features and matching	face_recognition, dlib, NumPy	Face detection/encoding and distance-based matching
  Data storage	SQLite	Registered face records
  Tests	pytest	Unit tests for matching and database operations
  Reporting	CSV, JSON	Benchmark results

How It Works

1. Enrollment: Capture an image containing exactly one face, compute its 128-dimensional embedding, and save the name and embedding in SQLite.
2. Capture: OpenCV reads frames from the selected webcam inside a camera worker thread.
3. Detection and encoding: The application detects face locations and generates embeddings.
4. Matching: Each embedding is compared with registered embeddings using face distance. If the nearest distance is at or below the configured threshold, its name is returned; otherwise, the result is Unknown.
5. Display: The desktop UI shows the camera preview, labels, and performance indicators.
6. Evaluation: An independent labeled image set is used to calculate offline metrics and save reports.
   Important: The system uses pretrained face-recognition models; enrollment stores embeddings. It does not train a new neural network from scratch.
   Project Structure
   face-recognition-python/
   ├── src/
   │   ├── main.py                 # PySide6 desktop interface
   │   ├── camera_worker.py        # Webcam QThread + live recognition and timing
   │   ├── people_manager.py       # Manage SQLite records
   │   ├── detector.py             # Face detection helpers
   │   ├── recognizer.py           # Face embedding matching and reload
   │   ├── trainer.py              # Enroll images from data/Pic/
   │   ├── benchmark.py            # Offline evaluation + CSV/JSON reporting
   │   └── utils.py                # Other image/display helpers
   ├── database/
   │   ├── Sqlite.py               # SQLite connection and schema
   │   ├── insert_faces.py         # Store face embeddings
   │   ├── load_faces.py           # Load registered embeddings
   │   └── __init__.py
   ├── tests/
   │   ├── test_recognizer.py      # Recognition logic tests
   │   └── test_database.py        # SQLite CRUD tests
   ├── data/                       # Local, private data (not committed)
   │   ├── face.db
   │   ├── Pic/
   │   └── benchmark/
   │       ├── known/<registered-name></registered>/
   │       └── unknown/
   ├── reports/                    # Locally generated benchmark reports
   ├── requirements.txt
   ├── .gitignore
   └── README.md
   Installation (Windows)
7. Clone the repository
   git clone https://github.com/Minh-Canh33/face-recognition-python.git
   cd face-recognition-python
8. Create a Python 3.11 virtual environment
   py -3.11 -m venv env-face
   .\env-face\Scripts\Activate.ps1
   If PowerShell blocks activation, use Command Prompt (env-face\Scripts\activate.bat) or invoke env-face\Scripts\python.exe directly. Do not change execution policy unnecessarily.
9. Install dependencies
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   Make sure requirements.txt includes PySide6, opencv-python, face_recognition, numpy, and pytest (or install pytest separately). face_recognition depends on dlib and may require compatible prebuilt packages or C++ build tools on Windows; installation can vary by machine. Python 3.11 is the version used during this project's reported tests.
10. Initialize the database
    python -m database.Sqlite
11. Launch the desktop application
    python -m src.main
    Run commands from the repository root, not from inside src/.
    Using the Application
    Register a person
12. Open Live Recognition and click Start Camera.
13. Make sure exactly one face is visible in the current frame.
14. Open People Management, enter the person's name, and click Register from Camera.
15. The name and face embedding are saved in SQLite and made available for recognition without restarting the app.
    You can also rename or delete selected database records in People Management. Each row currently represents a stored face embedding, not necessarily a distinct real-world person; repeated names are possible.
    Recognize faces
16. Go to Live Recognition and click Start Camera.
17. The app displays matching names or Unknown.
18. Adjust the threshold if needed; threshold tuning should be validated on separate data rather than selected based on one example.
19. Click Stop Camera before exiting if desired; the worker also releases the webcam on shutdown.
    Read the performance indicators

- Display FPS: Processed/displayed frames per second, including frames without a fresh recognition pass.
- Recognition FPS: Number of completed detection/recognition passes per second under the current frame-skipping configuration.
- AI Latency: Time spent in the most recent detection-and-recognition pass, in milliseconds; not an average and not end-to-end UI latency.
  The current live pipeline processes recognition only on selected frames (process_every, default 5), so Display FPS and Recognition FPS measure different things.
  Offline Benchmark
  The benchmark compares predictions with ground-truth labels derived from folder names. It does not automatically know a person's identity.
  Prepare independent test images
  data/benchmark/
  ├── known/
  │   ├── Canh/
  │   │   ├── test1.jpg
  │   │   └── test2.jpg
  │   └── Binh/
  │       └── test1.jpg
  └── unknown/
  └── stranger1.jpg
- Names under known/ must exactly match names registered in SQLite (including capitalization and spaces).
- Put only people not enrolled in SQLite under unknown/.
- Use images that are different from the enrollment images, ideally taken under varied conditions.
- Each test image should contain exactly one clearly visible face; invalid images are counted separately and excluded from accuracy calculations.
- Use only images you have permission to process. Do not commit test photos or biometric data.
  Run the benchmark
  python -m src.benchmark --threshold 0.50
  Optionally compare thresholds:
  python -m src.benchmark --threshold 0.40
  python -m src.benchmark --threshold 0.60
  Choose thresholds on a validation set, then report final metrics on a separate test set. Do not tune and evaluate on the same small collection.
  Metrics
  Metric	Meaning
  Identification accuracy	Correct predictions divided by valid test images (known and unknown combined)
  FAR	Unknown-person images incorrectly accepted as any enrolled identity / valid unknown-person images
  FRR	Known-person images predicted Unknown / valid known-person images
  Wrong-identity rate	Known-person images incorrectly assigned another enrolled identity / valid known-person images
  Mean latency	Average processing time per valid test image, including image load, detection, encoding, and matching
  P95 latency	95th percentile of those per-image processing times

Benchmark reports are written to reports/ as detailed CSV, summary CSV, and summary JSON files.
Example result from a local smoke test
The following results were measured in a small local test on November 11, 2026, at threshold 0.50:
Metric	Measured value
Enrolled identities / embeddings	4 / 4
Test images	5 (4 known, 1 unknown)
Valid / invalid images	5 / 0
Correct predictions	5 / 5
Identification accuracy on this set	100.0%
FAR on this set	0 / 1 (0.0%)
FRR on this set	0 / 4 (0.0%)
Wrong-identity predictions	0 / 4
Mean processing latency	657.314 ms/image
P95 processing latency	879.851 ms/image

Interpretation: This was a successful functional smoke test, not evidence of 100% real-world face recognition accuracy. One unknown-person image is insufficient for a reliable FAR estimate. A larger, diverse, independent dataset is required before making general performance claims.
Automated Tests
Run from the repository root:
python -m pytest tests/ -v
Recorded result (November 11, 2026): 13 tests passed in 2.73 seconds under Windows / Python 3.11.9 / pytest 9.2.0. These unit tests cover SQLite CRUD and embedding-matching logic; they are not a substitute for testing actual recognition accuracy on images.
Known Limitations and Future Work

- Small benchmark dataset: Increase the number of known/unknown identities and independent images; test different lighting, pose, and camera conditions.
- Basic enrollment data model: One SQLite row represents one embedding. A future version could use separate persons and face_samples tables to support multiple enrollment samples per person more cleanly.
- Threshold selection: Evaluate FAR/FRR trade-offs on a validation set and document the chosen threshold.
- Performance: Recognition occurs on sampled frames; measure and profile detection/encoding separately before optimizing.
- Robustness: Face recognition may fail or misidentify people under poor lighting, occlusion, motion blur, or other changes.
- Security: No liveness detection or anti-spoofing. Do not use this prototype to grant physical or account access.
  Privacy and Data Handling
  Face images and embeddings are sensitive biometric data. Only enroll people who consent. Keep data/face.db, test images, screenshots containing personal information, and generated private reports out of Git. Support deleting data when no longer needed, and avoid distributing a real database with the source code.

Recommended .gitignore coverage:
env-face/
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.idea/
.vscode/
data/
reports/
Before publishing, review git status and previously tracked files: .gitignore does not automatically untrack files already committed.
Author
Đinh Minh Cảnh (Minh Canh)
GitHub: Minh-Canh33
Repository: https://github.com/Minh-Canh33/face-recognition-python

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

```text
Face_recognizer/
│
├── src/                    # Core logic
│   ├── detector.py         # Detect faces from image/frame
│   ├── recognizer.py       # Recognize faces
│   ├── trainer.py          # Train face encodings
│   ├── utils.py            # Helper functions
│   └── main.py             # Main pipeline (camera + processing)
│
├── database/               # Database handling
│   ├── Sqlite.py
│   ├── insert_faces.py
│   ├── load_faces.py
│   └── __init__.py
│
├── data/                   # (Ignored) database + raw data
├── env-face/               # Virtual environment
├── testtk.py               # Tkinter UI demo
├── requirements.txt
└── .gitignore
```

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
```
