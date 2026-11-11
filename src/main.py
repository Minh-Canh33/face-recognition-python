
import sys
import logging

import cv2
import face_recognition

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFrame,
    QLabel,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QInputDialog,
    QDoubleSpinBox
)

from src.camera_worker import CameraWorker
from src.people_manager import PeopleManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


class FaceApp(QMainWindow):

    def __init__(self):
        super().__init__()

        self.manager = PeopleManager()
        self.worker = None
        self.current_frame = None
        self.last_names = []
        self.closing = False

        self.setWindowTitle(
            "FaceID Studio - AI Recognition"
        )
        self.resize(1200, 760)
        self.setMinimumSize(950, 600)

        self.setup_ui()
        self.setup_style()
        self.refresh_people()

    # ====================================
    # MAIN UI
    # ====================================

    def setup_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(230)

        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(
            18, 30, 18, 20
        )
        side_layout.setSpacing(14)

        logo = QLabel("FACEID STUDIO")
        logo.setObjectName("logo")
        side_layout.addWidget(logo)

        version = QLabel(
            "AI Recognition System V2"
        )
        version.setObjectName("muted")
        side_layout.addWidget(version)

        side_layout.addSpacing(30)

        self.live_nav = QPushButton(
            "Live Recognition"
        )
        self.people_nav = QPushButton(
            "People Management"
        )

        self.live_nav.clicked.connect(
            lambda: self.change_page(0)
        )
        self.people_nav.clicked.connect(
            lambda: self.change_page(1)
        )

        side_layout.addWidget(self.live_nav)
        side_layout.addWidget(self.people_nav)
        side_layout.addStretch()

        side_layout.addWidget(
            QLabel("Python + OpenCV + PySide6")
        )

        layout.addWidget(sidebar)

        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)

        self.build_live_page()
        self.build_people_page()
        self.change_page(0)

    def change_page(self, index):
        self.pages.setCurrentIndex(index)

        self.live_nav.setProperty(
            "active", index == 0
        )
        self.people_nav.setProperty(
            "active", index == 1
        )

        for btn in (
            self.live_nav,
            self.people_nav
        ):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        if index == 1:
            self.refresh_people()

    # ====================================
    # LIVE PAGE
    # ====================================

    def build_live_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(
            30, 28, 30, 28
        )
        layout.setSpacing(18)

        title = QLabel("Live Recognition")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        desc = QLabel(
            "Real-time facial recognition "
            "using your webcam"
        )
        desc.setObjectName("muted")
        layout.addWidget(desc)

        self.camera_label = QLabel(
            "Camera is not running"
        )
        self.camera_label.setObjectName(
            "cameraView"
        )
        self.camera_label.setAlignment(
            Qt.AlignCenter
        )
        self.camera_label.setMinimumHeight(350)

        layout.addWidget(
            self.camera_label, 1
        )

        buttons = QHBoxLayout()

        self.start_btn = QPushButton(
            "Start Camera"
        )
        self.stop_btn = QPushButton(
            "Stop Camera"
        )

        self.start_btn.clicked.connect(
            self.start_camera
        )
        self.stop_btn.clicked.connect(
            self.stop_camera
        )

        self.stop_btn.setEnabled(False)

        buttons.addWidget(self.start_btn)
        buttons.addWidget(self.stop_btn)
        buttons.addStretch()

        layout.addLayout(buttons)

        
        stats = QHBoxLayout()
        stats.setSpacing(20)

        self.status_label = QLabel("Status: Idle")
        self.fps_label = QLabel("Display FPS: --")

        self.recognition_fps_label = QLabel(
            "Recognition FPS: --"
        )

        self.latency_label = QLabel(
            "AI Latency: --"
        )

        self.result_label = QLabel("Detected: --")

        stats.addWidget(self.status_label)
        stats.addWidget(self.fps_label)
        stats.addWidget(self.recognition_fps_label)
        stats.addWidget(self.latency_label)
        stats.addWidget(self.result_label)

        layout.addLayout(stats)


        settings = QHBoxLayout()

        settings.addWidget(
            QLabel("Recognition threshold:")
        )

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(
            0.30, 0.70
        )
        self.threshold_spin.setSingleStep(
            0.05
        )
        self.threshold_spin.setValue(0.50)

        self.threshold_spin.valueChanged.connect(
            self.update_threshold
        )

        settings.addWidget(
            self.threshold_spin
        )
        settings.addStretch()

        layout.addLayout(settings)
        self.pages.addWidget(page)

    # ====================================
    # PEOPLE PAGE
    # ====================================

    def build_people_page(self):
        page = QWidget()

        layout = QVBoxLayout(page)
        layout.setContentsMargins(
            30, 28, 30, 28
        )
        layout.setSpacing(16)

        title = QLabel("People Management")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        self.count_label = QLabel()
        self.count_label.setObjectName("muted")
        layout.addWidget(self.count_label)

        form = QHBoxLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText(
            "Enter person's name"
        )

        self.register_btn = QPushButton(
            "Register from Camera"
        )
        self.register_btn.clicked.connect(
            self.register_person
        )

        form.addWidget(self.name_input, 1)
        form.addWidget(self.register_btn)

        layout.addLayout(form)

        self.people_table = QTableWidget()
        self.people_table.setColumnCount(2)
        self.people_table.setHorizontalHeaderLabels(
            ["ID", "Name"]
        )
        self.people_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.people_table.setSelectionBehavior(
            QTableWidget.SelectRows
        )
        self.people_table.setSelectionMode(
            QTableWidget.SingleSelection
        )
        self.people_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )

        layout.addWidget(
            self.people_table, 1
        )

        actions = QHBoxLayout()

        self.refresh_btn = QPushButton(
            "Refresh"
        )
        self.rename_btn = QPushButton(
            "Rename Selected"
        )
        self.delete_btn = QPushButton(
            "Delete Selected"
        )

        self.refresh_btn.clicked.connect(
            self.refresh_people
        )
        self.rename_btn.clicked.connect(
            self.rename_person
        )
        self.delete_btn.clicked.connect(
            self.delete_person
        )

        actions.addWidget(self.refresh_btn)
        actions.addWidget(self.rename_btn)
        actions.addWidget(self.delete_btn)

        layout.addLayout(actions)

        info = QLabel(
            "Start the camera on Live Recognition, "
            "then return here to register a face."
        )
        info.setWordWrap(True)
        info.setObjectName("muted")

        layout.addWidget(info)

        self.pages.addWidget(page)

    # ====================================
    # DATABASE
    # ====================================

    def refresh_people(self):
        rows = self.manager.get_all()

        self.people_table.setRowCount(
            len(rows)
        )

        for row_index, (person_id, name) in enumerate(rows):
            id_item = QTableWidgetItem(
                str(person_id)
            )
            name_item = QTableWidgetItem(name)

            self.people_table.setItem(
                row_index, 0, id_item
            )
            self.people_table.setItem(
                row_index, 1, name_item
            )

        self.count_label.setText(
            f"Registered face records: {len(rows)}"
        )

    def selected_person_id(self):
        row = self.people_table.currentRow()

        if row < 0:
            return None

        item = self.people_table.item(row, 0)

        if item is None:
            return None

        return int(item.text())

    def reload_profiles(self):
        if self.worker is None:
            return

        names, encodings = (
            self.manager.load_profiles()
        )

        self.worker.set_profiles(
            names, encodings
        )

    def register_person(self):
        name = self.name_input.text().strip()

        if not name:
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Please enter a name."
            )
            return

        if (
            self.worker is None
            or self.current_frame is None
        ):
            QMessageBox.warning(
                self,
                "Camera Required",
                "Start the camera first."
            )
            return

        try:
            frame = self.current_frame.copy()

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            locations = (
                face_recognition.face_locations(rgb)
            )

            if len(locations) != 1:
                QMessageBox.warning(
                    self,
                    "Face Enrollment",
                    "Exactly one face must be "
                    "visible in the camera."
                )
                return

            encodings = (
                face_recognition.face_encodings(
                    rgb, locations
                )
            )

            if len(encodings) != 1:
                raise RuntimeError(
                    "Cannot generate face encoding."
                )

            record_id = self.manager.register(
                name,
                encodings[0]
            )

            self.reload_profiles()
            self.refresh_people()
            self.name_input.clear()

            QMessageBox.information(
                self,
                "Registration Successful",
                f"Registered {name}\n"
                f"Record ID: {record_id}"
            )

        except Exception as exc:
            logging.exception(
                "Face registration failed"
            )
            QMessageBox.critical(
                self,
                "Registration Error",
                str(exc)
            )

    def rename_person(self):
        person_id = self.selected_person_id()

        if person_id is None:
            QMessageBox.warning(
                self, "Selection Required",
                "Select a person first."
            )
            return

        row = self.people_table.currentRow()
        old_name = self.people_table.item(
            row, 1
        ).text()

        new_name, ok = QInputDialog.getText(
            self,
            "Rename Person",
            "New name:",
            text=old_name
        )

        if not ok:
            return

        try:
            changed = self.manager.rename(
                person_id, new_name
            )

            if changed:
                self.refresh_people()
                self.reload_profiles()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Rename Error",
                str(exc)
            )

    def delete_person(self):
        person_id = self.selected_person_id()

        if person_id is None:
            QMessageBox.warning(
                self, "Selection Required",
                "Select a record first."
            )
            return

        answer = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete face record #{person_id}?",
            QMessageBox.Yes | QMessageBox.No
        )

        if answer != QMessageBox.Yes:
            return

        try:
            self.manager.delete(person_id)

            self.refresh_people()
            self.reload_profiles()

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Delete Error",
                str(exc)
            )

    # ====================================
    # CAMERA
    # ====================================

    def start_camera(self):
        if self.worker is not None:
            return

        self.worker = CameraWorker(
            threshold=self.threshold_spin.value()
        )

        try:
            self.reload_profiles()

            self.worker.frame_ready.connect(
                self.show_frame
            )
            self.worker.error.connect(
                self.camera_error
            )
            self.worker.finished.connect(
                self.camera_finished
            )

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)

            self.status_label.setText(
                "Status: Starting..."
            )

            self.worker.start()

        except Exception as exc:
            self.worker = None
            QMessageBox.critical(
                self,
                "Camera Error",
                str(exc)
            )

    def stop_camera(self):
        if self.worker is None:
            return

        self.worker.stop()
        self.stop_btn.setEnabled(False)

        self.status_label.setText(
            "Status: Stopping..."
        )

    def camera_finished(self):
        self.worker = None
        self.current_frame = None
        self.last_names = []

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        self.camera_label.clear()
        self.camera_label.setText(
            "Camera is not running"
        )

        self.status_label.setText(
            "Status: Idle"
        )
        
        self.fps_label.setText("Display FPS: --")
        self.recognition_fps_label.setText(
            "Recognition FPS: --"
        )
        self.latency_label.setText(
            "AI Latency: --"
        )
        self.result_label.setText(
            "Detected: --"
        )

    def camera_error(self, message):
        self.status_label.setText(
            "Status: Camera Error"
        )

        QMessageBox.warning(
            self,
            "Camera Error",
            message
        )

        
    def show_frame(self,frames, names,fps,recognition_fps,recognition_latency):
        if self.worker is None:
            return

        original, display = frames

        self.current_frame = original.copy()
        self.last_names = list(names)

        rgb = cv2.cvtColor(
            display, cv2.COLOR_BGR2RGB
        )

        height, width, channels = rgb.shape

        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(image)

        pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.camera_label.setPixmap(pixmap)

        self.status_label.setText("Status: Running")

        self.fps_label.setText(
            f"Display FPS: {fps:.1f}"
        )

        self.recognition_fps_label.setText(
            f"Recognition FPS: {recognition_fps:.1f}"
        )

        self.latency_label.setText(
            f"AI Latency: {recognition_latency:.1f} ms"
        )

        self.result_label.setText(
            "Detected: " +
            (", ".join(names) if names else "--")
        )

    def update_threshold(self, value):
        if self.worker is not None:
            self.worker.set_threshold(value)

    # ====================================
    # STYLING
    # ====================================

    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0f172a;
                color: #e2e8f0;
                font-family: "Segoe UI";
                font-size: 13px;
            }

            QFrame#sidebar {
                background-color: #111827;
                border-right: 1px solid #334155;
            }

            QLabel#logo {
                font-size: 21px;
                font-weight: bold;
                color: #38bdf8;
            }

            QLabel#pageTitle {
                font-size: 26px;
                font-weight: bold;
                color: #f8fafc;
            }

            QLabel#muted {
                color: #94a3b8;
            }

            QLabel#cameraView {
                background-color: #020617;
                border: 1px solid #334155;
                border-radius: 12px;
                color: #94a3b8;
            }

            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 11px 15px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #1d4ed8;
            }

            QPushButton:disabled {
                background-color: #475569;
                color: #cbd5e1;
            }

            QPushButton[active="true"] {
                background-color: #0369a1;
            }

            QLineEdit, QDoubleSpinBox {
                background-color: #1e293b;
                border: 1px solid #475569;
                border-radius: 7px;
                padding: 9px;
                color: #f8fafc;
            }

            QTableWidget {
                background-color: #1e293b;
                border: 1px solid #334155;
                color: #f8fafc;
                gridline-color: #334155;
                selection-background-color: #1d4ed8;
            }

            QHeaderView::section {
                background-color: #334155;
                color: white;
                padding: 10px;
                border: none;
            }
        """)

    # ====================================
    # CLOSE APP
    # ====================================

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()

            if not self.worker.wait(5000):
                event.ignore()

                QMessageBox.warning(
                    self,
                    "Camera Busy",
                    "Camera processing has not "
                    "stopped yet. Try closing again."
                )
                return

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceApp()
    window.show()
    sys.exit(app.exec())
