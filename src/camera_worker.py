
import sys
import time
import logging
from threading import Event, Lock

import cv2
import numpy as np
import face_recognition

from PySide6.QtCore import QThread, Signal
from src.detector import detect_faces


logger = logging.getLogger(__name__)


class CameraWorker(QThread):
    frame_ready = Signal(object, list, float, float, float)
    error = Signal(str)

    def __init__(
        self,
        camera_index=0,
        threshold=0.50,
        process_every=5
    ):
        super().__init__()

        self.camera_index = camera_index
        self.process_every = max(
            1, int(process_every)
        )

        self._stop = Event()
        self._lock = Lock()

        self._threshold = float(threshold)
        self._names = []
        self._encodings = np.empty(
            (0, 128), dtype=np.float64
        )

    def set_profiles(self, names, encodings):
        """Cap nhat du lieu nhan dien."""
        if len(names) != len(encodings):
            raise ValueError(
                "Names and encodings count mismatch."
            )

        if len(encodings):
            matrix = np.asarray(
                encodings, dtype=np.float64
            ).reshape(-1, 128).copy()
        else:
            matrix = np.empty(
                (0, 128), dtype=np.float64
            )

        with self._lock:
            self._names = list(names)
            self._encodings = matrix

    def set_threshold(self, threshold):
        with self._lock:
            self._threshold = float(threshold)

    def stop(self):
        self._stop.set()

    def _recognize(self, rgb, locations):
        encodings = face_recognition.face_encodings(
            rgb, locations
        )

        with self._lock:
            known_names = self._names.copy()
            known_encodings = self._encodings.copy()
            threshold = self._threshold

        results = []

        for encoding in encodings:
            if len(known_names) == 0:
                results.append("Unknown")
                continue

            distances = face_recognition.face_distance(
                known_encodings, encoding
            )

            index = int(np.argmin(distances))

            if distances[index] <= threshold:
                results.append(known_names[index])
            else:
                results.append("Unknown")

        return results

    @staticmethod
    def _draw(frame, locations, names):
        output = frame.copy()

        for (top, right, bottom, left), name in zip(
            locations, names
        ):
            # Detector xu ly anh ty le 0.25
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (
                (0, 180, 80)
                if name != "Unknown"
                else (0, 80, 240)
            )

            cv2.rectangle(
                output,
                (left, top),
                (right, bottom),
                color, 2
            )

            cv2.putText(
                output,
                name,
                (left, max(25, top - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        return output

    def run(self):
        cap = None

        last_locations = []
        last_names = []

        frame_number = 0

        fps_counter = 0
        fps = 0.0
        fps_start = time.perf_counter()

        recognition_count = 0
        recognition_fps = 0.0
        recognition_latency = 0.0
        recognition_start = time.perf_counter()

        try:
            if sys.platform == "win32":
                cap = cv2.VideoCapture(
                    self.camera_index,
                    cv2.CAP_DSHOW
                )
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(
                        self.camera_index
                    )
            else:
                cap = cv2.VideoCapture(
                    self.camera_index
                )

            if not cap.isOpened():
                self.error.emit(
                    "Cannot open webcam."
                )
                return

            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH, 640
            )
            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT, 480
            )

            while not self._stop.is_set():
                success, frame = cap.read()

                if not success or frame is None:
                    self.error.emit(
                        "Cannot read camera frame."
                    )
                    break

                frame_number += 1

                if (
                    frame_number == 1
                    or frame_number % self.process_every == 0
                ):
                    ai_start = time.perf_counter()

                    locations, rgb_small = detect_faces(frame)

                    names = self._recognize(
                        rgb_small, locations
                    )

                    recognition_latency = (
                        time.perf_counter() - ai_start
                    ) * 1000

                    recognition_count += 1

                    last_locations = locations
                    last_names = names

                display = self._draw(
                    frame,
                    last_locations,
                    last_names
                )

                
                # Count processed frames
                fps_counter += 1

                now = time.perf_counter()
                elapsed = now - fps_start

                # Update FPS every second
                if elapsed >= 1.0:
                    fps = fps_counter / elapsed

                    recognition_fps = (
                        recognition_count / elapsed
                    )

                    fps_counter = 0
                    recognition_count = 0
                    fps_start = now

                self.frame_ready.emit(
                    (frame.copy(), display),
                    last_names.copy(),
                    fps,
                    recognition_fps,
                    recognition_latency
                )

                self.msleep(20)


                self.frame_ready.emit(
                    (frame.copy(), display),
                    last_names.copy(),
                    fps,
                    recognition_fps,
                    recognition_latency
                )

                self.msleep(20)

        except Exception as exc:
            logger.exception(
                "Camera worker failed"
            )
            self.error.emit(str(exc))

        finally:
            if cap is not None:
                cap.release()

            logger.info("Camera released")
