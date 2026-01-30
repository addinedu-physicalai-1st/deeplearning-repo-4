import sys
import os
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
    def run(self):
        while self._run_flag:
            start_time = time.time()
            ret, cv_img = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
            
            elapsed = time.time() - start_time
            delay = max(0.001, 0.033 - elapsed)
            time.sleep(delay)
            
    def stop(self):
        self._run_flag = False
        self.wait()
        self.cap.release()

class ModelComparator(QMainWindow):
    tasks_result_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MP Legacy vs MP Tasks Comparison")
        self.resize(1280, 520)
        
        # UI Elements
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        
        # Legacy View (Left)
        self.legacy_layout = QVBoxLayout()
        self.legacy_label = QLabel("Legacy MP")
        self.legacy_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.legacy_video = QLabel()
        self.legacy_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.legacy_layout.addWidget(self.legacy_label)
        self.legacy_layout.addWidget(self.legacy_video)
        self.layout.addLayout(self.legacy_layout)
        
        # Tasks View (Right)
        self.tasks_layout = QVBoxLayout()
        self.tasks_label = QLabel("Tasks MP (New)")
        self.tasks_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tasks_video = QLabel()
        self.tasks_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tasks_layout.addWidget(self.tasks_label)
        self.tasks_layout.addWidget(self.tasks_video)
        self.layout.addLayout(self.tasks_layout)

        # ---------------------------------------------------------
        # 1. Setup Legacy MP
        # ---------------------------------------------------------
        self.mp_hands_legacy = mp.solutions.hands
        self.hands_legacy = self.mp_hands_legacy.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing_legacy = mp.solutions.drawing_utils

        # ---------------------------------------------------------
        # 2. Setup Tasks MP
        # ---------------------------------------------------------
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        model_path = os.path.join(project_root, 'models', 'hand_landmarker.task')
        
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", f"Model not found at: {model_path}")
            sys.exit(1)

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self.handle_tasks_result
        )
        self.landmarker_tasks = vision.HandLandmarker.create_from_options(options)

        # State for Tasks results (async)
        self.latest_tasks_landmarks = None
        
        # Video Thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.process_frame)
        self.thread.start()
        
        self.tasks_result_signal.connect(self.update_tasks_landmarks)

    def closeEvent(self, event):
        self.thread.stop()
        self.hands_legacy.close()
        self.landmarker_tasks.close()
        event.accept()

    def handle_tasks_result(self, result, output_image: mp.Image, timestamp_ms: int):
        if result.hand_landmarks:
            self.tasks_result_signal.emit(result.hand_landmarks)
        else:
            self.tasks_result_signal.emit([])

    def update_tasks_landmarks(self, landmarks):
        if landmarks:
            self.latest_tasks_landmarks = landmarks[0]
        else:
            self.latest_tasks_landmarks = None

    def process_frame(self, cv_img):
        # We need two copies for visualization
        img_legacy = cv_img.copy()
        img_tasks = cv_img.copy()
        
        # ---------------------------------------------------------
        # Process Legacy
        # ---------------------------------------------------------
        img_rgb_legacy = cv2.cvtColor(img_legacy, cv2.COLOR_BGR2RGB)
        results_legacy = self.hands_legacy.process(img_rgb_legacy)
        
        if results_legacy.multi_hand_landmarks:
            for hand_landmarks in results_legacy.multi_hand_landmarks:
                self.mp_drawing_legacy.draw_landmarks(
                    img_legacy, hand_landmarks, self.mp_hands_legacy.HAND_CONNECTIONS)
                
                # Placeholder for Model Inference
                cv2.putText(img_legacy, "Model: Legacy (Untrained)", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ---------------------------------------------------------
        # Process Tasks
        # ---------------------------------------------------------
        # Async call
        img_rgb_tasks = cv2.cvtColor(img_tasks, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb_tasks)
        timestamp_ms = int(time.time() * 1000)
        self.landmarker_tasks.detect_async(mp_image, timestamp_ms)
        
        # Draw LATEST result
        if self.latest_tasks_landmarks:
            self.draw_tasks_landmarks(img_tasks, self.latest_tasks_landmarks)
            cv2.putText(img_tasks, "Model: Tasks (Untrained)", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ---------------------------------------------------------
        # Update UI
        # ---------------------------------------------------------
        self.legacy_video.setPixmap(self.convert_cv_qt(img_legacy))
        self.tasks_video.setPixmap(self.convert_cv_qt(img_tasks))

    def draw_tasks_landmarks(self, image, landmarks):
        h, w, c = image.shape
        # Draw Points
        points = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            
        # Draw Connections
        connections = self.mp_hands_legacy.HAND_CONNECTIONS
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModelComparator()
    window.show()
    sys.exit(app.exec())
