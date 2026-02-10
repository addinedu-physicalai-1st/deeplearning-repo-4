"""
웹캠 영상 패널 위젯
"""

from PyQt6.QtWidgets import (
<<<<<<< HEAD
    QGroupBox, QVBoxLayout, QLabel, QWidget, QGridLayout, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt
=======
    QGroupBox, QVBoxLayout, QLabel, QWidget, QGridLayout, QFrame,
    QGraphicsOpacityEffect,
)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311
from PyQt6.QtGui import QFont

import config
from app.widgets.gesture_display import GestureDisplayWidget
<<<<<<< HEAD
from app.widgets.neon_frame import NeonFrameWidget
=======
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311


class WebcamPanelWidget(QGroupBox):
    """웹캠 영상 라벨과 제스처 표시 위젯을 담는 그룹박스."""

    def __init__(self, parent=None):
<<<<<<< HEAD
        super().__init__("", parent)
        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet("""
            QGroupBox {
                border: none;
                margin-top: 0px;
                padding-top: 0px;
            }
        """)

        # 비율 유지를 위한 중앙 정렬 레이아웃
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(10)

        # 카메라 컨테이너
        self.camera_container = QWidget()
        self.cam_layout = QVBoxLayout(self.camera_container)
        self.cam_layout.setContentsMargins(0, 0, 0, 0)
        self.cam_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.webcam_label = QLabel()
        self.webcam_label.setMinimumSize(10, 10)
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setStyleSheet(f"""
            background-color: #000000;
            color: {config.COLOR_TEXT_PRIMARY};
            border: 1px solid rgba(0, 255, 255, 0.2);
            border-radius: 5px;
        """)
        
        self.neon_frame = NeonFrameWidget(self.webcam_label)
        self.neon_frame.setMinimumSize(10, 10)
        self.cam_layout.addWidget(self.neon_frame)
        
        # Gesture Display Widget
        self.gesture_display = GestureDisplayWidget()
        
        self.main_layout.addWidget(self.camera_container, 1)
        self.main_layout.addWidget(self.gesture_display)
        self.setLayout(self.main_layout)

    def resizeEvent(self, event):
        """카메라 뷰의 4:3 비율 유지"""
        super().resizeEvent(event)
        w = self.camera_container.width()
        h = self.camera_container.height()
        
        if w > 0 and h > 0:
            target_ratio = 4.0 / 3.0
            curr_ratio = w / h
            
            if curr_ratio > target_ratio:
                new_h = h
                new_w = int(h * target_ratio)
            else:
                new_w = w
                new_h = int(w / target_ratio)
                
            self.neon_frame.setFixedSize(new_w, new_h)

    def set_recording(self, active: bool):
        """Webcam indicator removed."""
        pass
=======
        super().__init__("웹캠 영상", parent)
        self._opacity_effect = None
        self._opacity_animation = None
        self._init_ui()

    def _init_ui(self):
        self.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 14px;
                color: {config.COLOR_TEXT_PRIMARY};
                border: 2px solid {config.COLOR_SECONDARY};
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }}
        """)
        layout = QVBoxLayout()
        layout.setSpacing(10)

        webcam_container = QWidget()
        grid = QGridLayout(webcam_container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.webcam_label = QLabel()
        self.webcam_label.setMinimumSize(640, 480)
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setStyleSheet(f"""
            background-color: #F3F4F6;
            border: 2px dashed {config.COLOR_SECONDARY};
            border-radius: 10px;
            color: {config.COLOR_TEXT_SECONDARY};
        """)
        self.webcam_label.setText("웹캠 영상이 여기에 표시됩니다")
        self.webcam_label.setFont(QFont("Arial", 12))
        grid.addWidget(self.webcam_label, 0, 0)

        indicator_wrapper = QFrame()
        indicator_wrapper.setStyleSheet("background: transparent;")
        wrapper_layout = QGridLayout(indicator_wrapper)
        wrapper_layout.setContentsMargins(0, 12, 12, 0)
        wrapper_layout.setSpacing(0)
        self.recording_indicator = QLabel()
        self.recording_indicator.setFixedSize(14, 14)
        self.recording_indicator.setStyleSheet(
            "background-color: #EF4444; border-radius: 7px;"
        )
        self.recording_indicator.setVisible(False)
        self._opacity_effect = QGraphicsOpacityEffect(self.recording_indicator)
        self.recording_indicator.setGraphicsEffect(self._opacity_effect)
        wrapper_layout.addWidget(self.recording_indicator, 0, 0)
        grid.addWidget(indicator_wrapper, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        layout.addWidget(webcam_container)
        self.gesture_display = GestureDisplayWidget()
        layout.addWidget(self.gesture_display)
        self.setLayout(layout)

    def _setup_opacity_animation(self):
        if self._opacity_animation is not None:
            return
        self._opacity_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._opacity_animation.setDuration(1000)
        self._opacity_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._opacity_animation.setStartValue(1.0)
        self._opacity_animation.setEndValue(0.5)
        self._opacity_animation.finished.connect(self._reverse_opacity_animation)

    def _reverse_opacity_animation(self):
        if self._opacity_animation is None:
            return
        start = self._opacity_animation.startValue()
        end = self._opacity_animation.endValue()
        self._opacity_animation.setStartValue(end)
        self._opacity_animation.setEndValue(start)
        self._opacity_animation.start()

    def set_recording(self, active: bool):
        """웹캠 영상 우상단에 녹화 표시등 표시. opacity 1 ↔ 0.5 부드럽게 전환."""
        if active:
            self._setup_opacity_animation()
            self._opacity_effect.setOpacity(1.0)
            self.recording_indicator.setVisible(True)
            self._opacity_animation.setStartValue(1.0)
            self._opacity_animation.setEndValue(0.5)
            self._opacity_animation.start()
        else:
            if self._opacity_animation is not None:
                self._opacity_animation.stop()
                self._opacity_animation.finished.disconnect(self._reverse_opacity_animation)
                self._opacity_animation = None
            self.recording_indicator.setVisible(False)
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311
