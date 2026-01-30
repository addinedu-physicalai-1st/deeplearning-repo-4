"""
메인 윈도우 (UI 전용)
"""

import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QButtonGroup,
    QSlider, QStatusBar, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.app.gesture_display import GestureDisplayWidget


class MainWindow(QMainWindow):
    start_detection = pyqtSignal()
    stop_detection = pyqtSignal()
    sensitivity_changed = pyqtSignal(int)
    mode_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.current_mode = "PPT"
        self.sensitivity = config.SENSITIVITY_DEFAULT
        self.is_detecting = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setGeometry(100, 100, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.setStyleSheet(f"background-color: {config.COLOR_BACKGROUND};")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        content_layout.addWidget(self.create_webcam_panel(), 2)

        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        right_layout.addWidget(self.create_logo_section())
        right_layout.addWidget(self.create_control_panel(), 1)
        right_panel.setLayout(right_layout)
        content_layout.addWidget(right_panel, 1)

        main_layout.addLayout(content_layout)
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"background-color: {config.COLOR_BACKGROUND}; color: {config.COLOR_TEXT_SECONDARY};")
        self.status_bar.showMessage("준비됨")
        self.setStatusBar(self.status_bar)
        central_widget.setLayout(main_layout)
        self.ppt_radio.setChecked(True)

    def closeEvent(self, event):
        event.accept()

    def create_logo_section(self):
        logo_widget = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        logo_path = os.path.join(config.ASSETS_DIR, "gesto-light.png")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            pixmap = QPixmap(logo_path)
            scaled_pixmap = pixmap.scaled(180, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo_label)
        else:
            logo_text = QLabel(config.APP_NAME)
            logo_text.setFont(QFont("Arial", 18, QFont.Weight.Bold))
            logo_text.setStyleSheet(f"color: {config.COLOR_PRIMARY};")
            logo_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo_text)
        logo_widget.setLayout(layout)
        return logo_widget

    def create_webcam_panel(self):
        group_box = QGroupBox("웹캠 영상")
        group_box.setStyleSheet(f"""
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
        layout.addWidget(self.webcam_label)
        self.gesture_display = GestureDisplayWidget()
        layout.addWidget(self.gesture_display)
        group_box.setLayout(layout)
        return group_box

    def create_control_panel(self):
        group_box = QGroupBox("제어")
        group_box.setStyleSheet(f"""
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
        layout.setSpacing(20)

        mode_group = QGroupBox("모드 선택")
        mode_group.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY};")
        mode_layout = QVBoxLayout()
        self.mode_button_group = QButtonGroup()
        self.ppt_radio = QRadioButton("PPT 모드")
        self.youtube_radio = QRadioButton("유투브 모드")
        self.game_radio = QRadioButton("게임 모드")
        self.mode_button_group.addButton(self.ppt_radio, 0)
        self.mode_button_group.addButton(self.youtube_radio, 1)
        self.mode_button_group.addButton(self.game_radio, 2)
        self.ppt_radio.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY}; font-size: 12px;")
        self.youtube_radio.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY}; font-size: 12px;")
        self.game_radio.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY}; font-size: 12px;")
        self.ppt_radio.toggled.connect(lambda: self.on_mode_changed("PPT"))
        self.youtube_radio.toggled.connect(lambda: self.on_mode_changed("YOUTUBE"))
        self.game_radio.toggled.connect(lambda: self.on_mode_changed("GAME"))
        mode_layout.addWidget(self.ppt_radio)
        mode_layout.addWidget(self.youtube_radio)
        mode_layout.addWidget(self.game_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        sensitivity_group = QGroupBox("감도 설정")
        sensitivity_group.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY};")
        sensitivity_layout = QVBoxLayout()
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setMinimum(config.SENSITIVITY_MIN)
        self.sensitivity_slider.setMaximum(config.SENSITIVITY_MAX)
        self.sensitivity_slider.setValue(config.SENSITIVITY_DEFAULT)
        self.sensitivity_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ border: 1px solid {config.COLOR_SECONDARY}; height: 8px; background: #E5E7EB; border-radius: 4px; }}
            QSlider::handle:horizontal {{ background: {config.COLOR_SECONDARY}; border: 2px solid {config.COLOR_PRIMARY}; width: 20px; margin: -2px 0; border-radius: 10px; }}
            QSlider::handle:horizontal:hover {{ background: {config.COLOR_BUTTON_HOVER}; }}
        """)
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_changed)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        self.sensitivity_label = QLabel(f"감도: {config.SENSITIVITY_DEFAULT}%")
        self.sensitivity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sensitivity_label.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY}; font-size: 12px; font-weight: bold;")
        sensitivity_layout.addWidget(self.sensitivity_label)
        sensitivity_group.setLayout(sensitivity_layout)
        layout.addWidget(sensitivity_group)

        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)
        self.start_button = QPushButton("시작")
        self.start_button.setStyleSheet(f"""
            QPushButton {{ background-color: {config.COLOR_SECONDARY}; color: white; border: none; padding: 12px; border-radius: 8px; font-size: 14px; font-weight: bold; }}
            QPushButton:hover {{ background-color: {config.COLOR_BUTTON_HOVER}; }}
            QPushButton:pressed {{ background-color: {config.COLOR_PRIMARY}; }}
        """)
        self.start_button.clicked.connect(self.on_start_clicked)
        button_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("종료")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {{ background-color: #EF4444; color: white; border: none; padding: 12px; border-radius: 8px; font-size: 14px; font-weight: bold; }}
            QPushButton:hover {{ background-color: #DC2626; }}
            QPushButton:disabled {{ background-color: #D1D5DB; color: #9CA3AF; }}
        """)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)
        layout.addStretch()
        group_box.setLayout(layout)
        return group_box

    def set_detection_state(self, is_active: bool):
        self.is_detecting = is_active
        self.start_button.setEnabled(not is_active)
        self.stop_button.setEnabled(is_active)
        if is_active:
            self.status_bar.showMessage("제스처 인식 시작됨")
            self.gesture_display.update_status("감지 중", None)
            self.start_detection.emit()
        else:
            self.status_bar.showMessage("제스처 인식 중지됨")
            self.gesture_display.update_status("대기 중", None)
            self.stop_detection.emit()

    def on_start_clicked(self):
        self.set_detection_state(True)

    def on_stop_clicked(self):
        self.set_detection_state(False)

    def on_sensitivity_changed(self, value):
        self.sensitivity = value
        self.sensitivity_label.setText(f"감도: {value}%")
        self.sensitivity_changed.emit(value)

    def on_mode_changed(self, mode):
        self.current_mode = mode
        self.mode_changed.emit(mode)
        self.status_bar.showMessage(f"{mode} 모드로 전환됨")

    def update_webcam_frame(self, pixmap):
        """웹캠 프레임 표시 (백엔드 연결 시 사용)."""
        if pixmap and not pixmap.isNull():
            label_size = self.webcam_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            else:
                scaled_pixmap = pixmap
            if scaled_pixmap and not scaled_pixmap.isNull():
                self.webcam_label.setPixmap(scaled_pixmap)
                if self.webcam_label.text():
                    self.webcam_label.setText("")

    def update_gesture(self, gesture_name: str):
        """인식된 제스처 표시 (백엔드 연결 시 사용)."""
        if self.is_detecting:
            self.gesture_display.update_status("감지 중", gesture_name)
