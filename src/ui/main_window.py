"""
메인 윈도우
"""

import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QRadioButton, QButtonGroup,
    QSlider, QStatusBar, QGroupBox
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont, QIcon
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from .gesture_display import GestureDisplayWidget
from src.gesture.gesture_manager import GestureManager
from src.control.controller_manager import ControllerManager


class MainWindow(QMainWindow):
    """메인 윈도우 클래스"""
    
    # 시그널 정의
    start_detection = pyqtSignal()
    stop_detection = pyqtSignal()
    sensitivity_changed = pyqtSignal(int)
    mode_changed = pyqtSignal(str)  # "COMMON", "PPT", "YOUTUBE", "GAME"
    
    def __init__(self):
        super().__init__()
        self.current_mode = "PPT"
        self.sensitivity = config.SENSITIVITY_DEFAULT
        self.is_detecting = False
        
        # 제스처 매니저 초기화
        self.current_mode = "PPT" 
        self.gesture_manager = GestureManager(self, mode="PPT")
        self.gesture_manager.gesture_detected.connect(self.on_gesture_detected)
        self.gesture_manager.frame_ready.connect(self.update_webcam_frame)
        
        # 컨트롤러 매니저 초기화
        self.controller_manager = ControllerManager()
        self.controller_manager.set_mode(self.current_mode)
        
        self.init_ui()
        
        # 초기 감도 설정
        self.gesture_manager.set_sensitivity(self.sensitivity)
        
        # 프로그램 시작 시 웹캠 자동 초기화
        self._initialize_camera()
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setGeometry(100, 100, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        self.setStyleSheet(f"background-color: {config.COLOR_BACKGROUND};")
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # 컨텐츠 영역 (수평 레이아웃)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # 왼쪽: 웹캠 영상 영역
        left_panel = self.create_webcam_panel()
        content_layout.addWidget(left_panel, 2)
        
        # 오른쪽: 로고 + 제어 패널 (수직 레이아웃)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        
        # 로고 영역
        logo_widget = self.create_logo_section()
        right_layout.addWidget(logo_widget)
        
        # 제어 패널
        control_panel = self.create_control_panel()
        right_layout.addWidget(control_panel, 1)
        
        right_panel.setLayout(right_layout)
        content_layout.addWidget(right_panel, 1)
        
        main_layout.addLayout(content_layout)
        
        # 상태바
        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(f"background-color: {config.COLOR_BACKGROUND}; color: {config.COLOR_TEXT_SECONDARY};")
        self.status_bar.showMessage("준비됨")
        self.setStatusBar(self.status_bar)
        
        self.setStatusBar(self.status_bar)
        
        central_widget.setLayout(main_layout)
        
        # Set default mode safely after UI is fully initialized
        self.ppt_radio.setChecked(True)
    
    def _initialize_camera(self):
        """웹캠 자동 초기화 (프로그램 시작 시) - 비동기 처리"""
        # QTimer를 사용하여 초기화를 지연시켜 UI가 먼저 표시되도록 함
        from PyQt6.QtCore import QTimer
        init_timer = QTimer(self)
        init_timer.setSingleShot(True)
        init_timer.timeout.connect(self._do_initialize_camera)
        init_timer.start(100)  # 100ms 후 초기화
    
    def _do_initialize_camera(self):
        """실제 웹캠 초기화 수행"""
        if self.gesture_manager.initialize():
            self.gesture_manager.start_detection()  # QThread.start() 호출
            self.status_bar.showMessage("웹캠이 초기화되었습니다. 모션을 감지할 준비가 되었습니다.")
            self.gesture_display.update_status("대기 중", None)
            # 웹캠 영상이 곧 표시될 것임을 알림
            self.webcam_label.setText("웹캠 영상을 불러오는 중...")
        else:
            self.status_bar.showMessage("웹캠 초기화 실패. 웹캠을 확인해주세요.")
            self.webcam_label.setText("웹캠을 찾을 수 없습니다")
            self.webcam_label.setStyleSheet(f"""
                background-color: #FEE2E2;
                border: 2px dashed #EF4444;
                border-radius: 10px;
                color: #DC2626;
            """)
    
    def showEvent(self, event):
        """윈도우 표시 이벤트"""
        super().showEvent(event)
        # 윈도우가 표시된 후 웹캠 초기화 (이미 _initialize_camera에서 처리)
    
    def closeEvent(self, event):
        """윈도우 종료 이벤트 - 리소스 정리"""
        if hasattr(self, 'gesture_manager'):
            self.gesture_manager.stop_detection()  # 스레드 종료
            self.gesture_manager.release()
        event.accept()
    
    def create_logo_section(self):
        """로고 섹션 생성"""
        logo_widget = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 로고 이미지
        logo_path = os.path.join(config.ASSETS_DIR, "gesto-light.png")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            pixmap = QPixmap(logo_path)
            # 로고 크기 조정 (작게)
            scaled_pixmap = pixmap.scaled(180, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo_label)
        else:
            # 로고가 없을 경우 텍스트로 대체
            logo_text = QLabel(config.APP_NAME)
            logo_text.setFont(QFont("Arial", 18, QFont.Weight.Bold))
            logo_text.setStyleSheet(f"color: {config.COLOR_PRIMARY};")
            logo_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(logo_text)
        
        logo_widget.setLayout(layout)
        return logo_widget
    
    def create_webcam_panel(self):
        """웹캠 영상 패널 생성"""
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
        
        # 웹캠 영상 표시 영역 (더미 이미지)
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
        
        # 제스처 표시 위젯
        self.gesture_display = GestureDisplayWidget()
        layout.addWidget(self.gesture_display)
        
        group_box.setLayout(layout)
        return group_box
    
    def create_control_panel(self):
        """제어 패널 생성"""
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
        
        # 모드 선택
        mode_group = QGroupBox("모드 선택")
        mode_group.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY};")
        mode_layout = QVBoxLayout()
        
        self.mode_button_group = QButtonGroup()
        self.ppt_radio = QRadioButton("PPT 모드")
        self.youtube_radio = QRadioButton("유투브 모드")
        self.game_radio = QRadioButton("게임 모드")
        self.game_radio = QRadioButton("게임 모드")
        # self.game_radio.setChecked(True) # Moved to after connection
        
        self.mode_button_group.addButton(self.ppt_radio, 0)
        self.mode_button_group.addButton(self.youtube_radio, 1)
        self.mode_button_group.addButton(self.game_radio, 2)
        
        self.ppt_radio.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY}; font-size: 12px;")
        self.youtube_radio.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY}; font-size: 12px;")
        self.game_radio.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY}; font-size: 12px;")
        
        self.ppt_radio.toggled.connect(lambda: self.on_mode_changed("PPT"))
        self.youtube_radio.toggled.connect(lambda: self.on_mode_changed("YOUTUBE"))
        self.game_radio.toggled.connect(lambda: self.on_mode_changed("GAME"))

        # No setChecked here to avoid premature signal firing
        
        mode_layout.addWidget(self.ppt_radio)
        mode_layout.addWidget(self.youtube_radio)
        mode_layout.addWidget(self.game_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # 감도 설정
        sensitivity_group = QGroupBox("감도 설정")
        sensitivity_group.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY};")
        sensitivity_layout = QVBoxLayout()
        
        # 감도 슬라이드
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setMinimum(config.SENSITIVITY_MIN)
        self.sensitivity_slider.setMaximum(config.SENSITIVITY_MAX)
        self.sensitivity_slider.setValue(config.SENSITIVITY_DEFAULT)
        self.sensitivity_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {config.COLOR_SECONDARY};
                height: 8px;
                background: #E5E7EB;
                border-radius: 4px;
            }}
            QSlider::handle:horizontal {{
                background: {config.COLOR_SECONDARY};
                border: 2px solid {config.COLOR_PRIMARY};
                width: 20px;
                margin: -2px 0;
                border-radius: 10px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {config.COLOR_BUTTON_HOVER};
            }}
        """)
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_changed)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        
        # 감도 값 표시
        self.sensitivity_label = QLabel(f"감도: {config.SENSITIVITY_DEFAULT}%")
        self.sensitivity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sensitivity_label.setStyleSheet(f"color: {config.COLOR_TEXT_PRIMARY}; font-size: 12px; font-weight: bold;")
        sensitivity_layout.addWidget(self.sensitivity_label)
        
        sensitivity_group.setLayout(sensitivity_layout)
        layout.addWidget(sensitivity_group)
        
        # 시작/종료 버튼
        button_layout = QVBoxLayout()
        button_layout.setSpacing(10)
        
        self.start_button = QPushButton("시작")
        self.start_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {config.COLOR_SECONDARY};
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {config.COLOR_BUTTON_HOVER};
            }}
            QPushButton:pressed {{
                background-color: {config.COLOR_PRIMARY};
            }}
        """)
        self.start_button.clicked.connect(self.on_start_clicked)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("종료")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #EF4444;
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #DC2626;
            }}
            QPushButton:pressed {{
                background-color: #B91C1C;
            }}
            QPushButton:disabled {{
                background-color: #D1D5DB;
                color: #9CA3AF;
            }}
        """)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        group_box.setLayout(layout)
        return group_box
    
    def set_detection_state(self, is_active: bool):
        """
        시작/종료 상태를 통합 관리
        모션 인식과 버튼 클릭 모두 이 메서드를 사용
        
        Args:
            is_active: True면 감지 활성화, False면 감지 비활성화
        """
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
        """시작 버튼 클릭 핸들러"""
        self.set_detection_state(True)
    
    def on_stop_clicked(self):
        """종료 버튼 클릭 핸들러"""
        self.set_detection_state(False)
    
    def on_gesture_detected(self, gesture: str):
        """
        제스처 인식 결과 처리
        
        Args:
            gesture: 인식된 제스처 ("START", "STOP", "NONE")
        """
        # 제스처 표시 업데이트
        if gesture == "START" or gesture == "START_SYSTEM" or gesture == "WAKE_UP":
            self.gesture_display.update_status("감지 중", "시작 모션")
            # 시작 모션 감지 시 상태 변경
            if not self.is_detecting:
                self.set_detection_state(True)
        elif gesture == "STOP":
            self.gesture_display.update_status("감지 중", "종료 모션")
            # 종료 모션 감지 시 상태 변경
            if self.is_detecting:
                self.set_detection_state(False)
        elif gesture == "OPEN_GAME":
            self.gesture_display.update_status("감지됨", "게임 열기 (Spiderman)")
            # Spiderman 감지 시 자동으로 게임 모드로 전환 및 감지 시작
            if self.current_mode != "GAME":
                self.game_radio.setChecked(True)
            
            # 항상 감지 상태로 전환 (Wake Up)
            self.set_detection_state(True)
        elif gesture == "POINT_UP":
            self.gesture_display.update_status("감지됨", "위로 이동 (검지 위)")
        elif gesture == "POINT_DOWN":
            self.gesture_display.update_status("감지됨", "아래로 이동 (검지 아래)")
        elif gesture == "POINT_LEFT":
            self.gesture_display.update_status("감지됨", "왼쪽 이동 (검지 왼쪽)")
        elif gesture == "POINT_RIGHT":
            self.gesture_display.update_status("감지됨", "오른쪽 이동 (검지 오른쪽)")
        else:
            # NONE 또는 기타 제스처
            if self.is_detecting:
                self.gesture_display.update_status("감지 중", None)
        
        # 액션 실행 (감지 중일 때만)
        if self.is_detecting and gesture != "NONE":
            self.controller_manager.execute_action(gesture)
    
    def on_sensitivity_changed(self, value):
        """감도 변경 핸들러"""
        self.sensitivity = value
        self.sensitivity_label.setText(f"감도: {value}%")
        # 제스처 매니저에 감도 전달
        if hasattr(self, 'gesture_manager'):
            self.gesture_manager.set_sensitivity(value)
        self.sensitivity_changed.emit(value)
    
    def on_mode_changed(self, mode):
        """모드 변경 핸들러"""
        self.current_mode = mode
        self.mode_changed.emit(mode)
        self.status_bar.showMessage(f"{mode} 모드로 전환됨")
        
        # Explicitly update GestureManager
        if hasattr(self, 'gesture_manager'):
            self.gesture_manager.set_mode(mode)
            
        # Update ControllerManager
        if hasattr(self, 'controller_manager'):
            self.controller_manager.set_mode(mode)
            
        # 게임 모드 진입 시 자동으로 감지 활성화 (사용자 요청)
        if mode == "GAME" and not self.is_detecting:
            self.set_detection_state(True)
    
    def update_webcam_frame(self, pixmap):
        """웹캠 프레임 업데이트"""
        if pixmap and not pixmap.isNull():
            # 라벨 크기가 0이면 원본 크기 사용
            label_size = self.webcam_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = pixmap.scaled(
                    label_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            else:
                scaled_pixmap = pixmap
            
            # Pixmap이 유효한지 확인
            if scaled_pixmap and not scaled_pixmap.isNull():
                self.webcam_label.setPixmap(scaled_pixmap)
                # 텍스트 제거 (프레임이 표시되면)
                if self.webcam_label.text():
                    self.webcam_label.setText("")
    
    def update_gesture(self, gesture_name: str):
        """인식된 제스처 업데이트"""
        if self.is_detecting:
            self.gesture_display.update_status("감지 중", gesture_name)