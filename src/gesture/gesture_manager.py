"""
제스처 인식 통합 관리 모듈
웹캠, Mediapipe, 제스처 인식을 통합하여 관리
"""

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from typing import Optional
import sys
import os
import time

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.utils.camera import Camera, frame_to_qpixmap
from src.gesture.mediapipe_handler import MediapipeHandler
from src.gesture.gesture_detector import GestureDetector

# 제스처 클래스들을 import하여 레지스트리에 자동 등록되도록 함
from src.gesture.common import StartGesture, StopGesture
from src.gesture.ppt import NextGesture, PrevGesture, ShowStartGesture
from src.gesture.youtube import (
    PlayPauseGesture, VolumeUpGesture, VolumeDownGesture,
    MuteGesture, FullscreenGesture
)


class GestureManager(QThread):
    """제스처 인식 통합 관리 클래스 (별도 스레드에서 실행)"""
    
    # 시그널 정의
    gesture_detected = pyqtSignal(str)  # 인식된 제스처 이름 (모드별 제스처 포함: "START", "STOP", "NEXT", "PREV", etc. 또는 "NONE")
    frame_ready = pyqtSignal(object)  # QPixmap
    
    def __init__(self, parent=None, mode: str = "COMMON"):
        """
        제스처 매니저 초기화
        
        Args:
            parent: 부모 QObject
            mode: 초기 모드 ("COMMON", "PPT", "YOUTUBE")
        """
        super().__init__(parent)
        
        # 모듈 초기화
        self.camera = Camera()
        self.mediapipe_handler = MediapipeHandler()
        self.gesture_detector = GestureDetector(mode=mode)
        
        # 상태 관리
        self.is_running = False
        self._should_stop = False
        self.current_mode = mode
    
    def initialize(self) -> bool:
        """
        웹캠 및 제스처 인식 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        if not self.camera.initialize():
            return False
        
        self.is_running = True
        return True
    
    def start_detection(self):
        """제스처 인식 시작"""
        if not self.is_running:
            if not self.initialize():
                return False
        
        if not self.isRunning():
            self._should_stop = False
            super().start()  # QThread.start() 호출
        
        return True
    
    def stop_detection(self):
        """제스처 인식 중지"""
        self._should_stop = True
        if self.isRunning():
            self.wait()  # 스레드 종료 대기
    
    def release(self):
        """리소스 해제"""
        self.stop_detection()
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        if self.mediapipe_handler:
            self.mediapipe_handler.release()
    
    def run(self):
        """스레드 실행 메서드 - 프레임 읽기 및 처리 루프"""
        # FPS 제어를 위한 시간 간격 계산
        fps = config.GESTURE_DETECTION_FPS
        frame_time = 1.0 / fps
        
        while not self._should_stop and self.is_running:
            loop_start = time.time()
            
            # 프레임 읽기
            frame = self.camera.read_frame()
            if frame is None:
                time.sleep(0.01)  # 프레임이 없으면 짧은 대기
                continue
            
            # 프레임이 비어있는지 확인
            if frame.size == 0:
                continue
            
            # 좌우 반전 (거울 모드)
            frame = cv2.flip(frame, 1)
            
            # Mediapipe로 랜드마크 추출
            annotated_frame = None
            landmarks_list = None
            try:
                annotated_frame, landmarks_list = self.mediapipe_handler.process(frame)
                # Mediapipe가 None을 반환하면 원본 프레임 사용
                if annotated_frame is None:
                    annotated_frame = frame
            except Exception as e:
                # Mediapipe 오류 시 원본 프레임 사용
                annotated_frame = frame
                landmarks_list = None
            
            # 제스처 인식
            gesture = self.gesture_detector.detect(landmarks_list)
            
            # 시그널 발생 (비동기로 UI 업데이트)
            self.gesture_detected.emit(gesture)
            
            # 프레임을 QPixmap으로 변환하여 UI에 전달
            # annotated_frame이 None이면 원본 frame 사용
            frame_to_display = annotated_frame if annotated_frame is not None else frame
            if frame_to_display is not None and frame_to_display.size > 0:
                pixmap = frame_to_qpixmap(frame_to_display)
                if pixmap and not pixmap.isNull():
                    # 시그널을 통해 비동기로 UI 업데이트 (UI 스레드 블로킹 방지)
                    self.frame_ready.emit(pixmap)
            
            # FPS 제어 - 프레임 처리 시간을 고려하여 대기
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def set_mode(self, mode: str):
        """
        모드 설정
        
        모드 전환 시 제스처 인식기가 새로운 모드의 제스처를 인식하도록 업데이트합니다.
        
        Args:
            mode: 모드 이름 ("COMMON", "PPT", "YOUTUBE")
        """
        if mode in ["COMMON", "PPT", "YOUTUBE"]:
            self.current_mode = mode
            # GestureDetector의 모드 변경 (시퀀스 버퍼도 자동 초기화됨)
            self.gesture_detector.set_mode(mode)
    
    def get_mode(self) -> str:
        """
        현재 모드 반환
        
        Returns:
            str: 현재 모드
        """
        return self.current_mode
    
    def set_sensitivity(self, sensitivity: int):
        """
        감도 설정 (0-100)
        
        감도 값을 임계값으로 변환하여 GestureDetector에 전달합니다.
        
        Args:
            sensitivity: 감도 값 (0-100)
        """
        # 감도를 0.0-1.0 범위로 변환하여 임계값에 적용
        # 높은 감도 = 낮은 임계값 (더 관대한 인식)
        # 낮은 감도 = 높은 임계값 (더 엄격한 인식)
        threshold = 1.0 - (sensitivity / 100.0) * 0.5  # 0.5 ~ 1.0 범위
        self.gesture_detector.set_threshold(threshold)
    
    def is_camera_available(self) -> bool:
        """웹캠 사용 가능 여부 확인"""
        return self.camera.is_opened()
