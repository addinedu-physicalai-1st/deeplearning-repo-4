"""
웹캠 관리 모듈
"""

import cv2
import numpy as np
from PyQt6.QtGui import QPixmap, QImage
from typing import Optional, Tuple
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class Camera:
    """웹캠 관리 클래스"""
    
    def __init__(self, camera_index: int = None, width: int = None, height: int = None, fps: int = None):
        """
        웹캠 초기화
        
        Args:
            camera_index: 웹캠 인덱스 (기본값: config.CAMERA_INDEX)
            width: 프레임 너비 (기본값: config.CAMERA_WIDTH)
            height: 프레임 높이 (기본값: config.CAMERA_HEIGHT)
            fps: FPS (기본값: config.CAMERA_FPS)
        """
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.width = width if width is not None else config.CAMERA_WIDTH
        self.height = height if height is not None else config.CAMERA_HEIGHT
        self.fps = fps if fps is not None else config.CAMERA_FPS
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """
        웹캠 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            # macOS에서는 AVFoundation 백엔드를 명시적으로 사용
            # 여러 인덱스를 시도하여 작동하는 웹캠 찾기
            working_cap = None
            working_index = None
            
            # 시도할 인덱스 목록
            # 가상 카메라(OBS 등)가 0번을 점유할 수 있으므로 1, 2를 먼저 시도
            # 현재 인덱스가 0이 아니면 먼저 시도
            if self.camera_index != 0:
                indices_to_try = [self.camera_index, 1, 2, 0]
            else:
                indices_to_try = [1, 2, 0]  # 0번은 마지막에 시도
            
            for idx in indices_to_try:
                test_cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
                
                if not test_cap.isOpened():
                    test_cap.release()
                    continue
                
                # 웹캠 설정
                test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                test_cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # 프레임 읽기 테스트
                ret, test_frame = test_cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    working_cap = test_cap
                    working_index = idx
                    break
                
                test_cap.release()
            
            if working_cap is None:
                raise Exception(f"작동하는 웹캠을 찾을 수 없습니다.")
            
            self.cap = working_cap
            self.camera_index = working_index
            
            # 웹캠 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 버퍼 크기를 1로 설정하여 지연 최소화
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 초기 프레임 버리기 (웹캠 준비)
            # 첫 몇 프레임은 버퍼에 쌓인 오래된 프레임일 수 있으므로 버리기
            for _ in range(10):
                self.cap.read()
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.is_initialized = False
            return False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        프레임 읽기
        
        Returns:
            Optional[np.ndarray]: BGR 프레임, 실패 시 None
        """
        if not self.is_initialized or self.cap is None:
            return None
        
        if not self.cap.isOpened():
            return None
        
        # 프레임 읽기
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return None
        
        return frame
    
    def release(self):
        """웹캠 해제"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_initialized = False
    
    def is_opened(self) -> bool:
        """웹캠이 열려있는지 확인"""
        return self.is_initialized and self.cap is not None and self.cap.isOpened()
    
    def __del__(self):
        """소멸자: 웹캠 자동 해제"""
        self.release()


def frame_to_qpixmap(frame: np.ndarray) -> Optional[QPixmap]:
    """
    OpenCV BGR 프레임을 QPixmap으로 변환
    
    Args:
        frame: BGR 프레임 (numpy array)
        
    Returns:
        QPixmap: 변환된 QPixmap, 실패 시 None
    """
    if frame is None:
        return None
    
    try:
        # 프레임이 비어있는지 확인
        if frame.size == 0:
            return None
        
        # BGR to RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # numpy array를 QImage로 변환
        # macOS에서는 데이터를 복사해야 안정적으로 작동합니다
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        
        # 데이터를 contiguous array로 변환 (메모리 레이아웃 보장)
        rgb_frame_contiguous = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
        
        # macOS에서는 데이터를 명시적으로 복사해야 할 수 있습니다
        # QImage 생성 (데이터 직접 참조)
        q_image = QImage(
            rgb_frame_contiguous.data, 
            width, 
            height, 
            bytes_per_line, 
            QImage.Format.Format_RGB888
        )
        
        # QImage가 유효한지 확인
        if q_image.isNull():
            # 대안: 데이터를 복사하여 생성
            rgb_frame_bytes = rgb_frame_contiguous.tobytes()
            q_image = QImage(rgb_frame_bytes, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            if q_image.isNull():
                return None
        
        # QImage를 QPixmap으로 변환
        pixmap = QPixmap.fromImage(q_image)
        if pixmap.isNull():
            return None
        
        return pixmap
        
    except Exception as e:
        return None
