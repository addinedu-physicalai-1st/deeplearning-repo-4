"""
제스처 인식 프로토타입 모듈
확장 가능한 구조로 설계하여 다양한 모션 인식 방법을 실험할 수 있도록 함
"""

import numpy as np
from typing import Optional, List, Dict
from enum import Enum


class GestureType(Enum):
    """제스처 타입"""
    NONE = "NONE"
    START = "START"
    STOP = "STOP"


class GestureDetector:
    """제스처 인식 프로토타입 클래스
    
    현재는 간단한 규칙 기반 인식을 구현하지만,
    나중에 LSTM 모델이나 다른 방법으로 쉽게 교체할 수 있도록 설계됨
    """
    
    def __init__(self):
        """제스처 인식기 초기화"""
        self.current_state = GestureType.NONE
        self.detection_threshold = 0.5  # 기본 임계값
    
    def detect(self, landmarks_list: Optional[List]) -> str:
        """
        랜드마크에서 제스처 인식
        
        Args:
            landmarks_list: 손 랜드마크 리스트 (각 손마다 하나의 리스트)
                           각 리스트는 21개의 랜드마크 포인트를 포함
        
        Returns:
            str: 인식된 제스처 ("START", "STOP", "NONE")
        """
        if landmarks_list is None or len(landmarks_list) == 0:
            return GestureType.NONE.value
        
        # 첫 번째 손만 사용 (나중에 양손 지원 가능)
        landmarks = landmarks_list[0]
        
        if len(landmarks) < 21:
            return GestureType.NONE.value
        
        # 간단한 규칙 기반 제스처 인식
        # TODO: 팀원들이 다양한 모션을 실험할 수 있도록 이 부분을 수정
        gesture = self._detect_simple_gesture(landmarks)
        
        return gesture
    
    def _detect_simple_gesture(self, landmarks: List[Dict]) -> str:
        """
        간단한 규칙 기반 제스처 인식 (프로토타입)
        
        현재 구현:
        - 주먹 (손가락이 모두 접힌 상태) → STOP
        - 손 펴기 (손가락이 모두 펴진 상태) → START
        
        Args:
            landmarks: 21개의 랜드마크 포인트 리스트
        
        Returns:
            str: 인식된 제스처
        """
        # 손가락 끝 포인트 인덱스 (Mediapipe Hands)
        # 엄지: 4, 검지: 8, 중지: 12, 약지: 16, 새끼: 20
        finger_tips = [4, 8, 12, 16, 20]
        # 손가락 관절 포인트 인덱스 (손가락이 접혔는지 판단)
        finger_pips = [3, 6, 10, 14, 18]  # 각 손가락의 PIP 관절
        
        fingers_up = []
        
        # 엄지 처리 (x 좌표로 판단)
        if landmarks[4]['x'] > landmarks[3]['x']:
            fingers_up.append(1)
        else:
            fingers_up.append(0)
        
        # 나머지 손가락 처리 (y 좌표로 판단)
        for i in range(1, 5):
            tip_idx = finger_tips[i]
            pip_idx = finger_pips[i]
            if landmarks[tip_idx]['y'] < landmarks[pip_idx]['y']:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        total_fingers = sum(fingers_up)
        
        # 제스처 판단
        if total_fingers == 0:
            # 주먹 → STOP
            return GestureType.STOP.value
        elif total_fingers >= 4:
            # 손 펴기 → START
            return GestureType.START.value
        else:
            # 기타 → NONE
            return GestureType.NONE.value
    
    def set_threshold(self, threshold: float):
        """
        인식 임계값 설정 (감도 조절용, 현재는 미사용)
        
        Args:
            threshold: 임계값 (0.0 - 1.0)
        """
        self.detection_threshold = max(0.0, min(1.0, threshold))
    
    def reset(self):
        """상태 초기화"""
        self.current_state = GestureType.NONE
