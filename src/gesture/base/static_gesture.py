"""
정적 제스처 추상 클래스
단일 프레임에서 인식 가능한 정적 제스처를 위한 기본 클래스
"""

from abc import abstractmethod
from typing import Optional, List, Dict
from .gesture_base import BaseGesture


class StaticGesture(BaseGesture):
    """정적 제스처 추상 클래스
    
    단일 프레임의 랜드마크만으로 인식 가능한 제스처를 위한 기본 클래스.
    예: 주먹, 손 펴기, 특정 손가락 제스처 등
    """
    
    gesture_type: str = "static"
    
    def __init__(self):
        """정적 제스처 초기화"""
        super().__init__()
    
    def detect(self, landmarks_list: Optional[List]) -> str:
        """
        랜드마크에서 정적 제스처 인식
        
        Args:
            landmarks_list: 손 랜드마크 리스트 (각 손마다 하나의 리스트)
        
        Returns:
            str: 인식된 제스처 이름 (인식되지 않으면 "NONE")
        """
        if landmarks_list is None or len(landmarks_list) == 0:
            return "NONE"
        
        # 첫 번째 손만 사용
        landmarks = landmarks_list[0]
        
        if len(landmarks) < 21:
            return "NONE"
        
        return self.detect_static(landmarks)
    
    @abstractmethod
    def detect_static(self, landmarks: List[Dict]) -> str:
        """
        단일 프레임 랜드마크에서 정적 제스처 인식
        
        Args:
            landmarks: 21개의 랜드마크 포인트 리스트
                      각 포인트는 {'x': float, 'y': float, 'z': float} 형태
        
        Returns:
            str: 인식된 제스처 이름 (인식되지 않으면 "NONE")
        """
        pass
