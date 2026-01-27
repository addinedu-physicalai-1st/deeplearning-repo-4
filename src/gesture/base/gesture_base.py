"""
제스처 기본 추상 클래스
모든 제스처의 기본이 되는 추상 클래스
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict


class BaseGesture(ABC):
    """모든 제스처의 기본 추상 클래스
    
    모든 제스처 클래스는 이 클래스를 상속받아야 합니다.
    """
    
    # 제스처 메타데이터 (하위 클래스에서 정의)
    name: str = ""  # 제스처 이름 (예: "START", "NEXT")
    mode: str = ""  # 제스처 모드 (예: "COMMON", "PPT", "YOUTUBE")
    gesture_type: str = ""  # 제스처 타입 ("static" 또는 "dynamic")
    
    def __init__(self):
        """제스처 초기화"""
        pass
    
    @abstractmethod
    def detect(self, landmarks_list: Optional[List]) -> str:
        """
        랜드마크에서 제스처 인식
        
        Args:
            landmarks_list: 손 랜드마크 리스트 (각 손마다 하나의 리스트)
                           각 리스트는 21개의 랜드마크 포인트를 포함
        
        Returns:
            str: 인식된 제스처 이름 (인식되지 않으면 "NONE")
        """
        pass
    
    def get_name(self) -> str:
        """
        제스처 이름 반환
        
        Returns:
            str: 제스처 이름
        """
        return self.name
    
    def get_mode(self) -> str:
        """
        제스처 모드 반환
        
        Returns:
            str: 제스처 모드
        """
        return self.mode
    
    def get_gesture_type(self) -> str:
        """
        제스처 타입 반환
        
        Returns:
            str: 제스처 타입 ("static" 또는 "dynamic")
        """
        return self.gesture_type
