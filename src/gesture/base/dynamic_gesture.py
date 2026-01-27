"""
동적 제스처 추상 클래스
시퀀스 기반으로 인식하는 동적 제스처를 위한 기본 클래스
"""

from abc import abstractmethod
from typing import Optional, List, Dict
import numpy as np
from .gesture_base import BaseGesture


class DynamicGesture(BaseGesture):
    """동적 제스처 추상 클래스
    
    여러 프레임의 랜드마크 시퀀스를 기반으로 인식하는 제스처를 위한 기본 클래스.
    예: 손 흔들기, 손 움직임 등
    """
    
    gesture_type: str = "dynamic"
    sequence_length: int = 30  # 시퀀스 길이 (프레임 수)
    
    def __init__(self, sequence_length: int = 30):
        """
        동적 제스처 초기화
        
        Args:
            sequence_length: 시퀀스 길이 (프레임 수, 기본값: 30)
        """
        super().__init__()
        self.sequence_length = sequence_length
    
    def detect(self, landmarks_list: Optional[List]) -> str:
        """
        랜드마크에서 동적 제스처 인식
        
        주의: 동적 제스처는 시퀀스 버퍼가 필요하므로,
        이 메서드는 GestureDetector에서 시퀀스 버퍼를 관리한 후
        detect_dynamic()을 직접 호출하는 것을 권장합니다.
        
        Args:
            landmarks_list: 손 랜드마크 리스트 (각 손마다 하나의 리스트)
        
        Returns:
            str: 인식된 제스처 이름 (인식되지 않으면 "NONE")
        """
        # 동적 제스처는 시퀀스 기반이므로 단일 프레임만으로는 인식 불가
        return "NONE"
    
    @abstractmethod
    def detect_dynamic(self, landmark_sequence: np.ndarray) -> str:
        """
        랜드마크 시퀀스에서 동적 제스처 인식
        
        Args:
            landmark_sequence: 랜드마크 시퀀스 배열
                              shape: (sequence_length, 21, 3)
                              또는 (sequence_length, 63) - flatten된 형태
        
        Returns:
            str: 인식된 제스처 이름 (인식되지 않으면 "NONE")
        """
        pass
    
    def get_sequence_length(self) -> int:
        """
        시퀀스 길이 반환
        
        Returns:
            int: 시퀀스 길이 (프레임 수)
        """
        return self.sequence_length
