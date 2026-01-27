"""
규칙 기반 제스처 인식기
규칙 기반으로 제스처를 인식하는 인식기
"""

from typing import Optional, List, Dict
import numpy as np
from src.gesture.base.gesture_base import BaseGesture


class RuleBasedRecognizer:
    """규칙 기반 제스처 인식기
    
    규칙 기반 알고리즘을 사용하여 제스처를 인식합니다.
    """
    
    def __init__(self):
        """규칙 기반 인식기 초기화"""
        pass
    
    def detect_static(self, gesture: BaseGesture, landmarks: List[Dict]) -> str:
        """
        정적 제스처 인식 (규칙 기반)
        
        Args:
            gesture: 제스처 인스턴스
            landmarks: 21개의 랜드마크 포인트 리스트
        
        Returns:
            str: 인식된 제스처 이름
        """
        # 제스처 클래스의 detect_static 메서드 호출
        return gesture.detect_static(landmarks)
    
    def detect_dynamic(self, gesture: BaseGesture, landmark_sequence: np.ndarray) -> str:
        """
        동적 제스처 인식 (규칙 기반)
        
        Args:
            gesture: 제스처 인스턴스
            landmark_sequence: 랜드마크 시퀀스 배열
        
        Returns:
            str: 인식된 제스처 이름
        """
        # 제스처 클래스의 detect_dynamic 메서드 호출
        return gesture.detect_dynamic(landmark_sequence)
