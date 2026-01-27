"""
LSTM 기반 제스처 인식기
LSTM 모델을 사용하여 제스처를 인식하는 인식기
"""

from typing import Optional, List, Dict
import numpy as np
from src.gesture.base.gesture_base import BaseGesture


class LSTMRecognizer:
    """LSTM 기반 제스처 인식기
    
    LSTM 모델을 사용하여 제스처를 인식합니다.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        LSTM 인식기 초기화
        
        Args:
            model_path: LSTM 모델 파일 경로 (None이면 모델 미사용)
        """
        self.model_path = model_path
        self.model = None  # TODO: LSTM 모델 로드
    
    def detect_static(self, gesture: BaseGesture, landmarks: List[Dict]) -> str:
        """
        정적 제스처 인식 (LSTM 기반)
        
        Args:
            gesture: 제스처 인스턴스
            landmarks: 21개의 랜드마크 포인트 리스트
        
        Returns:
            str: 인식된 제스처 이름
        """
        # TODO: LSTM 모델을 사용한 정적 제스처 인식
        # 현재는 규칙 기반으로 폴백
        if hasattr(gesture, 'detect_static'):
            return gesture.detect_static(landmarks)
        return "NONE"
    
    def detect_dynamic(self, gesture: BaseGesture, landmark_sequence: np.ndarray) -> str:
        """
        동적 제스처 인식 (LSTM 기반)
        
        Args:
            gesture: 제스처 인스턴스
            landmark_sequence: 랜드마크 시퀀스 배열
        
        Returns:
            str: 인식된 제스처 이름
        """
        # TODO: LSTM 모델을 사용한 동적 제스처 인식
        # 현재는 규칙 기반으로 폴백
        if hasattr(gesture, 'detect_dynamic'):
            return gesture.detect_dynamic(landmark_sequence)
        return "NONE"
