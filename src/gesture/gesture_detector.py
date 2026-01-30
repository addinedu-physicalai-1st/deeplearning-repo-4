"""
통합 제스처 인식 모듈
레지스트리 기반으로 정적/동적 제스처를 모두 지원하는 통합 인식기
"""

import numpy as np
from collections import deque
from typing import Optional, List, Dict
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.gesture.registry.gesture_registry import GestureRegistry
from src.gesture.recognizers.rule_based_recognizer import RuleBasedRecognizer
from src.gesture.recognizers.lstm_recognizer import LSTMRecognizer


class GestureDetector:
    """통합 제스처 인식기
    
    레지스트리 기반으로 정적/동적 제스처를 모두 지원합니다.
    규칙 기반 및 LSTM 기반 인식을 모두 지원합니다.
    """
    
    def __init__(self, mode: str = "COMMON", recognition_method: str = "rule"):
        """
        제스처 인식기 초기화
        
        Args:
            mode: 현재 모드 ("COMMON", "PPT", "YOUTUBE", "GAME")
            recognition_method: 인식 방법 ("rule", "lstm", "auto")
                              "auto"면 제스처별 기본 방법 사용
        """
        self.mode = mode
        self.recognition_method = recognition_method
        self.detection_threshold = 0.7  # 기본 임계값
        
        # 레지스트리 및 인식기 초기화
        self.registry = GestureRegistry()
        self.rule_recognizer = RuleBasedRecognizer()
        self.lstm_recognizer = LSTMRecognizer()  # TODO: 모델 경로 설정
        
        # 동적 제스처용 시퀀스 버퍼
        self.sequence_buffers: Dict[str, deque] = {}
    
    def detect(self, landmarks_list: Optional[List]) -> List[str]:
        """
        랜드마크에서 제스처 인식 (정적/동적 자동 분기)
        
        Args:
            landmarks_list: 손 랜드마크 리스트 (각 손마다 하나의 리스트)
                           각 리스트는 21개의 랜드마크 포인트를 포함
        
        Returns:
            List[str]: 인식된 제스처 이름 리스트 (없으면 ["NONE"])
        """
        if landmarks_list is None or len(landmarks_list) == 0:
            return ["NONE"]
            
        detected_gestures = []
            
        # 모든 감지된 손에 대해 제스처 확인
        for landmarks in landmarks_list:
            if len(landmarks) < 21:
                continue
                
            # 현재 모드의 제스처 목록 조회
            gesture_names = self.registry.get_gestures_by_mode(self.mode)
            # COMMON 모드 제스처도 항상 함께 검사 (Spiderman 등 전역 제스처 지원)
            if self.mode != "COMMON":
                common_gestures = self.registry.get_gestures_by_mode("COMMON")
                # 중복 제거 및 GAME 모드에서 STOP, START, OPEN_GAME 제외
                for g in common_gestures:
                    if self.mode == "GAME" and g in ["STOP", "START", "OPEN_GAME"]:
                        continue
                    if g not in gesture_names:
                        gesture_names.append(g)
            
            # 정적 제스처 먼저 인식
            for gesture_name in gesture_names:
                # GAME 모드에서 제외된 제스처 건너뛰기 (Double check)
                if self.mode == "GAME" and gesture_name in ["STOP", "START", "OPEN_GAME"]:
                    continue
                    
                # print(f"Checking Gesture: {gesture_name}") # DEBUG
                gesture = self.registry.get_gesture(gesture_name)
                if gesture is None:
                    continue
                
                # 정적 제스처 인식
                if gesture.get_gesture_type() == "static":
                    result = self._detect_static_gesture(gesture, landmarks)
                    if result != "NONE":
                        print(f"Detected Gesture: {result}") # DEBUG
                        detected_gestures.append(result)
                        break 
                        # 한 손당 하나만? 아니면 한 손에서도 여러개? 
                        # 보통 한 손은 하나의 모양만 가짐. 
                        # 따라서 이 손에 대해 우선순위 높은거 찾으면 다음 손으로 넘어가는게 맞음.

        # 동적 제스처는 시퀀스 버퍼에 추가하고 인식
        # 현재 동적 제스처 로직은 단일 손 제어에 맞춰져 있어 복잡함.
        # 일단 정적 제스처가 있으면 동적 제스처는 건너뜀 (안전성)
        if detected_gestures:
            return detected_gestures
            
        return ["NONE"]
            
            # 동적 제스처는 지금은 첫 번째 손만 처리하거나 스킵 (복잡성 방지)
            # 여기서는 정적 제스처가 우선이므로 루프 계속
        
        # 동적 제스처는 시퀀스 버퍼에 추가하고 인식
        for gesture_name in gesture_names:
            # print(f"Checking Gesture: {gesture_name}") # DEBUG
            gesture = self.registry.get_gesture(gesture_name)
            if gesture is None:
                continue
            
            # 동적 제스처 인식
            if gesture.get_gesture_type() == "dynamic":
                result = self._detect_dynamic_gesture(gesture, landmarks)
                if result != "NONE":
                    return result
        
        return "NONE"
    
    def _detect_static_gesture(self, gesture, landmarks: List[Dict]) -> str:
        """
        정적 제스처 인식
        
        Args:
            gesture: 제스처 인스턴스
            landmarks: 21개의 랜드마크 포인트 리스트
        
        Returns:
            str: 인식된 제스처 이름
        """
        # 인식 방법 결정
        method = self._get_recognition_method(gesture)
        
        if method == "rule":
            return self.rule_recognizer.detect_static(gesture, landmarks)
        elif method == "lstm" and self.lstm_recognizer:
            return self.lstm_recognizer.detect_static(gesture, landmarks)
        
        return "NONE"
    
    def _detect_dynamic_gesture(self, gesture, landmarks: List[Dict]) -> str:
        """
        동적 제스처 인식 (시퀀스 기반)
        
        Args:
            gesture: 제스처 인스턴스
            landmarks: 21개의 랜드마크 포인트 리스트
        
        Returns:
            str: 인식된 제스처 이름
        """
        gesture_name = gesture.get_name()
        
        # 시퀀스 버퍼 초기화
        if gesture_name not in self.sequence_buffers:
            sequence_length = getattr(gesture, 'sequence_length', 30)
            self.sequence_buffers[gesture_name] = deque(maxlen=sequence_length)
        
        # 랜드마크를 벡터로 변환하여 버퍼에 추가
        landmark_vector = self._landmarks_to_vector(landmarks)
        self.sequence_buffers[gesture_name].append(landmark_vector)
        
        # 버퍼가 충분히 채워지지 않았으면 NONE 반환
        if len(self.sequence_buffers[gesture_name]) < self.sequence_buffers[gesture_name].maxlen:
            return "NONE"
        
        # 시퀀스 배열 생성
        sequence = np.array(list(self.sequence_buffers[gesture_name]))
        
        # 인식 방법 결정
        method = self._get_recognition_method(gesture)
        
        if method == "rule":
            return self.rule_recognizer.detect_dynamic(gesture, sequence)
        elif method == "lstm" and self.lstm_recognizer:
            return self.lstm_recognizer.detect_dynamic(gesture, sequence)
        
        return "NONE"
    
    def _landmarks_to_vector(self, landmarks: List[Dict]) -> np.ndarray:
        """
        랜드마크를 벡터로 변환
        
        Args:
            landmarks: 21개의 랜드마크 포인트 리스트
        
        Returns:
            np.ndarray: 랜드마크 벡터 (63차원: 21개 포인트 × 3차원)
        """
        vector = []
        for landmark in landmarks:
            vector.extend([landmark['x'], landmark['y'], landmark['z']])
        return np.array(vector)
    
    def _get_recognition_method(self, gesture) -> str:
        """
        제스처별 인식 방법 결정
        
        Args:
            gesture: 제스처 인스턴스
        
        Returns:
            str: 인식 방법 ("rule" 또는 "lstm")
        """
        if self.recognition_method != "auto":
            return self.recognition_method
        
        # 제스처별 기본 방법 사용 (현재는 모두 규칙 기반)
        # TODO: 제스처 메타데이터에서 기본 방법 읽기
        return "rule"
    
    def set_mode(self, mode: str):
        """
        모드 설정
        
        Args:
            mode: 모드 이름 ("COMMON", "PPT", "YOUTUBE", "GAME")
        """
        if mode in ["COMMON", "PPT", "YOUTUBE", "GAME"]:
            self.mode = mode
            # 모드 변경 시 시퀀스 버퍼 초기화
            self.sequence_buffers.clear()
    
    def set_threshold(self, threshold: float):
        """
        인식 임계값 설정 (감도 조절용)
        
        Args:
            threshold: 임계값 (0.0 - 1.0)
        """
        self.detection_threshold = max(0.0, min(1.0, threshold))
    
    def set_recognition_method(self, method: str):
        """
        인식 방법 설정
        
        Args:
            method: 인식 방법 ("rule", "lstm", "auto")
        """
        if method in ["rule", "lstm", "auto"]:
            self.recognition_method = method
    
    def reset(self):
        """상태 초기화"""
        self.sequence_buffers.clear()
