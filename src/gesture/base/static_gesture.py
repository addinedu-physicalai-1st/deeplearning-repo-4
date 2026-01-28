"""
정적 제스처 추상 클래스
단일 프레임에서 인식 가능한 정적 제스처를 위한 기본 클래스
"""

from abc import abstractmethod
from typing import Optional, List, Dict
import math
from .gesture_base import BaseGesture
from src.gesture.registry.action_mapper import maps_to_action


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
    def _distance(self, p1: Dict, p2: Dict) -> float:
        """두 점 사이의 유클리드 거리 계산"""
        dx = p1['x'] - p2['x']
        dy = p1['y'] - p2['y']
        return math.sqrt(dx*dx + dy*dy)

    def _get_finger_ratio(self, landmarks: List[Dict], finger_name: str) -> float:
        """손가락 길이 비율 계산"""
        finger_indices = {
            "index": {"tip": 8, "mcp": 5},
            "middle": {"tip": 12, "mcp": 9},
            "ring": {"tip": 16, "mcp": 13},
            "pinky": {"tip": 20, "mcp": 17}
        }
        
        idx = finger_indices[finger_name]
        wrist = landmarks[0]
        mcp = landmarks[idx["mcp"]]
        tip = landmarks[idx["tip"]]
        
        palm_len = self._distance(wrist, mcp)
        if palm_len == 0: return 0.0
        
        finger_len = self._distance(mcp, tip)
        return finger_len / palm_len

    def _is_finger_extended(self, landmarks: List[Dict], finger_name: str) -> bool:
        """
        손가락 펴짐 여부 확인 (Tip-MCP 거리 비율 기반)
        Wrist-MCP 거리(손바닥 길이)를 기준으로 Tip-MCP 거리 비율을 계산
        """
        ratio = self._get_finger_ratio(landmarks, finger_name)
        
        # print(f"Finger: {finger_name}, Ratio: {ratio:.2f}") # DEBUG
        
        # 펴짐 임계값: 일반적으로 0.8 이상이면 펴짐
        # 접힘 임계값: 0.6 이하면 접힘
        return ratio > 0.65

    def is_v_sign(self, landmarks: List[Dict]) -> bool:
        """V-Sign (가위/브이) 제스처 여부 확인"""
        # 검지, 중지 펴짐
        index_extended = self._is_finger_extended(landmarks, "index")
        middle_extended = self._is_finger_extended(landmarks, "middle")
        
        # 약지, 소지 접힘 (Not Extended)
        ring_folded = not self._is_finger_extended(landmarks, "ring")
        pinky_folded = not self._is_finger_extended(landmarks, "pinky")
        
        if index_extended and middle_extended and ring_folded and pinky_folded:
            # 검지와 중지 사이 거리 확인 (V 모양)
            dist = self._distance(landmarks[8], landmarks[12])
            return dist > 0.05
        return False

    def is_spiderman(self, landmarks: List[Dict]) -> bool:
        """Spiderman 제스처 확인"""
        # Custom Logic for Spiderman (More lenient on folded fingers)
        # 검지, 소지 펴짐
        index_ratio = self._get_finger_ratio(landmarks, "index")
        pinky_ratio = self._get_finger_ratio(landmarks, "pinky")
        
        index_extended = index_ratio > 0.65
        pinky_extended = pinky_ratio > 0.65
        
        # 중지, 약지 접힘 (Lenient threshold: < 0.85)
        # Ring finger sometimes looks extended in logs (around 0.76), so we allow up to 0.85
        middle_ratio = self._get_finger_ratio(landmarks, "middle")
        ring_ratio = self._get_finger_ratio(landmarks, "ring")
        
        middle_folded = middle_ratio < 0.85
        ring_folded = ring_ratio < 0.85
        
        # print(f"Spiderman Check: I={index_extended}, P={pinky_extended}, M_Fold={middle_folded}, R_Fold={ring_folded}")
        
        return index_extended and pinky_extended and middle_folded and ring_folded

    def is_pointing(self, landmarks: List[Dict], direction: str) -> bool:
        """검지 포인팅 확인"""
        # 1. 검지 펴짐
        if not self._is_finger_extended(landmarks, "index"):
            return False
            
        # 2. 나머지 접힘
        if self._is_finger_extended(landmarks, "middle") or \
           self._is_finger_extended(landmarks, "ring") or \
           self._is_finger_extended(landmarks, "pinky"):
            return False
            
        # 3. 방향 확인 (손목 기준 검지 끝의 상대 위치)
        wrist = landmarks[0]
        index_tip = landmarks[8]
        
        dx = index_tip['x'] - wrist['x']
        dy = index_tip['y'] - wrist['y']
        
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # 임계값 (노이즈 방지)
        THRESHOLD = 0.1
        
        if direction == "UP":
            # 위로: dy가 음수 (화면상 위쪽), 수직 성분이 더 큼
            return dy < -THRESHOLD and abs_dy > abs_dx
            
        elif direction == "DOWN":
            # 아래로: dy가 양수, 수직 성분이 더 큼
            return dy > THRESHOLD and abs_dy > abs_dx
            
        elif direction == "LEFT":
            # 왼쪽: dx가 음수 (화면상 왼쪽 -> 사용자의 오른쪽?), 수평 성분이 더 큼
            # 미러링 여부에 따라 다름. 여기서는 화면 좌표 기준 (왼쪽이 x 작음)
            return dx < -THRESHOLD and abs_dx > abs_dy
            
        elif direction == "RIGHT":
            # 오른쪽: dx가 양수, 수평 성분이 더 큼
            return dx > THRESHOLD and abs_dx > abs_dy
            
        return False


@maps_to_action("COMMON", "OPEN_GAME")
class SpidermanGesture(StaticGesture):
    """Spiderman 제스처 (게임 열기)"""
    name = "SPIDERMAN"
    mode = "COMMON"
    gesture_type = "static"
    
    def detect_static(self, landmarks: List[Dict]) -> str:
        if self.is_spiderman(landmarks):
            return "OPEN_GAME"
        return "NONE"


@maps_to_action("GAME", "UP")
class PointUpGesture(StaticGesture):
    """위로 가리키기 (화살표 위)"""
    name = "POINT_UP"
    mode = "GAME"
    gesture_type = "static"
    
    def detect_static(self, landmarks: List[Dict]) -> str:
        if self.is_pointing(landmarks, "UP"):
            return "POINT_UP"
        return "NONE"


@maps_to_action("GAME", "DOWN")
class PointDownGesture(StaticGesture):
    """아래로 가리키기 (화살표 아래)"""
    name = "POINT_DOWN"
    mode = "GAME"
    gesture_type = "static"
    
    def detect_static(self, landmarks: List[Dict]) -> str:
        if self.is_pointing(landmarks, "DOWN"):
            return "POINT_DOWN"
        return "NONE"


@maps_to_action("GAME", "LEFT")
class PointLeftGesture(StaticGesture):
    """왼쪽으로 가리키기 (화살표 왼쪽)"""
    name = "POINT_LEFT"
    mode = "GAME"
    gesture_type = "static"
    
    def detect_static(self, landmarks: List[Dict]) -> str:
        if self.is_pointing(landmarks, "LEFT"):
            return "POINT_LEFT"
        return "NONE"


@maps_to_action("GAME", "RIGHT")
class PointRightGesture(StaticGesture):
    """오른쪽으로 가리키기 (화살표 오른쪽)"""
    name = "POINT_RIGHT"
    mode = "GAME"
    gesture_type = "static"
    
    def detect_static(self, landmarks: List[Dict]) -> str:
        if self.is_pointing(landmarks, "RIGHT"):
            return "POINT_RIGHT"
        return "NONE"
