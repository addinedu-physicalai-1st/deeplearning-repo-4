"""
PPT 제스처 클래스들
PPT 제어를 위한 제스처들의 집합
"""

from typing import List, Dict
from src.gesture.base.static_gesture import StaticGesture
from src.gesture.registry.action_mapper import maps_to_action


@maps_to_action("PPT", "WAKE_UP")
class WakeUpGesture(StaticGesture):
    """WAKE_UP 제스처 클래스 (V-Sign)
    
    시스템을 깨우기 위한 제스처입니다.
    정적 제스처: 브이 사인
    """
    
    name = "WAKE_UP"
    mode = "PPT"
    gesture_type = "static"
    
    def detect_static(self, landmarks: List[Dict]) -> str:
        """WAKE_UP 제스처 인식"""
        if self.is_v_sign(landmarks):
            return "WAKE_UP"
        return "NONE"


    def detect_static(self, landmarks: List[Dict]) -> str:
        """WAKE_UP 제스처 인식"""
        if self.is_v_sign(landmarks):
            return "WAKE_UP"
        return "NONE"
