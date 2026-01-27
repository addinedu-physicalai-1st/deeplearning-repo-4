"""
VOLUME_UP 제스처
볼륨 증가를 위한 제스처
"""

from typing import List, Dict
from src.gesture.base.static_gesture import StaticGesture
from src.gesture.registry.action_mapper import maps_to_action


@maps_to_action("YOUTUBE", "VOLUME_UP")
class VolumeUpGesture(StaticGesture):
    """VOLUME_UP 제스처 클래스
    
    볼륨 증가를 위한 제스처입니다.
    정적 또는 동적 제스처로 구현 가능합니다.
    """
    
    name = "VOLUME_UP"
    mode = "YOUTUBE"
    gesture_type = "static"  # 정적 제스처로 시작, 필요시 dynamic으로 변경 가능
    
    def __init__(self):
        """VOLUME_UP 제스처 초기화"""
        super().__init__()
    
    def detect_static(self, landmarks: List[Dict]) -> str:
        """
        단일 프레임 랜드마크에서 VOLUME_UP 제스처 인식
        
        Args:
            landmarks: 21개의 랜드마크 포인트 리스트
                      각 포인트는 {'x': float, 'y': float, 'z': float} 형태
        
        Returns:
            str: "VOLUME_UP" (인식된 경우) 또는 "NONE" (인식되지 않은 경우)
        """
        # TODO: VOLUME_UP 제스처 인식 로직 구현
        # 예: 위로 손 흔들기, 특정 손가락 제스처 등
        # 동적 제스처가 필요한 경우 DynamicGesture를 상속받아 변경
        return "NONE"
