"""
START 제스처
동작 감지 시작을 위한 제스처
"""

from typing import List, Dict
from src.gesture.base.static_gesture import StaticGesture
from src.gesture.registry.action_mapper import maps_to_action


@maps_to_action("COMMON", "START")
class StartGesture(StaticGesture):
    """START 제스처 클래스
    
    동작 감지 시작을 위한 정적 제스처입니다.
    예: 손 펴기, 특정 손가락 제스처 등
    """
    
    name = "START"
    mode = "COMMON"
    gesture_type = "static"
    
    def __init__(self):
        """START 제스처 초기화"""
        super().__init__()
    
    def detect_static(self, landmarks: List[Dict]) -> str:
        """
        단일 프레임 랜드마크에서 START 제스처 인식
        
        손가락이 모두 펴진 상태를 감지합니다.
        - 엄지: 4번 포인트가 3번 포인트보다 오른쪽(x 좌표가 큼)
        - 나머지 손가락: 끝 포인트가 기저부보다 위(y 좌표가 작음)
        
        Args:
            landmarks: 21개의 랜드마크 포인트 리스트
                      각 포인트는 {'x': float, 'y': float, 'z': float} 형태
        
        Returns:
            str: "START" (인식된 경우) 또는 "NONE" (인식되지 않은 경우)
        """
        if len(landmarks) < 21:
            return "NONE"
        
        fingers_up = 0
        
        # 엄지 (x 좌표 비교: 4번이 3번보다 오른쪽에 있으면 펴진 것)
        if landmarks[4]['x'] > landmarks[3]['x']:
            fingers_up += 1
        
        # 검지, 중지, 약지, 새끼손가락 (y 좌표 비교: 끝 포인트가 기저부보다 위에 있으면 펴진 것)
        # 검지: 8번(끝) vs 7번(기저부)
        # 중지: 12번(끝) vs 11번(기저부)
        # 약지: 16번(끝) vs 15번(기저부)
        # 새끼손가락: 20번(끝) vs 19번(기저부)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [7, 11, 15, 19]
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks[tip]['y'] < landmarks[pip]['y']:
                fingers_up += 1
        
        # 4개 이상 손가락이 펴져 있으면 START
        if fingers_up >= 4:
            return "START"
        
        return "NONE"
