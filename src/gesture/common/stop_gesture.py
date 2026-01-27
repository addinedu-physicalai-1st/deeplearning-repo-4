"""
STOP 제스처
동작 감지 종료를 위한 제스처
"""

import numpy as np
from src.gesture.base.dynamic_gesture import DynamicGesture
from src.gesture.registry.action_mapper import maps_to_action


@maps_to_action("COMMON", "STOP")
class StopGesture(DynamicGesture):
    """STOP 제스처 클래스
    
    동작 감지 종료를 위한 동적 제스처입니다.
    손을 좌우로 흔드는 동작을 감지합니다.
    """
    
    name = "STOP"
    mode = "COMMON"
    gesture_type = "dynamic"
    sequence_length = 20  # 시퀀스 길이 (프레임 수) - 더 짧게 조정하여 빠른 인식
    
    def __init__(self):
        """STOP 제스처 초기화"""
        super().__init__(sequence_length=self.sequence_length)
    
    def detect_dynamic(self, landmark_sequence: np.ndarray) -> str:
        """
        랜드마크 시퀀스에서 STOP 제스처 인식
        
        손을 좌우로 흔드는 동작을 감지합니다.
        손목(랜드마크 0)의 x 좌표 변화를 분석하여 좌우 왕복 움직임을 감지합니다.
        
        Args:
            landmark_sequence: 랜드마크 시퀀스 배열
                              shape: (sequence_length, 63)
                              각 행은 21개 포인트의 (x, y, z) 좌표를 flatten한 형태
        
        Returns:
            str: "STOP" (인식된 경우) 또는 "NONE" (인식되지 않은 경우)
        """
        if landmark_sequence is None or landmark_sequence.shape[0] < 10:
            return "NONE"
        
        # landmark_sequence shape: (sequence_length, 63)
        # 각 행은 [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20] 형태
        # 손목의 x 좌표는 인덱스 0에 위치
        wrist_x = landmark_sequence[:, 0]
        
        # x 좌표의 최대값과 최소값 차이 (움직임 범위)
        x_range = np.max(wrist_x) - np.min(wrist_x)
        
        # x 좌표의 표준편차 (변동성)
        x_std = np.std(wrist_x)
        
        # 변화 방향이 바뀌는 횟수 계산 (좌우 왕복 감지)
        x_diff = np.diff(wrist_x)  # 연속된 프레임 간 x 좌표 차이
        if len(x_diff) > 0:
            direction_changes = np.sum(np.diff(np.sign(x_diff)) != 0)  # 부호가 바뀌는 횟수
        else:
            direction_changes = 0
        
        # 개선된 임계값: 여러 조건을 조합하여 더 정확하게 인식
        # 조건 1: 표준편차가 0.025 이상 (변동성이 큼) - 가장 민감한 조건
        # 조건 2: 범위가 0.05 이상이고 방향 변화가 1회 이상 (명확한 왕복)
        # 조건 3: 범위가 0.07 이상 (큰 움직임, 방향 변화 없어도 인식)
        has_shake = (
            x_std > 0.025 or  # 변동성이 크면
            (x_range > 0.05 and direction_changes >= 1) or  # 명확한 왕복
            x_range > 0.07  # 큰 움직임
        )
        
        if has_shake:
            return "STOP"
        
        return "NONE"
