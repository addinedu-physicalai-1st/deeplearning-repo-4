"""
Common 제스처 모듈
공통 필수 기능을 위한 제스처 (START, STOP)
"""

# 제스처 클래스들을 import하여 자동 등록되도록 함
from .start_gesture import StartGesture
from .start_gesture import StartGesture
from .stop_gesture import StopGesture
from src.gesture.base.static_gesture import SpidermanGesture, PointUpGesture, PointDownGesture, PointLeftGesture, PointRightGesture

# 레지스트리에 자동 등록
from src.gesture.registry.gesture_registry import GestureRegistry
registry = GestureRegistry()
registry.register(StartGesture)
registry.register(StopGesture)
registry.register(SpidermanGesture)
registry.register(PointUpGesture)
registry.register(PointDownGesture)
registry.register(PointLeftGesture)
registry.register(PointRightGesture)

__all__ = ['StartGesture', 'StopGesture', 'SpidermanGesture', 'PointUpGesture', 'PointDownGesture', 'PointLeftGesture', 'PointRightGesture']
