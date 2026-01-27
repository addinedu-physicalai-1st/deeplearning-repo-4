"""
Common 제스처 모듈
공통 필수 기능을 위한 제스처 (START, STOP)
"""

# 제스처 클래스들을 import하여 자동 등록되도록 함
from .start_gesture import StartGesture
from .stop_gesture import StopGesture

# 레지스트리에 자동 등록
from src.gesture.registry.gesture_registry import GestureRegistry
registry = GestureRegistry()
registry.register(StartGesture)
registry.register(StopGesture)

__all__ = ['StartGesture', 'StopGesture']
