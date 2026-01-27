"""
PPT 제스처 모듈
PPT 제어를 위한 제스처 (NEXT, PREV, SHOW_START)
"""

# 제스처 클래스들을 import하여 자동 등록되도록 함
from .next_gesture import NextGesture
from .prev_gesture import PrevGesture
from .show_start_gesture import ShowStartGesture

# 레지스트리에 자동 등록
from src.gesture.registry.gesture_registry import GestureRegistry
registry = GestureRegistry()
registry.register(NextGesture)
registry.register(PrevGesture)
registry.register(ShowStartGesture)

__all__ = ['NextGesture', 'PrevGesture', 'ShowStartGesture']
