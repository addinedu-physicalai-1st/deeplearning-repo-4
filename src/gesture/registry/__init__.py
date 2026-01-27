"""
제스처 레지스트리 및 액션 매핑 모듈
"""

from .gesture_registry import GestureRegistry
from .action_mapper import ActionMapper, maps_to_action

__all__ = ['GestureRegistry', 'ActionMapper', 'maps_to_action']
