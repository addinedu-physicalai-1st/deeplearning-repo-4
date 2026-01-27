"""
YouTube 제스처 모듈
YouTube 제어를 위한 제스처 (PLAY_PAUSE, VOLUME_UP, VOLUME_DOWN, MUTE, FULLSCREEN)
"""

# 제스처 클래스들을 import하여 자동 등록되도록 함
from .play_pause_gesture import PlayPauseGesture
from .volume_up_gesture import VolumeUpGesture
from .volume_down_gesture import VolumeDownGesture
from .mute_gesture import MuteGesture
from .fullscreen_gesture import FullscreenGesture

# 레지스트리에 자동 등록
from src.gesture.registry.gesture_registry import GestureRegistry
registry = GestureRegistry()
registry.register(PlayPauseGesture)
registry.register(VolumeUpGesture)
registry.register(VolumeDownGesture)
registry.register(MuteGesture)
registry.register(FullscreenGesture)

__all__ = [
    'PlayPauseGesture',
    'VolumeUpGesture',
    'VolumeDownGesture',
    'MuteGesture',
    'FullscreenGesture'
]
