"""
제어 매니저 모듈
모드 전환 및 제스처-액션 매핑 관리
"""

from typing import Optional, Dict
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.control.base.action_base import BaseAction
from src.control.common.common_actions import (
    StartAction, StopAction,
    OpenGameAction, UpAction, DownAction, LeftAction, RightAction
)
from src.control.ppt.ppt_actions import NextAction, PrevAction, ShowStartAction
from src.control.youtube.youtube_actions import (
    PlayPauseAction, VolumeUpAction, VolumeDownAction,
    MuteAction, FullscreenAction
)
from src.gesture.registry.action_mapper import ActionMapper

# 제스처 클래스들을 import하여 데코레이터가 실행되도록 함
from src.gesture.common import (
    StartGesture, StopGesture,
    SpidermanGesture, PointUpGesture, PointDownGesture, PointLeftGesture, PointRightGesture
)
from src.gesture.ppt import NextGesture, PrevGesture, ShowStartGesture
from src.gesture.youtube import (
    PlayPauseGesture, VolumeUpGesture, VolumeDownGesture,
    MuteGesture, FullscreenGesture
)


class ControllerManager:
    """제어 매니저 클래스
    
    모드 전환 관리 및 제스처-액션 매핑을 담당합니다.
    액션 매퍼를 사용하여 제스처와 액션을 연결합니다.
    """
    
    def __init__(self):
        """제어 매니저 초기화"""
        # 모드별 액션 인스턴스
        self.actions: Dict[str, BaseAction] = {
            # Common 액션
            "START": StartAction(),
            "STOP": StopAction(),
            # PPT 액션
            "NEXT": NextAction(),
            "PREV": PrevAction(),
            "SHOW_START": ShowStartAction(),
            # YouTube 액션
            "PLAY_PAUSE": PlayPauseAction(),
            "VOLUME_UP": VolumeUpAction(),
            "VOLUME_DOWN": VolumeDownAction(),
            "MUTE": MuteAction(),
            "FULLSCREEN": FullscreenAction(),
            "FULLSCREEN": FullscreenAction(),
            # GAME 액션
            "OPEN_GAME": OpenGameAction(),
            "UP": UpAction(),
            "DOWN": DownAction(),
            "LEFT": LeftAction(),
            "RIGHT": RightAction(),
        }
        
        # 현재 모드
        self.current_mode = "PPT"  # "COMMON", "PPT", "YOUTUBE", "GAME"
        
        # 액션 매퍼
        self.action_mapper = ActionMapper()
    
    def set_mode(self, mode: str):
        """
        모드 전환
        
        모드 전환 시 해당 모드의 제스처-액션 매핑이 사용됩니다.
        액션 매퍼를 통해 현재 모드에 맞는 액션만 실행됩니다.
        
        Args:
            mode: 모드 이름 ("COMMON", "PPT", "YOUTUBE", "GAME")
        """
        if mode in ["COMMON", "PPT", "YOUTUBE", "GAME"]:
            self.current_mode = mode
    
    def get_current_mode(self) -> str:
        """
        현재 모드 반환
        
        Returns:
            str: 현재 모드 ("COMMON", "PPT", "YOUTUBE")
        """
        return self.current_mode
    
    def execute_action(self, gesture: str) -> bool:
        """
        제스처에 해당하는 액션 실행
        
        액션 매퍼를 사용하여 제스처를 액션으로 매핑하고 실행합니다.
        공통 제스처(START/STOP)는 모든 모드에서 처리되며,
        모드별 제스처는 현재 모드에 따라 액션 매퍼를 통해 조회됩니다.
        
        Args:
            gesture: 인식된 제스처 이름
        
        Returns:
            bool: 실행 성공 여부
        """
        if gesture == "NONE" or gesture is None:
            return False
        
        # 액션 매퍼를 통해 제스처에 매핑된 액션 이름 조회
        # 먼저 현재 모드에서 조회
        action_name = self.action_mapper.get_action(gesture, self.current_mode)
        
        # 현재 모드에 없으면 COMMON 모드에서 조회 (공통 제스처 지원)
        if action_name is None and self.current_mode != "COMMON":
            action_name = self.action_mapper.get_action(gesture, "COMMON")
        
        # 매퍼에 없으면 제스처 이름을 직접 액션 이름으로 사용 (하위 호환성)
        if action_name is None:
            action_name = gesture
        
        # 액션 실행 (비동기 처리로 UI 블로킹 방지)
        if action_name in self.actions:
            # 스레드로 실행
            import threading
            def run_wrapper():
                try:
                    self.actions[action_name].execute()
                except Exception as e:
                    print(f"Action Execution Failed: {e}")
                    import traceback
                    traceback.print_exc()

            threading.Thread(
                target=run_wrapper, 
                daemon=True
            ).start()
            return True
        
        return False
    
    def enable_controller(self, mode: Optional[str] = None):
        """
        컨트롤러 활성화
        
        Args:
            mode: 활성화할 모드 (None이면 현재 모드)
        """
        # 현재는 특별한 활성화 로직이 없음
        # 필요시 각 액션 클래스에 enable/disable 메서드 추가 가능
        pass
    
    def disable_controller(self, mode: Optional[str] = None):
        """
        컨트롤러 비활성화
        
        Args:
            mode: 비활성화할 모드 (None이면 현재 모드)
        """
        # 현재는 특별한 비활성화 로직이 없음
        # 필요시 각 액션 클래스에 enable/disable 메서드 추가 가능
        pass
