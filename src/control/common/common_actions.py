"""
Common 액션 클래스들
공통 필수 기능을 위한 액션들
"""

from src.control.base.action_base import BaseAction
from pynput.keyboard import Key, Controller
import time
import webbrowser

# Initialize Keyboard Controller
keyboard = Controller()


class StartAction(BaseAction):
    """START 액션 클래스
    
    동작 감지 시작을 위한 액션입니다.
    """
    
    name = "START"
    mode = "COMMON"
    
    def __init__(self):
        """START 액션 초기화"""
        super().__init__()
    
    def execute(self) -> bool:
        """
        START 액션 실행
        
        Returns:
            bool: 실행 성공 여부
        """
        # TODO: START 액션 실행 로직 구현
        # 예: 제스처 인식 활성화 등
        return True


class StopAction(BaseAction):
    """STOP 액션 클래스
    
    동작 감지 종료를 위한 액션입니다.
    """
    
    name = "STOP"
    mode = "COMMON"
    
    def __init__(self):
        """STOP 액션 초기화"""
        super().__init__()
    
    def execute(self) -> bool:
        """
        STOP 액션 실행
        
        Returns:
            bool: 실행 성공 여부
        """
        # TODO: STOP 액션 실행 로직 구현
        # 예: 제스처 인식 비활성화 등
        return True


import time

class OpenGameAction(BaseAction):
    """게임 열기 액션"""
    name = "OPEN_GAME"
    mode = "COMMON"
    
    _last_execution_time = 0
    _cooldown = 3.0  # 3 seconds cooldown
    _is_opened = False # Track if game has been opened
    
    def execute(self) -> bool:
        current_time = time.time()
        if current_time - self._last_execution_time < self._cooldown:
            return False
            
        self._last_execution_time = current_time
        
        # Only open if not already opened
        if self._is_opened:
            print("Game already opened, skipping browser launch")
            return True
            
        self._is_opened = True
        
        url = "https://ratlabyrinth.netlify.app/"
        try:
            # Try to grab google-chrome specifically
            browser = webbrowser.get('google-chrome')
            browser.open(url)
        except webbrowser.Error:
            # Fallback if chrome not found or registered under that name
            webbrowser.open(url)
        return True


class UpAction(BaseAction):
    """위로 이동 액션"""
    name = "UP"
    mode = "GAME"
    
    def execute(self) -> bool:
        # print("Action: UP") # DEBUG
        keyboard.press(Key.up)
        time.sleep(0.1)
        keyboard.release(Key.up)
        return True


class DownAction(BaseAction):
    """아래로 이동 액션"""
    name = "DOWN"
    mode = "GAME"
    
    def execute(self) -> bool:
        # print("Action: DOWN") # DEBUG
        keyboard.press(Key.down)
        time.sleep(0.1)
        keyboard.release(Key.down)
        return True


class LeftAction(BaseAction):
    """왼쪽으로 이동 액션"""
    name = "LEFT"
    mode = "GAME"
    
    def execute(self) -> bool:
        # print("Action: LEFT") # DEBUG
        keyboard.press(Key.left)
        time.sleep(0.1)
        keyboard.release(Key.left)
        return True


class RightAction(BaseAction):
    """오른쪽으로 이동 액션"""
    name = "RIGHT"
    mode = "GAME"
    
    def execute(self) -> bool:
        # print("Action: RIGHT") # DEBUG
        keyboard.press(Key.right)
        time.sleep(0.1)
        keyboard.release(Key.right)
        return True
