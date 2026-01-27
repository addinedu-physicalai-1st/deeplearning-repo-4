"""
PPT 액션 클래스들
PPT 제어를 위한 액션들
"""

from src.control.base.action_base import BaseAction


class NextAction(BaseAction):
    """NEXT 액션 클래스
    
    다음 슬라이드로 이동하는 액션입니다.
    """
    
    name = "NEXT"
    mode = "PPT"
    
    def __init__(self):
        """NEXT 액션 초기화"""
        super().__init__()
    
    def execute(self) -> bool:
        """
        NEXT 액션 실행
        
        Returns:
            bool: 실행 성공 여부
        """
        # TODO: NEXT 액션 실행 로직 구현
        # 예: pyautogui.press('right') 또는 PowerPoint 자동화
        return True


class PrevAction(BaseAction):
    """PREV 액션 클래스
    
    이전 슬라이드로 이동하는 액션입니다.
    """
    
    name = "PREV"
    mode = "PPT"
    
    def __init__(self):
        """PREV 액션 초기화"""
        super().__init__()
    
    def execute(self) -> bool:
        """
        PREV 액션 실행
        
        Returns:
            bool: 실행 성공 여부
        """
        # TODO: PREV 액션 실행 로직 구현
        # 예: pyautogui.press('left') 또는 PowerPoint 자동화
        return True


class ShowStartAction(BaseAction):
    """SHOW_START 액션 클래스
    
    슬라이드 쇼 시작 액션입니다.
    """
    
    name = "SHOW_START"
    mode = "PPT"
    
    def __init__(self):
        """SHOW_START 액션 초기화"""
        super().__init__()
    
    def execute(self) -> bool:
        """
        SHOW_START 액션 실행
        
        Returns:
            bool: 실행 성공 여부
        """
        # TODO: SHOW_START 액션 실행 로직 구현
        # 예: pyautogui.hotkey('f5') 또는 PowerPoint 자동화
        return True
