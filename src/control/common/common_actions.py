"""
Common 액션 클래스들
공통 필수 기능을 위한 액션들
"""

from src.control.base.action_base import BaseAction


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
