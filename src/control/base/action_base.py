"""
액션 기본 추상 클래스
모든 액션의 기본이 되는 추상 클래스
"""

from abc import ABC, abstractmethod


class BaseAction(ABC):
    """모든 액션의 기본 추상 클래스
    
    모든 액션 클래스는 이 클래스를 상속받아야 합니다.
    """
    
    # 액션 메타데이터 (하위 클래스에서 정의)
    name: str = ""  # 액션 이름 (예: "START", "NEXT")
    mode: str = ""  # 액션 모드 (예: "COMMON", "PPT", "YOUTUBE")
    
    def __init__(self):
        """액션 초기화"""
        pass
    
    @abstractmethod
    def execute(self) -> bool:
        """
        액션 실행
        
        Returns:
            bool: 실행 성공 여부
        """
        pass
    
    def get_name(self) -> str:
        """
        액션 이름 반환
        
        Returns:
            str: 액션 이름
        """
        return self.name
    
    def get_mode(self) -> str:
        """
        액션 모드 반환
        
        Returns:
            str: 액션 모드
        """
        return self.mode
