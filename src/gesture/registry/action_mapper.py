"""
액션 매핑 데코레이터 및 관리자
제스처와 액션을 연결하는 데코레이터 시스템
"""

from functools import wraps
from typing import Dict, Optional, Callable


class ActionMapper:
    """제스처-액션 매핑 관리자
    
    데코레이터로 등록된 제스처와 액션의 매핑을 관리합니다.
    """
    
    _instance: Optional['ActionMapper'] = None
    _mappings: Dict[str, Dict[str, str]] = {}  # {gesture_name: {mode: action_name}}
    
    def __new__(cls):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, gesture_name: str, mode: str, action_name: str):
        """
        제스처-액션 매핑 등록
        
        Args:
            gesture_name: 제스처 이름
            mode: 모드 이름 ("COMMON", "PPT", "YOUTUBE")
            action_name: 액션 이름
        """
        if gesture_name not in self._mappings:
            self._mappings[gesture_name] = {}
        self._mappings[gesture_name][mode] = action_name
    
    def get_action(self, gesture_name: str, mode: Optional[str] = None) -> Optional[str]:
        """
        제스처에 매핑된 액션 조회
        
        Args:
            gesture_name: 제스처 이름
            mode: 모드 이름 (None이면 모든 모드에서 검색)
        
        Returns:
            Optional[str]: 액션 이름 (없으면 None)
        """
        if gesture_name not in self._mappings:
            return None
        
        if mode is None:
            # 첫 번째 매핑 반환
            if self._mappings[gesture_name]:
                return next(iter(self._mappings[gesture_name].values()))
            return None
        
        return self._mappings[gesture_name].get(mode)
    
    def get_all_mappings(self) -> Dict[str, Dict[str, str]]:
        """
        모든 매핑 조회
        
        Returns:
            Dict[str, Dict[str, str]]: 모든 제스처-액션 매핑
        """
        return self._mappings.copy()
    
    def clear(self):
        """모든 매핑 초기화"""
        self._mappings.clear()


def maps_to_action(mode: str, action_name: str):
    """
    제스처 클래스에 액션 매핑을 추가하는 데코레이터
    
    사용 예:
        @maps_to_action("COMMON", "START")
        class StartGesture(StaticGesture):
            ...
    
    Args:
        mode: 모드 이름 ("COMMON", "PPT", "YOUTUBE")
        action_name: 액션 이름
    
    Returns:
        데코레이터 함수
    """
    def decorator(cls):
        """제스처 클래스를 데코레이트하여 액션 매핑 등록"""
        # 클래스의 name 속성을 사용하여 매핑 등록
        gesture_name = getattr(cls, 'name', None)
        if gesture_name:
            mapper = ActionMapper()
            mapper.register(gesture_name, mode, action_name)
        
        # 원본 클래스 반환 (메타데이터만 추가, 클래스 수정 없음)
        return cls
    
    return decorator
