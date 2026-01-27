"""
제스처 레지스트리
제스처 클래스의 자동 등록 및 조회 시스템
"""

from typing import Dict, List, Optional, Type
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from src.gesture.base.gesture_base import BaseGesture


class GestureRegistry:
    """제스처 레지스트리
    
    제스처 클래스를 자동으로 등록하고 조회할 수 있는 시스템입니다.
    """
    
    _instance: Optional['GestureRegistry'] = None
    _gestures: Dict[str, Type[BaseGesture]] = {}  # {gesture_name: gesture_class}
    _gestures_by_mode: Dict[str, List[str]] = {}  # {mode: [gesture_names]}
    _gesture_instances: Dict[str, BaseGesture] = {}  # {gesture_name: gesture_instance}
    
    def __new__(cls):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, gesture_class: Type[BaseGesture]):
        """
        제스처 클래스 등록
        
        Args:
            gesture_class: 제스처 클래스 (BaseGesture를 상속받은 클래스)
        """
        # 클래스 인스턴스 생성하여 메타데이터 확인
        try:
            instance = gesture_class()
            gesture_name = instance.get_name()
            mode = instance.get_mode()
            
            if not gesture_name:
                return  # 이름이 없으면 등록하지 않음
            
            # 제스처 등록
            self._gestures[gesture_name] = gesture_class
            self._gesture_instances[gesture_name] = instance
            
            # 모드별 인덱스 업데이트
            if mode not in self._gestures_by_mode:
                self._gestures_by_mode[mode] = []
            if gesture_name not in self._gestures_by_mode[mode]:
                self._gestures_by_mode[mode].append(gesture_name)
        
        except Exception:
            # 인스턴스 생성 실패 시 등록하지 않음
            pass
    
    def get_gesture(self, gesture_name: str) -> Optional[BaseGesture]:
        """
        제스처 인스턴스 조회
        
        Args:
            gesture_name: 제스처 이름
        
        Returns:
            Optional[BaseGesture]: 제스처 인스턴스 (없으면 None)
        """
        return self._gesture_instances.get(gesture_name)
    
    def get_gesture_class(self, gesture_name: str) -> Optional[Type[BaseGesture]]:
        """
        제스처 클래스 조회
        
        Args:
            gesture_name: 제스처 이름
        
        Returns:
            Optional[Type[BaseGesture]]: 제스처 클래스 (없으면 None)
        """
        return self._gestures.get(gesture_name)
    
    def get_gestures_by_mode(self, mode: str) -> List[str]:
        """
        모드별 제스처 이름 목록 조회
        
        Args:
            mode: 모드 이름 ("COMMON", "PPT", "YOUTUBE")
        
        Returns:
            List[str]: 제스처 이름 목록
        """
        return self._gestures_by_mode.get(mode, []).copy()
    
    def get_all_gestures(self) -> List[str]:
        """
        모든 제스처 이름 목록 조회
        
        Returns:
            List[str]: 모든 제스처 이름 목록
        """
        return list(self._gestures.keys())
    
    def clear(self):
        """모든 제스처 등록 초기화"""
        self._gestures.clear()
        self._gestures_by_mode.clear()
        self._gesture_instances.clear()
