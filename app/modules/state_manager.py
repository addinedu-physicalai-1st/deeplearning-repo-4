"""
Stage Manager — 현재 모드·감지 on/off 단일 소스, 제스처 → pynput 출력.
"""

from typing import Literal

from PyQt6.QtCore import QObject, pyqtSignal

ModeName = Literal["PPT", "YOUTUBE", "GAME"]


class StateManager(QObject):
    """현재 모드(PPT/YOUTUBE/GAME) 및 모션 감지 시작/정지 단일 소스."""

    VALID_MODES: tuple[ModeName, ...] = ("PPT", "YOUTUBE", "GAME")
    detection_state_changed = pyqtSignal(bool)  # 감지 시작/정지 시 UI용
    mode_changed = pyqtSignal(str)  # 모드 변경 시 알림

    def __init__(self, initial_mode: ModeName = "GAME", parent=None):
        super().__init__(parent)
        self._mode: ModeName = initial_mode if initial_mode in self.VALID_MODES else "PPT"
        self._is_detecting = False

    def set_mode(self, mode: ModeName) -> None:
        if mode in self.VALID_MODES:
            if self._mode == mode:
                return
            self._mode = mode
            self.mode_changed.emit(mode)

    def get_mode(self) -> ModeName:
        return self._mode

    def set_detection_state(self, is_active: bool) -> None:
        """공통 트리거(시작/정지)에서 호출. 감지 on/off 상태 갱신 후 UI에 시그널."""
        if self._is_detecting == is_active:
            return
        self._is_detecting = is_active
        self.detection_state_changed.emit(is_active)

    def get_detection_state(self) -> bool:
        return self._is_detecting
