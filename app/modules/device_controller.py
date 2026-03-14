from PyQt6.QtCore import QObject, pyqtSlot
from pynput.keyboard import Controller as KeyController, Key
import config

class DeviceController(QObject):
    """하드웨어 키보드 출력을 담당하며, 제스처와 키 매핑을 관리합니다.
    모드나 감지 상태를 저장하지 않고, 명령(시그널)을 받으면 즉시 수행합니다.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._keyboard = KeyController()
        # 제스처 매핑: (mode, gesture_name) -> list of key actions
        self._gesture_to_key = self._build_gesture_mapping()
        # GAME 모드: 현재 눌려 있는 키 관리
        self._last_game_keys = set()

    def _build_gesture_mapping(self) -> dict[tuple[str, str], list]:
        """config.GESTURE_ACTION_MAP에서 매핑을 로드. ','로 구분된 시퀀스 지원."""
        mapping = {}
        for mode, gestures in config.GESTURE_ACTION_MAP.items():
            for g_name, action_str in gestures.items():
                # 시퀀스(,) 분리
                steps = []
                for step_str in action_str.split(","):
                    steps.append(self._resolve_key(step_str))
                mapping[(mode, g_name)] = steps
        return mapping

    def _resolve_key(self, key_str: str) -> object:
        """문자열을 pynput Key(또는 문자) 또는 조합키 리스트로 변환."""
        key_str = key_str.strip()
        if not key_str:
            return None
            
        if "+" in key_str:
            # 조합키: 순서대로 press, 역순 release
            parts = [p.strip().lower() for p in key_str.split("+") if p.strip()]
            resolved = []
            for p in parts:
                if len(p) == 1:
                    resolved.append(p)
                else:
                    try:
                        resolved.append(getattr(Key, p))
                    except AttributeError:
                        resolved.append(p)
            return resolved
        if len(key_str) == 1:
            return key_str
        try:
            return getattr(Key, key_str.lower())
        except AttributeError:
            return key_str

    @pyqtSlot(str, str)
    def execute_gesture(self, gesture_name: str, mode: str) -> None:
        """모드와 제스처 이름에 따라 키보드 이벤트를 실행합니다."""
        raw = (gesture_name or "").strip()
        if not raw or raw.lower() == "unknown":
            return
        names = [s.strip() for s in raw.split("|") if s.strip()]
        
        # 1. 제스처에 따른 액션 수집
        all_actions = []
        for name in names:
            actions = self._gesture_to_key.get((mode, name))
            if actions:
                all_actions.extend(actions)

        if not all_actions:
            return

        # 2. GAME 모드 로직 (연속 입력)
        if mode == "GAME":
            keys_set = set(all_actions)
            if keys_set != self._last_game_keys:
                self.release_all()
                if keys_set:
                    try:
                        for k in keys_set:
                            if k: self._keyboard.press(k)
                        self._last_game_keys = keys_set
                    except Exception:
                        pass
            return

        # 3. PPT/YOUTUBE 로직 (순차적 tap 또는 조합키)
        try:
            for action in all_actions:
                if not action:
                    continue
                
                if isinstance(action, list):
                    # Combination (e.g. ctrl+f5)
                    for k in action:
                        self._keyboard.press(k)
                    for k in reversed(action):
                        self._keyboard.release(k)
                else:
                    # Single Key Tap
                    self._keyboard.press(action)
                    self._keyboard.release(action)
                
                # Small delay between sequence steps
                time.sleep(0.08) 
        except Exception:
            pass

    @pyqtSlot()
    def release_all(self) -> None:
        """현재 눌려 있는 모든 키를 해제합니다."""
        if not self._last_game_keys:
            return
            
        try:
            for k in self._last_game_keys:
                self._keyboard.release(k)
        except Exception:
            pass
        self._last_game_keys = set()
