"""
효과음 재생 워커 — QThread에서 MP3 재생. 재생 로직·워커·API 통합.
"""

import os
<<<<<<< HEAD
import time as _time
=======
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311
import queue
import shutil
import subprocess
import sys
from typing import Optional

from PyQt6.QtCore import QThread

import config

# PyQt6 QtMultimedia (선택)
_Player: Optional[type] = None
_AudioOutput: Optional[type] = None
_QUrl: Optional[type] = None

try:
    from PyQt6.QtCore import QUrl
    from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer

    _Player = QMediaPlayer
    _AudioOutput = QAudioOutput
    _QUrl = QUrl
except ImportError:
    _Player = _AudioOutput = _QUrl = None

_players: list = []
_worker: Optional["SoundPlaybackWorker"] = None
<<<<<<< HEAD
_last_play_time: float = 0.0
=======
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311


def start_playback_worker() -> None:
    """재생 워커 스레드 시작. main에서 앱 기동 후 한 번 호출."""
    global _worker
    if _worker is not None:
        return
    _worker = SoundPlaybackWorker()
    _worker.start()


def stop_playback_worker() -> None:
    """재생 워커 스레드 정지. main에서 aboutToQuit 시 호출."""
    global _worker
    if _worker is not None:
        _worker.stop()
        _worker.wait(2000)
        _worker = None


def play_trigger_start() -> None:
    """모션 감지 시작 효과음 재생."""
<<<<<<< HEAD
    _enqueue_play(config.ASSETS_DIR, "motion-trigger.wav", volume=0.7)
=======
    _enqueue_play(config.ASSETS_DIR, "motion-trigger-start.mp3")
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311


def play_trigger_stop() -> None:
    """모션 감지 종료 효과음 재생."""
<<<<<<< HEAD
    _enqueue_play(config.ASSETS_DIR, "motion-trigger.wav", volume=0.6)


def play_mode_sound(mode: str) -> None:
    """모드 전환 시 효과음 재생."""
    _enqueue_play(config.ASSETS_DIR, "mode-switch.wav", volume=0.8)


def play_aot_on() -> None:
    """항상 위에 고정 시 효과음."""
    _enqueue_play(config.ASSETS_DIR, "aot-toggle.wav", volume=0.7)


def play_aot_off() -> None:
    """항상 위에 고정 해제 시 효과음."""
    _enqueue_play(config.ASSETS_DIR, "aot-toggle.wav", volume=0.6)


def play_gesture_success() -> None:
    """제스처 인식 성공 효과음."""
    _enqueue_play(config.ASSETS_DIR, "gesture-success.wav", volume=0.6)


def play_ui_click() -> None:
    """UI 버튼 클릭 효과음."""
    _enqueue_play(config.ASSETS_DIR, "ui-click.wav", volume=1.0)


def play_app_startup() -> None:
    """앱 시작 시 오프닝 효과음."""
    _enqueue_play(config.ASSETS_DIR, "gesture-success.wav", volume=0.8)


def _enqueue_play(assets_dir: str, filename: str, volume: float = 1.0) -> None:
    global _last_play_time
    path = os.path.join(assets_dir, filename)
    if not os.path.isfile(path):
        return
        
    # 동일한 사운드가 너무 짧은 간격으로 재생되는 것을 방지 (0.2초 쿨다운)
    current_time = _time.time()
    if current_time - _last_play_time < 0.2:
        return
    _last_play_time = current_time

    if _worker is not None:
        _worker.enqueue(path, volume)
    else:
        _play_mp3(path, volume)


def _play_mp3(path: str, volume: float = 1.0) -> None:
    if not os.path.isfile(path):
        return
    if _Player is not None and _AudioOutput is not None and _QUrl is not None:
        # Linux에서 .wav 파일은 aplay가 더 안정적인 경우가 많음
        if sys.platform == "linux" and path.endswith(".wav") and shutil.which("aplay"):
            _play_subprocess(path, volume)
        else:
            _play_qt(path, volume)
    else:
        _play_subprocess(path, volume)




def _play_qt(path: str, volume: float = 1.0) -> None:
    try:
        player = _Player()
        audio_output = _AudioOutput()
        audio_output.setVolume(volume)
        player.setAudioOutput(audio_output)
        player.setSource(_QUrl.fromLocalFile(path))
        player.play()
        
        # Keep both alive
        _players.append((player, audio_output))

        def _cleanup():
            try:
                for p, a in _players[:]:
                    if p == player:
                        _players.remove((p, a))
                        break
=======
    _enqueue_play(config.ASSETS_DIR, "motion-trigger-stop.mp3")


def play_mode_sound(mode: str) -> None:
    """모드 전환 시 해당 모드 효과음 재생 (PPT / YOUTUBE / GAME)."""
    mode_upper = (mode or "").upper()
    filename = f"mode-{mode_upper.lower()}.mp3"
    _enqueue_play(config.ASSETS_DIR, filename)


def _enqueue_play(assets_dir: str, filename: str) -> None:
    path = os.path.join(assets_dir, filename)
    if not os.path.isfile(path):
        return
    if _worker is not None:
        _worker.enqueue(path)
    else:
        _play_mp3(path)


def _play_mp3(path: str) -> None:
    if not os.path.isfile(path):
        return
    if _Player is not None and _AudioOutput is not None and _QUrl is not None:
        _play_qt(path)
    else:
        _play_subprocess(path)


def _play_qt(path: str) -> None:
    try:
        player = _Player()
        audio_output = _AudioOutput()
        player.setAudioOutput(audio_output)
        player.setSource(_QUrl.fromLocalFile(path))
        player.play()
        _players.append(player)

        def _cleanup():
            try:
                _players.remove(player)
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311
            except ValueError:
                pass

        player.mediaStatusChanged.connect(
            lambda s: _cleanup() if s == _Player.MediaStatus.EndOfMedia else None
        )
    except Exception:
        _play_subprocess(path)


<<<<<<< HEAD
def _play_subprocess(path: str, volume: float = 1.0) -> None:
=======
def _play_subprocess(path: str) -> None:
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311
    """시스템 플레이어 (macOS afplay, Ubuntu ffplay 등)."""
    cmd: Optional[list[str]] = None
    if sys.platform == "darwin":
        if shutil.which("afplay"):
<<<<<<< HEAD
            # afplay volume is 0 to 255ish but 1.0 is standard
            cmd = ["afplay", "-v", str(volume), path]
    elif sys.platform == "linux":
        if shutil.which("ffplay"):
            # ffplay volume is 0 to 100
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-volume", str(int(volume * 100)), path]
        elif shutil.which("aplay") and path.endswith(".wav"):
            cmd = ["aplay", "-q", path] # aplay doesn't have easy volume flag
        elif shutil.which("mpg123"):
            cmd = ["mpg123", "-q", "-f", str(int(volume * 32768)), path]
=======
            cmd = ["afplay", path]
    elif sys.platform == "linux":
        if shutil.which("ffplay"):
            cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]
        elif shutil.which("mpg123"):
            cmd = ["mpg123", "-q", path]
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311
    if cmd:
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=(sys.platform == "darwin"),
            )
        except (FileNotFoundError, OSError):
            pass


class SoundPlaybackWorker(QThread):
    """효과음 재생 QThread. 경로 큐를 받아 워커 스레드에서 _play_subprocess 호출."""

    def __init__(self, parent=None):
        super().__init__(parent)
<<<<<<< HEAD
        self._path_queue: queue.Queue[Optional[tuple[str, float]]] = queue.Queue(maxsize=32)
        self._running = True

    def enqueue(self, path: str, volume: float = 1.0) -> None:
        try:
            self._path_queue.put_nowait((path, volume))
=======
        self._path_queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=32)
        self._running = True

    def enqueue(self, path: str) -> None:
        try:
            self._path_queue.put_nowait(path)
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311
        except queue.Full:
            pass

    def run(self) -> None:
        while self._running:
            try:
<<<<<<< HEAD
                item = self._path_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            path, volume = item
            _play_mp3(path, volume)
=======
                path = self._path_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if path is None:
                break
            _play_subprocess(path)
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311

    def stop(self) -> None:
        self._running = False
        try:
            self._path_queue.put_nowait(None)
        except queue.Full:
            pass
