"""모드별 감지기 등록 — get_mode_detector."""

<<<<<<< HEAD
from typing import Callable, Optional

=======
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311
from app.recognition.ppt import PPTDetector
from app.recognition.youtube import YouTubeDetector
from app.recognition.game import GameDetector


<<<<<<< HEAD
def get_mode_detector(
    mode: str,
    get_confidence_threshold: Optional[Callable[[], float]] = None,
):
    """모드 문자열에 해당하는 감지기 인스턴스 반환.
    get_confidence_threshold: LSTM 계열(PPT 등)에서 감도 실시간 반영용. 호출 시 0.3~0.9 반환.
    """
    mode_upper = (mode or "").upper()
    if mode_upper == "PPT":
        return PPTDetector(get_confidence_threshold=get_confidence_threshold)
    if mode_upper == "YOUTUBE":
        return YouTubeDetector(get_confidence_threshold=get_confidence_threshold)
    if mode_upper == "GAME":
        return GameDetector()
    return PPTDetector(get_confidence_threshold=get_confidence_threshold)
=======
def get_mode_detector(mode: str):
    """모드 문자열에 해당하는 감지기 인스턴스 반환."""
    mode_upper = (mode or "").upper()
    if mode_upper == "PPT":
        return PPTDetector()
    if mode_upper == "YOUTUBE":
        return YouTubeDetector()
    if mode_upper == "GAME":
        return GameDetector()
    return PPTDetector()  # 기본
>>>>>>> d1bd67f5dcb6706aacd57c6cdd4a254dd5041311
