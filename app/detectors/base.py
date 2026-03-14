from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any

class Detector(ABC):
    """
    Abstract base class for all mode-specific detectors.
    Ensures a consistent interface for the DetectionWorker.
    """

    @abstractmethod
    def process_landmarks(self, multi_hand_landmarks: Any, multi_handedness: Any) -> tuple[Optional[str], float]:
        """
        Process hand landmarks and return the detected gesture name and confidence.
        
        Args:
            multi_hand_landmarks: MediaPipe multi_hand_landmarks.
            multi_handedness: MediaPipe multi_handedness.
            
        Returns:
            Tuple of (gesture_name, confidence). gesture_name is None if no gesture is detected.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release resources if necessary."""
        pass

    @property
    @abstractmethod
    def cooldown_until(self) -> float:
        """Return the timestamp (time.monotonic()) until which the detector is in cooldown."""
        pass

    @property
    @abstractmethod
    def last_probs(self) -> Dict[str, float]:
        """Return a dictionary of gesture class probabilities for the last processed frame."""
        pass

    @property
    @abstractmethod
    def last_11ch_means(self) -> Optional[List[float]]:
        """GESTURE_DEBUG: Return the means of the 11-channel features for the last frame."""
        pass

    @property
    @abstractmethod
    def last_fist_debug(self) -> Any:
        """GESTURE_DEBUG: Return debug information about fist detection."""
        pass
