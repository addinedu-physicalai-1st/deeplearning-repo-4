"""
MediaPipe Solutions API vs Task API — 실시간 Side-by-Side 비교 GUI
──────────────────────────────────────────────────────────────────
웹캠 하나로 두 모델을 동시에 실행하여 좌/우 화면으로 비교합니다.
각 모델의 랜드마크를 동일한 LSTM(lstm_legacy.tflite)에 넣어
인식된 제스처와 확률을 실시간 표시합니다.

좌측: mp.solutions.hands (Legacy Solution API)
우측: hand_landmarker.task (Task API)

조작:
  q : 종료
  s : 현재 화면 스크린샷 저장 (validation/results/)

사용법:
    python3 validation/compare_side_by_side.py
"""

import os
import sys
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# ─── 경로 설정 ───────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TASK_MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# config import
sys.path.insert(0, str(PROJECT_ROOT))
try:
    import config
    CAMERA_INDEX = config.CAMERA_INDEX
    MODELS_DIR = config.MODELS_DIR
except ImportError:
    CAMERA_INDEX = 0
    MODELS_DIR = str(PROJECT_ROOT / "app" / "models")

# ─── LSTM 관련 상수 ──────────────────────────────────
SEQUENCE_LENGTH = 30
LANDMARKS_COUNT = 42
NUM_CHANNELS = 11  # xyz(3) + left_feats(4) + right_feats(4)

# 랜드마크 인덱스
WRIST = 0
THUMB_TIP = 4
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_PIP = 14
RING_TIP = 16
PINKY_PIP = 18
PINKY_TIP = 20

# ─── 브랜드 테마 컬러 (BGR) ──────────────────────────
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BG_DARK = (10, 10, 26)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)

# ─── 랜드마크 연결 ────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]


def _load_gesture_labels() -> list[str]:
    path = os.path.join(MODELS_DIR, "lstm_legacy_labels.txt")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return [l.strip() for l in f if l.strip()]
    return ["Pinch_In_Left", "Pinch_In_Right", "Pinch_Out_Left", "Pinch_Out_Right",
            "Swipe_Left", "Swipe_Right"]


def _normalize_landmarks(data: np.ndarray) -> np.ndarray:
    """손목 기준 상대 좌표 정규화. (frames, 42, N) — xyz 채널만 정규화."""
    normalized = data.copy().astype(np.float32)
    for start in (0, 21):
        end = start + 21
        wrist = data[:, start:start+1, :3]
        part = data[:, start:end, :3] - wrist
        scale = np.max(np.abs(part), axis=(1, 2), keepdims=True) + 1e-6
        normalized[:, start:end, :3] = part / scale
    return normalized


def _get_tflite_interpreter():
    """TFLite 인터프리터 로드."""
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter
    except Exception:
        pass
    raise RuntimeError("tensorflow 또는 tflite-runtime이 필요합니다.")


def _euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)


def _is_fist(landmarks_21):
    """손이 주먹 상태인지 판별 (4개 손가락 모두 접혀야 함)."""
    wrist = landmarks_21[WRIST]
    fingers = [(INDEX_PIP, INDEX_TIP), (MIDDLE_PIP, MIDDLE_TIP),
               (RING_PIP, RING_TIP), (PINKY_PIP, PINKY_TIP)]
    curled = sum(1 for pip_i, tip_i in fingers
                 if _euclidean_dist(wrist, landmarks_21[tip_i]) <
                    _euclidean_dist(wrist, landmarks_21[pip_i]))
    return 1.0 if curled == 4 else 0.0


def _hand_features(landmarks_21, prev_21):
    """[Is_Fist, Pinch_Dist, Thumb_V, Index_Z_V] — 4개 값."""
    fist = _is_fist(landmarks_21)
    pinch = _euclidean_dist(landmarks_21[THUMB_TIP], landmarks_21[INDEX_TIP])
    thumb_v = 0.0
    index_z_v = 0.0
    if prev_21 is not None:
        thumb_v = landmarks_21[THUMB_TIP][1] - prev_21[THUMB_TIP][1]
        index_z_v = landmarks_21[INDEX_TIP][2] - prev_21[INDEX_TIP][2]
    return [fist, pinch, thumb_v, index_z_v]


class LightLSTM:
    """LSTM tflite 추론기. 11채널 (xyz + hand features) 버퍼."""

    def __init__(self):
        tflite_path = os.path.join(MODELS_DIR, "lstm_legacy.tflite")
        if not os.path.isfile(tflite_path):
            raise RuntimeError(f"LSTM 모델 없음: {tflite_path}")

        self.labels = _load_gesture_labels()
        InterpreterClass = _get_tflite_interpreter()
        self._interpreter = InterpreterClass(model_path=tflite_path)
        self._interpreter.allocate_tensors()
        self._input_index = self._interpreter.get_input_details()[0]["index"]
        self._output_index = self._interpreter.get_output_details()[0]["index"]

        self._buffer = np.zeros((SEQUENCE_LENGTH, LANDMARKS_COUNT * NUM_CHANNELS), dtype=np.float32)
        self._buffer_count = 0
        self._prev_right = None
        self._prev_left = None
        self._feature_array = np.zeros((LANDMARKS_COUNT, NUM_CHANNELS), dtype=np.float32)

    def _construct_11ch(self, landmarks_42x3: np.ndarray) -> np.ndarray:
        """(42, 3) → (42, 11) 피처 배열."""
        right = landmarks_42x3[0:21, :]
        left = landmarks_42x3[21:42, :]

        left_feats = _hand_features(left, self._prev_left)
        right_feats = _hand_features(right, self._prev_right)

        self._prev_right = right.copy()
        self._prev_left = left.copy()

        self._feature_array.fill(0)
        self._feature_array[:, 0:3] = landmarks_42x3
        self._feature_array[:, 3] = left_feats[0]
        self._feature_array[:, 4] = left_feats[1]
        self._feature_array[:, 5] = left_feats[2]
        self._feature_array[:, 6] = left_feats[3]
        self._feature_array[:, 7] = right_feats[0]
        self._feature_array[:, 8] = right_feats[1]
        self._feature_array[:, 9] = right_feats[2]
        self._feature_array[:, 10] = right_feats[3]
        return self._feature_array

    def feed(self, landmarks_42x3: np.ndarray):
        """(42, 3) 랜드마크 한 프레임 추가. 버퍼가 차면 추론."""
        data_11ch = self._construct_11ch(landmarks_42x3)
        data_norm = _normalize_landmarks(data_11ch[np.newaxis, ...])[0]
        row = data_norm.reshape(-1).astype(np.float32)  # (462,)

        if self._buffer_count < SEQUENCE_LENGTH:
            self._buffer[self._buffer_count] = row
            self._buffer_count += 1
        else:
            self._buffer[:-1] = self._buffer[1:]
            self._buffer[-1] = row

        if self._buffer_count < SEQUENCE_LENGTH:
            return None, {}, 0.0

        input_data = self._buffer[np.newaxis, ...]  # (1, 30, 462)
        self._interpreter.set_tensor(self._input_index, input_data)
        self._interpreter.invoke()
        probs = self._interpreter.get_tensor(self._output_index)[0]

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        all_probs = {self.labels[i]: float(probs[i]) for i in range(len(probs))}
        gesture = self.labels[idx] if conf > 0.5 else None
        return gesture, all_probs, conf

    def reset(self):
        self._buffer.fill(0)
        self._buffer_count = 0
        self._prev_right = None
        self._prev_left = None


# ════════════════════════════════════════════════════
#  랜드마크 추출 헬퍼
# ════════════════════════════════════════════════════
def _extract_42_from_solutions(results) -> np.ndarray | None:
    """mp.solutions.hands 결과 → (42, 3) ndarray. handedness 기반 슬롯 배치."""
    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None
    zero = np.zeros((21, 3), dtype=np.float32)
    right_slot, left_slot = zero.copy(), zero.copy()
    for hlm, hnd in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = hnd.classification[0].label if hnd.classification else ""
        arr = np.array([[lm.x, lm.y, lm.z] for lm in hlm.landmark], dtype=np.float32)
        if label == "Right":
            right_slot = arr
        else:
            left_slot = arr
    return np.vstack([right_slot, left_slot])


def _extract_42_from_task(result) -> np.ndarray | None:
    """Task API 결과 → (42, 3) ndarray. handedness 기반 슬롯 배치."""
    if not result.hand_landmarks or not result.handedness:
        return None
    zero = np.zeros((21, 3), dtype=np.float32)
    right_slot, left_slot = zero.copy(), zero.copy()
    for hlm, hnd in zip(result.hand_landmarks, result.handedness):
        label = hnd[0].category_name if hnd else ""
        arr = np.array([[lm.x, lm.y, lm.z] for lm in hlm], dtype=np.float32)
        if label == "Right":
            right_slot = arr
        else:
            left_slot = arr
    return np.vstack([right_slot, left_slot])


# ════════════════════════════════════════════════════
#  시각화 헬퍼
# ════════════════════════════════════════════════════
def draw_landmarks(frame, landmarks_list, accent_color):
    """랜드마크를 프레임 위에 그리기."""
    h, w = frame.shape[:2]
    for landmarks in landmarks_list:
        pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]
        for i, j in HAND_CONNECTIONS:
            cv2.line(frame, pts[i], pts[j], accent_color, 3, cv2.LINE_AA)
            cv2.line(frame, pts[i], pts[j], WHITE, 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 5, accent_color, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 2, WHITE, -1, cv2.LINE_AA)


def draw_info_panel(frame, model_name: str, fps: float, detected: bool,
                    accent_color, panel_y=0):
    """상단 정보 패널."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, panel_y), (w, panel_y + 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, model_name, (12, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, accent_color, 2, cv2.LINE_AA)
    fps_color = GREEN if fps >= 25 else YELLOW if fps >= 15 else RED
    cv2.putText(frame, f"FPS: {fps:.1f}", (12, panel_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2, cv2.LINE_AA)

    status = "Detected" if detected else "No Hand"
    status_color = GREEN if detected else RED
    cv2.putText(frame, status, (w - 130, panel_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2, cv2.LINE_AA)
    cv2.line(frame, (0, panel_y + 69), (w, panel_y + 69), accent_color, 2)


def draw_gesture_result(frame, gesture: str | None, confidence: float,
                        accent_color, y_offset: int = 80):
    """인식된 제스처명 + 확률을 프레임에 표시."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_offset), (w, y_offset + 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if gesture:
        text = f"{gesture} ({confidence*100:.0f}%)"
        cv2.putText(frame, text, (12, y_offset + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, accent_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Waiting...", (12, y_offset + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1, cv2.LINE_AA)


def draw_prob_bars(frame, probs: dict, accent_color, y_start: int = 135,
                   bar_height: int = 14, max_width: int = 200):
    """각 클래스별 확률 바를 하단에 표시."""
    h, w = frame.shape[:2]
    if not probs:
        return

    overlay = frame.copy()
    total_bar_area = len(probs) * (bar_height + 4) + 8
    cv2.rectangle(overlay, (0, y_start), (max_width + 120, y_start + total_bar_area),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    y = y_start + 4
    for label, prob in sorted_probs:
        # 단축 라벨
        short = label.replace("Pinch_In_", "PIn_").replace("Pinch_Out_", "POut_") \
                      .replace("Swipe_", "Sw_").replace("Volume_", "Vol_") \
                      .replace("Play_Pause_", "PP_")
        bar_w = int(prob * max_width)
        bar_color = accent_color if prob > 0.5 else GRAY
        cv2.rectangle(frame, (100, y), (100 + bar_w, y + bar_height), bar_color, -1)
        cv2.rectangle(frame, (100, y), (100 + max_width, y + bar_height), GRAY, 1)
        cv2.putText(frame, short, (4, y + bar_height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{prob*100:.0f}%", (100 + max_width + 4, y + bar_height - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, WHITE, 1, cv2.LINE_AA)
        y += bar_height + 4


def draw_match_indicator(combined, sol_gesture, task_gesture):
    """중앙에 일치/불일치 아이콘 표시."""
    h, w = combined.shape[:2]
    mid_x = w // 2
    if sol_gesture and task_gesture:
        match = sol_gesture == task_gesture
        icon = "MATCH" if match else "DIFF"
        color = GREEN if match else RED
        bg_color = (0, 80, 0) if match else (0, 0, 80)
        # 배경 원
        cv2.circle(combined, (mid_x, 100), 30, bg_color, -1, cv2.LINE_AA)
        cv2.circle(combined, (mid_x, 100), 30, color, 2, cv2.LINE_AA)
        cv2.putText(combined, icon, (mid_x - 28, 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)


def draw_center_divider(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    for y in range(0, h, 12):
        cv2.line(frame, (mid, y), (mid, y + 6), GRAY, 2)


def draw_watermark(frame):
    h, w = frame.shape[:2]
    text = "q: Quit | s: Screenshot"
    cv2.putText(frame, text, (w // 2 - 120, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)


class FPSCounter:
    def __init__(self, window=30):
        self._times = deque(maxlen=window)

    def tick(self, elapsed: float):
        self._times.append(elapsed)

    @property
    def fps(self) -> float:
        if not self._times:
            return 0.0
        return 1.0 / (sum(self._times) / len(self._times))


# ════════════════════════════════════════════════════
#  메인
# ════════════════════════════════════════════════════
def main():
    if not TASK_MODEL_PATH.exists():
        print(f"[ERROR] 모델 파일이 없습니다: {TASK_MODEL_PATH}")
        sys.exit(1)

    # ── 1. Solutions API 초기화 ──
    mp_hands = mp.solutions.hands
    sol_hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── 2. Task API 초기화 ──
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    base_options = mp_tasks.BaseOptions(model_asset_path=str(TASK_MODEL_PATH))
    task_options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    task_detector = vision.HandLandmarker.create_from_options(task_options)

    # ── 3. LSTM 제스처 인식기 (각 API마다 별도 버퍼) ──
    sol_lstm = LightLSTM()
    task_lstm = LightLSTM()
    print(f"[INFO] LSTM 라벨: {sol_lstm.labels}")

    # ── 4. 카메라 열기 ──
    if sys.platform == "linux":
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] 카메라(index={CAMERA_INDEX})를 열 수 없습니다.")
        print("  main.py를 먼저 종료한 후 다시 시도하세요.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    sol_fps_counter = FPSCounter()
    task_fps_counter = FPSCounter()
    screenshot_count = 0

    sol_gesture, sol_conf, sol_probs = None, 0.0, {}
    task_gesture, task_conf, task_probs = None, 0.0, {}

    print("[INFO] Side-by-Side 비교 시작! (제스처 인식 포함)")
    print("       q: 종료 | s: 스크린샷 저장")

    cv2.namedWindow("MediaPipe: Solutions API vs Task API", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MediaPipe: Solutions API vs Task API", 1280, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Solutions API ──
        sol_frame = frame.copy()
        t0 = time.perf_counter()
        sol_results = sol_hands.process(rgb)
        t1 = time.perf_counter()
        sol_fps_counter.tick(t1 - t0)

        sol_detected = False
        if sol_results.multi_hand_landmarks:
            sol_detected = True
            sol_lm_list = [
                [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                for hand in sol_results.multi_hand_landmarks
            ]
            draw_landmarks(sol_frame, sol_lm_list, CYAN)

        # LSTM 제스처 인식 (Solutions)
        lm42 = _extract_42_from_solutions(sol_results)
        if lm42 is not None:
            sol_gesture, sol_probs, sol_conf = sol_lstm.feed(lm42)
        else:
            sol_lstm.reset()
            sol_gesture, sol_probs, sol_conf = None, {}, 0.0

        draw_info_panel(sol_frame, "Solutions API (mp.solutions)", sol_fps_counter.fps, sol_detected, CYAN)
        draw_gesture_result(sol_frame, sol_gesture, sol_conf, CYAN)
        draw_prob_bars(sol_frame, sol_probs, CYAN)

        # ── Task API ──
        task_frame = frame.copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        t0 = time.perf_counter()
        task_results = task_detector.detect(mp_image)
        t1 = time.perf_counter()
        task_fps_counter.tick(t1 - t0)

        task_detected = False
        if task_results.hand_landmarks:
            task_detected = True
            task_lm_list = [
                [(lm.x, lm.y, lm.z) for lm in hand]
                for hand in task_results.hand_landmarks
            ]
            draw_landmarks(task_frame, task_lm_list, MAGENTA)

        # LSTM 제스처 인식 (Task)
        lm42_task = _extract_42_from_task(task_results)
        if lm42_task is not None:
            task_gesture, task_probs, task_conf = task_lstm.feed(lm42_task)
        else:
            task_lstm.reset()
            task_gesture, task_probs, task_conf = None, {}, 0.0

        draw_info_panel(task_frame, "Task API (hand_landmarker.task)", task_fps_counter.fps, task_detected, MAGENTA)
        draw_gesture_result(task_frame, task_gesture, task_conf, MAGENTA)
        draw_prob_bars(task_frame, task_probs, MAGENTA)

        # ── 합성 ──
        combined = np.hstack([sol_frame, task_frame])
        draw_center_divider(combined)
        draw_match_indicator(combined, sol_gesture, task_gesture)
        draw_watermark(combined)

        cv2.imshow("MediaPipe: Solutions API vs Task API", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            screenshot_count += 1
            ts = int(time.time())
            path = RESULTS_DIR / f"side_by_side_{ts}_{screenshot_count}.png"
            cv2.imwrite(str(path), combined)
            print(f"[SCREENSHOT] 저장됨: {path}")

    cap.release()
    sol_hands.close()
    task_detector.close()
    cv2.destroyAllWindows()
    print("[INFO] 종료.")


if __name__ == "__main__":
    main()
