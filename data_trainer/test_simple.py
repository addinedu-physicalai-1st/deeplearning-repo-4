
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import time
from PIL import ImageFont, ImageDraw, Image
import sys

# --- 설정 ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'simple_lstm.h5')
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_collector', 'data', 'legacy'))
SEQUENCE_LENGTH = 45
LANDMARKS_COUNT = 21

# --- 폰트 설정 ---
FONT_PATH = ""
if sys.platform == "linux":
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            FONT_PATH = path
            break
elif sys.platform == "darwin":
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Apple SD Gothic Neo.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            FONT_PATH = path
            break

# --- 헬퍼 클래스/함수 ---
class MovingAverage:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = []
    
    def update(self, new_data):
        self.buffer.append(new_data)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        return np.mean(self.buffer, axis=0)

def normalize_landmarks(data):
    wrist = data[:, 0:1, :]
    normalized = data - wrist
    scale = np.max(np.abs(normalized), axis=(1, 2), keepdims=True) + 1e-6
    normalized = normalized / scale
    return normalized

def extract_features(landmark_list):
    data = []
    for lm in landmark_list.landmark:
        data.append([lm.x, lm.y, lm.z])
    return np.array(data)

def is_fist(landmarks):
    lm = landmarks.landmark
    fingers = [(8, 5), (12, 9), (16, 13), (20, 17)]
    folded_count = 0
    start_point = lm[0]
    for tip_idx, mcp_idx in fingers:
        tip = lm[tip_idx]
        pip_idx = mcp_idx + 1
        pip = lm[pip_idx]
        dist_tip = np.sqrt((tip.x - start_point.x)**2 + (tip.y - start_point.y)**2)
        dist_pip = np.sqrt((pip.x - start_point.x)**2 + (pip.y - start_point.y)**2)
        if dist_tip < dist_pip:
             folded_count += 1
    return folded_count >= 3

def put_text_korean(img, text, position, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        if FONT_PATH:
            font = ImageFont.truetype(FONT_PATH, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

def load_labels(data_dir):
    label_map = {}
    current_id = 0
    modes = ['Gesture', 'Posture']
    for mode in modes:
        mode_path = os.path.join(data_dir, mode)
        if not os.path.exists(mode_path): continue
        gestures = sorted(os.listdir(mode_path))
        for g in gestures:
            if os.path.isdir(os.path.join(mode_path, g)):
                if g not in label_map:
                    label_map[g] = current_id
                    current_id += 1
    return {v: k for k, v in label_map.items()}

# --- 카메라 설정 ---
def nothing(x):
    pass

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"모델 파일 없음: {MODEL_PATH}")
        return
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = load_labels(DATA_DIR)
    
    cap = cv2.VideoCapture(0)
    
    WINDOW_NAME = 'Simple Hand Gesture Test'
    cv2.namedWindow(WINDOW_NAME)
    
    # 트랙바 생성 (초기값은 카메라마다 다르므로 128 중간값 설정)
    cv2.createTrackbar('Brightness', WINDOW_NAME, 128, 255, nothing)
    cv2.createTrackbar('Contrast', WINDOW_NAME, 128, 255, nothing)
    cv2.createTrackbar('Saturation', WINDOW_NAME, 128, 255, nothing)
    cv2.createTrackbar('Gain', WINDOW_NAME, 0, 255, nothing)
    # 노출: 보통 -10 ~ 0, 혹은 0 ~ 10000 등 다양함. 
    # 여기선 0~100으로 하고 -10~0으로 매핑하거나, 수동 모드면 그냥 값 전달
    # v4l2loopback 등에서는 다를 수 있음.
    # 일단 0~200 범위 정중앙 100을 0(auto)으로 가정하거나, -10 ~ +10 매핑
    cv2.createTrackbar('Exposure', WINDOW_NAME, 50, 100, nothing) 
    
    # 초기값 카메라에 적용 (Auto 모드 해제 시도)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 0.25 or 0.75 often matches 'Manual' on Linux v4l2
        # 또는 1 (Manual), 3 (Auto)
    except:
        pass
        
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    
    history_buffers = {'Right': [], 'Left': []}
    smoothers = {'Right': MovingAverage(3), 'Left': MovingAverage(3)}
    
    prev_settings = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- 카메라 설정 업데이트 ---
        # 매번 set하면 느려질 수 있으므로 값 변경 확인
        bri = cv2.getTrackbarPos('Brightness', WINDOW_NAME)
        con = cv2.getTrackbarPos('Contrast', WINDOW_NAME)
        sat = cv2.getTrackbarPos('Saturation', WINDOW_NAME)
        gain = cv2.getTrackbarPos('Gain', WINDOW_NAME)
        exp = cv2.getTrackbarPos('Exposure', WINDOW_NAME)
        
        current_settings = {'bri': bri, 'con': con, 'sat': sat, 'gain': gain, 'exp': exp}
        
        if current_settings != prev_settings:
            # 밝기 등 조절
            cap.set(cv2.CAP_PROP_BRIGHTNESS, bri)
            cap.set(cv2.CAP_PROP_CONTRAST, con)
            cap.set(cv2.CAP_PROP_SATURATION, sat)
            cap.set(cv2.CAP_PROP_GAIN, gain)
            
            # 노출 매핑: 슬라이더 0~100 -> 실제 -10 ~ +10 or similar
            # Linux v4l2: exposure_absolute usually
            # 단순히 슬라이더 값 그대로 전달 시도하거나, 매핑
            # 여기선 슬라이더 50을 기준으로 -5 ~ +5 매핑해봄
            exp_mapped = (exp - 50) 
            # 그러나 많은 웹캠은 -10 ~ -1 등의 음수 로그 스케일 사용
            # 혹은 0 ~ 10000.
            # 사용자 요청: "드래그해도 값이 안바뀜" -> cap.set을 안해서임. 이제 함.
            # 노출은 카메라마다 특성이 많이 타서, 일단 값 전달해보고 안되면 Auto 모드 문제일 수 있음.
            cap.set(cv2.CAP_PROP_EXPOSURE, exp_mapped) 
            
            prev_settings = current_settings

        # ------------------------
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        h, w, c = frame.shape
        status_text = "상태: 양손 대기 중"
        trigger_active = False 
        active_hand_label = None 
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hand_statuses = {} 
            hand_landmarks_dict = {}
            for idx, hand_class in enumerate(results.multi_handedness):
                label = hand_class.classification[0].label 
                lms = results.multi_hand_landmarks[idx]
                
                raw_coords = extract_features(lms)
                smoothed_coords = smoothers[label].update(raw_coords)
                hand_landmarks_dict[label] = { 'raw': lms, 'smoothed': smoothed_coords }
                
                is_fisted = is_fist(lms)
                hand_statuses[label] = is_fisted
                
                mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)
                txt = "주먹" if is_fisted else "펴짐"
                color = (255, 0, 0) if is_fisted else (0, 255, 0)
                cx, cy = int(lms.landmark[0].x * w), int(lms.landmark[0].y * h)
                cv2.putText(frame, label, (cx, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            right_fist = hand_statuses.get('Right', False)
            left_fist = hand_statuses.get('Left', False)
            
            if right_fist and left_fist:
                status_text = "⚠️ 안전 모드 (양손 주먹)"
                cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 255), -1) 
            elif right_fist:
                status_text = "▶ 트리거: 오른손 (왼손 인식 중...)"
                trigger_active = True
                active_hand_label = 'Left'
            elif left_fist:
                status_text = "◀ 트리거: 왼손 (오른손 인식 중...)"
                trigger_active = True
                active_hand_label = 'Right'
            else:
                status_text = "✋ 대기 중 (한 손만 주먹을 쥐세요)"
            
            result_display = ""
            conf_display = 0.0
            
            if trigger_active and active_hand_label in hand_landmarks_dict:
                lms = hand_landmarks_dict[active_hand_label]['raw']
                x_min = int(min([lm.x for lm in lms.landmark]) * w)
                y_min = int(min([lm.y for lm in lms.landmark]) * h)
                x_max = int(max([lm.x for lm in lms.landmark]) * w)
                y_max = int(max([lm.y for lm in lms.landmark]) * h)
                cv2.rectangle(frame, (x_min-10, y_min-10), (x_max+10, y_max+10), (0, 255, 0), 2)
                
                target_data = hand_landmarks_dict[active_hand_label]['smoothed']
                history_buffers[active_hand_label].append(target_data)
                if len(history_buffers[active_hand_label]) > SEQUENCE_LENGTH:
                    history_buffers[active_hand_label].pop(0)
                
                if len(history_buffers[active_hand_label]) == SEQUENCE_LENGTH:
                    input_seq = np.array(history_buffers[active_hand_label])
                    norm_seq = normalize_landmarks(input_seq)
                    flat_seq = norm_seq.reshape(SEQUENCE_LENGTH, -1)
                    final_input = np.expand_dims(flat_seq, axis=0)
                    prediction = model.predict(final_input, verbose=0)
                    pred_idx = np.argmax(prediction)
                    confidence = prediction[0][pred_idx]
                    gesture_name = labels.get(pred_idx, "알 수 없음")
                    if confidence > 0.8:
                        result_display = gesture_name
                        conf_display = confidence
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = put_text_korean(frame, status_text, (20, 10), 20, (255, 255, 255))
            if result_display:
                msg = f"인식됨: {result_display} ({conf_display*100:.1f}%)"
                frame = put_text_korean(frame, msg, (50, h - 60), 40, (0, 255, 0)) 
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = put_text_korean(frame, "손을 카메라에 비춰주세요", (20, 20), 20, (200, 200, 200))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q'): # ESC or q to quit
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
