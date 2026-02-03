
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- 경로 설정 ---
# 기존 legacy 데이터 폴더 사용
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_collector', 'data', 'legacy'))
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
SAVE_PATH = os.path.join(MODELS_DIR, 'simple_lstm.h5')

# --- 하이퍼파라미터 ---
SEQUENCE_LENGTH = 45
LANDMARKS_COUNT = 21
COORDS_COUNT = 3
INPUT_SHAPE = (SEQUENCE_LENGTH, LANDMARKS_COUNT * COORDS_COUNT)
EPOCHS = 50
BATCH_SIZE = 16

def normalize_landmarks(data):
    """
    랜드마크 정규화: (Frames, 21, 3) -> 손목 기준 상대 좌표
    """
    wrist = data[:, 0:1, :]
    normalized = data - wrist
    # 간단한 스케일링
    scale = np.max(np.abs(normalized), axis=(1, 2), keepdims=True) + 1e-6
    normalized = normalized / scale
    return normalized

def load_data(data_dir):
    X = []
    y = []
    label_map = {}
    current_label_id = 0
    count_per_class = {}

    if not os.path.exists(data_dir):
        print(f"오류: 데이터 폴더를 찾을 수 없습니다: {data_dir}")
        return np.array(X), np.array(y), label_map

    # Gesture 및 Posture 폴더 탐색
    modes = ['Gesture', 'Posture']
    
    for mode in modes:
        mode_path = os.path.join(data_dir, mode)
        if not os.path.exists(mode_path):
            continue
            
        gestures = sorted(os.listdir(mode_path))
        for gesture in gestures:
            gesture_path = os.path.join(mode_path, gesture)
            if not os.path.isdir(gesture_path):
                continue
                
            if gesture not in label_map:
                label_map[gesture] = current_label_id
                count_per_class[gesture] = 0
                current_label_id += 1
            
            label_id = label_map[gesture]
            
            # .npy 파일 로드
            for file in os.listdir(gesture_path):
                if file.endswith('.npy'):
                    file_path = os.path.join(gesture_path, file)
                    try:
                        data = np.load(file_path)
                        
                        # 전처리
                        data = normalize_landmarks(data)
                        
                        # 시퀀스 길이 맞추기
                        if data.shape[0] > SEQUENCE_LENGTH:
                            data = data[:SEQUENCE_LENGTH]
                        elif data.shape[0] < SEQUENCE_LENGTH:
                            padding = np.zeros((SEQUENCE_LENGTH - data.shape[0], 21, 3))
                            data = np.vstack((data, padding))
                        
                        # 평탄화 (45, 21, 3) -> (45, 63)
                        data_flat = data.reshape(SEQUENCE_LENGTH, -1)
                        
                        X.append(data_flat)
                        y.append(label_id)
                        count_per_class[gesture] += 1
                        
                    except Exception as e:
                        print(f"Error loading {file}: {e}")

    # 데이터 개수 출력
    print("\n" + "="*40)
    print("📊 학습 데이터 통계")
    print("="*40)
    total_count = 0
    for gesture, count in count_per_class.items():
        print(f" - {gesture}: {count} 개")
        total_count += count
    print("-" * 40)
    print(f" 총 데이터 개수: {total_count} 개")
    print("="*40 + "\n")

    return np.array(X), np.array(y), label_map

def create_simple_model(num_classes):
    """
    필수 기능만 포함한 심플 LSTM 모델
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=INPUT_SHAPE),
        LSTM(32, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("데이터 로딩 중...")
    X, y, label_map = load_data(DATA_DIR)
    
    if len(X) == 0:
        print("데이터가 없습니다. collect_mp_legacy.py를 사용하여 데이터를 먼저 수집하세요.")
        return

    # One-hot encoding
    num_classes = len(label_map)
    y_encoded = to_categorical(y, num_classes=num_classes)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    print(f"학습 데이터: {X_train.shape}, 검증 데이터: {X_test.shape}")
    
    # Model
    model = create_simple_model(num_classes)
    model.summary()
    
    print("\n학습 시작...")
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))
    
    # Save
    model.save(SAVE_PATH)
    print(f"\n✅ 모델 저장 완료: {SAVE_PATH}")

if __name__ == "__main__":
    main()
