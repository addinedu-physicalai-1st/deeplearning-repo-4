# 제스처 등록 및 인식 가이드

이 문서는 Gesto 프로젝트에서 새로운 제스처를 등록하고 인식시키는 방법을 설명합니다.

## 목차

- [정적 제스처 등록](#정적-제스처-static-gesture-등록)
- [동적 제스처 등록](#동적-제스처-dynamic-gesture-등록)
- [제스처 인식 흐름](#제스처-인식-흐름)
- [참고사항](#참고사항)

## 정적 제스처 (Static Gesture) 등록

정적 제스처는 단일 프레임의 랜드마크만으로 인식 가능한 제스처입니다 (예: 손 펴기, 주먹).

### 1. 제스처 클래스 생성

해당 모드 폴더에 새 파일을 생성하고 `StaticGesture`를 상속받는 클래스를 만듭니다:

```python
# src/mediapipe/gestures.py 에 클래스 추가
from typing import List, Dict
from src.mediapipe.gestures import StaticGesture
from src.mediapipe.registry import maps_to_action

@maps_to_action("COMMON", "MY_ACTION")  # 모드, 액션 이름
class MyStaticGesture(StaticGesture):
    name = "MY_GESTURE"      # 제스처 이름
    mode = "COMMON"          # 모드 ("COMMON", "PPT", "YOUTUBE")
    gesture_type = "static"  # 제스처 타입
    
    def __init__(self):
        super().__init__()
    
    def detect_static(self, landmarks: List[Dict]) -> str:
        """
        단일 프레임 랜드마크에서 제스처 인식
        
        Args:
            landmarks: 21개의 랜드마크 포인트 리스트
                      각 포인트는 {'x': float, 'y': float, 'z': float} 형태
        
        Returns:
            str: 제스처 이름 (인식된 경우) 또는 "NONE"
        """
        # 인식 로직 구현
        # landmarks[0] ~ landmarks[20]: 손목부터 새끼손가락 끝까지
        if 조건:
            return "MY_GESTURE"
        return "NONE"
```

### 2. 레지스트리에 등록

`src/mediapipe/gestures.py`에 새 제스처 클래스를 추가한 뒤, 파일 하단의 `_register_all()` 목록에 클래스를 추가합니다:

```python
# src/mediapipe/gestures.py 하단
def _register_all() -> None:
    reg = GestureRegistry()
    for cls in (
        ...
        MyStaticGesture,  # 추가
    ):
        reg.register(cls)
```

### 3. 액션 매핑

`@maps_to_action` 데코레이터로 제스처와 액션을 연결합니다. 액션 클래스는 `src/input_simulator/actions.py`에 추가합니다.

## 동적 제스처 (Dynamic Gesture) 등록

동적 제스처는 여러 프레임의 랜드마크 시퀀스를 기반으로 인식하는 제스처입니다 (예: 손 흔들기, 손 움직임).

### 1. 제스처 클래스 생성

해당 모드 폴더에 새 파일을 생성하고 `DynamicGesture`를 상속받는 클래스를 만듭니다:

```python
# src/mediapipe/gestures.py 에 클래스 추가
import numpy as np
from src.mediapipe.gestures import DynamicGesture
from src.mediapipe.registry import maps_to_action

@maps_to_action("COMMON", "MY_ACTION")  # 모드, 액션 이름
class MyDynamicGesture(DynamicGesture):
    name = "MY_GESTURE"      # 제스처 이름
    mode = "COMMON"          # 모드
    gesture_type = "dynamic" # 제스처 타입
    sequence_length = 30     # 시퀀스 길이 (프레임 수)
    
    def __init__(self):
        super().__init__(sequence_length=self.sequence_length)
    
    def detect_dynamic(self, landmark_sequence: np.ndarray) -> str:
        """
        랜드마크 시퀀스에서 제스처 인식
        
        Args:
            landmark_sequence: 랜드마크 시퀀스 배열
                              shape: (sequence_length, 63)
                              각 행은 21개 포인트의 (x, y, z) 좌표를 flatten한 형태
                              [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
        
        Returns:
            str: 제스처 이름 (인식된 경우) 또는 "NONE"
        """
        if landmark_sequence is None or landmark_sequence.shape[0] < 10:
            return "NONE"
        
        # 시퀀스 분석 로직 구현
        # 예: 손목(인덱스 0)의 x 좌표 추출
        wrist_x = landmark_sequence[:, 0]  # shape: (sequence_length,)
        
        # 인식 로직 (예: 좌우 움직임 감지)
        x_range = np.max(wrist_x) - np.min(wrist_x)
        if x_range > 임계값:
            return "MY_GESTURE"
        
        return "NONE"
```

### 2. 레지스트리에 등록

정적 제스처와 동일하게 `src/mediapipe/gestures.py`의 `_register_all()` 목록에 새 동적 제스처 클래스를 추가합니다.

## 제스처 인식 흐름

1. **Mediapipe**: 웹캠 프레임에서 손 랜드마크 추출
2. **GestureDetector**: 현재 모드의 제스처 목록 조회
3. **정적 제스처**: 단일 프레임 랜드마크로 즉시 인식
4. **동적 제스처**: 시퀀스 버퍼에 누적 후 인식 (sequence_length 프레임 필요)
5. **인식 결과**: 제스처 이름 반환 → 액션 매퍼를 통해 액션 실행

## 참고사항

- **정적 제스처**: 즉시 인식 가능, 빠른 반응
- **동적 제스처**: 시퀀스 버퍼가 채워질 때까지 대기 필요 (기본 30프레임 ≈ 1초)
- **인식 방법**: 현재는 규칙 기반(`rule`) 사용, LSTM은 추후 지원 예정
- **모드별 분리**: 각 모드(`COMMON`, `PPT`, `YOUTUBE`)는 독립적인 제스처 세트를 가짐

## 예시

실제 구현 예시는 `src/mediapipe/gestures.py` 내 `StartGesture`(정적), `StopGesture`(동적) 클래스를 참고하세요.
