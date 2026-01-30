# Gesto 구현 계획 (System Architecture 기반)

## 1. 문서 개요

- **제목**: Gesto 구현 계획 (System Architecture 기반)
- **목적**: 시스템 아키텍처에 맞춰 Mode Controller 중심의 실시간 인식 및 Pynput 제어를 구현한다.
- **참조**:
  - [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) — 데이터 흐름, 레이어 역할, 폴더 매핑
  - [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) — 시스템 요구사항
  - 폴더 구조: [src/app](src/app), [src/capture](src/capture), [src/mode_controller](src/mode_controller), [src/mediapipe](src/mediapipe), [src/input_simulator](src/input_simulator)

## 2. 시스템 아키텍처

- **상세 내용은 [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) 참조.** (다이어그램, 레이어 역할, 폴더 매핑이 해당 문서에 정의됨.)
- **요약**: Mode Controller가 현재 모드에 따라 해당 모드 인식기를 선택하고, 인식 결과를 Pynput에 명령으로 전달한다. 이미지는 opencv에서 **바로 Mediapipe**로 전달하며, YOLO는 범위에서 제외한다.

## 3. Mode Controller 역할 (핵심)

- **입력**: UI에서의 모드 변경(`set_mode`)만. (YOLO/인물 영역은 현재 범위에서 제외.)
- **책임**:
  1. **현재 모드** 단일 소스로 유지.
  2. 실시간 루프에서 **현재 모드에 맞는 인식기**만 사용 (Game → Posture, PPT → Gesture+LSTM, YouTube → Gesture+LSTM).
  3. 인식 결과(제스처/자세 ID 또는 액션 이름)를 **명령**으로 변환하여 **Pynput(input_simulator)**에 전달.
- **구현 방향**: [src/mode_controller/mode_controller.py](src/mode_controller/mode_controller.py)가 단순 상태 저장을 넘어, (1) 파이프라인/앱에서 **현재 모드용 인식기**를 조회하는 진입점을 제공하거나, (2) 인식 결과 → 명령 → Pynput 호출을 **Mode Controller가 오케스트레이션**하도록 단계적으로 확장한다.

## 4. 구현 단계 (Phase) — 체크리스트

### Phase 1: Capture 및 PyQT UI

- [ ] 웹캠(opencv) 확보 — [src/capture/camera.py](src/capture/camera.py)
- [ ] UI에서 모드 선택·시작/종료·감도 표시 — [src/app/main_window.py](src/app/main_window.py), [src/app/gesture_display.py](src/app/gesture_display.py)
- [ ] UI 이벤트가 Mode Controller에 모드 전달 (`set_mode(mode)` 수신)
- [ ] S-01: 프로그램 실행/종료
- [ ] S-04: 자신 모습 표시 (웹캠 영상)
- [ ] S-05: 선택 모드 표시
- [ ] S-02/S-02-TRIG-01/S-02-TRIG-02: 트리거 시작/종료 모션
- [ ] S-03: 모션인식 시작·종료 확인

### Phase 2: Mode Controller 및 실시간 루프 오케스트레이션

- [ ] "현재 모드 → 해당 모드 인식기 사용 → 인식 결과 → 명령 → Pynput" 실시간 루프 정립
- [ ] [src/mode_controller/mode_controller.py](src/mode_controller/mode_controller.py) 확장
- [ ] [src/mediapipe/pipeline.py](src/mediapipe/pipeline.py)가 Mode Controller에서 현재 모드 읽기
- [ ] 파이프라인이 해당 모드용 detector/recognizer만 사용하도록 변경
- [ ] 모드 변경 시 Game/PPT/YouTube 중 해당 인식 경로만 사용

### Phase 3: Mediapipe — 모드별 인식

- [ ] **Game (Posture)**: [src/mediapipe/](src/mediapipe/) 내 Posture 인식 (직진/후진/좌회전/우회전)
- [ ] **PPT (Gesture + LSTM)**: 다음/이전/쇼 시작 제스처 (규칙 기반 + LSTM 도입 시)
- [ ] **YouTube (Gesture + LSTM)**: 재생/정지, 10초 앞/뒤, 음소거, 전체화면 (규칙 기반 + LSTM 도입 시)
- [ ] **공통**: 트리거(시작/종료) 모션 모든 모드에서 인식
- [ ] S-06-GME-01~04: Game 직진/후진/좌회전/우회전
- [ ] S-06-YTB-01~06: YouTube 제스처
- [ ] S-06-PPT-01~02: PPT 다음/이전 슬라이드

### Phase 4: Pynput 연동 (명령 실행)

- [ ] Mode Controller(또는 파이프라인)에서 내려준 명령을 [src/input_simulator/manager.py](src/input_simulator/manager.py), [src/input_simulator/actions.py](src/input_simulator/actions.py)로 전달
- [ ] 키/마우스 입력 실행
- [ ] 기존 제스처→액션 매핑(registry/action_mapper) 일관성 유지

### Phase 5: 통합 테스트 및 문서

- [ ] 통합 테스트: 모드 전환 → 해당 모드 인식만 동작 → Pynput 명령 실행 E2E
- [ ] README에 아키텍처·Mode Controller 역할 반영
- [ ] [GESTURE_GUIDE.md](GESTURE_GUIDE.md) 등 문서에 아키텍처·Mode Controller 역할 반영

## 5. 기존 계획과의 차이

- **제거**: 1~7일차 일정형 체크리스트, "초기 단계" 등 과거 상태 설명.
- **유지**: 현재 디렉터리 구조(app, capture, mode_controller, mediapipe, input_simulator), config, requirements, SYSTEM_REQUIREMENTS 매핑.
- **추가/강조**: 아키텍처 다이어그램(SYSTEM_ARCHITECTURE.md), Mode Controller의 "현재 모드 → 모드별 모델 선택 → 실시간 인식 → Pynput 명령" 역할, Phase 단위 구현 순서. **YOLO는 범위에서 제외하며, 이미지는 opencv에서 바로 Mediapipe로 전달.**
