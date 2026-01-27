"""
Mediapipe 기반 손 랜드마크 추출 모듈
Mediapipe 0.10.32 Task API 사용
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, List, Tuple
import sys
import os
import urllib.request
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class MediapipeHandler:
    """Mediapipe Task API를 사용한 손 랜드마크 추출 (Mediapipe 0.10.32+)"""
    
    def __init__(self, 
                 num_hands: int = 2,
                 min_hand_detection_confidence: float = 0.5,
                 min_hand_presence_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_path: Optional[str] = None):
        """
        Mediapipe HandLandmarker 초기화 (Task API)
        
        Args:
            num_hands: 최대 손 개수
            min_hand_detection_confidence: 최소 손 검출 신뢰도
            min_hand_presence_confidence: 최소 손 존재 신뢰도
            min_tracking_confidence: 최소 추적 신뢰도
            model_path: 모델 파일 경로 (.task 파일). None이면 Mediapipe가 자동으로 모델을 찾거나 다운로드
        """
        # 모델 경로 설정 (None이면 Mediapipe가 기본 모델 사용)
        # Mediapipe 0.10.32는 모델 파일이 필요하므로, 없으면 자동으로 다운로드하거나 에러 발생
        # 실제 사용 시 모델 파일을 다운로드하거나 패키지에 포함시켜야 함
        model_asset_path = model_path if model_path else self._get_default_model_path()
        
        # Task API 옵션 설정
        base_options = python.BaseOptions(
            model_asset_path=model_asset_path
        )
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO  # 비디오 모드
        )
        
        # HandLandmarker 생성
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # 손 연결 정보 (21개 랜드마크 포인트 간 연결)
        # Mediapipe Hands 표준 연결 정보
        self.HAND_CONNECTIONS = [
            # 손목에서 손가락 기저부
            (0, 1), (1, 2), (2, 3), (3, 4),  # 엄지
            (0, 5), (5, 6), (6, 7), (7, 8),  # 검지
            (0, 9), (9, 10), (10, 11), (11, 12),  # 중지
            (0, 13), (13, 14), (14, 15), (15, 16),  # 약지
            (0, 17), (17, 18), (18, 19), (19, 20),  # 새끼손가락
            # 손가락 간 연결
            (5, 9), (9, 13), (13, 17)  # 손가락 기저부 연결
        ]
        
        # 시각화 색상 설정
        self.landmark_color = (0, 255, 0)  # 초록색 (BGR)
        self.connection_color = (255, 0, 0)  # 파란색 (BGR)
        self.landmark_radius = 5
        self.connection_thickness = 2
        
        # 프레임 카운터 (비디오 모드에서 timestamp로 사용)
        self.frame_timestamp_ms = 0
    
    def _get_default_model_path(self) -> str:
        """
        기본 모델 경로 반환
        모델 파일이 없으면 다운로드
        
        Returns:
            str: 모델 파일 경로
        """
        # 모델 파일 저장 경로
        models_dir = Path(config.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "hand_landmarker.task"
        
        # 모델 파일이 없으면 다운로드
        if not model_path.exists():
            print("Hand landmarker 모델 파일을 다운로드하는 중...")
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"모델 파일 다운로드 완료: {model_path}")
            except Exception as e:
                print(f"모델 파일 다운로드 실패: {e}")
                raise FileNotFoundError(
                    f"Hand landmarker 모델 파일을 찾을 수 없습니다. "
                    f"수동으로 다운로드하여 {model_path}에 저장해주세요. "
                    f"다운로드 URL: {model_url}"
                )
        
        return str(model_path)
    
    def process(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """
        프레임에서 손 랜드마크 추출
        
        Args:
            frame: BGR 프레임
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[List]]: 
                - 랜드마크가 그려진 프레임 (시각화용)
                - 랜드마크 데이터 리스트 (각 손마다 하나의 리스트)
        """
        if frame is None:
            return None, None
        
        # 프레임이 비어있는지 확인
        if frame.size == 0:
            return None, None
        
        # 시각화용 프레임 복사 (원본 프레임 보존)
        annotated_frame = frame.copy()
        
        try:
            # BGR to RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Mediapipe Image 객체 생성
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # 손 랜드마크 검출 (비디오 모드)
            detection_result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
            
            # 프레임 카운터 증가 (밀리초 단위, 30fps 가정)
            self.frame_timestamp_ms += int(1000 / 30)
            
            landmarks_list = []
            
            if detection_result.hand_landmarks:
                frame_height, frame_width = annotated_frame.shape[:2]
                
                for hand_landmarks in detection_result.hand_landmarks:
                    # 랜드마크 데이터 추출 및 시각화
                    landmarks = []
                    landmark_points = []  # 픽셀 좌표로 변환된 포인트들
                    
                    for landmark in hand_landmarks:
                        # 정규화된 좌표를 픽셀 좌표로 변환
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        landmark_points.append((x, y))
                        
                        # 랜드마크 데이터 저장 (정규화된 좌표)
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z
                        })
                    
                    # 랜드마크 포인트 그리기 (원)
                    for point in landmark_points:
                        cv2.circle(annotated_frame, point, self.landmark_radius, 
                                  self.landmark_color, -1)
                    
                    # 랜드마크 연결선 그리기
                    for connection in self.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                            start_point = landmark_points[start_idx]
                            end_point = landmark_points[end_idx]
                            cv2.line(annotated_frame, start_point, end_point,
                                    self.connection_color, self.connection_thickness)
                    
                    landmarks_list.append(landmarks)
            
            # 랜드마크가 없어도 원본 프레임 반환
            return annotated_frame, landmarks_list if landmarks_list else None
            
        except Exception as e:
            print(f"Mediapipe 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            # 오류 발생 시 원본 프레임 반환
            return annotated_frame, None
    
    def normalize_landmarks(self, landmarks: List, frame_width: int, frame_height: int) -> np.ndarray:
        """
        랜드마크를 정규화 (픽셀 좌표로 변환)
        
        Args:
            landmarks: 랜드마크 리스트 (21개 포인트)
            frame_width: 프레임 너비
            frame_height: 프레임 높이
            
        Returns:
            np.ndarray: 정규화된 랜드마크 배열 (21, 3)
        """
        normalized = []
        for landmark in landmarks:
            normalized.append([
                landmark['x'] * frame_width,
                landmark['y'] * frame_height,
                landmark['z']
            ])
        return np.array(normalized)
    
    def get_landmarks_array(self, landmarks: List) -> np.ndarray:
        """
        랜드마크를 numpy 배열로 변환 (정규화된 좌표 사용)
        
        Args:
            landmarks: 랜드마크 리스트 (21개 포인트)
            
        Returns:
            np.ndarray: 랜드마크 배열 (21, 3) - x, y, z는 0-1 범위
        """
        array = []
        for landmark in landmarks:
            array.append([landmark['x'], landmark['y'], landmark['z']])
        return np.array(array)
    
    def release(self):
        """리소스 해제"""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
