import sys
import os
import time

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PyQt6.QtCore import QCoreApplication
from app.modules import CameraWorker

class LandmarkTester:
    def __init__(self):
        self.last_print_time = 0
        self.worker = CameraWorker()
        
        # Connect signals
        self.worker.landmarks_updated.connect(self.print_landmarks_structure)
        self.worker.error_occurred.connect(lambda err: print(f"Error: {err}"))

    def print_landmarks_structure(self, multi_hand_landmarks, multi_handedness):
        if not multi_hand_landmarks:
            # We could print "No hands" every now and then, but usually constant update is fine for landmarks
            return

        current_time = time.time()
        # Print every 1 second to avoid console spam
        if current_time - self.last_print_time >= 1.0:
            self.last_print_time = current_time
            
            print(f"\n--- Landmark Data ({time.strftime('%H:%M:%S')}) ---")
            print(len(multi_hand_landmarks))
            print(multi_handedness)
            print("-" * 40)

    def start(self):
        print("CameraWorker 시작... (Ctrl+C로 종료)")
        self.worker.start()

    def stop(self):
        self.worker.stop()
        self.worker.wait()

if __name__ == "__main__":
    app = QCoreApplication(sys.argv)
    
    tester = LandmarkTester()
    tester.start()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\n종료 중...")
        tester.stop()
        print("테스트 종료")