"""
Gesto 메인 실행 파일
"""

import sys
from PyQt6.QtWidgets import QApplication
from src.app.main_window import MainWindow
import config


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 애플리케이션 정보 설정
    app.setApplicationName(config.APP_NAME)
    app.setApplicationVersion(config.APP_VERSION)
    
    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()
    
    # 이벤트 루프 실행
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
