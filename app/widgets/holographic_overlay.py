import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, QPoint, QRectF, pyqtProperty, QPropertyAnimation
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QLinearGradient, QFont

import config

class HolographicOverlay(QWidget):
    """
    웹캠 위에 겹쳐지는 홀로그래픽 인터페이스:
    - 스캔라인 애니메이션
    - 모서리 타겟팅 브래킷
    - 데이터 일련번호 데코레이션
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        
        self._scanline_pos = 0.0
        self._bracket_margin = 20
        self._is_locking = False
        self._lock_anim = 0.0
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(30)
    
    def set_locking(self, active: bool):
        self._is_locking = active
        self.update()

    def _animate(self):
        self._scanline_pos = (self._scanline_pos + 0.01) % 1.0
        if self._is_locking:
            self._lock_anim = min(1.0, self._lock_anim + 0.1)
        else:
            self._lock_anim = max(0.0, self._lock_anim - 0.1)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        primary = QColor(config.COLOR_PRIMARY)
        primary.setAlpha(150)
        
        # 1. Scanline Effect
        scan_y = int(self._scanline_pos * h)
        grad = QLinearGradient(0, scan_y - 20, 0, scan_y + 20)
        grad.setColorAt(0, QColor(0, 0, 0, 0))
        grad.setColorAt(0.5, QColor(0, 255, 255, 40))
        grad.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(0, scan_y - 20, w, 40, grad)
        
        # Static thin lines
        painter.setPen(QPen(QColor(0, 255, 255, 10), 1))
        for i in range(0, h, 6):
            painter.drawLine(0, i, w, i)

        # 2. Corner Brackets
        margin = self._bracket_margin - (5 * self._lock_anim)
        length = 30 + (10 * self._lock_anim)
        
        pen = QPen(primary if not self._is_locking else QColor("white"))
        pen.setWidth(2)
        painter.setPen(pen)
        
        # Top Left
        painter.drawLine(int(margin), int(margin), int(margin + length), int(margin))
        painter.drawLine(int(margin), int(margin), int(margin), int(margin + length))
        
        # Top Right
        painter.drawLine(int(w - margin), int(margin), int(w - margin - length), int(margin))
        painter.drawLine(int(w - margin), int(margin), int(w - margin), int(margin + length))
        
        # Bottom Left
        painter.drawLine(int(margin), int(h - margin), int(margin + length), int(h - margin))
        painter.drawLine(int(margin), int(h - margin), int(margin), int(h - margin - length))
        
        # Bottom Right
        painter.drawLine(int(w - margin), int(h - margin), int(w - margin - length), int(h - margin))
        painter.drawLine(int(w - margin), int(h - margin), int(w - margin), int(h - margin - length))
        
        # 3. Data Readout
        font = painter.font()
        font.setPointSize(8)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 2)
        painter.setFont(font)
        painter.setPen(QPen(QColor(0, 255, 255, 100)))
        
        painter.drawText(int(margin + 5), int(h - margin - 5), "G-SYSTEM: READY")
        painter.drawText(int(w - margin - 100), int(margin + 15), "FEED: HD_60FPS")
        
        painter.end()
