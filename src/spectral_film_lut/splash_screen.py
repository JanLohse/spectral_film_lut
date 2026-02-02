import ctypes
import sys
from ctypes import wintypes

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QProgressBar, QVBoxLayout, QWidget, QLabel

PROGRESS_BACKGROUND = "#2d2d2d"
PROGRESS_COLOR = "#dddddd"
BACKGROUND_COLOR = '#272727'
TEXT_PRIMARY = '#dddddd'
BUTTON_RADIUS = 6


class SplashScreen(QWidget):
    def __init__(self, name, version, total_items=100):
        super().__init__()
        self.setWindowTitle(f"{name} {version}")
        self.setFixedSize(400, 180)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.CoverWindow)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Background color
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};")

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Heading label
        self.heading = QLabel(name)
        self.heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heading.setStyleSheet("font-weight: bold; font-size: 18px;")

        # Sub-label
        self.label = QLabel("Starting up...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 13px;")

        # Progress bar
        self.total_items = total_items
        self.progress = QProgressBar()
        self.progress.setRange(0, total_items)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        self.update_style_sheet(0)

        layout.addWidget(self.heading)
        layout.addWidget(self.label)
        layout.addWidget(self.progress)

        self.show()
        QApplication.processEvents()

    def update_style_sheet(self, progress, colour=None):
        # Style the progress bar
        if colour is not None and progress:
            progress_color = colour.convert((0.5, 0.08, progress), "Oklch", "Hexadecimal")
        else:
            progress_color = "transparent"
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border-radius: {BUTTON_RADIUS}px;
                text-align: center;
                background-color: {PROGRESS_BACKGROUND};
                height: 16px;  /* thicker bar */
                color: {TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background-color: {progress_color};
                border-radius: {BUTTON_RADIUS}px;
            }}
        """)

    def set_total_items(self, total_items):
        self.progress.setRange(0, total_items)

    def update(self, current, total, name, colour):
        self.progress.setValue(current)
        self.update_style_sheet(current / self.total_items, colour=colour)
        self.label.setText(f"Loading {name} ({current}/{total})")
        QApplication.processEvents()


DWMWA_USE_IMMERSIVE_DARK_MODE = 20


def set_dark_title_bar(hwnd):
    value = ctypes.c_int(1)
    ctypes.windll.dwmapi.DwmSetWindowAttribute(wintypes.HWND(hwnd), wintypes.DWORD(DWMWA_USE_IMMERSIVE_DARK_MODE),
                                               ctypes.byref(value), ctypes.sizeof(value))


class DarkApp(QApplication):
    def notify(self, receiver, event):
        result = super().notify(receiver, event)
        if event.type() == event.Type.Show and isinstance(receiver, QWidget) and receiver.isWindow():
            try:
                hwnd = int(receiver.winId())
                set_dark_title_bar(hwnd)
            except Exception:
                pass
        return result


def launch_splash_screen(name, version):
    if sys.platform == "win32":
        app = DarkApp(sys.argv)
    else:
        app = QApplication(sys.argv)

    splash_screen = SplashScreen(name, version)

    return app, splash_screen
