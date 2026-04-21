import ctypes
import sys

from PyQt6.QtCore import QObject, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QIcon

from spectral_film_lut import BASE_DIR, __version__
from spectral_film_lut.splash_screen import launch_splash_screen


def run():
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "jan_lohse.spectral_film_lut"
        )

    app, splash_screen = launch_splash_screen("Spectral Film LUT", __version__)

    icon = QIcon()
    for size in [256, 128, 64, 48, 32, 16]:
        path = f"{BASE_DIR}/resources/spectral_film_lut_{size}.png"
        icon.addFile(path, QSize(size, size))

    app.setWindowIcon(icon)

    class LoaderWorker(QObject):
        finished = pyqtSignal()

        def run(self):

            self.finished.emit()

    thread = QThread()
    worker = LoaderWorker()
    worker.moveToThread(thread)

    def on_finished():
        from spectral_film_lut.film_loader import load_ui
        from spectral_film_lut.gui import MainWindow

        load_ui(MainWindow, splash_screen, app)

    worker.finished.connect(on_finished)
    thread.started.connect(worker.run)

    thread.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    run()
