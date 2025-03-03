import os
import sys
from pathlib import Path

import ffmpeg
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, \
    QComboBox, QFileDialog, QLineEdit
from colour.models import RGB_COLOURSPACES

from negative_film.kodak_5207 import Kodak5207
from negative_film.kodak_portra_400 import KodakPortra400
from print_film.kodak_2383 import Kodak2383
from print_film.kodak_2393 import Kodak2393
from print_film.kodak_endura_premier import KodakEnduraPremier
from reversal_film.kodachrome_64 import Kodachrome64
from spectral_lut import create_lut


class MainWindow(QMainWindow):
    def __init__(self, filmstocks):
        super().__init__()

        self.setWindowTitle("Spectral Film LUT")

        self.filmstocks = filmstocks

        pagelayout = QHBoxLayout()
        sidelayout = QVBoxLayout()

        self.image_selector = FileSelector("Image:", None)
        sidelayout.addWidget(self.image_selector)

        colourspaces = ["None"] + list(RGB_COLOURSPACES.data.keys())
        self.input_colourspace_selector = QComboBox()
        self.input_colourspace_selector.addItems(colourspaces)
        self.input_colourspace_selector.setCurrentText("ARRI Wide Gamut 4")
        sidelayout.addWidget(self.input_colourspace_selector)

        self.negative_selector = QComboBox()
        self.negative_selector.addItems(list(filmstocks.keys()))
        sidelayout.addWidget(self.negative_selector)

        filmstocks["None"] = None
        self.print_selector = QComboBox()
        self.print_selector.addItems(["None"] + list(filmstocks.keys()))
        self.print_selector.setCurrentText("Kodak2393")
        sidelayout.addWidget(self.print_selector)

        self.output_colourspace_selector = QComboBox()
        self.output_colourspace_selector.addItems(colourspaces)
        self.output_colourspace_selector.setCurrentText("sRGB")
        sidelayout.addWidget(self.output_colourspace_selector)

        self.image = QLabel("Select a reference image for the preview")
        self.image.setScaledContents(True)
        self.image.setMinimumSize(QSize(256, 256))
        self.image.setMaximumSize(QSize(1024, 1024))
        self.image.minimumSize()
        pagelayout.addWidget(self.image)
        pagelayout.addLayout(sidelayout)

        self.input_colourspace_selector.currentTextChanged.connect(self.parameter_changed)
        self.negative_selector.currentTextChanged.connect(self.parameter_changed)
        self.output_colourspace_selector.currentTextChanged.connect(self.parameter_changed)
        self.print_selector.currentTextChanged.connect(self.parameter_changed)
        self.image_selector.textChanged.connect(self.parameter_changed)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        self.parameter_changed()

    def parameter_changed(self, **kwargs):
        if self.image_selector.currentText() == "" or not os.path.isfile(self.image_selector.currentText()):
            return

        negative_film = self.filmstocks[self.negative_selector.currentText()]
        print_film = self.filmstocks[self.print_selector.currentText()]
        input_colourspace = self.input_colourspace_selector.currentText()
        if input_colourspace == "None": input_colourspace = None

        output_colourspace = self.output_colourspace_selector.currentText()
        if output_colourspace == "None": output_colourspace = None
        lut = create_lut(negative_film, print_film, name="temp", matrix_method=False, size=33,
                         input_colourspace=input_colourspace, output_colourspace=output_colourspace)
        src = self.image_selector.currentText()
        target = "temp.jpg"
        if os.path.isfile(target):
            os.remove(target)
        try:
            ffmpeg.input(src).filter('lut3d', file=lut).output(target, loglevel="quiet").run()
        except:
            return
        self.image.setPixmap(QPixmap(target))

class FileSelector(QWidget):
    def __init__(self, name="File", default=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout()
        self.setLayout(layout)

        # file selection
        file_browse = QPushButton('Browse')
        file_browse.clicked.connect(self.open_file_dialog)
        self.filename_edit = QLineEdit()
        self.filename_edit.setText(default)

        layout.addWidget(QLabel(name))
        layout.addWidget(self.filename_edit)
        layout.addWidget(file_browse)

        self.textChanged = self.filename_edit.textChanged

        self.show()

    def open_file_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(self, "Select a File", "D:\\icons\\avatar\\", "Images (*.png *.jpg *)")
        if filename:
            path = Path(filename)
            self.filename_edit.setText(str(path))

    def currentText(self):
        return self.filename_edit.text()




if __name__ == '__main__':
    filmstocks = [Kodak5207, KodakPortra400, Kodachrome64, Kodak2383, Kodak2393, KodakEnduraPremier]
    filmstocks = [x() for x in filmstocks]
    filmstocks = {stock.__class__.__name__: stock for stock in filmstocks}

    app = QApplication(sys.argv)
    w = MainWindow(filmstocks)
    w.show()
    app.exec()
