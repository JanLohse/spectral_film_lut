import os
import sys
from pathlib import Path

import ffmpeg
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap, QIntValidator
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QWidget, QHBoxLayout, QComboBox, \
    QFileDialog, QLineEdit, QGridLayout, QSizePolicy
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
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))
        sidelayout = QGridLayout()
        widget.setLayout(sidelayout)
        sidelayout.setAlignment(Qt.AlignmentFlag.AlignBottom)

        self.image = QLabel("Select a reference image for the preview")
        self.image.setMinimumSize(QSize(10, 10))
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(QSize(256, 256))
        self.pixmap = QPixmap()

        pagelayout.addWidget(self.image)
        pagelayout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignBottom)

        self.side_counter = -1

        def add_option(widget, name=None):
            self.side_counter += 1
            sidelayout.addWidget(widget, self.side_counter, 1)
            sidelayout.addWidget(QLabel(name, alignment=(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)),
                                 self.side_counter, 0)

        self.image_selector = FileSelector()
        add_option(self.image_selector, "Reference image:")

        colourspaces = ["CIE XYZ 1931"] + list(RGB_COLOURSPACES.data.keys())
        self.input_colourspace_selector = QComboBox()
        self.input_colourspace_selector.addItems(colourspaces)
        self.input_colourspace_selector.setCurrentText("ARRI Wide Gamut 4")
        add_option(self.input_colourspace_selector, "Input colourspace:")

        self.negative_selector = QComboBox()
        self.negative_selector.addItems(list(filmstocks.keys()))
        add_option(self.negative_selector, "Negativ stock:")

        filmstocks["None"] = None
        self.print_selector = QComboBox()
        self.print_selector.addItems(["None"] + list(filmstocks.keys()))
        self.print_selector.setCurrentText("Kodak2393")
        add_option(self.print_selector, "Print stock:")

        self.output_colourspace_selector = QComboBox()
        self.output_colourspace_selector.addItems(colourspaces)
        self.output_colourspace_selector.setCurrentText("sRGB")
        add_option(self.output_colourspace_selector, "Output colourspace:")

        self.lut_size = QComboBox()
        self.lut_size.addItems(["17", "33", "67"])
        self.lut_size.setEditable(True)
        self.lut_size.setCurrentText("33")
        self.lut_size.setValidator(QIntValidator())
        add_option(self.lut_size, "LUT size:")

        self.save_lut_button = QPushButton("Save LUT")
        self.save_lut_button.released.connect(self.save_lut)
        add_option(self.save_lut_button)

        self.input_colourspace_selector.currentTextChanged.connect(self.parameter_changed)
        self.negative_selector.currentTextChanged.connect(self.parameter_changed)
        self.output_colourspace_selector.currentTextChanged.connect(self.parameter_changed)
        self.print_selector.currentTextChanged.connect(self.parameter_changed)
        self.image_selector.textChanged.connect(self.parameter_changed)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        self.resize(QSize(1024, 612))

    def scale_pixmap(self):
        if not self.pixmap.isNull():
            scaled_pixmap = self.pixmap.scaled(self.image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
            self.image.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.scale_pixmap()
        super().resizeEvent(event)

    def generate_lut(self, name="temp", size=33):
        negative_film = self.filmstocks[self.negative_selector.currentText()]
        print_film = self.filmstocks[self.print_selector.currentText()]
        input_colourspace = self.input_colourspace_selector.currentText()
        if input_colourspace == "CIE XYZ 1931": input_colourspace = None

        output_colourspace = self.output_colourspace_selector.currentText()
        if output_colourspace == "CIE XYZ 1931": output_colourspace = None
        lut = create_lut(negative_film, print_film, name=name, matrix_method=False, size=size,
                         input_colourspace=input_colourspace, output_colourspace=output_colourspace)
        return lut

    def parameter_changed(self, **kwargs):
        if self.image_selector.currentText() == "" or not os.path.isfile(self.image_selector.currentText()):
            return

        lut = self.generate_lut()

        src = self.image_selector.currentText()
        target = "temp.jpg"
        if os.path.isfile(target):
            os.remove(target)
        try:
            ffmpeg.input(src).filter('lut3d', file=lut).output(target, loglevel="quiet").run()
        except:
            return
        self.pixmap = QPixmap(target)
        self.image.setPixmap(self.pixmap)
        self.scale_pixmap()
        os.remove(target)
        os.remove(lut)

    def save_lut(self):
        filename, ok = QFileDialog.getSaveFileName(self)

        if ok:
            self.generate_lut(filename, int(self.lut_size.currentText()))


class FileSelector(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.minimumSize()

        self.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))

        # file selection
        file_browse = QPushButton('Browse')
        file_browse.setFixedWidth(55)
        file_browse.clicked.connect(self.open_file_dialog)
        self.filename_edit = QLineEdit()

        layout.addWidget(self.filename_edit)
        layout.addWidget(file_browse)

        self.textChanged = self.filename_edit.textChanged

        self.show()

    def open_file_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(self, "Select a File", "D:\\icons\\avatar\\",
                                                   "Images (*.png *.jpg *)")
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
