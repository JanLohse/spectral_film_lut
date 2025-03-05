import os
import sys
import traceback
from pathlib import Path

import ffmpeg
import numpy as np
from PyQt6.QtCore import QSize, Qt, QObject, pyqtSignal, QRunnable, pyqtSlot, QThreadPool
from PyQt6.QtGui import QPixmap, QIntValidator
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QWidget, QHBoxLayout, QComboBox, \
    QFileDialog, QLineEdit, QGridLayout, QSizePolicy, QSlider, QCheckBox, QLayout
from colour.models import RGB_COLOURSPACES

from negative_film.kodak_5207 import Kodak5207
from negative_film.kodak_portra_400 import KodakPortra400
from print_film.kodak_2383 import Kodak2383
from print_film.kodak_2393 import Kodak2393
from print_film.kodak_endura_premier import KodakEnduraPremier
from reversal_film.fuji_instax_color import FujiInstaxColor
from reversal_film.kodachrome_64 import Kodachrome64
from spectral_lut import create_lut


class WorkerSignals(QObject):
    """Signals from a running worker thread.

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc())

    result
        object data returned from processing, anything

    progress
        float indicating % progress
    """

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(float)

class Worker(QRunnable):
    """Worker thread.

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread.
                     Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

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
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(QSize(256, 256))
        self.pixmap = QPixmap()

        pagelayout.addWidget(self.image)
        pagelayout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignBottom)

        self.side_counter = -1

        def add_option(widget, name=None, default=None, setter=None):
            self.side_counter += 1
            sidelayout.addWidget(widget, self.side_counter, 1)
            label = QLabel(name, alignment=(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            sidelayout.addWidget(label,
                                 self.side_counter, 0)
            if default is not None and setter is not None:
                label.mouseDoubleClickEvent = lambda *args: setter(default)
                setter(default)

        self.image_selector = FileSelector()
        add_option(self.image_selector, "Reference image:")

        colourspaces = ["CIE XYZ 1931"] + list(RGB_COLOURSPACES.data.keys())
        self.input_colourspace_selector = QComboBox()
        self.input_colourspace_selector.addItems(colourspaces)
        add_option(self.input_colourspace_selector, "Input colourspace:", "ARRI Wide Gamut 4", self.input_colourspace_selector.setCurrentText)

        self.exp_comp = Slider()
        self.exp_comp.setMinMaxTicks(-2, 2, 1, 6)
        add_option(self.exp_comp, "Exposure:", 0, self.exp_comp.setValue)

        self.negative_selector = QComboBox()
        self.negative_selector.addItems(list(filmstocks.keys()))
        add_option(self.negative_selector, "Negativ stock:", "Kodak5207", self.negative_selector.setCurrentText)

        self.red_light = Slider()
        self.red_light.setMinMaxTicks(-2, 2, 1, 6)
        add_option(self.red_light, "Red printer light:", 0, self.red_light.setValue)
        self.green_light = Slider()
        self.green_light.setMinMaxTicks(-2, 2, 1, 6)
        add_option(self.green_light, "Green printer light:", 0, self.green_light.setValue)
        self.blue_light = Slider()
        self.blue_light.setMinMaxTicks(-2, 2, 1, 6)
        add_option(self.blue_light, "Blue printer light:", 0, self.blue_light.setValue)

        self.link_lights = QCheckBox()
        self.link_lights.setChecked(True)
        self.link_lights.setText("link lights")
        add_option(self.link_lights)

        filmstocks["None"] = None
        self.print_selector = QComboBox()
        self.print_selector.addItems(["None"] + list(filmstocks.keys()))
        add_option(self.print_selector, "Print stock:", "Kodak2383", self.print_selector.setCurrentText)

        self.projector_kelvin = Slider()
        self.projector_kelvin.setMinMaxTicks(2700, 10000, 100)
        add_option(self.projector_kelvin, "Projector wb:", 6500, self.projector_kelvin.setValue)

        self.output_colourspace_selector = QComboBox()
        self.output_colourspace_selector.addItems(colourspaces)
        add_option(self.output_colourspace_selector, "Output colourspace:", "sRGB", self.output_colourspace_selector.setCurrentText)

        self.lut_size = QComboBox()
        self.lut_size.addItems(["17", "33", "67"])
        self.lut_size.setEditable(True)
        self.lut_size.setCurrentText("33")
        self.lut_size.setValidator(QIntValidator())
        add_option(self.lut_size, "LUT size:", "33", self.lut_size.setCurrentText)

        self.save_lut_button = QPushButton("Save LUT")
        self.save_lut_button.released.connect(self.save_lut)
        add_option(self.save_lut_button)

        self.input_colourspace_selector.currentTextChanged.connect(self.parameter_changed)
        self.negative_selector.currentTextChanged.connect(self.parameter_changed)
        self.output_colourspace_selector.currentTextChanged.connect(self.parameter_changed)
        self.print_selector.currentTextChanged.connect(self.print_light_changed)
        self.image_selector.textChanged.connect(self.parameter_changed)
        self.projector_kelvin.valueChanged.connect(self.parameter_changed)
        self.exp_comp.valueChanged.connect(self.parameter_changed)
        self.red_light.valueChanged.connect(self.lights_changed)
        self.green_light.valueChanged.connect(self.lights_changed)
        self.blue_light.valueChanged.connect(self.lights_changed)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        self.resize(QSize(1024, 512))

        self.waiting = False
        self.running = False

        self.threadpool = QThreadPool()

    def scale_pixmap(self):
        if not self.pixmap.isNull():
            scaled_pixmap = self.pixmap.scaled(self.image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
            self.image.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.scale_pixmap()
        super().resizeEvent(event)

    def generate_lut(self, name="temp", size=17):
        negative_film = self.filmstocks[self.negative_selector.currentText()]
        print_film = self.filmstocks[self.print_selector.currentText()]
        input_colourspace = self.input_colourspace_selector.currentText()
        projector_kelvin = self.projector_kelvin.getValue()
        exp_comp = self.exp_comp.getValue()
        printer_light_comp = np.array([self.red_light.getValue(), self.green_light.getValue(), self.blue_light.getValue()])
        if input_colourspace == "CIE XYZ 1931": input_colourspace = None
        output_colourspace = self.output_colourspace_selector.currentText()
        if output_colourspace == "CIE XYZ 1931": output_colourspace = None
        lut = create_lut(negative_film, print_film, name=name, matrix_method=False, size=size,
                         input_colourspace=input_colourspace, output_colourspace=output_colourspace,
                         projector_kelvin=projector_kelvin, exp_comp=exp_comp, printer_light_comp=printer_light_comp)
        return lut

    def lights_changed(self, value):
        if self.link_lights.isChecked():
            if value == self.red_light.getPosition() == self.green_light.getPosition() == self.blue_light.getPosition():
                self.parameter_changed()
            else:
                self.red_light.setPosition(value)
                self.green_light.setPosition(value)
                self.blue_light.setPosition(value)
        else:
            self.parameter_changed()

    def print_light_changed(self):
        if self.print_selector.currentText() == "None":
            self.red_light.setDisabled(True)
            self.green_light.setDisabled(True)
            self.blue_light.setDisabled(True)
            self.link_lights.setDisabled(True)
        else:
            self.red_light.setDisabled(False)
            self.green_light.setDisabled(False)
            self.blue_light.setDisabled(False)
            self.link_lights.setDisabled(False)
        self.parameter_changed()

    def print_output(self, s):
        print(s)

    def update_finished(self):
        self.running = False
        if self.waiting:
            self.waiting = False
            self.parameter_changed()

    def progress_fn(self, n):
        print(f"{n:.1f}% done")

    def parameter_changed(self):
        if self.running:
            self.waiting = True
            return
        else:
            self.running = True
        worker = Worker(
            self.update_preview
        )
        worker.signals.finished.connect(self.update_finished)
        worker.signals.progress.connect(self.progress_fn)

        self.threadpool.start(worker)

    def update_preview(self, *args, **kwargs):
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


class Slider(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

        self.slider = QSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.setMinMaxTicks(0, 1, 1)

        self.text = QLabel()
        self.text.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.text.setFixedWidth(30)

        layout.addWidget(self.slider)
        layout.addWidget(self.text)

        self.slider.valueChanged.connect(self.sliderValueChanged)

        self.valueChanged = self.slider.valueChanged

    def setMinMaxTicks(self, min, max, enumerator=1, denominator=1):
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(((max - min) * denominator) / enumerator))
        self.min = min
        self.enumerator = enumerator
        self.denominator = denominator

    def sliderValueChanged(self):
        value = self.getValue()
        if value.is_integer():
            self.text.setText(str(int(value)))
        else:
            self.text.setText(f"{value:.2f}")

    def getValue(self):
        return self.slider.value() * self.enumerator / self.denominator + self.min

    def setValue(self, value):
        self.slider.setValue(round((value - self.min) * self.denominator / self.enumerator))

    def setPosition(self, position):
        self.slider.setValue(position)

    def getPosition(self):
        return self.slider.value()


if __name__ == '__main__':
    filmstocks = [Kodak5207, KodakPortra400, Kodachrome64, Kodak2383, Kodak2393, KodakEnduraPremier, FujiInstaxColor]
    filmstocks = [x() for x in filmstocks]
    filmstocks = {stock.__class__.__name__: stock for stock in filmstocks}

    app = QApplication(sys.argv)
    w = MainWindow(filmstocks)
    w.show()
    app.exec()
