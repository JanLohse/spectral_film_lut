import ffmpeg
import numpy as np
from PyQt6.QtCore import QSize, QThreadPool
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QGridLayout, QSizePolicy, QCheckBox
from colour.models import RGB_COLOURSPACES

from spectral_film_lut.reversal_film.kodak_ektachrome_100d import KodakEktachrome100D
from spectral_film_lut.negative_film.kodak_5207 import Kodak5207
from spectral_film_lut.negative_film.kodak_ektar_100 import KodakEktar100
from spectral_film_lut.negative_film.kodak_portra_400 import KodakPortra400
from spectral_film_lut.print_film.kodak_2383 import Kodak2383
from spectral_film_lut.print_film.kodak_2393 import Kodak2393
from spectral_film_lut.print_film.kodak_endura_premier import KodakEnduraPremier
from spectral_film_lut.reversal_film.fuji_instax_color import FujiInstaxColor
from spectral_film_lut.reversal_film.kodachrome_64 import Kodachrome64
from spectral_film_lut.print_film.fuji_ca_dpII import FujiCrystalArchiveDPII
from spectral_film_lut.utils import *


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
            sidelayout.addWidget(label, self.side_counter, 0)
            if default is not None and setter is not None:
                label.mouseDoubleClickEvent = lambda *args: setter(default)
                setter(default)

        self.image_selector = FileSelector()
        add_option(self.image_selector, "Reference image:")

        colourspaces = ["CIE XYZ 1931"] + list(RGB_COLOURSPACES.data.keys())
        self.input_colourspace_selector = QComboBox()
        self.input_colourspace_selector.addItems(colourspaces)
        add_option(self.input_colourspace_selector, "Input colourspace:", "ARRI Wide Gamut 4",
                   self.input_colourspace_selector.setCurrentText)

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

        self.white_point = Slider()
        self.white_point.setMinMaxTicks(.5, 2., 1, 20)
        add_option(self.white_point, "White point:", 1., self.white_point.setValue)

        self.output_colourspace_selector = QComboBox()
        self.output_colourspace_selector.addItems(colourspaces)
        add_option(self.output_colourspace_selector, "Output colourspace:", "sRGB",
                   self.output_colourspace_selector.setCurrentText)

        self.lut_size = Slider()
        self.lut_size.setMinMaxTicks(2, 67)
        add_option(self.lut_size, "LUT size:", 33, self.lut_size.setValue)

        self.mode = QComboBox()
        self.mode.addItems(['full', 'negative', 'print'])
        add_option(self.mode, "Mode:", "full", self.mode.setCurrentText)

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
        self.lut_size.valueChanged.connect(self.parameter_changed)
        self.white_point.valueChanged.connect(self.parameter_changed)
        self.mode.currentTextChanged.connect(self.parameter_changed)

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

    def generate_lut(self, name="temp"):
        negative_film = self.filmstocks[self.negative_selector.currentText()]
        print_film = self.filmstocks[self.print_selector.currentText()]
        input_colourspace = self.input_colourspace_selector.currentText()
        projector_kelvin = self.projector_kelvin.getValue()
        exp_comp = self.exp_comp.getValue()
        printer_light_comp = np.array(
            [self.red_light.getValue(), self.green_light.getValue(), self.blue_light.getValue()])
        if input_colourspace == "CIE XYZ 1931": input_colourspace = None
        output_colourspace = self.output_colourspace_selector.currentText()
        if output_colourspace == "CIE XYZ 1931": output_colourspace = None
        size = int(self.lut_size.getValue())
        white_point = self.white_point.getValue()
        mode = self.mode.currentText()
        lut = create_lut(negative_film, print_film, name=name, matrix_method=False, size=size,
                         input_colourspace=input_colourspace, output_colourspace=output_colourspace,
                         projector_kelvin=projector_kelvin, exp_comp=exp_comp, printer_light_comp=printer_light_comp,
                         white_point=white_point, mode=mode)
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
        return

    def update_finished(self):
        self.running = False
        if self.waiting:
            self.waiting = False
            self.parameter_changed()

    def progress_fn(self, n):
        return

    def parameter_changed(self):
        if self.running:
            self.waiting = True
            return
        else:
            self.running = True
        worker = Worker(self.update_preview)
        worker.signals.finished.connect(self.update_finished)
        worker.signals.progress.connect(self.progress_fn)

        self.threadpool.start(worker)

    def update_preview(self, verbose=False, *args, **kwargs):
        if self.image_selector.currentText() == "" or not os.path.isfile(self.image_selector.currentText()):
            return

        lut = self.generate_lut()

        src = self.image_selector.currentText()
        target = "temp.jpg"
        if os.path.isfile(target):
            os.remove(target)
        try:
            start = time.time()
            run(ffmpeg.input(src).filter('scale', "1024", "-1").filter('lut3d', file=lut).output(target,
                                                                                                 loglevel="quiet"))
            if verbose:
                print(f"applied lut in {time.time() - start:.2f} seconds")
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
            self.generate_lut(filename)


def main():
    filmstocks = [Kodak5207, KodakPortra400, Kodachrome64, Kodak2383, Kodak2393, KodakEnduraPremier, FujiInstaxColor,
                  KodakEktachrome100D, FujiCrystalArchiveDPII, KodakEktar100]
    filmstocks = [x() for x in filmstocks]
    filmstocks = {stock.__class__.__name__: stock for stock in filmstocks}

    app = QApplication(sys.argv)
    w = MainWindow(filmstocks)
    w.show()
    app.exec()


if __name__ == '__main__':
    main()
