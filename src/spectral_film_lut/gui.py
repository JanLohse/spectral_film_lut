from PyQt6.QtCore import QSize, QThreadPool
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QMainWindow, QComboBox, QGridLayout, QSizePolicy, QCheckBox
from colour.models import RGB_COLOURSPACES

from spectral_film_lut.film_loader import load_ui
from spectral_film_lut.filmstock_selector import FilmStockSelector
from spectral_film_lut.reversal_film.technicolor_iv import *


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

        self.exp_wb = Slider()
        self.exp_wb.setMinMaxTicks(2000, 15000, 100)
        add_option(self.exp_wb, "WB:", 6500, self.exp_wb.setValue)

        self.tint = Slider()
        self.tint.setMinMaxTicks(-1, 1, 1, 100)
        add_option(self.tint, "Tint:", 0, self.tint.setValue)

        filmstock_info = {x: {'Year': filmstocks[x].year, 'Manufacturer': filmstocks[x].manufacturer, 'Type':
            {"camerapositive": "Slide", "cameranegative": "Negative", "printnegative": "Print",
             "printpositive": "SlidePrint"}[filmstocks[x].stage + filmstocks[x].type], 'Medium': filmstocks[x].medium,
                              'Sensitivity': f"ISO {filmstocks[x].iso}" if filmstocks[x].iso is not None else None,
                              'sensitivity': filmstocks[x].iso if filmstocks[x].iso is not None else None,
                              'resolution': f"{filmstocks[x].resolution} lines/mm" if filmstocks[
                                                                                          x].resolution is not None else None,
                              'Resolution': filmstocks[x].resolution if filmstocks[x].resolution is not None else None,
                              'Granularity': f"{filmstocks[x].rms} rms" if filmstocks[x].rms is not None else None,
                              'Decade': f"{filmstocks[x].year // 10 * 10}s" if filmstocks[x].year is not None else None,
                              'stage': filmstocks[x].stage,
                              'Chromaticity': 'BW' if filmstocks[x].density_measure == 'bw' else 'Color'} for x in
                          filmstocks}
        negative_info = {x: y for x, y in filmstock_info.items() if y['stage'] == 'camera'}
        sort_keys_negative = ["Name", "Year", "Resolution", "Granularity", "sensitivity"]
        group_keys_negative = ["Manufacturer", "Type", "Decade", "Medium"]
        list_keys_negative = ["Manufacturer", "Type", "Year", "Sensitivity", "Chromaticity"]
        sidebar_keys_negative = ["Manufacturer", "Type", "Year", "Sensitivity", "resolution", "Granularity", "Medium",
                                 "Chromaticity"]
        self.filmstocks["None"] = None
        self.negative_selector = FilmStockSelector(negative_info, sort_keys=sort_keys_negative,
                                                   group_keys=group_keys_negative, list_keys=list_keys_negative,
                                                   sidebar_keys=sidebar_keys_negative, default_group="Manufacturer")
        add_option(self.negative_selector, "Negativ stock:", "Kodak5207", self.negative_selector.setCurrentText)

        self.red_light = Slider()
        self.red_light.setMinMaxTicks(-1, 1, 1, 20)
        add_option(self.red_light, "Red printer light:", 0, self.red_light.setValue)
        self.green_light = Slider()
        self.green_light.setMinMaxTicks(-1, 1, 1, 20)
        add_option(self.green_light, "Green printer light:", 0, self.green_light.setValue)
        self.blue_light = Slider()
        self.blue_light.setMinMaxTicks(-1, 1, 1, 20)
        add_option(self.blue_light, "Blue printer light:", 0, self.blue_light.setValue)

        self.link_lights = QCheckBox()
        self.link_lights.setChecked(True)
        self.link_lights.setText("link lights")
        add_option(self.link_lights)

        print_info = {x: y for x, y in filmstock_info.items() if y['stage'] == 'print'}
        print_info["None"] = {}
        sort_keys_print = ["Name", "Year"]
        group_keys_print = ["Manufacturer", "Type", "Decade", "Medium"]
        list_keys_print = ["Manufacturer", "Type", "Year", "Chromaticity"]
        sidebar_keys_print = ["Manufacturer", "Type", "Year", "Medium", "Chromaticity"]
        self.print_selector = FilmStockSelector(print_info, sort_keys=sort_keys_print, group_keys=group_keys_print,
                                                list_keys=list_keys_print, sidebar_keys=sidebar_keys_print,
                                                default_group="Manufacturer")
        add_option(self.print_selector, "Print stock:", "Kodak2383", self.print_selector.setCurrentText)

        self.projector_kelvin = Slider()
        self.projector_kelvin.setMinMaxTicks(2700, 10000, 100)
        add_option(self.projector_kelvin, "Projector wb:", 6500, self.projector_kelvin.setValue)

        self.white_point = Slider()
        self.white_point.setMinMaxTicks(.5, 2., 1, 20)
        add_option(self.white_point, "White point:", 1., self.white_point.setValue)

        self.black_offset = Slider()
        self.black_offset.setMinMaxTicks(-2, 2, 1, 10)
        add_option(self.black_offset, "Black offset", 0., self.black_offset.setValue)

        self.output_colourspace_selector = QComboBox()
        self.output_colourspace_selector.addItems(colourspaces)
        add_option(self.output_colourspace_selector, "Output colourspace:", "sRGB",
                   self.output_colourspace_selector.setCurrentText)

        self.lut_size = Slider()
        self.lut_size.setMinMaxTicks(2, 67)
        add_option(self.lut_size, "LUT size:", 33, self.lut_size.setValue)

        self.color_masking = Slider()
        self.color_masking.setMinMaxTicks(0, 1, 1, 10)
        add_option(self.color_masking, "Color masking:", 1, self.color_masking.setValue)

        self.mode = QComboBox()
        self.mode.addItems(['full', 'negative', 'print'])
        add_option(self.mode, "Mode:", "full", self.mode.setCurrentText)

        self.save_lut_button = QPushButton("Save LUT")
        self.save_lut_button.released.connect(self.save_lut)
        add_option(self.save_lut_button)

        self.input_colourspace_selector.currentTextChanged.connect(self.parameter_changed)
        self.negative_selector.currentTextChanged.connect(self.negative_changed)
        self.output_colourspace_selector.currentTextChanged.connect(self.parameter_changed)
        self.print_selector.currentTextChanged.connect(self.print_light_changed)
        self.image_selector.textChanged.connect(self.parameter_changed)
        self.projector_kelvin.valueChanged.connect(self.parameter_changed)
        self.exp_comp.valueChanged.connect(self.parameter_changed)
        self.exp_wb.valueChanged.connect(self.parameter_changed)
        self.tint.valueChanged.connect(self.parameter_changed)
        self.red_light.valueChanged.connect(self.lights_changed)
        self.green_light.valueChanged.connect(self.lights_changed)
        self.blue_light.valueChanged.connect(self.lights_changed)
        self.lut_size.valueChanged.connect(self.parameter_changed)
        self.black_offset.valueChanged.connect(self.parameter_changed)
        self.color_masking.valueChanged.connect(self.parameter_changed)
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
        red_light = self.red_light.getValue()
        green_light = self.green_light.getValue()
        blue_light = self.blue_light.getValue()
        if input_colourspace == "CIE XYZ 1931": input_colourspace = None
        output_colourspace = self.output_colourspace_selector.currentText()
        if output_colourspace == "CIE XYZ 1931": output_colourspace = None
        size = int(self.lut_size.getValue())
        white_point = self.white_point.getValue()
        black_offset = self.black_offset.getValue()
        color_masking = self.color_masking.getValue()
        mode = self.mode.currentText()
        exp_wb = self.exp_wb.getValue()
        tint = self.tint.getValue()
        lut = create_lut(negative_film, print_film, name=name, matrix_method=False, lut_size=size,
                         input_colourspace=input_colourspace, output_colourspace=output_colourspace,
                         projector_kelvin=projector_kelvin, exp_comp=exp_comp, white_point=white_point,
                         exposure_kelvin=exp_wb, mode=mode, red_light=red_light, green_light=green_light,
                         blue_light=blue_light, black_offset=black_offset, color_masking=color_masking, tint=tint)
        return lut

    def lights_changed(self, value):
        if self.link_lights.isChecked():
            if value == self.red_light.getPosition() == self.green_light.getPosition() == self.blue_light.getPosition():
                self.parameter_changed()
            else:
                self.red_light.setValue(value)
                self.green_light.setValue(value)
                self.blue_light.setValue(value)
                self.parameter_changed()
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

    def negative_changed(self, negative_film):
        self.color_masking.setValue(self.filmstocks[negative_film].color_masking)
        self.parameter_changed()

    def update_preview(self, verbose=False, *args, **kwargs):
        if self.image_selector.currentText() == "" or not os.path.isfile(self.image_selector.currentText()):
            return

        lut = self.generate_lut()

        src = self.image_selector.currentText()
        start = time.time()
        image = iio.imread(src)
        height, width, _ = image.shape
        height_target = self.image.height()
        width_target = self.image.width()
        scale_factor = min(max(width_target / width, height_target / height), 1)
        scale_factor = math.floor(1 / scale_factor)
        image = image[::scale_factor, ::scale_factor, :]
        height, width, _ = image.shape
        process = run_async(
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb48', s=f'{width}x{height}').filter('lut3d',
                                                                                                    file=lut).output(
                'pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1, loglevel='quiet'), pipe_stdin=True,
            pipe_stdout=True)
        process.stdin.write(image.tobytes())
        process.stdin.close()
        image = process.stdout.read(width * height * 3)
        process.wait()
        os.remove(lut)
        image = np.frombuffer(image, np.uint8).reshape([height, width, 3])
        image = QImage(np.require(image, np.uint8, 'C'), width, height, 3 * width, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image)
        self.image.setPixmap(self.pixmap)
        self.scale_pixmap()
        if verbose:
            print(f"applied lut in {time.time() - start:.2f} seconds")

    def save_lut(self):
        filename, ok = QFileDialog.getSaveFileName(self)

        if ok:
            self.generate_lut(filename)


def main():
    load_ui(MainWindow)


if __name__ == '__main__':
    main()
