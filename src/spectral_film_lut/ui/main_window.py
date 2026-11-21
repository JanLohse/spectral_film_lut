"""The main GUI implementation."""

import math
import os
import time
from functools import lru_cache

import imageio.v3 as iio
import numpy as np
from PyQt6.QtCore import QSize, Qt, QThreadPool
from PyQt6.QtGui import QAction, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from spectral_film_lut import BASE_DIR, __version__
from spectral_film_lut.core.utils import apply_lut_tetrahedral_int, create_lut
from spectral_film_lut.ui.grain_generation import ExportGrainDialog
from spectral_film_lut.ui.panels import (
    build_export_panel,
    build_filmstock_panel,
    build_preview_and_source_panel,
)
from spectral_film_lut.ui.state import collect_lut_settings
from spectral_film_lut.ui.theme import BASE_COLOR, BORDER_RADIUS
from spectral_film_lut.ui.widgets.controls import AboutDialog, Worker


class MainWindow(QMainWindow):
    """The main window of Raw2Film."""

    def __init__(self, filmstocks):
        super().__init__()

        self.setWindowTitle(f"Spectral Film LUT {__version__}")

        icon = QIcon()
        for size in [256, 128, 64, 48, 32, 16]:
            path = f"{BASE_DIR}/resources/spectral_film_lut_{size}.png"
            icon.addFile(path, QSize(size, size))

        self.setWindowIcon(icon)

        self.filmstocks = filmstocks

        pagelayout = QHBoxLayout()
        left_layout = QVBoxLayout()
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))
        sidelayout = QGridLayout()
        widget.setLayout(sidelayout)
        widget.setObjectName("sidelayout")
        widget.setStyleSheet(
            f"#sidelayout {{background-color: {BASE_COLOR}; border-radius: {BORDER_RADIUS};}}"
        )
        sidelayout.setAlignment(Qt.AlignmentFlag.AlignBottom)

        pagelayout.addLayout(left_layout)
        pagelayout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignBottom)

        self.side_counter = -1

        def add_option(control, name=None, default=None, setter=None, tool_tip=None):
            self.side_counter += 1
            sidelayout.addWidget(control, self.side_counter, 1)
            from PyQt6.QtWidgets import QLabel

            label = QLabel(name)
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            sidelayout.addWidget(label, self.side_counter, 0)
            if default is not None and setter is not None:
                label.mouseDoubleClickEvent = lambda *args: setter(default)
                setter(default)
            if tool_tip is not None:
                label.setToolTip(tool_tip)
                control.setToolTip(tool_tip)

        build_preview_and_source_panel(self, left_layout, add_option)
        build_filmstock_panel(self, add_option)
        build_export_panel(self, add_option)
        left_layout.insertWidget(0, self.mode)

        self.input_colorspace_selector.currentTextChanged.connect(self.parameter_changed)
        self.output_gamut.currentTextChanged.connect(self.parameter_changed)
        self.output_gamma.currentTextChanged.connect(self.parameter_changed)
        self.print_selector.currentTextChanged.connect(self.print_light_changed)
        self.image_selector.textChanged.connect(self.parameter_changed)
        self.projector_kelvin.valueChanged.connect(self.parameter_changed)
        self.inversion_gamma.valueChanged.connect(self.parameter_changed)
        self.exp_comp.valueChanged.connect(self.parameter_changed)
        self.exp_wb.valueChanged.connect(self.parameter_changed)
        self.tint.valueChanged.connect(self.parameter_changed)
        self.push_pull.valueChanged.connect(self.parameter_changed)
        self.dye_inhibition.valueChanged.connect(self.parameter_changed)
        self.red_light.valueChanged.connect(self.lights_changed)
        self.green_light.valueChanged.connect(self.lights_changed)
        self.blue_light.valueChanged.connect(self.lights_changed)
        self.lut_size.valueChanged.connect(self.parameter_changed)
        self.shadow_comp.valueChanged.connect(self.parameter_changed)
        self.color_masking.valueChanged.connect(self.parameter_changed)
        self.idealized_curve.stateChanged.connect(self.parameter_changed)
        self.white_clip.stateChanged.connect(self.parameter_changed)
        self.white_balance.stateChanged.connect(self.parameter_changed)
        self.mode.currentTextChanged.connect(self.parameter_changed)
        self.sat_adjust.valueChanged.connect(self.parameter_changed)
        self.adx_scale.currentTextChanged.connect(self.parameter_changed)
        self.apd_intermediate.stateChanged.connect(self.parameter_changed)
        self.filter_mode.currentTextChanged.connect(self.parameter_changed)
        self.spectral_upsampling_method.currentTextChanged.connect(self.parameter_changed)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        self.resize(QSize(1024, 512))

        self.waiting = False
        self.running = False

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)

        about_action = QAction("About This App", self)
        about_action.triggered.connect(self.show_about)
        self.addAction(about_action)

        self.threadpool = QThreadPool()

    def resizeEvent(self, event):
        self.update_preview()
        super().resizeEvent(event)

    def generate_lut(self, name="temp", cube=True):
        settings = collect_lut_settings(self)
        kwargs = settings.create_kwargs(name=name, cube=cube)
        return create_lut(**kwargs)

    def lights_changed(self, value):

        if self.link_lights.isChecked():
            if (
                value
                == self.red_light.getPosition()
                == self.green_light.getPosition()
                == self.blue_light.getPosition()
            ):
                self.parameter_changed()

            else:
                self.red_light.setValue(value)

                self.green_light.setValue(value)

                self.blue_light.setValue(value)

                self.parameter_changed()

        else:
            self.parameter_changed()

    def print_light_changed(self):
        if (
            self.print_selector.currentText() == "None"
            or "Inversion" in self.print_selector.currentText()
        ):
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

    @lru_cache(maxsize=8)
    def load_image_data(self, src):
        image = iio.imread(src)
        if image.dtype == np.uint8:
            image = image.astype(np.uint16) * 255
        return image

    def update_preview(self, verbose=False, *args, **kwargs):
        if verbose:
            start = time.time()
        if self.image_selector.currentText() == "" or not os.path.isfile(
            self.image_selector.currentText()
        ):
            return

        lut = self.generate_lut(cube=False)

        src = self.image_selector.currentText()
        image = self.load_image_data(src)
        height, width, _ = image.shape
        image_widget_size = self.image.size() * self.devicePixelRatioF()
        height_target = image_widget_size.height()
        width_target = image_widget_size.width()
        target_ratio = width_target / height_target
        image_ratio = width / height
        if target_ratio > image_ratio:
            scale_factor = min(height_target / height, 1)
        else:
            scale_factor = min(width_target / width, 1)
        scale_factor = math.floor(1 / scale_factor)
        image = image[::scale_factor, ::scale_factor, :]
        height, width, _ = image.shape
        lut = (lut * (2**8 - 1)).astype(np.uint8)
        image = apply_lut_tetrahedral_int(image, lut)

        image = QImage(image, width, height, 3 * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)

        scaled_pixmap = pixmap.scaled(
            image_widget_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        scaled_pixmap.setDevicePixelRatio(self.devicePixelRatioF())

        self.image.setPixmap(scaled_pixmap)
        if verbose:
            print(f"applied lut in {time.time() - start:.2f} seconds")

    def save_lut(self):
        mode = self.mode.currentText()
        size = f"{self.lut_size.getValue():.0f}"

        parts = {
            "full": [
                self.negative_selector.currentText(),
                self.print_selector.currentText(),
                self.input_colorspace_selector.currentText(),
                size,
            ],
            "negative": [
                self.negative_selector.currentText(),
                self.input_colorspace_selector.currentText(),
                size,
            ],
            "print": [
                self.print_selector.currentText(),
                "print",
                size,
            ],
            "grain": [
                self.negative_selector.currentText(),
                "grain",
                size,
            ],
        }

        default_name = "_".join(parts[mode]).replace(" ", "-")

        filename, ok = QFileDialog.getSaveFileName(self, "Save LUT", default_name, ".cube")

        if ok:
            self.generate_lut(filename)

    def export_noise(self):
        dialog = ExportGrainDialog(self)
        dialog.exec()

    def show_about(self):
        app_links = {
            "GitHub Page": "https://github.com/JanLohse/spectral_film_lut",
            "PyPI Page": "https://pypi.org/project/spectral_film_lut/",
            "Documentation": "https://janlohse.github.io/spectral_film_lut",
        }

        about_box = AboutDialog(
            parent=self,
            app_name="Spectral Film LUT",
            version=__version__,
            author="Jan Lohse",
            year="2026",
            license_type="MIT License",
            links=app_links,
        )
        about_box.exec()
