"""The main GUI implementation."""

import math
import os
import time
from functools import lru_cache

import imageio.v3 as iio
import numpy as np
from colour.models import RGB_COLOURSPACES
from PyQt6.QtCore import QSize, Qt, QThreadPool
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QWidget,
)

from spectral_film_lut import BASE_DIR, __version__
from spectral_film_lut.color_space import COLOR_SPACES, GAMMA_FUNCTIONS
from spectral_film_lut.css_theme import BASE_COLOR, BORDER_RADIUS
from spectral_film_lut.filmstock_selector import FilmStockSelector
from spectral_film_lut.grain_generation import ExportGrainDialog
from spectral_film_lut.gui_objects import (
    AnimatedButton,
    FileSelector,
    Slider,
    SliderLog,
    WideComboBox,
    Worker,
)
from spectral_film_lut.utils import apply_lut_tetrahedral_int, create_lut


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
        widget = QWidget()
        widget.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        )
        sidelayout = QGridLayout()
        widget.setLayout(sidelayout)
        widget.setObjectName("sidelayout")
        widget.setStyleSheet(
            f"#sidelayout {{background-color: {BASE_COLOR}; border-radius: "
            f"{BORDER_RADIUS};}}"
        )
        sidelayout.setAlignment(Qt.AlignmentFlag.AlignBottom)

        self.image = QLabel("Select a reference image for the preview")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        self.image.setMinimumSize(QSize(256, 256))

        pagelayout.addWidget(self.image)
        pagelayout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignBottom)

        self.side_counter = -1

        def add_option(widget, name=None, default=None, setter=None, tool_tip=None):
            self.side_counter += 1
            sidelayout.addWidget(widget, self.side_counter, 1)
            label = QLabel(
                name,
                alignment=(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            )
            sidelayout.addWidget(label, self.side_counter, 0)
            if default is not None and setter is not None:
                label.mouseDoubleClickEvent = lambda *args: setter(default)
                setter(default)
            if tool_tip is not None:
                label.setToolTip(tool_tip)
                widget.setToolTip(tool_tip)

        self.image_selector = FileSelector()
        """
        Select which image to use for the preview. Should be encoded in the target
        color space of the LUT.
        """
        add_option(
            self.image_selector,
            "Reference image",
            tool_tip="Select which image to use for the preview. Should be encoded in\n"
            "the target color space of the LUT.",
        )

        colourspaces = ["CIE XYZ 1931"] + list(RGB_COLOURSPACES.data.keys())
        self.input_colorspace_selector = WideComboBox()
        """
        What color space is the reference image in?
        """
        self.input_colorspace_selector.addItems(colourspaces)
        add_option(
            self.input_colorspace_selector,
            "Input colorspace",
            "ARRI Wide Gamut 4",
            self.input_colorspace_selector.setCurrentText,
            tool_tip="What color space is the reference image in?",
        )

        self.exp_comp = Slider()
        """
        Compensate for exposure. Default assumes middle gray to be at the default from
        the color space spec.
        """
        self.exp_comp.setMinMaxTicks(-2, 2, 1, 6)
        add_option(
            self.exp_comp,
            "Exposure",
            0,
            self.exp_comp.setValue,
            tool_tip="Compensate for exposure. Default assumes middle gray to be at\n"
            "the default from the color space spec.",
        )

        self.exp_wb = SliderLog()
        """
        Adjust the white balance of the image. Default assumes white to be at 6500
        kelvin.
        """
        self.exp_wb.setMinMaxSteps(2700, 16000, 120, 6500, -2)
        self.exp_wb.set_color_gradient(
            np.array([2 / 3, 0.14, 0.65277]), np.array([2 / 3, 0.14, 0.15277])
        )
        add_option(
            self.exp_wb,
            "WB",
            6500,
            self.exp_wb.setValue,
            tool_tip="Adjust the white balance of the image. Default assumes white to\n"
            "be at 6500 kelvin.",
        )

        self.tint = Slider()
        """
        Adjust the tint along the green to red/purple axis.
        """
        self.tint.setMinMaxTicks(-1, 1, 1, 100)
        self.tint.set_color_gradient(
            np.array([2 / 3, 0.14, 0.90277]), np.array([2 / 3, 0.14, 0.40277])
        )
        add_option(
            self.tint,
            "Tint",
            0,
            self.tint.setValue,
            tool_tip="Adjust the tint along the green to red/purple axis.",
        )

        filmstock_info = {
            x: {
                "Year": filmstocks[x].year,
                "Manufacturer": filmstocks[x].manufacturer,
                "Type": {
                    "camerapositive": "Slide",
                    "cameranegative": "Negative",
                    "printnegative": "Print",
                    "printpositive": "SlidePrint",
                }[filmstocks[x].stage + filmstocks[x].film_type],
                "Medium": filmstocks[x].medium,
                "Sensitivity": f"ISO {filmstocks[x].iso}"
                if filmstocks[x].iso is not None
                else None,
                "sensitivity": filmstocks[x].iso
                if filmstocks[x].iso is not None
                else None,
                "resolution": f"{filmstocks[x].resolution} lines/mm"
                if filmstocks[x].resolution is not None
                else None,
                "Resolution": filmstocks[x].resolution
                if filmstocks[x].resolution is not None
                else None,
                "Granularity": f"{filmstocks[x].rms} rms"
                if filmstocks[x].rms is not None
                else None,
                "Decade": f"{filmstocks[x].year // 10 * 10}s"
                if filmstocks[x].year is not None
                else None,
                "stage": filmstocks[x].stage,
                "Chromaticity": "BW"
                if filmstocks[x].density_measure == "bw"
                else "Color",
                "image": QImage(
                    np.require(filmstocks[x].color_checker, np.uint8, "C"),
                    6,
                    4,
                    18,
                    QImage.Format.Format_RGB888,
                ),
                "Gamma": round(filmstocks[x].gamma, 3),
                "Alias": filmstocks[x].alias,
                "Comment": filmstocks[x].comment,
                "D-max": f"{filmstocks[x].d_max.max():.2f}",
            }
            for x in filmstocks
        }
        negative_info = {
            x: y for x, y in filmstock_info.items() if y["stage"] == "camera"
        }
        sort_keys_negative = [
            "Name",
            "Year",
            "Resolution",
            "Granularity",
            "sensitivity",
            "Gamma",
            "D-max",
        ]
        group_keys_negative = ["Manufacturer", "Type", "Decade", "Medium"]
        list_keys_negative = [
            "Manufacturer",
            "Type",
            "Year",
            "Sensitivity",
            "Chromaticity",
        ]
        sidebar_keys_negative = [
            "Manufacturer",
            "Type",
            "Year",
            "Sensitivity",
            "resolution",
            "Granularity",
            "Medium",
            "Chromaticity",
            "Gamma",
            "Alias",
            "Comment",
            "D-max",
        ]
        self.filmstocks["None"] = None
        self.filmstocks["Inversion"] = None
        self.negative_selector = FilmStockSelector(
            negative_info,
            self,
            self,
            sort_keys=sort_keys_negative,
            group_keys=group_keys_negative,
            list_keys=list_keys_negative,
            sidebar_keys=sidebar_keys_negative,
            default_group="Manufacturer",
            image_key="image",
        )
        """Select the camera film stock that is emulated."""
        add_option(
            self.negative_selector,
            "Negativ stock",
            "Kodak Vision3 250D 5207",
            self.negative_selector.setCurrentText,
            tool_tip="Select the camera film stock that is emulated.",
        )

        self.push_pull = Slider()
        """
        How much to push or pull the film, adjusting contrast. Works linearly scaling
        the characteristic curve of the film. Not based on measured data, but a rough
        approximation, useful for controlling contrast.

        Not recommended for use with slide film.
        """
        self.push_pull.setMinMaxTicks(-1.5, 1.5, 1, 20)
        add_option(
            self.push_pull,
            "Push/pull",
            0.0,
            self.push_pull.setValue,
            tool_tip="How much to push or pull the film, adjusting contrast.\n"
            "Works linearly scaling the characteristic curve of the film.\n"
            "Not based on measured data, but a rough approximation useful\n"
            "for controlling contrast.\n"
            "Not recommended for use with slide film.",
        )

        self.color_masking = Slider()
        """
        How effective the orange color mask of the film is. Value of 1 perfectly
        compensates for color layer cross contamination. An increased value leads to
        higher saturation. There is no documented data on this, so you can play around
        with this to your liking.

        For film without a color mask like slide film this can be used to simulate other
        inter-layer effects. Should probably set lower, but should be experimented with.
        """
        self.color_masking.setMinMaxTicks(0, 2, 1, 10, 1)
        self.color_masking.set_color_gradient(
            np.array(
                [
                    0.666,
                    0.0,
                    0.0,
                ]
            ),
            np.array([0.666, 0.25, 2.0]),
            20,
            False,
        )
        add_option(
            self.color_masking,
            "Color masking",
            1.0,
            self.color_masking.setValue,
            tool_tip="How effective the orange color mask of the film is. Value of 1\n"
            "perfectly compensates for color layer cross contamination. An\n"
            "increased value leads to higher saturation. There is no\n"
            "documented data on this, so you can play around with this to\n"
            "your liking.\n"
            "For film without a color mask like slide film this\n"
            "can be used to simulate other inter-layer effects. Should\n"
            "probably set lower, but should be experimented with.",
        )

        luma_bright = 0.8
        luma_dark = 0.4
        chroma = 0.2
        hue_offset = 0.06111111
        self.red_light = Slider()
        """Intensity of the red light during printing."""
        self.red_light.setMinMaxTicks(-0.5, 0.5, 1, 100)
        self.red_light.set_color_gradient(
            np.array([luma_bright, chroma, hue_offset + 0 / 6]),
            np.array([luma_dark, chroma, hue_offset + 3 / 6]),
        )
        add_option(
            self.red_light,
            "Red printer light",
            0,
            self.red_light.setValue,
            tool_tip="Intensity of the red light during printing.",
        )
        self.green_light = Slider()
        """Intensity of the green light during printing."""
        self.green_light.setMinMaxTicks(-0.5, 0.5, 1, 100)
        self.green_light.set_color_gradient(
            np.array([luma_bright, chroma, hue_offset + 2 / 6]),
            np.array([luma_dark, chroma, hue_offset + 5 / 6]),
        )
        add_option(
            self.green_light,
            "Green printer light",
            0,
            self.green_light.setValue,
            tool_tip="Intensity of the green light during printing.",
        )
        self.blue_light = Slider()
        """Intensity of the blue light during printing."""
        self.blue_light.setMinMaxTicks(-0.5, 0.5, 1, 100)
        self.blue_light.set_color_gradient(
            np.array([luma_bright, chroma, hue_offset + 4 / 6]),
            np.array([luma_dark, chroma, hue_offset + 1 / 6]),
        )
        add_option(
            self.blue_light,
            "Blue printer light",
            0,
            self.blue_light.setValue,
            tool_tip="Intensity of the blue light during printing.",
        )

        self.link_lights = QCheckBox()
        """Connect the sliders of all printer lights together."""
        self.link_lights.setChecked(True)
        self.link_lights.setText("link lights")
        add_option(
            self.link_lights,
            tool_tip="Connect the sliders of all printer lights together.",
        )

        print_info = {x: y for x, y in filmstock_info.items() if y["stage"] == "print"}
        print_info["None"] = {}
        print_info["Inversion"] = {}
        sort_keys_print = ["Name", "Year", "Gamma", "D-max"]
        group_keys_print = ["Manufacturer", "Type", "Decade", "Medium"]
        list_keys_print = ["Manufacturer", "Type", "Year", "Chromaticity"]
        sidebar_keys_print = [
            "Alias",
            "Manufacturer",
            "Type",
            "Year",
            "Medium",
            "Chromaticity",
            "Gamma",
            "Comment",
            "D-max",
        ]
        self.print_selector = FilmStockSelector(
            print_info,
            self,
            self,
            sort_keys=sort_keys_print,
            group_keys=group_keys_print,
            list_keys=list_keys_print,
            sidebar_keys=sidebar_keys_print,
            default_group="Manufacturer",
            image_key="image",
        )
        """
        What print material to simulate.
        For slide film it should normally be set to `None`.
        If `None` is selected for negative film a digital scan with a simple
        mathematical inversion is simulated.
        """
        add_option(
            self.print_selector,
            "Print stock",
            "Kodak Vision 2383",
            self.print_selector.setCurrentText,
            tool_tip="What print material to simulate. For slide film it should\n"
            "normally be set to `None`. If `None` is selected for negative\n"
            "film a digital scan with a simple mathematical inversion is\n"
            "simulated.",
        )

        self.projector_kelvin = SliderLog()
        """The color of the projection lamp or the viewing lamp for paper prints."""
        self.projector_kelvin.setMinMaxSteps(2700, 16000, 120, 6500, -2)
        self.projector_kelvin.set_color_gradient(
            np.array([2 / 3, 0.14, 0.15277]), np.array([2 / 3, 0.14, 0.65277])
        )
        add_option(
            self.projector_kelvin,
            "Projector WB",
            6500,
            self.projector_kelvin.setValue,
            tool_tip="The color of the projection lamp or the viewing lamp for paper \n"
            "prints.",
        )

        self.inversion_gamma = Slider()
        """The gamma applied using the inversion if 'Inversion' is selected."""
        self.inversion_gamma.setMinMaxTicks(1, 7, 1, 10)
        add_option(
            self.inversion_gamma,
            "Inversion gamma",
            4.0,
            self.inversion_gamma.setValue,
            tool_tip="The gamma applied using the inversion if 'Inversion' is\n"
            "selected.",
        )

        self.idealized_curve = QCheckBox("Pure curve")
        """
        Replace the characteristic curve of the print film with an ideal gamma curve.
        Preserves the sensitivity and dye densities of the print film.
        When activated, the gamma is controlled by the inversion gamma.
        """
        self.idealized_curve.setToolTip(
            "Replace the characteristic curve of the print film with an ideal gamma\n"
            "curve. Preserves the sensitivity and dye densities of the print film.\n"
            "When activated, the gamma is controlled by the inversion gamma."
        )

        self.white_clip = QCheckBox("Clip")
        """
        When viewing print film brightness will be increased to clip at exactly 1.0.
        When viewing slide film white balancing is applied, so that a gray patch will
        actually produce the color temperature specified by the  projector kelvin.
        """
        self.white_clip.setToolTip(
            "When viewing print film brightness will be increased to clip at\n"
            "exactly 1.0. When viewing slide film white balancing is\n"
            "applied, so that a gray patch will actually produce the color\n"
            "temperature specified by the  projector kelvin."
        )

        self.white_balance = QCheckBox("WB")
        """Whether to white balance slide film."""
        self.white_balance.setToolTip("Whether to white balance slide film.")

        checker_widget = QWidget()
        checker_widget_layout = QHBoxLayout(checker_widget)
        checker_widget.setLayout(QHBoxLayout())
        checker_widget_layout.addWidget(self.idealized_curve)
        checker_widget_layout.addWidget(self.white_clip)
        checker_widget_layout.addWidget(self.white_balance)
        add_option(checker_widget)

        self.sat_adjust = Slider()
        self.sat_adjust.set_color_gradient(
            np.array(
                [
                    0.666,
                    0.0,
                    0.0,
                ]
            ),
            np.array([0.666, 0.25, 2.0]),
            20,
            False,
        )
        self.sat_adjust.setMinMaxTicks(0, 2, 1, 100, 1)
        """A simple post processing saturation slider. Not physically based."""
        add_option(
            self.sat_adjust,
            "Sat",
            1,
            self.sat_adjust.setValue,
            tool_tip="A simple post processing saturation slider. Not physically\n"
            "based.",
        )

        self.shadow_comp = Slider()
        """
        Lift or lower dark areas. For 1 or -1 it acts like an OOTF or inverse OOTF
        respectively.
        """
        self.shadow_comp.setMinMaxTicks(-2, 2, 1, 50)
        add_option(
            self.shadow_comp,
            "Shadow comp.",
            0.0,
            self.shadow_comp.setValue,
            tool_tip="Lift or lower dark areas. For 1 or -1 it acts like an OOTF or\n"
            "inverse OOTF respectively.",
        )
        # TODO: add film emulation based shadow comp, not EOTF based

        self.output_gamut = WideComboBox(self)
        """In what color space to encode the output."""
        self.output_gamut.addItems(COLOR_SPACES.keys())
        add_option(
            self.output_gamut,
            "Output gamut",
            "Rec. 709",
            self.output_gamut.setCurrentText,
            tool_tip="In what color space to encode the output.",
        )

        self.output_gamma = WideComboBox(self)
        """Gamma function to apply for encoding."""
        self.output_gamma.addItems(GAMMA_FUNCTIONS.keys())
        add_option(
            self.output_gamma,
            "Output gamma",
            "Gamma 2.4",
            self.output_gamma.setCurrentText,
            tool_tip="Gamma function to apply for encoding.",
        )

        self.lut_size = Slider()
        """The size of the LUT table."""
        self.lut_size.setMinMaxTicks(2, 67, default=33)
        add_option(
            self.lut_size,
            "LUT size",
            33,
            self.lut_size.setValue,
            tool_tip="The size of the LUT table.",
        )

        self.mode = WideComboBox(self)
        """
        What part of the pipeline to simulate. Using *negative* + *print* in conjunction
        should give the same result as using *full*. *Grain* expects as input the output
        of *negative* and is to be used as a multiplicative intensity scale for a grain
        overlay.
        """
        self.mode.addItems(["full", "negative", "print", "grain"])
        add_option(
            self.mode,
            "Mode",
            "full",
            self.mode.setCurrentText,
            tool_tip="What part of the pipeline to simulate. Using *negative* +\n"
            "*print* in conjunction should give the same result as using\n"
            "*full*. *Grain* expects as input the output of *negative* and\n"
            "is to be used as a multiplicative intensity scale for a grain\n"
            "overlay.",
        )

        self.adx_scale = WideComboBox(self)
        """
        What density does 100% output of the negative LUT respond to. Matching values
        have to be used for print and grain LUTs. No effect when using the full mode.

        Density 2 matches ADX10 and should be used for Cineon like workflows. Density 8
        matches ADX16. When using density 2 there is risk of clipping for some negative
        film stocks.
        """
        self.adx_scale.addItems(["Density 2", "Density 4", "Density 8"])
        add_option(
            self.adx_scale,
            "ADX d-max",
            "Density 2",
            self.adx_scale.setCurrentText,
            tool_tip="What density does 100% output of the negative LUT respond to.\n"
            "Matching values have to be used for print and grain LUTs. No\n"
            "effect when using the full mode.\n"
            "Density 2 matches ADX10 and should be used for Cineon like\n"
            "workflows. Density 8 matches ADX16. When using density 2 there\n"
            "is risk of clipping for some negative film stocks.",
        )

        self.save_lut_button = AnimatedButton("Save LUT")
        """Export the LUT."""
        self.save_lut_button.clicked.connect(self.save_lut)
        add_option(self.save_lut_button, tool_tip="Export the LUT.")

        self.noise_button = AnimatedButton("Export Grain")
        """Open the grain overlay export dialog."""
        self.noise_button.clicked.connect(self.export_noise)
        add_option(self.noise_button, tool_tip="Open the grain overlay export dialog.")

        self.input_colorspace_selector.currentTextChanged.connect(
            self.parameter_changed
        )
        self.negative_selector.currentTextChanged.connect(self.negative_changed)
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

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        self.resize(QSize(1024, 512))

        self.waiting = False
        self.running = False

        self.threadpool = QThreadPool()

    def resizeEvent(self, event):
        self.update_preview()
        super().resizeEvent(event)

    def generate_lut(self, name="temp", cube=True):
        negative_film = self.filmstocks[self.negative_selector.currentText()]
        print_film = self.filmstocks[self.print_selector.currentText()]
        inversion = self.print_selector.currentText() == "Inversion"
        input_colorspace = self.input_colorspace_selector.currentText()
        projector_kelvin = self.projector_kelvin.getValue()
        inversion_gamma = self.inversion_gamma.getValue()
        exp_comp = self.exp_comp.getValue()
        red_light = self.red_light.getValue()
        green_light = self.green_light.getValue()
        blue_light = self.blue_light.getValue()
        if input_colorspace == "CIE XYZ 1931":
            input_colorspace = None
        output_gamut = self.output_gamut.currentText()
        gamma_func = self.output_gamma.currentText()
        if output_gamut == "CIE XYZ 1931":
            output_gamut = None
        size = int(self.lut_size.getValue())
        white_clip = self.white_clip.isChecked()
        white_balance = self.white_balance.isChecked()
        shadow_comp = self.shadow_comp.getValue()
        color_masking = self.color_masking.getValue()
        mode = self.mode.currentText()
        exp_wb = self.exp_wb.getValue()
        tint = self.tint.getValue()
        sat_adjust = self.sat_adjust.getValue()
        push_pull = self.push_pull.getValue()
        idealized_curve = self.idealized_curve.isChecked()

        adx_scaling = {"Density 2": 4.0, "Density 4": 2.0, "Density 8": 1.0}[
            self.adx_scale.currentText()
        ]

        lut = create_lut(
            negative_film,
            print_film,
            name=name,
            lut_size=size,
            cube=cube,
            input_colorspace=input_colorspace,
            output_gamut=output_gamut,
            gamma_func=gamma_func,
            projector_kelvin=projector_kelvin,
            exp_comp=exp_comp,
            white_clip=white_clip,
            white_balance=white_balance,
            exp_kelvin=exp_wb,
            mode=mode,
            red_light=red_light,
            green_light=green_light,
            blue_light=blue_light,
            shadow_comp=shadow_comp,
            color_masking=color_masking,
            tint=tint,
            sat_adjust=sat_adjust,
            adx_scaling=adx_scaling,
            push_pull=push_pull,
            inversion=inversion,
            inversion_gamma=inversion_gamma,
            idealized_curve=idealized_curve,
        )
        return lut

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
            or self.print_selector.currentText() == "Inversion"
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

    def negative_changed(self, negative_film):
        self.color_masking.setValue(self.filmstocks[negative_film].color_masking)
        self.parameter_changed()

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
        bit_depth = 16
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
        lut = (lut * (2**bit_depth - 1)).astype(np.uint)
        image = apply_lut_tetrahedral_int(image, lut, bit_depth=bit_depth)

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
        filename, ok = QFileDialog.getSaveFileName(self)

        if ok:
            self.generate_lut(filename)

    def export_noise(self):
        dialog = ExportGrainDialog(self)
        dialog.exec()
