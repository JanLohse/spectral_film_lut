"""Scene-input controls for the main window."""

import numpy as np
from colour.models import RGB_COLOURSPACES

from spectral_film_lut.ui.widgets.controls import FileSelector, Slider, SliderLog, WideComboBox


def build_preview_and_source_panel(window, left_layout, add_option) -> None:
    """Build the preview label and scene-input controls."""
    from PyQt6.QtCore import QSize, Qt
    from PyQt6.QtWidgets import QLabel, QSizePolicy

    window.image = QLabel("Select a reference image for the preview")
    window.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
    window.image.setSizePolicy(
        QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    )
    window.image.setMinimumSize(QSize(256, 256))
    left_layout.addWidget(window.image)

    window.image_selector = FileSelector()
    add_option(
        window.image_selector,
        "Reference image",
        tool_tip="Select an image encoded in the target LUT color space.",
    )

    colorspaces = ["CIE XYZ 1931"] + list(RGB_COLOURSPACES.data.keys())
    window.input_colorspace_selector = WideComboBox()
    window.input_colorspace_selector.addItems(colorspaces)
    add_option(
        window.input_colorspace_selector,
        "Input colorspace",
        "ARRI Wide Gamut 4",
        window.input_colorspace_selector.setCurrentText,
        tool_tip="What color space is the reference image in?",
    )

    window.exp_comp = Slider()
    window.exp_comp.setMinMaxTicks(-2, 2, 1, 6)
    add_option(
        window.exp_comp,
        "Exposure",
        0,
        window.exp_comp.setValue,
        tool_tip="Compensate exposure; middle gray follows the color-space convention.",
    )

    window.exp_wb = SliderLog()
    window.exp_wb.setMinMaxSteps(2700, 16000, 120, 6500, -2)
    window.exp_wb.set_color_gradient(
        np.array([2 / 3, 0.14, 0.65277]), np.array([2 / 3, 0.14, 0.15277])
    )
    add_option(
        window.exp_wb,
        "WB",
        6500,
        window.exp_wb.setValue,
        tool_tip="Adjust the scene white balance in kelvin.",
    )

    window.tint = Slider()
    window.tint.setMinMaxTicks(-1, 1, 1, 100)
    window.tint.set_color_gradient(
        np.array([2 / 3, 0.14, 0.90277]), np.array([2 / 3, 0.14, 0.40277])
    )
    add_option(
        window.tint,
        "Tint",
        0,
        window.tint.setValue,
        tool_tip="Adjust tint along the green to red/purple axis.",
    )
