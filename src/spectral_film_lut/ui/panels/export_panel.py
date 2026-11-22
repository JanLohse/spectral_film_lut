"""Output and LUT export controls."""

from typing import get_args

import numpy as np
from PyQt6.QtWidgets import QCheckBox

from spectral_film_lut.core.color_space import COLOR_SPACES, GAMMA_FUNCTIONS
from spectral_film_lut.core.xy_lut import UpsampleMethod
from spectral_film_lut.ui.widgets.controls import AnimatedButton, ModeSelector, Slider, WideComboBox


def build_export_panel(window, add_option) -> None:
    """Build post-processing, LUT, mode, and export controls."""
    window.sat_adjust = Slider()
    window.sat_adjust.set_color_gradient(
        np.array([0.666, 0.0, 0.0]),
        np.array([0.666, 0.25, 2.0]),
        20,
        False,
    )
    window.sat_adjust.setMinMaxTicks(0, 2, 1, 100, 1)
    add_option(window.sat_adjust, "Sat", 1, window.sat_adjust.setValue)

    window.shadow_comp = Slider()
    window.shadow_comp.setMinMaxTicks(-1, 1, 1, 50)
    add_option(
        window.shadow_comp,
        "Shadow comp.",
        0.0,
        window.shadow_comp.setValue,
        tool_tip="Lift or lower shadows; 1 and -1 act as OOTF and inverse OOTF.",
    )

    window.output_gamut = WideComboBox(window)
    window.output_gamut.addItems(COLOR_SPACES.keys())
    add_option(
        window.output_gamut,
        "Output gamut",
        "Rec. 709",
        window.output_gamut.setCurrentText,
        tool_tip="Color space used to encode the output.",
    )

    window.output_gamma = WideComboBox(window)
    window.output_gamma.addItems(GAMMA_FUNCTIONS.keys())
    add_option(
        window.output_gamma,
        "Output gamma",
        "Gamma 2.4",
        window.output_gamma.setCurrentText,
        tool_tip="Gamma or transfer function used for display output.",
    )

    window.lut_size = Slider()
    window.lut_size.setMinMaxTicks(2, 67, default=33)
    add_option(window.lut_size, "LUT size", 33, window.lut_size.setValue)

    window.mode = ModeSelector(window, items=["full", "negative", "print", "grain"])
    window.mode.setToolTip(
        "Select full emulation or an intermediate negative, print, or grain output."
    )

    window.adx_scale = WideComboBox(window)
    window.adx_scale.addItems(["Density 2", "Density 4", "Density 8"])
    add_option(
        window.adx_scale,
        "ADX d-max",
        "Density 2",
        window.adx_scale.setCurrentText,
        tool_tip="Maximum output density for ADX intermediate LUTs.",
    )

    window.apd_intermediate = QCheckBox()
    add_option(
        window.apd_intermediate,
        "APD intermediate",
        True,
        window.apd_intermediate.setChecked,
        tool_tip="Encode intermediate negative values as Academy Printing Density.",
    )

    window.filter_mode = WideComboBox(window)
    window.filter_mode.addItems(["No filtering", "Oversampling", "Shift"])
    add_option(
        window.filter_mode,
        "LUT filter",
        "No filtering",
        window.filter_mode.setCurrentText,
        tool_tip="Select oversampling or offset-grid median filtering.",
    )

    window.spectral_upsampling_method = WideComboBox(window)
    window.spectral_upsampling_method.addItems(get_args(UpsampleMethod))
    add_option(
        window.spectral_upsampling_method,
        "Spectral upsampling",
        "SFL upsampling",
        window.spectral_upsampling_method.setCurrentText,
        tool_tip="Method used to reconstruct spectra from CIE XYZ.",
    )

    window.save_lut_button = AnimatedButton("Save LUT")
    window.save_lut_button.clicked.connect(window.save_lut)
    add_option(window.save_lut_button, tool_tip="Export the LUT.")

    window.noise_button = AnimatedButton("Export Grain")
    window.noise_button.clicked.connect(window.export_noise)
    add_option(window.noise_button, tool_tip="Open the grain-overlay export dialog.")
