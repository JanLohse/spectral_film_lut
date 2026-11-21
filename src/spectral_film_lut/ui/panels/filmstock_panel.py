"""Film-stock and print-control panel."""

import numpy as np
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QWidget

from spectral_film_lut.ui.filmstock_selector import FilmStockSelector
from spectral_film_lut.ui.metadata import build_filmstock_info
from spectral_film_lut.ui.widgets.controls import Slider, SliderLog


def build_filmstock_panel(window, add_option) -> None:
    """Build camera-stock, grading, printer-light, and print-stock controls."""
    filmstock_info = build_filmstock_info(window.filmstocks)

    negative_info = {
        name: info for name, info in filmstock_info.items() if info["stage"] == "camera"
    }
    window.negative_selector = FilmStockSelector(
        negative_info,
        window,
        window,
        sort_keys=["Name", "Year", "Resolution", "Granularity", "sensitivity", "Gamma", "D-max"],
        group_keys=["Manufacturer", "Type", "Decade", "Medium"],
        list_keys=["Manufacturer", "Type", "Year", "Sensitivity", "Chromaticity"],
        sidebar_keys=[
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
        ],
        default_group="Manufacturer",
        image_key="image",
    )
    add_option(
        window.negative_selector,
        "Negative stock",
        "Kodak Vision3 250D 5207",
        window.negative_selector.setCurrentText,
        tool_tip="Select the camera film stock that is emulated.",
    )

    window.push_pull = Slider()
    window.push_pull.setMinMaxTicks(-1.5, 1.5, 1, 20)
    add_option(
        window.push_pull,
        "Push/pull",
        0.0,
        window.push_pull.setValue,
        tool_tip="Adjust characteristic-curve contrast; not recommended for slide film.",
    )

    window.dye_inhibition = Slider()
    window.dye_inhibition.setMinMaxTicks(0, 1, 1, 20, 0.5)
    window.dye_inhibition.set_color_gradient(
        np.array([0.666, 0.0, 0.0]), np.array([0.666, 0.25, 2.0]), 20, False
    )
    add_option(
        window.dye_inhibition,
        "Dye inhibition",
        0.5,
        window.dye_inhibition.setValue,
        tool_tip="Strength of inter-layer dye inhibition; 0 disables it.",
    )

    window.color_masking = Slider()
    window.color_masking.setMinMaxTicks(0, 1, 1, 20, 0.5)
    window.color_masking.set_color_gradient(
        np.array([0.666, 0.0, 0.0]), np.array([0.666, 0.25, 2.0]), 20, False
    )
    add_option(
        window.color_masking,
        "Color masking",
        0.5,
        window.color_masking.setValue,
        tool_tip="Intensity of color masking; 1 compensates full layer cross-contamination.",
    )

    hue_offset = 0.06111111
    window.red_light = Slider(continuous=False)
    window.red_light.setMinMaxTicks(-0.5, 0.5, 1, 100)
    window.red_light.set_color_gradient(
        np.array([0.8, 0.2, hue_offset]), np.array([0.4, 0.2, hue_offset + 0.5])
    )
    add_option(window.red_light, "Red printer light", 0, window.red_light.setValue)

    window.green_light = Slider(continuous=False)
    window.green_light.setMinMaxTicks(-0.5, 0.5, 1, 100)
    window.green_light.set_color_gradient(
        np.array([0.8, 0.2, hue_offset + 2 / 6]),
        np.array([0.4, 0.2, hue_offset + 5 / 6]),
    )
    add_option(window.green_light, "Green printer light", 0, window.green_light.setValue)

    window.blue_light = Slider(continuous=False)
    window.blue_light.setMinMaxTicks(-0.5, 0.5, 1, 100)
    window.blue_light.set_color_gradient(
        np.array([0.8, 0.2, hue_offset + 4 / 6]),
        np.array([0.4, 0.2, hue_offset + 1 / 6]),
    )
    add_option(window.blue_light, "Blue printer light", 0, window.blue_light.setValue)

    window.link_lights = QCheckBox("link lights")
    window.link_lights.setChecked(True)
    add_option(window.link_lights, tool_tip="Link the printer-light sliders.")

    print_info = {name: info for name, info in filmstock_info.items() if info["stage"] == "print"}
    print_info.update({"None": {}, "Inversion (ACES like)": {}})
    window.filmstocks["None"] = None
    window.filmstocks["Inversion (ACES like)"] = None
    window.print_selector = FilmStockSelector(
        print_info,
        window,
        window,
        sort_keys=["Name", "Year", "Gamma", "D-max"],
        group_keys=["Manufacturer", "Type", "Decade", "Medium"],
        list_keys=["Manufacturer", "Type", "Year", "Chromaticity"],
        sidebar_keys=[
            "Alias",
            "Manufacturer",
            "Type",
            "Year",
            "Medium",
            "Chromaticity",
            "Gamma",
            "Comment",
            "D-max",
        ],
        default_group="Manufacturer",
        image_key="image",
    )
    add_option(
        window.print_selector,
        "Print stock",
        "Kodak Vision 2383",
        window.print_selector.setCurrentText,
        tool_tip="Select the print stock or an inversion mode.",
    )

    window.projector_kelvin = SliderLog()
    window.projector_kelvin.setMinMaxSteps(2700, 16000, 120, 6500, -2)
    window.projector_kelvin.set_color_gradient(
        np.array([2 / 3, 0.14, 0.15277]), np.array([2 / 3, 0.14, 0.65277])
    )
    add_option(
        window.projector_kelvin,
        "Projector WB",
        6500,
        window.projector_kelvin.setValue,
        tool_tip="Projection or viewing illuminant in kelvin.",
    )

    window.inversion_gamma = Slider()
    window.inversion_gamma.setMinMaxTicks(1, 5, 1, 10)
    add_option(
        window.inversion_gamma,
        "Inversion gamma",
        3.0,
        window.inversion_gamma.setValue,
        tool_tip="Gamma used by digital inversion modes.",
    )

    window.idealized_curve = QCheckBox("Pure curve")
    window.idealized_curve.setToolTip(
        "Replace the print characteristic curve with an ideal gamma curve."
    )

    window.white_clip = QCheckBox("Clip")
    window.white_clip.setToolTip("Scale print-film brightness to clip at exactly 1.0.")

    window.white_balance = QCheckBox("WB")
    window.white_balance.setTristate(True)
    window.white_balance.setToolTip("Slide-film white balance mode.")

    checker_widget = QWidget()
    checker_layout = QHBoxLayout(checker_widget)
    checker_layout.addWidget(window.idealized_curve)
    checker_layout.addWidget(window.white_clip)
    checker_layout.addWidget(window.white_balance)
    add_option(checker_widget)
