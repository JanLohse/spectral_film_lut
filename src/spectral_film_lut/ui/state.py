"""Collect LUT-generation settings from the main window."""

from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import Qt


@dataclass(frozen=True)
class LUTSettings:
    """Complete values needed to generate or preview a LUT."""

    negative_film: Any
    print_film: Any
    lut_kwargs: dict[str, Any]

    def create_kwargs(self, name: str, cube: bool) -> dict[str, Any]:
        """Build keyword arguments for `create_lut`."""
        lut_kwargs = self.lut_kwargs.copy()
        return {
            "negative_film": self.negative_film,
            "print_film": self.print_film,
            "name": name,
            "lut_size": lut_kwargs.pop("lut_size"),
            "cube": cube,
            **lut_kwargs,
        }


def collect_lut_settings(window) -> LUTSettings:
    """Read the current controls and return complete LUT settings."""
    input_colorspace = window.input_colorspace_selector.currentText()
    if input_colorspace == "CIE XYZ 1931":
        input_colorspace = None

    output_gamut = window.output_gamut.currentText()
    if output_gamut == "CIE XYZ 1931":
        output_gamut = None

    white_balance = {
        Qt.CheckState.Unchecked: 0,
        Qt.CheckState.PartiallyChecked: 1,
        Qt.CheckState.Checked: 2,
    }[window.white_balance.checkState()]

    adx_scaling = {
        "Density 2": 4.0,
        "Density 4": 2.0,
        "Density 8": 1.0,
    }[window.adx_scale.currentText()]

    filter_mode = {
        "Oversampling": "oversampling",
        "Shift": "shift",
    }.get(window.filter_mode.currentText())

    print_name = window.print_selector.currentText()
    lut_kwargs = {
        "input_colorspace": input_colorspace,
        "output_gamut": output_gamut,
        "gamma_func": window.output_gamma.currentText(),
        "projector_kelvin": window.projector_kelvin.getValue(),
        "exp_comp": window.exp_comp.getValue(),
        "white_clip": window.white_clip.isChecked(),
        "white_balance": white_balance,
        "exp_kelvin": window.exp_wb.getValue(),
        "mode": window.mode.currentText(),
        "red_light": window.red_light.getValue(),
        "green_light": window.green_light.getValue(),
        "blue_light": window.blue_light.getValue(),
        "shadow_comp": window.shadow_comp.getValue(),
        "dye_inhibition": window.dye_inhibition.getValue(),
        "color_masking": window.color_masking.getValue(),
        "tint": window.tint.getValue(),
        "sat_adjust": window.sat_adjust.getValue(),
        "adx_scaling": adx_scaling,
        "push_pull": window.push_pull.getValue(),
        "inversion": "Inversion" in print_name,
        "inversion_gamma": window.inversion_gamma.getValue(),
        "idealized_curve": window.idealized_curve.isChecked(),
        "apd_intermediate": window.apd_intermediate.isChecked(),
        "filter_mode": filter_mode,
        "upsampling_method": window.spectral_upsampling_method.currentText(),
        "reference_negative": window.filmstocks["Kodak Vision3 250D 5207"],
        "lut_size": int(window.lut_size.getValue()),
    }
    return LUTSettings(
        window.filmstocks[window.negative_selector.currentText()],
        window.filmstocks[print_name],
        lut_kwargs,
    )
