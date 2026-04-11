"""Define and convert to common display color spaces."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass
class ColorSpace:
    """Store color space gamut data and provide CIE XYZ conversion matrices."""

    w_x: float
    w_y: float
    r_x: float
    r_y: float
    g_x: float
    g_y: float
    b_x: float
    b_y: float

    @cached_property
    def rgb_to_xyz(self) -> np.ndarray:
        return self._compute_rgb_to_xyz()

    @cached_property
    def xyz_to_rgb(self) -> np.ndarray:
        return np.linalg.inv(self.rgb_to_xyz)

    def _compute_rgb_to_xyz(self) -> np.ndarray:
        def xy_to_xyz(x, y):
            return np.array([x / y, 1.0, (1 - x - y) / y])

        Xr, Yr, Zr = xy_to_xyz(self.r_x, self.r_y)
        Xg, Yg, Zg = xy_to_xyz(self.g_x, self.g_y)
        Xb, Yb, Zb = xy_to_xyz(self.b_x, self.b_y)

        M = np.array(
            [
                [Xr, Xg, Xb],
                [Yr, Yg, Yb],
                [Zr, Zg, Zb],
            ]
        )

        Xw, Yw, Zw = xy_to_xyz(self.w_x, self.w_y)
        S = np.linalg.solve(M, np.array([Xw, Yw, Zw]))
        return M * S


COLOR_SPACES = {
    "Rec. 709": ColorSpace(0.3127, 0.3290, 0.64, 0.33, 0.3, 0.6, 0.15, 0.06),
    "Display P3": ColorSpace(0.3127, 0.3290, 0.68, 0.32, 0.2651, 0.69, 0.15, 0.06),
    "Rec. 2020": ColorSpace(0.3127, 0.329, 0.708, 0.292, 0.17, 0.797, 0.131, 0.046),
    "ACES AP1": ColorSpace(0.32168, 0.33767, 0.713, 0.293, 0.165, 0.830, 0.128, 0.044),
    "ACES AP0": ColorSpace(0.32168, 0.33767, 0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.0770),
    "CIE XYZ": None,
    "DCI-P3": ColorSpace(0.314, 0.351, 0.68, 0.32, 0.2651, 0.69, 0.15, 0.06),
    "DCI-P3 D60": ColorSpace(0.32168, 0.33767, 0.68, 0.32, 0.2651, 0.69, 0.15, 0.06),
}
"""The Default output color spaces."""


def rec_709_encoding(x):
    """The Rec. 709 OETF."""
    return np.where(x < 0.018, 4.5 * x, 1.099 * x**0.45 - 0.099)


def srgb_encoding(x):
    """The sRGB OETF."""
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def hlg_encoding(x, peak=203):
    """
    HLG (Hybrid Log-Gamma) OETF as per ARIB STD-B67
    """
    a = 0.17883277
    b = 0.28466892
    c = 0.55991073
    x *= peak / 1000
    return np.where(x <= 1 / 12, np.sqrt(3 * x), a * np.log(12 * x - b) + c)


def pq_encoding(x, peak=203):
    """PQ (SMPTE ST 2084) OETF"""
    m1 = 2610 / 16384
    m2 = 2523 / 32
    c1 = 107 / 128
    c2 = 2413 / 128
    c3 = 2392 / 128
    x *= peak / 10000
    x = np.clip(x, 0, 1)
    return ((c1 + c2 * x**m1) / (1 + c3 * x**m1)) ** m2


GAMMA_FUNCTIONS = {
    "Linear": lambda x: x,
    "Gamma 1.8": lambda x: x ** (1 / 1.8),
    "Gamma 2.0": lambda x: x ** (1 / 2.0),
    "Gamma 2.2": lambda x: x ** (1 / 2.2),
    "Gamma 2.4": lambda x: x ** (1 / 2.4),
    "Gamma 2.6": lambda x: x ** (1 / 2.6),
    "Rec. 709": rec_709_encoding,
    "sRGB": srgb_encoding,
    "HLG": hlg_encoding,
    "PQ": pq_encoding,
}
"""Different gamma functions for output encoding."""
