"""
Color processing transforms not directly related to film.
"""

import colour
import numpy as np

from spectral_film_lut.color_space import COLOR_SPACES, GAMMA_FUNCTIONS
from spectral_film_lut.config import DEFAULT_DTYPE
from spectral_film_lut.xy_lut import apply_2d_lut, xyS_to_XYZ


def CCT_to_xy(CCT):
    """Convert from a color temperature in kelvin to the closest xy pair."""
    CCT_3 = CCT**3
    CCT_2 = CCT**2

    if CCT <= 7000:
        x = (
            -4.607 * 10**9 / CCT_3
            + 2.9678 * 10**6 / CCT_2
            + 0.09911 * 10**3 / CCT
            + 0.244063
        )
    else:
        x = (
            -2.0064 * 10**9 / CCT_3
            + 1.9018 * 10**6 / CCT_2
            + 0.24748 * 10**3 / CCT
            + 0.23704
        )

    y = -3.000 * x**2 + 2.870 * x - 0.275
    return np.array([x, y])


COLORCHECKER_2005 = np.array(
    [
        [0.3457, 0.3585, 100.0],
        [0.4316, 0.3777, 10.08],
        [0.4197, 0.3744, 34.95],
        [0.2760, 0.3016, 18.36],
        [0.3703, 0.4499, 13.25],
        [0.2999, 0.2856, 23.04],
        [0.2848, 0.3911, 41.78],
        [0.5295, 0.4055, 31.18],
        [0.2305, 0.2106, 11.26],
        [0.5012, 0.3273, 19.38],
        [0.3319, 0.2482, 6.37],
        [0.3984, 0.5008, 44.46],
        [0.4957, 0.4427, 43.57],
        [0.2018, 0.1692, 5.75],
        [0.3253, 0.5032, 23.18],
        [0.5686, 0.3303, 12.57],
        [0.4697, 0.4734, 59.81],
        [0.4159, 0.2688, 20.09],
        [0.2131, 0.3023, 19.30],
        [0.3469, 0.3608, 91.31],
        [0.3440, 0.3584, 58.94],
        [0.3432, 0.3581, 36.32],
        [0.3446, 0.3579, 19.15],
        [0.3401, 0.3548, 8.83],
        [0.3406, 0.3537, 3.11],
    ],
    DEFAULT_DTYPE,
)
"""The XYZ value for the colorchecker 2005 data."""
COLORCHECKER_2005 = colour.xyY_to_XYZ(COLORCHECKER_2005)
COLORCHECKER_2005 *= np.array([0.95047, 1.00000, 1.08883]) / COLORCHECKER_2005[0]
COLORCHECKER_2005 = COLORCHECKER_2005[1:]
COLORCHECKER_2005 = np.asarray(COLORCHECKER_2005, DEFAULT_DTYPE)


def gamut_compression(image: np.ndarray, strength=0.95):
    """
    A simple gamut compression that limits the maximal relative distance from the
    achromatic. Inspired by ACES Reference Gamut Compression. Has been simplified
    to be color space agnostic and use a simple clipping function instead of a
    roll-off. It is not perceptually neutral and should be used conservatively. It
    is intended to fix numeric issues in highly saturated colors, and not to be part
    of look creation.

    Args:
        image: The image to transform.
        strength: How strong to compress. strength=1 is uncompressed and strength=0
            is fully desaturated.

    Returns:
        The compressed image.
    """
    # Get achromatic value
    a = image.max(axis=-1, keepdims=True)

    # Compute and limit distance from achromatic value.
    d = np.where(a > 0, (a - image) / np.abs(a), 0)
    d = np.clip(d, 0, strength)

    # Reconstruct image with limited distance, resulting in limited saturation.
    image = a - d * np.abs(a)

    return image


def output_color_transform(
    image, output_gamut, sat_adjust: float, lut_size=33
) -> np.ndarray:
    """
    Transform from XYZ to the target gamut and adjust the saturation.
    Saturation adjustment is performed in OkLch for visual uniformity.
    Gamut compression is performed to avoid visual artifacts.
    To preserve high performance the transforms are applied as a 2D xy LUT.

    Args:
        image: The XYZ image to transform.
        output_gamut: The name of the output color space.
        sat_adjust: By what factor to multiply the saturation.
        lut_size: The size of the LUT. Default is 33x33.

    Returns:
        The transformed image in the output gamut.
    """
    # initialize 2.5D XYZ LUT
    x = np.linspace(0, 1, lut_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    lut_id = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    lut_XYZ = xyS_to_XYZ(lut_id)

    # adjust saturation
    if sat_adjust != 1:
        lut_oklab = colour.XYZ_to_Oklab(lut_XYZ)
        lut_oklab[..., 1:3] *= sat_adjust
        lut_XYZ = colour.Oklab_to_XYZ(lut_oklab)

    # convert to output color space
    if output_gamut != "CIE XYZ":
        lut_XYZ @= COLOR_SPACES[output_gamut].xyz_to_rgb.T

    # compress gamut
    lut_XYZ = gamut_compression(lut_XYZ)

    # apply to image safely
    image = np.clip(image, 0, None)
    image = apply_2d_lut(image, lut_XYZ)
    image = np.clip(image, 0, 1)

    return image


def shadow_compensation(image: np.ndarray, intensity) -> np.ndarray:
    """
    Raises or lowers shadows. Has been computed to act as an OOTF for the ITU-R
    BT.1886 curve. Setting gamma to Gamma 2.4 and Black offset to 1.0 will yield
    essentially Rec. 709. Setting gamma to Rec. 709 and Black offset to -1.0 in turn
    gives essentially Gamma 2.4. Has been computed from BT.1886 as to not be
    piecewise, and to not overfit to Rec. 709.

    Args:
        image: The image to transform. Assumed to be in linear gamma.
        intensity: How much to lift or lower particularly dark areas. For 0 no
            effect, 1 and -1 act as forward and inverse OOTFs respectively.
        gamma: The assumed viewing gamma for the OOTF modeling.
        black_level: The assumed viewing black level for OOTF modeling.

    Returns:
        The shadow compensated image.
    """
    # Make the intensity less sensitive close to 0.
    intensity *= abs(intensity)

    # black of 1.2% and gamma 2.4 chosen to match Rec. 709 closely
    black = 0.012 * abs(intensity)
    gamma = 2.4

    # a and b from ITU-R BT.1886
    a = (1 - black ** (1 / gamma)) ** gamma
    b = (black ** (1 / gamma)) / (1 - black ** (1 / gamma))

    if intensity > 0:
        image = (a * (image ** (1 / gamma) + b) ** gamma - black) / (1 - black)
    elif intensity < 0:
        image = (((image * (1 - black) + black) / a) ** (1 / gamma) - b) ** gamma

    return image


def output_transform(
    image: np.ndarray,
    output_gamut,
    sat_adjust: float,
    lut_size=33,
    shadow_comp=0.0,
    gamma_func="Gamma 2.4",
):
    image = output_color_transform(image, output_gamut, sat_adjust, lut_size)
    if shadow_comp:
        image = shadow_compensation(image, shadow_comp)
    image = GAMMA_FUNCTIONS[gamma_func](image)

    return image


def CCT_to_XYZ(CCT: float | int, Y=1.0, tint=0.0):
    """Converts from a color temperature in kelvin to a XYZ triplet."""
    xy = CCT_to_xy(CCT)
    xyY = (xy[0], xy[1], Y)
    XYZ = colour.xyY_to_XYZ(xyY)
    Lab = colour.XYZ_to_Oklab(XYZ)
    Lab += np.array([0, 0.9849548, -0.17281227]) * tint / 15
    XYZ = colour.Oklab_to_XYZ(Lab)
    return XYZ
