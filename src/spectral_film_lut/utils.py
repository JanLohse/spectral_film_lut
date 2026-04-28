"""
Additional utility functions.
"""

import os
import time
from typing import Literal

import colour
import numpy as np
from colour.hints import LiteralRGBColourspace
from numba import njit, prange

from spectral_film_lut.color_processing import CCT_to_XYZ, output_transform
from spectral_film_lut.color_space import COLOR_SPACE_KEYS, GAMMA_KEYS
from spectral_film_lut.config import DEFAULT_DTYPE

MODES = Literal["full", "print", "negative", "grain"]
"""Conversion modes, i.e., what steps of the pipeline to activate."""


def film_conversion(
    image: np.ndarray,
    negative_film,
    print_film=None,
    mode: MODES = "full",
    input_colorspace: LiteralRGBColourspace | None = None,
    exp_kelvin: int | float = 6500,
    tint: float = 0.0,
    exp_comp: float = 0.0,
    color_masking: None | float = None,
    adx_coding: bool = True,
    adx_scaling: float | int = 1.0,
    red_light: float = 0.0,
    green_light: float = 0.0,
    blue_light: float = 0.0,
    projector_kelvin: int | float = 6500,
    white_comp: bool = True,
    white_balance: bool = True,
    output_gamut: COLOR_SPACE_KEYS = "Rec. 709",
    sat_adjust: float = 1.0,
    shadow_comp: float = 0.0,
    gamma_func: GAMMA_KEYS = "Gamma 2.4",
    push_pull: float = 0.0,
    inversion: bool = False,
) -> np.ndarray:
    """
    Emulates the full film pipeline including exposure, printing, and projection.

    Args:
        image: The image (or LUT) containing the scene referred data.
        negative_film: The film stock to capture the scene.
        print_film: The optional print stock.
        mode: Which part of the pipeline to emulate.
        input_colorspace: The colorspace if the image. If None CIE XYZ is used.
        exp_kelvin: The scene WB in kelvin.
        tint: The tint adjustment on the green -- red/purple axis.
        exp_comp: Exposure compensation in stops.
        color_masking: How strong the color mask is on the negative film. 0 for no mask.
            1 for a perfectly optimized mask (no pollution of other layers). >1 for
            increased saturation. Most film stocks don't provide exact data. For films
            with orange mask can be close to 1, for other (e.g. slide film) should be
            set quite low.
        adx_scaling: When ADX encoding is used for negative/print/grain stages, what is
            by what factor to divide the output. For 1.0 we have pure ADX16 with a max
            density of 8, and for 4.0 we have ADX10 with a max density of 2.
        adx_coding: Whether to activate ADX encoding for partial pipelines. Ensures
            values stay within [0, 1] range. Necessary for save LUT generation.
        red_light: Offset of the red printer light from neutral.
        green_light: Offset of the green printer light from neutral.
        blue_light: Offset of the blue printer light from neutral.
        projector_kelvin: The white balance in kelvin of the projection or viewing
            illuminant.
        white_comp: Whether to adjust the output brightness that it clips at exactly 1.
        white_balance: Whether to white balance slide film.
        output_gamut: The gamut of the output color space.
        sat_adjust: A saturation adjustment factor applied in OkLab. 1.0 for neutral,
            0.0 for monochrome, and >1.0 for increased saturation.
        shadow_comp: Shadow compensation to smoothly brighten or dark areas without
            clipped details. Function is based on the ITU Bt.1886 function and acts like
            a OOTF or inverse OOTF function for values of 1.0 and -1.0.
        gamma_func: The gamma function the output is encoded with.
        push_pull: By how many stops to push/pull the negative to adjust contrast.
        inversion: Apply inversion to the final output.

    Returns:
        An image (or LUT) representing the transformed scene data after (partial) film
        emulation.
    """
    image = np.ascontiguousarray(image)

    if mode == "negative" or mode == "full":
        image = negative_film.input_transform(
            image,
            input_colorspace,
            exp_comp,
            exp_kelvin,
            tint,
            color_masking,
            push_pull,
        )

    if adx_coding and mode == "negative":
        image = negative_film.adx_encoding(image, adx_scaling, color_masking)

    if adx_coding and (mode == "print" or mode == "grain"):
        image = negative_film.adx_decoding(image, adx_scaling, color_masking)

    if mode == "print" or mode == "full":
        if print_film is not None:
            output_film = print_film
            image = negative_film.print_to(
                image, print_film, color_masking, red_light, green_light, blue_light
            )
            color_masking = None
        else:
            output_film = negative_film

        if inversion:
            projector_kelvin = 6500
            white_balance = False

        image, out_gray = output_film.project(
            image,
            projector_kelvin,
            color_masking,
            white_comp,
            white_balance and print_film is None,
        )

        if inversion:
            target_gray = 0.18
            if len(out_gray) == 3:
                target_gray = np.asarray(CCT_to_XYZ(6500, target_gray))
            exponent = np.log10(target_gray) / np.log10(1 - out_gray)
            image = (1 - image) ** exponent

        image = output_transform(
            image,
            output_gamut,
            sat_adjust=sat_adjust,
            shadow_comp=shadow_comp,
            gamma_func=gamma_func,
        )

    if mode == "grain":
        image = negative_film.grain_transform(image, std_div=0.001, scale=adx_scaling)

    return image


def create_lut(
    negative_film,
    print_film=None,
    lut_size: int = 33,
    name: str = "test",
    cube: bool = True,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray | str:
    """
    Creates a cube LUT from using `.film_spectral.FilmSpectral.generate_conversion`.
    """
    lut = colour.LUT3D(size=lut_size, name="test")
    start = time.time()
    table = film_conversion(lut.table, negative_film, print_film, **kwargs)
    if table.shape[-1] == 1:
        table = table.repeat(3, -1)
    if cube:
        lut.table = table
    else:
        return table
    end = time.time()
    path = f"{name}.cube"
    if not os.path.exists("../../LUTs"):
        os.makedirs("../../LUTs")
    colour.io.write_LUT(lut, path)
    if verbose:
        print(f"created {path} in {end - start:.2f} seconds")
    return path


def multi_channel_interp(
    x: np.ndarray,
    xps,
    fps,
    num_bins=1024,
    interpolate=False,
    left_extrapolate=False,
    right_extrapolate=False,
):
    """
    Resamples each (xp, fp) pair to a uniform grid for fast lookup.

    Returns:
        xp_common: np.ndarray, shape (num_bins,)
        fp_uniform: np.ndarray, shape (n_channels, num_bins)
    """
    n_channels = len(xps)
    xp_min = min(x[0] for x in xps)
    xp_max = max(x[-1] for x in xps)
    xp_common = np.linspace(xp_min, xp_max, num_bins).astype(DEFAULT_DTYPE)

    fp_uniform = np.empty((n_channels, num_bins), dtype=DEFAULT_DTYPE)
    for ch in range(n_channels):
        fp_uniform[ch] = np.interp(xp_common, xps[ch], fps[ch])
    return uniform_multi_channel_interp(
        x, xp_common, fp_uniform, interpolate, left_extrapolate, right_extrapolate
    )


@njit
def uniform_multi_channel_interp(
    x: np.ndarray,
    xp_common: np.ndarray,
    fp_uniform: np.ndarray,
    interpolate=True,
    left_extrapolate=False,
    right_extrapolate=False,
) -> np.ndarray:
    """Interpolate values in an N-D array over the last dimension.

    Interpolates values in an N-dimensional array ``x`` across the last
    dimension (channels) using a precomputed uniform grid defined by
    ``xp_common`` and ``fp_uniform``. Optionally performs linear
    interpolation or nearest-neighbor selection, and supports linear
    extrapolation on either end of the grid.

    Args:
        x : Input array of shape ``(..., channels)``.
        xp_common: Monotonically increasing 1D array of shape
            ``(num_bins,)`` representing the shared grid points.
        fp_uniform: Array of shape ``(channels, num_bins)``
            containing function values at each grid point for every channel.
        interpolate: If ``True``, perform linear interpolation.
            If ``False``, use nearest-neighbor selection.
        left_extrapolate: If ``True``, apply linear extrapolation
            for values less than ``xp_common[0]``.
        right_extrapolate: If ``True``, apply linear extrapolation
            for values greater than ``xp_common[-1]``.

    Returns:
        Interpolated array with the same shape as ``x``.
    """
    shape = x.shape
    ndim = len(shape)
    n_channels = shape[ndim - 1]

    flat_size = 1
    for i in range(ndim - 1):
        flat_size *= shape[i]

    num_bins = xp_common.shape[0]
    xp_min = xp_common[0]
    xp_max = xp_common[-1]
    bin_width = (xp_max - xp_min) / (num_bins - 1)

    x_contig = np.ascontiguousarray(x)
    result = np.empty_like(x_contig, dtype=DEFAULT_DTYPE)

    x_flat = x_contig.reshape(flat_size, n_channels)
    r_flat = result.reshape(flat_size, n_channels)

    for idx in range(flat_size):
        for ch in range(n_channels):
            xi = x_flat[idx, ch]
            if xi <= xp_min:
                if left_extrapolate:
                    x0 = xp_common[0]
                    x1 = xp_common[1]
                    y0 = fp_uniform[ch, 0]
                    y1 = fp_uniform[ch, 1]
                    slope = (y1 - y0) / (x1 - x0)
                    r_flat[idx, ch] = y0 + slope * (xi - x0)
                else:
                    r_flat[idx, ch] = fp_uniform[ch, 0]
            elif xi >= xp_max:
                if right_extrapolate:
                    x0 = xp_common[-2]
                    x1 = xp_common[-1]
                    y0 = fp_uniform[ch, -2]
                    y1 = fp_uniform[ch, -1]
                    slope = (y1 - y0) / (x1 - x0)
                    r_flat[idx, ch] = y1 + slope * (xi - x1)
                else:
                    r_flat[idx, ch] = fp_uniform[ch, -1]
            else:
                pos = (xi - xp_min) / bin_width
                i = int(pos)
                if interpolate:
                    f = pos - i
                    y0 = fp_uniform[ch, i]
                    y1 = fp_uniform[ch, i + 1]
                    r_flat[idx, ch] = y0 + f * (y1 - y0)
                else:
                    r_flat[idx, ch] = fp_uniform[ch, i]

    return result


@njit(parallel=True)
def apply_lut_tetrahedral_int(
    image: np.ndarray, lut: np.ndarray, bit_depth: int = 16, out_bit_depth: int = 8
) -> np.ndarray:
    """Apply a 3D LUT using tetrahedral interpolation.

    The input image is expected to have dtype ``uint16`` with values in
    the range ``[0, 65535]``. The output image has dtype ``uint8`` with
    values in the range ``[0, 255]``.

    Args:
        image: Input image of shape ``(H, W, 3)``,
            dtype ``uint16``.
        lut: 3D lookup table of shape
            ``(size, size, size, 3)``, dtype ``uint8``.

    Returns:
        Output image of shape ``(H, W, 3)``, dtype ``uint8``.
    """
    h, w, c = image.shape
    size = lut.shape[0]
    max_value = 2**bit_depth - 1
    scale = max_value // (size - 1)
    scale_out = scale * 2 ** (bit_depth - out_bit_depth)

    out = np.empty((h, w, 3), dtype=np.uint8)

    for y in prange(h):
        for x in prange(w):
            r = image[y, x, 0]
            g = image[y, x, 1]
            b = image[y, x, 2]

            r0 = r // scale
            g0 = g // scale
            b0 = b // scale

            dr = r % scale
            dg = g % scale
            db = b % scale

            r1 = min(r0 + 1, size - 1)
            g1 = min(g0 + 1, size - 1)
            b1 = min(b0 + 1, size - 1)

            # Fetch cube corners
            c000 = lut[r0, g0, b0]
            c100 = lut[r1, g0, b0]
            c010 = lut[r0, g1, b0]
            c001 = lut[r0, g0, b1]
            c110 = lut[r1, g1, b0]
            c101 = lut[r1, g0, b1]
            c011 = lut[r0, g1, b1]
            c111 = lut[r1, g1, b1]

            # Tetrahedral interpolation
            if dr >= dg:
                if dg >= db:
                    c = (
                        c000 * scale
                        + dr * (c100 - c000)
                        + dg * (c110 - c100)
                        + db * (c111 - c110)
                    )
                elif dr >= db:
                    c = (
                        c000 * scale
                        + dr * (c100 - c000)
                        + db * (c101 - c100)
                        + dg * (c111 - c101)
                    )
                else:
                    c = (
                        c000 * scale
                        + db * (c001 - c000)
                        + dr * (c101 - c001)
                        + dg * (c111 - c101)
                    )
            else:
                if db >= dg:
                    c = (
                        c000 * scale
                        + db * (c001 - c000)
                        + dg * (c011 - c001)
                        + dr * (c111 - c011)
                    )
                elif db >= dr:
                    c = (
                        c000 * scale
                        + dg * (c010 - c000)
                        + db * (c011 - c010)
                        + dr * (c111 - c011)
                    )
                else:
                    c = (
                        c000 * scale
                        + dg * (c010 - c000)
                        + dr * (c110 - c010)
                        + db * (c111 - c110)
                    )

            # Convert back to uint8 safely
            if out_bit_depth == 8:
                out[y, x, 0] = np.uint8(c[0] // scale_out)
                out[y, x, 1] = np.uint8(c[1] // scale_out)
                out[y, x, 2] = np.uint8(c[2] // scale_out)
            else:
                out[y, x, 0] = np.uint16(c[0] // scale_out)
                out[y, x, 1] = np.uint16(c[1] // scale_out)
                out[y, x, 2] = np.uint16(c[2] // scale_out)

    return out
