"""
Additional utility functions.
"""

import os
import time

import colour
import numpy as np
from numba import njit, prange

from spectral_film_lut.color_processing import (
    CCT_to_XYZ,
    output_transform,
)
from spectral_film_lut.config import DEFAULT_DTYPE
from spectral_film_lut.xy_lut import apply_2d_lut


def film_conversion(
    image,
    negative_film,
    print_film=None,
    mode="full",  # TODO: literal
    input_colorspace: None | str = None,  # TODO: literal
    exposure_kelvin=6500,
    tint=0.0,
    exp_comp=0.0,
    color_masking: None | float = None,
    adx_scaling=1.0,
    adx_coding=True,
    red_light=0.0,
    green_light=0.0,
    blue_light=0.0,
    projector_kelvin=6500,
    white_comp=True,
    output_gamut="Rec. 709",  # TODO: literal
    sat_adjust=1.0,
    shadow_comp=0.0,
    gamma_func="Gamma 2.4",  # TODO: literal
):
    image = np.ascontiguousarray(image)
    if mode == "negative" or mode == "full":
        # TODO: turn to own function
        if input_colorspace is not None:
            image = colour.RGB_to_XYZ(image, input_colorspace, apply_cctf_decoding=True)

        input_lut = negative_film.get_input_lut(exposure_kelvin, tint, exp_comp)

        image = apply_2d_lut(np.clip(image, 0, None), input_lut)

        # TODO: merge with log_exposure_to_density to single 1D LUT
        image = np.log10(np.clip(image, 10**-16, None))

        image = negative_film.log_exposure_to_density(image, color_masking)

    if adx_coding and mode == "negative":
        image = negative_film.adx_encoding(image, adx_scaling, color_masking)

    if adx_coding and (mode == "print" or mode == "grain"):
        image = negative_film.adx_decoding(image, adx_scaling, color_masking)

    if mode == "print" or mode == "full":
        # TODO: turn to own function
        if print_film is not None:
            output_film = print_film
            if negative_film.density_measure == print_film.density_measure == "bw":
                image = (
                    -image + print_film.log_H_ref + negative_film.d_ref + green_light
                )
            else:
                density_neg = negative_film.get_spectral_density(color_masking)
                printer_light = negative_film.compute_printer_light(
                    print_film, red_light, green_light, blue_light
                )
                printing_mat = (
                    print_film.sensitivity.T
                    * printer_light
                    * 10**-negative_film.d_min_sd
                ).T
                printing_mat = printing_mat.reshape(-1, 3, printing_mat.shape[-1]).sum(
                    axis=1
                )
                density_neg = density_neg.reshape(-1, 3, density_neg.shape[-1]).mean(
                    axis=1
                )
                image = np.log10(
                    np.clip(
                        10 ** -(image @ density_neg.T) @ printing_mat, 0.00001, None
                    )
                )

            image = print_film.log_exposure_to_density(image)
        else:
            output_film = negative_film

        if output_film.density_measure == "bw":
            image = 1 / 10**-output_film.d_min * 10**-image

            if not 6500 <= projector_kelvin <= 6505:
                image = image * np.asarray(CCT_to_XYZ(projector_kelvin))
        else:
            if output_film.density_measure == "status_a" and print_film is None:
                white_comp = False
            projection_light, xyz_cmfs = output_film.compute_projection_light(
                projector_kelvin=projector_kelvin, white_comp=white_comp
            )
            d_min_sd = output_film.d_min_sd
            if print_film is None:
                density_mat = output_film.get_spectral_density(color_masking)
            else:
                density_mat = output_film.get_spectral_density()

            output_mat = (xyz_cmfs.T * projection_light * 10**-d_min_sd).T
            output_mat = output_mat.reshape(-1, 3, 3).sum(axis=1)
            density_mat = density_mat.reshape(-1, 3, 3).mean(axis=1)

            image = 10 ** -(image @ density_mat.T) @ output_mat
            # TODO: return functionality
            # if (
            #     output_film.density_measure == "status_a"
            #     and print_film is None
            #     and white_balance
            # ):
            #     mid_gray = pipeline[-1][0](output_film.get_d_ref(color_masking))
            #     out_gray = np.asarray(CCT_to_XYZ(output_kelvin, mid_gray[1]))
            #     output_mat = np.asarray(
            #         colour.chromatic_adaptation(output_mat, mid_gray, out_gray)
            #     )

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
    lut_size=33,
    name="test",
    cube=True,
    verbose=False,
    **kwargs,
):
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
    x,
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
    image: np.ndarray, lut, bit_depth=16, out_bit_depth=8
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
