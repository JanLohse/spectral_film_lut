"""
Additional utility functions.
"""

import os
import time
import warnings

import colour
import numpy as np
from numba import njit, prange
from scipy import ndimage

warnings.filterwarnings("ignore", category=FutureWarning)


DEFAULT_DTYPE = np.float32
"""The dtype used for the color pipeline."""
colour.utilities.set_default_float_dtype(DEFAULT_DTYPE)
SPECTRAL_SHAPE = colour.SpectralShape(380, 780, 5)
"""The wavelengths used for all spectral simulations and data."""


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
    transform = negative_film.generate_conversion(negative_film, print_film, **kwargs)
    start = time.time()
    table = transform(lut.table)
    if table.shape[-1] == 1:
        table = table.repeat(3, -1)
    if not cube:
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


def construct_spectral_density(
    ref_density: colour.SpectralDistribution, sigma=25
) -> np.ndarray:
    """Split single density curve into separate layers using local extrema."""
    red_peak = wavelength_argmax(ref_density, 600, min(750, SPECTRAL_SHAPE.end))
    green_peak = wavelength_argmax(ref_density, 500, 600)
    blue_peak = wavelength_argmax(ref_density, max(400, SPECTRAL_SHAPE.start), 500)
    bg_cutoff = wavelength_argmin(ref_density, blue_peak, green_peak)
    gr_cutoff = wavelength_argmin(ref_density, green_peak, red_peak)

    wavelengths = np.asarray(ref_density.wavelengths)
    factors = np.stack(
        (
            np.where(gr_cutoff <= wavelengths, 1.0, 0.0),
            np.where((bg_cutoff < wavelengths) & (wavelengths < gr_cutoff), 1.0, 0.0),
            np.where(wavelengths <= bg_cutoff, 1.0, 0.0),
        )
    )
    factors = ndimage.gaussian_filter(
        factors, sigma=(0, sigma / SPECTRAL_SHAPE.interval)
    ).astype(DEFAULT_DTYPE)

    out = (factors * np.asarray(ref_density.values)).T
    return out


def wavelength_argmax(
    distribution: colour.SpectralDistribution, low=None, high=None
) -> int:
    """Gets the argmax of a spectral distribution."""
    range = distribution.copy()
    if low is not None and high is not None:
        range.trim(colour.SpectralShape(low, high, 1))
    peak = range.wavelengths[range.values.argmax()]
    return peak


def wavelength_argmin(
    distribution: colour.SpectralDistribution, low=None, high=None
) -> int:
    """Gets the argmax of a spectral distribution."""
    range = distribution.copy()
    if low is not None and high is not None:
        range.trim(colour.SpectralShape(low, high, 1))
    peak = range.wavelengths[range.values.argmin()]
    return peak


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
