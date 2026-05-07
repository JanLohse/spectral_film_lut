r"""
2D LUT application in chromaticity space using barycentric interpolation.

This module implements a lookup-table (LUT) transform in normalized chromaticity
coordinates derived from tristimulus values. The transform is suitable for
operators that scale linearly with overall intensity.

We operate in $xyS$ space instead of the conventional $xyY$ formulation for
improved numerical stability when $Y \to 0$:

\begin{align}
    S &= X + Y + Z \\
    x &= X / S \\
    y &= Y / S
\end{align}

Key assumptions and conventions:

- Inputs must be non-negative.
- For $S = 0$, chromaticity is undefined; we define $(x, y) = (0, 0)$ and
  the output is set to zero.
- The LUT is defined over the domain $(x, y, S=1)$, i.e. the unit simplex
  in chromaticity space.
- Interpolation is performed in 2D $(x, y)$ space using triangular
  (barycentric) interpolation within each grid cell.
- The interpolated result is scaled by the original intensity $S$.

LUT format:

- A LUT of resolution $n$ with $c$ output channels must have shape $(n, n, c)$.
- The grid spans $(x, y) \in [0, 1] \times [0, 1]$.
- Values represent the transform evaluated at $(x, y, S=1)$.

Pipeline:

1. Convert input $XYZ \mapsto xyS$.
2. Interpolate LUT values in $(x, y)$.
3. Rescale the result by $S$.
"""

import math

import colour
import numpy as np
from numba import njit, prange
from scipy.optimize import nnls

from spectral_film_lut.config import DEFAULT_DTYPE, SPECTRAL_SHAPE

XYZ_CMFS = np.asarray(
    colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    .align(SPECTRAL_SHAPE)
    .values,
    dtype=DEFAULT_DTYPE,
)
"""The CIE XYZ 1931 color matching functions."""


def xyS_to_XYZ(xyS):
    """Convert from $xyS$ to $XYZ$."""
    h, w, c = xyS.shape
    out = np.empty((h, w, c), dtype=np.float32)

    for j in prange(h):
        for i in prange(w):
            x = xyS[j, i, 0]
            y = xyS[j, i, 1]
            S = xyS[j, i, 2]

            X = x * S
            Y = y * S
            Z = S - X - Y

            out[j, i, 0] = X
            out[j, i, 1] = Y
            out[j, i, 2] = Z

    return out


@njit(parallel=True)
def XYZ_to_xyS(XYZ):
    """Convert from XYZ to xyS."""
    h, w, c = XYZ.shape
    out = np.empty((h, w, c), dtype=np.float32)

    for j in prange(h):
        for i in prange(w):
            X = XYZ[j, i, 0]
            Y = XYZ[j, i, 1]
            Z = XYZ[j, i, 2]

            S = X + Y + Z
            x = X / S
            y = Y / S

            out[j, i, 0] = x
            out[j, i, 1] = y
            out[j, i, 2] = S

    return out


@njit(parallel=True)
def apply_2d_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Apply a 2D lookup table (LUT) in chromaticity space with barycentric interpolation.

    The input is interpreted as tristimulus values $(X, Y, Z)$. Each pixel is
    mapped to $(x, y, S)$ with $S = X + Y + Z$, interpolated in $(x, y)$ using
    the LUT defined at $S = 1$, and finally scaled back by $S$.

    Interpolation is performed per pixel using triangular (barycentric)
    interpolation within each LUT grid cell.

    Args:
        image (np.ndarray):
            Input array of shape (..., 3) containing non-negative $XYZ$ values.
        lut (np.ndarray):
            2D LUT of shape (n, n, c), where n is the grid resolution and
            c is the number of output channels. The LUT encodes values for
            $(x, y, S=1)$ over the domain $[0, 1]^2$.

    Returns:
        ndarray:
            Output array of shape (..., c), where c is the number of LUT channels.

    Notes:
        - Pixels with $S = 0$ produce zero output.
        - The LUT is indexed assuming a uniform grid over $[0, 1]^2$.
        - Interpolation switches between two triangles per grid cell based on
          the fractional position within the cell.
        - No bounds checking is performed; inputs are assumed to map into
          the LUT domain.
    """
    orig_shape = image.shape
    c = orig_shape[-1]  # should be 3

    # flatten spatial dims
    n = 1
    for i in range(len(orig_shape) - 1):
        n *= orig_shape[i]

    image_flat = image.reshape(n, c)

    lut_size = lut.shape[0]
    k = lut.shape[2]  # output channels
    scaling = lut_size - 1

    out_flat = np.empty((n, k), dtype=np.float32)

    for i in prange(n):
        r = image_flat[i, 0]
        g = image_flat[i, 1]
        b = image_flat[i, 2]

        S = r + g + b

        if S < 1e-12:
            for ch in range(k):
                out_flat[i, ch] = 0.0
            continue

        inv_sum = scaling / S

        r *= inv_sum
        g *= inv_sum

        r_ind = int(math.floor(r))
        g_ind = int(math.floor(g))

        r_ind = min(max(r_ind, 0), lut_size - 2)
        g_ind = min(max(g_ind, 0), lut_size - 2)

        r_factor = r % 1
        g_factor = g % 1

        factor_sum = r_factor + g_factor

        if factor_sum <= 1.0:
            s_factor = 1.0 - factor_sum
            for ch in range(k):
                r_val = lut[r_ind + 1, g_ind, ch]
                g_val = lut[r_ind, g_ind + 1, ch]
                S_val = lut[r_ind, g_ind, ch]

                out_flat[i, ch] = (
                    r_val * r_factor + g_val * g_factor + S_val * s_factor
                ) * S
        else:
            s_factor = factor_sum - 1.0
            r_factor2 = 1.0 - g_factor
            g_factor2 = 1.0 - r_factor

            for ch in range(k):
                r_val = lut[r_ind + 1, g_ind, ch]
                g_val = lut[r_ind, g_ind + 1, ch]
                S_val = lut[r_ind + 1, g_ind + 1, ch]

                out_flat[i, ch] = (
                    r_val * r_factor2 + g_val * g_factor2 + S_val * s_factor
                ) * S

    out_shape = orig_shape[:-1] + (k,)
    return out_flat.reshape(out_shape)


def xy_to_spectrum_nnls(xy, loss_factor):
    """
    Get a non-negative spectrum that closely matches an xy pair, using least squares
    regression.
    """
    x, y = xy
    if x + y > 1:
        return np.ones(XYZ_CMFS.shape[0])
    if not y:
        y = 1e-16

    X = x / y
    Z = (1 - x - y) / y
    XYZ = np.array([X, 1, Z])

    A = XYZ_CMFS.T  # (3×N)
    N = A.shape[1]

    # first derivative smoothness
    D1 = np.eye(N, k=0) - np.eye(N, k=1)

    # stack system instead of forming normal equations
    C = np.vstack([A, np.sqrt(loss_factor) * D1])
    d = np.concatenate([XYZ, np.zeros(N)])

    spectrum, _ = nnls(C, d)
    return spectrum


def xyS_to_spectrum_nnls(xyS, loss_factor) -> np.ndarray:
    """
    Get a non-negative spectrum that closely matches an xyS triplet, using least squares
    regression.
    """
    x, y, s = xyS
    spectrum = xy_to_spectrum_nnls((x, y), loss_factor)
    s_spectrum = (spectrum @ XYZ_CMFS).sum()
    spectrum *= s / s_spectrum
    return spectrum


def generate_spectral_sample_table(n, loss_factor):
    """
    Generate a full spectral data table of shape (n, n, SPECTRAL_SHAPE).
    """
    x = np.linspace(0, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    lut_id = np.stack([xx, yy, np.ones_like(xx)], axis=-1)

    flat = lut_id.reshape(-1, 3)

    out_flat = np.array(
        [xyS_to_spectrum_nnls(point, loss_factor).astype(np.float32) for point in flat]
    )

    out = out_flat.reshape(n, n, -1)

    return out


SPECTRUM_LUT = generate_spectral_sample_table(33, loss_factor=5)
"""A look-up table with spectral distributions across the CIE 1931 xy space."""
