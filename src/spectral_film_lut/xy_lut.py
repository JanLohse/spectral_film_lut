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


@njit(parallel=True)
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


# Global setup configuration
LUT_SIZE = 33
SCALING = LUT_SIZE - 1
TEMPERATURES = ["A", "D65"]


def setup_static_matrices():
    """Compute base dimensions and normalized matrices for NNLS optimization."""
    a_raw = XYZ_CMFS.T  # Shape: (3, N_BINS)
    n_bins = a_raw.shape[1]
    d1_raw = np.eye(n_bins, k=0) - np.eye(n_bins, k=1)
    i_raw = np.eye(n_bins)

    norm_a = np.linalg.norm(a_raw)
    norm_d1 = np.linalg.norm(d1_raw)
    norm_i = np.linalg.norm(i_raw)

    return (
        n_bins,
        a_raw,
        norm_a,
        norm_i,
        a_raw / norm_a,
        d1_raw / norm_d1,
        i_raw / norm_i,
    )


# Extract global variables from static setup
(
    _N_BINS,
    _A_RAW,
    _NORM_A,
    _NORM_I,
    A_STATIC_SCALED,
    D1_STATIC_SCALED,
    I_STATIC_SCALED,
) = setup_static_matrices()


def compute_prior_spectra():
    """Generate and splat target illuminant datasets into a normalized prior grid."""
    rawtoaces = colour.characterisation.read_training_data_rawtoaces_v1()
    rawtoaces.align(SPECTRAL_SHAPE)
    rawtoaces_values = rawtoaces.values.T

    # Generate and stack the illuminant-reflected spectra
    all_training_runs = []
    for T in TEMPERATURES:
        blackbody_T = colour.SDS_ILLUMINANTS[T].align(SPECTRAL_SHAPE).values
        spectra_at_T = rawtoaces_values * blackbody_T
        all_training_runs.append(spectra_at_T)

    training_spectra = np.vstack(all_training_runs)

    # Allocate grids for barycentric splatting
    splat_spectra_grid = np.zeros((LUT_SIZE, LUT_SIZE, _N_BINS), dtype=np.float32)
    splat_weight_grid = np.zeros((LUT_SIZE, LUT_SIZE, 1), dtype=np.float32)

    # Populate the prior grid via barycentric splatting
    for spectrum in training_spectra:
        xyz = spectrum @ _A_RAW.T
        S = np.sum(xyz)
        if S < 1e-12:
            continue

        # Scale coordinates up to grid index space
        r = (xyz[0] / S) * SCALING
        g = (xyz[1] / S) * SCALING

        r_ind = min(max(int(math.floor(r)), 0), LUT_SIZE - 2)
        g_ind = min(max(int(math.floor(g)), 0), LUT_SIZE - 2)

        r_factor = r % 1
        g_factor = g % 1
        factor_sum = r_factor + g_factor

        if factor_sum <= 1.0:
            s_factor = 1.0 - factor_sum
            splat_spectra_grid[r_ind + 1, g_ind] += spectrum * r_factor
            splat_weight_grid[r_ind + 1, g_ind, 0] += r_factor

            splat_spectra_grid[r_ind, g_ind + 1] += spectrum * g_factor
            splat_weight_grid[r_ind, g_ind + 1, 0] += g_factor

            splat_spectra_grid[r_ind, g_ind] += spectrum * s_factor
            splat_weight_grid[r_ind, g_ind, 0] += s_factor
        else:
            s_factor = factor_sum - 1.0
            r_factor2 = 1.0 - g_factor
            g_factor2 = 1.0 - r_factor

            splat_spectra_grid[r_ind + 1, g_ind] += spectrum * r_factor2
            splat_weight_grid[r_ind + 1, g_ind, 0] += r_factor2

            splat_spectra_grid[r_ind, g_ind + 1] += spectrum * g_factor2
            splat_weight_grid[r_ind, g_ind + 1, 0] += g_factor2

            splat_spectra_grid[r_ind + 1, g_ind + 1] += spectrum * s_factor
            splat_weight_grid[r_ind + 1, g_ind + 1, 0] += s_factor

    # Average and normalize accumulated entries
    mask = splat_weight_grid[:, :, 0] > 0
    prior_spectra = np.zeros_like(splat_spectra_grid)
    prior_spectra[mask] = splat_spectra_grid[mask] / splat_weight_grid[mask]

    for i in range(LUT_SIZE):
        for j in range(LUT_SIZE):
            if mask[i, j]:
                spec = prior_spectra[i, j]
                Y_val = _A_RAW[1] @ spec
                if Y_val > 0:
                    prior_spectra[i, j] = (spec / Y_val) / _NORM_I

    return training_spectra, prior_spectra, mask


# Generate the shared prior dependencies
RAWTOACES_TRAINING_SPECTRA, PRIOR_SPECTRA, mask = compute_prior_spectra()


def xy_grid_to_spectrum(
    grid_indices,
    xy,
    smoothness_loss_factor: float = 0.1,
    data_loss_factor: float = 5.0,
):
    x, y = xy
    if x + y > 1 or y <= 0:
        return np.ones(_N_BINS)

    # Map to XYZ target
    X = x / y
    Z = (1 - x - y) / y
    XYZ_scaled = np.array([X, 1, Z]) / _NORM_A

    # Look up pre-splat data using current grid coordinate
    grid_r, grid_j = grid_indices
    ref_scaled = PRIOR_SPECTRA[grid_r, grid_j]
    has_prior = mask[grid_r, grid_j]

    weight_smooth = np.sqrt(smoothness_loss_factor)
    weight_dataset = np.sqrt(data_loss_factor) if has_prior else 0.0

    # Stack target system
    C = np.vstack(
        [
            A_STATIC_SCALED,
            weight_smooth * D1_STATIC_SCALED,
            weight_dataset * I_STATIC_SCALED,
        ]
    )

    d = np.concatenate([XYZ_scaled, np.zeros(_N_BINS), weight_dataset * ref_scaled])

    spectrum, _ = nnls(C, d)
    return spectrum


def generate_spectral_sample_table(
    n, smoothness_loss_factor: float = 0.1, data_loss_factor: float = 5.0
):
    x = np.linspace(0, 1, n, dtype=np.float32)
    out = np.empty((n, n, _N_BINS), dtype=np.float32)

    # Map grid coordinates to target spectral estimates
    for i in range(n):
        for j in range(n):
            xy = (x[i], x[j])
            spec = xy_grid_to_spectrum(
                (i, j), xy, smoothness_loss_factor, data_loss_factor
            )

            s_spectrum = (spec @ XYZ_CMFS).sum()
            if s_spectrum > 0:
                spec *= 1.0 / s_spectrum

            out[i, j, :] = spec.astype(np.float32)

    return out


SPECTRUM_LUT = generate_spectral_sample_table(
    33, smoothness_loss_factor=0.1, data_loss_factor=5.0
)
"""A look-up table with spectral distributions across the CIE 1931 xy space."""
