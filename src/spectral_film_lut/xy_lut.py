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
from scipy.spatial import Delaunay

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


RAWTOACES = colour.characterisation.read_training_data_rawtoaces_v1()
RAWTOACES.align(SPECTRAL_SHAPE)
TRANSMITTANCE_TRAINING = np.asarray(RAWTOACES.values.T, dtype=DEFAULT_DTYPE)

TEMPERATURES = ["A", "A", "D50", "D55", "D60", "D65", "D75"]

# Build multi-illuminant training sets and Delaunay trees
ILLUMINANT_SPECTRA = []
ILLUMINANT_TRIANGULATIONS = []

for T in TEMPERATURES:
    # Load and align illuminant SPD
    blackbody_T = np.asarray(
        colour.SDS_ILLUMINANTS[T].align(SPECTRAL_SHAPE).values,
        dtype=DEFAULT_DTYPE,
    )

    # Modulate transmittance by the source illuminant spectrum
    spectra_at_T = TRANSMITTANCE_TRAINING * blackbody_T
    ILLUMINANT_SPECTRA.append(spectra_at_T)

    # Project to 2D xy space to build the Delaunay structure for this illuminant
    xyz_samples = spectra_at_T @ XYZ_CMFS
    s_samples = np.sum(xyz_samples, axis=-1, keepdims=True)
    s_samples = np.where(s_samples < 1e-12, 1e-12, s_samples)
    xy_samples = xyz_samples[:, :2] / s_samples

    ILLUMINANT_TRIANGULATIONS.append(Delaunay(xy_samples))


def get_single_illuminant_prior(
    xy: tuple[float, float],
    triangulation,
    training_spectra: np.ndarray,
    cmfs: np.ndarray,
):
    """Computes the zero-drift XYZ matrix-solved prior for a single illuminant run."""
    simplex_idx = triangulation.find_simplex(xy)
    if simplex_idx < 0:
        return None, False

    vertex_indices = triangulation.simplices[simplex_idx]
    x, y = xy
    target_XYZ = np.array([x, y, 1.0 - x - y], dtype=np.float32)

    corner_spectra = training_spectra[vertex_indices]
    corner_XYZs = corner_spectra @ cmfs
    corner_S = np.sum(corner_XYZs, axis=-1, keepdims=True)
    corner_S = np.where(corner_S < 1e-12, 1e-12, corner_S)

    normalized_corner_XYZs = corner_XYZs / corner_S
    M = normalized_corner_XYZs.T

    try:
        weights = np.linalg.solve(M, target_XYZ)
    except np.linalg.LinAlgError:
        transform = triangulation.transform[simplex_idx]
        delta = xy - transform[2]
        c1_c2 = transform[:2].dot(delta)
        c3 = 1.0 - np.sum(c1_c2)
        weights = np.append(c1_c2, c3)

    weights = np.clip(weights, 0.0, 1.0)
    weights /= np.sum(weights)

    normalized_corner_spectra = corner_spectra / corner_S
    prior_spectrum = weights @ normalized_corner_spectra
    return prior_spectrum, True


def get_mixed_illuminant_prior(xy: tuple[float, float], cmfs: np.ndarray):
    """
    Evaluates the target coordinate against each illuminant dataset independently,
    then combines the zero-drift prior distributions.
    """
    valid_priors = []

    for triangulation, training_spectra in zip(
        ILLUMINANT_TRIANGULATIONS, ILLUMINANT_SPECTRA
    ):
        prior, inside_gamut = get_single_illuminant_prior(
            xy, triangulation, training_spectra, cmfs
        )
        if inside_gamut:
            # Normalize to S=1 space before mixing to keep scaling consistent
            prior_s = np.sum(prior @ cmfs)
            if prior_s > 1e-12:
                valid_priors.append(prior / prior_s)

    if not valid_priors:
        return None, False

    # Mix the valid priors together evenly
    mixed_prior = np.mean(valid_priors, axis=0)

    # Scale back to Y=1 standard normalization for the NNLS target matrix
    prior_y = mixed_prior @ cmfs[:, 1]
    if prior_y > 1e-12:
        mixed_prior /= prior_y

    return mixed_prior, True


def xy_to_spectrum_nnls(
    xy: tuple[float, float],
    smoothness_loss_factor: float,
    data_loss_factor: float,
    cmfs: np.ndarray,
):
    """Finds a non-negative spectrum guided by the mixed multi-illuminant prior."""
    x, y = xy
    if x + y > 1 or y <= 0:
        return np.ones(cmfs.shape[0], dtype=np.float32)

    X = x / y
    Z = (1 - x - y) / y
    XYZ_target = np.array([X, 1.0, Z], dtype=np.float32)

    A = cmfs.T
    n_bins = A.shape[1]
    D1 = np.eye(n_bins, k=0) - np.eye(n_bins, k=1)

    # Resolve mixed dynamic multi-illuminant prior
    prior_spectrum, has_prior = get_mixed_illuminant_prior(xy, cmfs)

    w_smooth = np.sqrt(smoothness_loss_factor)
    w_data = np.sqrt(data_loss_factor) if has_prior else 0.0

    C = np.vstack([A, w_smooth * D1, w_data * np.eye(n_bins)])

    if has_prior:
        d = np.concatenate([XYZ_target, np.zeros(n_bins), w_data * prior_spectrum])
    else:
        d = np.concatenate([XYZ_target, np.zeros(n_bins), np.zeros(n_bins)])

    spectrum, _ = nnls(C, d)
    return spectrum


def generate_spectral_sample_table(
    n, smoothness_loss_factor: float = 1.0, data_loss_factor: float = 1.0
):
    """Generate the full spectral lookup table across the unit simplex grid."""
    grid_coords = np.linspace(0, 1, n, dtype=np.float32)
    n_bins = XYZ_CMFS.shape[0]
    out = np.empty((n, n, n_bins), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            xy = (grid_coords[i], grid_coords[j])

            spec = xy_to_spectrum_nnls(
                xy, smoothness_loss_factor, data_loss_factor, XYZ_CMFS
            )

            # Final normalization to unity scale sum (s = X + Y + Z = 1)
            s_spectrum = (spec @ XYZ_CMFS).sum()
            if s_spectrum > 1e-12:
                spec /= s_spectrum
            else:
                spec = np.zeros(n_bins, dtype=np.float32)

            out[i, j, :] = spec

    return out


# Generate the data table using exact barycentric priors
SPECTRUM_LUT = generate_spectral_sample_table(
    33, smoothness_loss_factor=1.0, data_loss_factor=1.0
)
"""A look-up table with spectral distributions across the CIE 1931 xy space."""
