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


def xyS_to_XYZ(xyS: np.ndarray) -> np.ndarray:
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
def XYZ_to_xyS(XYZ: np.ndarray) -> np.ndarray:
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
    c = orig_shape[-1]

    n = 1
    for i in range(len(orig_shape) - 1):
        n *= orig_shape[i]

    image_flat = image.reshape(n, c)

    lut_size = lut.shape[0]
    k = lut.shape[2]
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
RAWTOACES = RAWTOACES.values.T

ILLUMINANT_KEYS = [
    "A",
    "D50",
    "D55",
    "D65",
    "D75",
    "FL2",
    "FL7",
    "FL11",
    "LED-B1",
    "LED-B3",
    "LED-B5",
    "LED-V1",
]


def _prepare_training_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Generates globally stacked, S-normalized training spectra and chromaticity
    coordinates.
    """
    raw_spectra_list = []
    for key in ILLUMINANT_KEYS:
        blackbody_spd = np.asarray(
            colour.SDS_ILLUMINANTS[key].align(SPECTRAL_SHAPE).values,
            dtype=DEFAULT_DTYPE,
        )
        raw_spectra_list.append(RAWTOACES * blackbody_spd)

    all_spectra = np.vstack(raw_spectra_list)
    all_xyz = all_spectra @ XYZ_CMFS
    all_s = np.sum(all_xyz, axis=-1, keepdims=True)

    all_spectra_norm = all_spectra / all_s
    all_xy = all_xyz[:, :2] / all_s

    return all_xy, all_spectra_norm


_ALL_XY, _ALL_SPECTRA_NORM = _prepare_training_data()


def build_binned_triangulation(grid_res: int) -> tuple[Delaunay, np.ndarray]:
    """
    Bins globally stacked S-normalized spectra into a 2D grid matching resolution
    grid_res.
    """
    scale = grid_res - 1

    x_indices = np.clip(np.floor(_ALL_XY[:, 0] * scale).astype(int), 0, grid_res - 2)
    y_indices = np.clip(np.floor(_ALL_XY[:, 1] * scale).astype(int), 0, grid_res - 2)
    bin_keys = y_indices * scale + x_indices

    unique_bins, inverse_indices = np.unique(bin_keys, return_inverse=True)
    n_unique = len(unique_bins)

    # Fast vectorized bin accumulation
    counts = np.bincount(inverse_indices)
    reduced_xy = np.zeros((n_unique, 2), dtype=np.float32)
    reduced_spectra = np.zeros((n_unique, _ALL_SPECTRA_NORM.shape[1]), dtype=np.float32)

    np.add.at(reduced_xy, inverse_indices, _ALL_XY)
    np.add.at(reduced_spectra, inverse_indices, _ALL_SPECTRA_NORM)

    reduced_xy /= counts[:, None]
    reduced_spectra /= counts[:, None]

    return Delaunay(reduced_xy), reduced_spectra


def get_barycentric_prior(
    xy: tuple[float, float],
    triangulation: Delaunay,
    reduced_spectra: np.ndarray,
    cmfs: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    """
    Computes zero-drift XYZ matrix-solved prior spectrum using the binned Delaunay mesh.
    """
    simplex_idx = triangulation.find_simplex(xy)
    if simplex_idx < 0:
        return None, False

    vertex_indices = triangulation.simplices[simplex_idx]
    x, y = xy
    target_XYZ = np.array([x, y, 1.0 - x - y], dtype=np.float32)

    corner_spectra = reduced_spectra[vertex_indices]
    corner_XYZs = corner_spectra @ cmfs
    corner_S = np.where(
        np.sum(corner_XYZs, axis=-1, keepdims=True) < 1e-12,
        1e-12,
        np.sum(corner_XYZs, axis=-1, keepdims=True),
    )

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

    prior_spectrum = weights @ (corner_spectra / corner_S)

    prior_y = prior_spectrum @ cmfs[:, 1]
    if prior_y > 1e-12:
        prior_spectrum /= prior_y

    return prior_spectrum, True


def xy_to_spectrum_nnls(
    xy: tuple[float, float],
    smoothness_loss_factor: float,
    data_loss_factor: float,
    cmfs: np.ndarray,
    triangulation: Delaunay,
    reduced_spectra: np.ndarray,
) -> np.ndarray:
    """Finds a non-negative spectrum guided by the binned multi-illuminant prior."""
    x, y = xy
    if x + y > 1 or y <= 0:
        return np.ones(cmfs.shape[0], dtype=np.float32)

    X = x / y
    Z = (1 - x - y) / y
    XYZ_target = np.array([X, 1.0, Z], dtype=np.float32)

    A = cmfs.T
    n_bins = A.shape[1]
    D1 = np.eye(n_bins, k=0) - np.eye(n_bins, k=1)

    prior_spectrum, has_prior = get_barycentric_prior(
        xy, triangulation, reduced_spectra, cmfs
    )

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
    n: int, smoothness_loss_factor: float = 1.0, data_loss_factor: float = 2.5
) -> np.ndarray:
    """Generate the full spectral lookup table across the unit simplex grid."""
    triangulation, reduced_spectra = build_binned_triangulation(grid_res=n)

    grid_coords = np.linspace(0, 1, n, dtype=np.float32)
    n_bins = XYZ_CMFS.shape[0]
    out = np.empty((n, n, n_bins), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            xy = (float(grid_coords[i]), float(grid_coords[j]))

            spec = xy_to_spectrum_nnls(
                xy,
                smoothness_loss_factor,
                data_loss_factor,
                XYZ_CMFS,
                triangulation,
                reduced_spectra,
            )

            s_spectrum = (spec @ XYZ_CMFS).sum()
            if s_spectrum > 1e-12:
                spec /= s_spectrum
            else:
                spec = np.zeros(n_bins, dtype=np.float32)

            out[i, j, :] = spec

    return out


SPECTRUM_LUT: np.ndarray = generate_spectral_sample_table(33)
"""A look-up table with spectral distributions across the CIE 1931 xy space."""
