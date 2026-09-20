r"""
2D LUT application in chromaticity space using barycentric interpolation and spectral
upsampling.

This module provides tools for transforming trichromatic colors (XYZ / xyS) into
high-dimensional spectral distributions across targeted wavelength bins, applying
high-performance 2D lookup tables in normalized chromaticity space, and caching
generated lookup tables on demand.

Supported Spectral Upsampling Algorithms:
------------------------------------------
1. SFL upsampling:
    A custom hybrid method combining empirical spectral data with constrained
    optimization. It builds a 2D Delaunay triangulation mesh in chromaticity space from
    training spectra (RawToACES dataset multiplied across 12 standard illuminants) to
    derive a barycentric prior spectrum, which is then refined using Non-Negative Least
    Squares (NNLS) with smoothness and data regularizers.

2. Pure NNLS Optimization:
    Reconstructs spectra by solving a non-negative least-squares problem matching target
    XYZ values, constrained purely by a 1D first-derivative smoothness loss penalty
    without relying on dataset priors.

3. Simple Matrix:
    A fast linear least-squares model (`XYZ -> Spectrum`) fitted to D65-illuminated
    spectral datasets. Reconstructed spectra are clamped to non-negative bounds and
    re-normalized to unit chromaticity sum S.

4. Mallett 2019:
    Utilizes `colour.XYZ_to_sd` implementation of the Mallett et al. (2019) algorithm.
    Generates smooth, physically plausible reflectance spectra designed for computer
    graphics and color management.

5. Otsu 2018:
    Utilizes `colour.XYZ_to_sd` implementation of the Otsu et al. (2018) algorithm.
    Constructs smooth non-negative spectra optimized for wide-gamut RGB/XYZ inputs.

6. Smits 1999:
    Utilizes `colour.XYZ_to_sd` implementation of Smits' (1999) classical algorithm.
    Reconstructs spectra using pre-calculated smooth basis curves for primary,
    secondary, and white colors.
"""

import functools
import math
from collections.abc import Callable
from typing import Literal

import colour
import numpy as np
from numba import njit, prange
from scipy.optimize import nnls
from scipy.spatial import Delaunay

from spectral_film_lut.config import DEFAULT_DTYPE, SPECTRAL_SHAPE

# Color Matching Functions (CMFs)

XYZ_CMFS = np.asarray(
    colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    .align(SPECTRAL_SHAPE)
    .values,
    dtype=DEFAULT_DTYPE,
)


_RAWTOACES = colour.characterisation.read_training_data_rawtoaces_v1()
_RAWTOACES.align(SPECTRAL_SHAPE)
_RAWTOACES = _RAWTOACES.values.T
RAWTOACES_XYZ = _RAWTOACES @ XYZ_CMFS


# Color Space Conversion & LUT Application Functions


def xyS_to_XYZ(xyS: np.ndarray) -> np.ndarray:
    """Convert from xyS to XYZ."""
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
    """Apply a 2D lookup table in chromaticity space with barycentric interpolation."""
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


# Lazy Data Loading Helpers


ALL_ILLUMINANT_KEYS: tuple[str, ...] = (
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
)


@functools.lru_cache(maxsize=8)
def _get_training_data(
    illuminants: tuple[str, ...], return_xyz: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Lazy loader and generator for spectral training datasets."""
    rawtoaces = colour.characterisation.read_training_data_rawtoaces_v1()
    rawtoaces.align(SPECTRAL_SHAPE)
    rawtoaces_data = rawtoaces.values.T

    raw_spectra_list = []
    for key in illuminants:
        spd = np.asarray(
            colour.SDS_ILLUMINANTS[key].align(SPECTRAL_SHAPE).values,
            dtype=DEFAULT_DTYPE,
        )
        raw_spectra_list.append(rawtoaces_data * spd)

    all_spectra = np.vstack(raw_spectra_list)
    all_xyz = all_spectra @ XYZ_CMFS

    if return_xyz:
        return all_xyz, all_spectra

    all_s = np.sum(all_xyz, axis=-1, keepdims=True)

    all_spectra_norm = all_spectra / all_s
    all_xy = all_xyz[:, :2] / all_s

    return all_xy, all_spectra_norm


def build_binned_triangulation(
    all_xy: np.ndarray, all_spectra_norm: np.ndarray, grid_res: int
) -> tuple[Delaunay, np.ndarray]:
    """Bins training spectra into grid resolution and creates Delaunay triangulation."""
    scale = grid_res - 1

    x_indices = np.clip(np.floor(all_xy[:, 0] * scale).astype(int), 0, grid_res - 2)
    y_indices = np.clip(np.floor(all_xy[:, 1] * scale).astype(int), 0, grid_res - 2)
    bin_keys = y_indices * scale + x_indices

    unique_bins, inverse_indices = np.unique(bin_keys, return_inverse=True)
    n_unique = len(unique_bins)

    counts = np.bincount(inverse_indices)
    reduced_xy = np.zeros((n_unique, 2), dtype=np.float32)
    reduced_spectra = np.zeros((n_unique, all_spectra_norm.shape[1]), dtype=np.float32)

    np.add.at(reduced_xy, inverse_indices, all_xy)
    np.add.at(reduced_spectra, inverse_indices, all_spectra_norm)

    reduced_xy /= counts[:, None]
    reduced_spectra /= counts[:, None]

    return Delaunay(reduced_xy), reduced_spectra


# Solver Core Logic


def get_barycentric_prior(
    xy: tuple[float, float],
    triangulation: Delaunay,
    reduced_spectra: np.ndarray,
    cmfs: np.ndarray,
) -> tuple[np.ndarray | None, bool]:
    """Computes matrix-solved prior spectrum using the binned Delaunay mesh."""
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
    triangulation: Delaunay | None = None,
    reduced_spectra: np.ndarray | None = None,
) -> np.ndarray:
    """Finds a non-negative spectrum, optionally guided by a barycentric prior."""
    x, y = xy
    if x + y > 1 or y <= 0:
        return np.ones(cmfs.shape[0], dtype=np.float32)

    X = x / y
    Z = (1 - x - y) / y
    XYZ_target = np.array([X, 1.0, Z], dtype=np.float32)

    A = cmfs.T
    n_bins = A.shape[1]
    D1 = np.eye(n_bins, k=0) - np.eye(n_bins, k=1)

    has_prior = False
    prior_spectrum = None
    if triangulation is not None and reduced_spectra is not None:
        prior_spectrum, has_prior = get_barycentric_prior(
            xy, triangulation, reduced_spectra, cmfs
        )

    w_smooth = np.sqrt(smoothness_loss_factor)
    w_data = np.sqrt(data_loss_factor) if has_prior else 0.0

    C = np.vstack([A, w_smooth * D1, w_data * np.eye(n_bins)])

    if has_prior and prior_spectrum is not None:
        d = np.concatenate([XYZ_target, np.zeros(n_bins), w_data * prior_spectrum])
    else:
        d = np.concatenate([XYZ_target, np.zeros(n_bins), np.zeros(n_bins)])

    spectrum, _ = nnls(C, d)
    return spectrum


def _sample_grid_nnls(
    resolution: int,
    cmfs: np.ndarray,
    triangulation: Delaunay | None = None,
    reduced_spectra: np.ndarray | None = None,
    smoothness_loss_factor: float = 1.0,
    data_loss_factor: float = 2.5,
) -> np.ndarray:
    """Helper to sample NNLS across the full 2D grid resolution."""
    grid_coords = np.linspace(0, 1, resolution, dtype=np.float32)
    n_bins = cmfs.shape[0]
    out = np.empty((resolution, resolution, n_bins), dtype=np.float32)

    for i in range(resolution):
        for j in range(resolution):
            xy = (float(grid_coords[i]), float(grid_coords[j]))

            spec = xy_to_spectrum_nnls(
                xy,
                smoothness_loss_factor,
                data_loss_factor,
                cmfs,
                triangulation,
                reduced_spectra,
            )

            s_spectrum = (spec @ cmfs).sum()
            if s_spectrum > 1e-12:
                spec /= s_spectrum
            else:
                spec = np.zeros(n_bins, dtype=np.float32)

            out[i, j, :] = spec

    return out


# Upsampling Method Registry & Strategy Definitions


UpsampleMethod = Literal[
    "SFL upsampling",
    "Pure NNLS optimization",
    "Simple Matrix",
    "Mallett 2019",
    "Otsu 2018",
    "Smits 1999",
]

MethodHandler = Callable[[int, np.ndarray], np.ndarray]
_METHOD_REGISTRY: dict[str, MethodHandler] = {}


def register_method(name: str):
    """Decorator to register a new spectral upsampling LUT generator."""

    def decorator(fn: MethodHandler) -> MethodHandler:
        _METHOD_REGISTRY[name] = fn
        return fn

    return decorator


@register_method("SFL upsampling")
def _generate_barycentric_nnls_all(resolution: int, cmfs: np.ndarray) -> np.ndarray:
    all_xy, all_spectra_norm = _get_training_data(ALL_ILLUMINANT_KEYS)
    tri, reduced_spec = build_binned_triangulation(all_xy, all_spectra_norm, resolution)
    return _sample_grid_nnls(resolution, cmfs, tri, reduced_spec)


@register_method("Pure NNLS optimization")
def _generate_pure_nnls(resolution: int, cmfs: np.ndarray) -> np.ndarray:
    return _sample_grid_nnls(resolution, cmfs, triangulation=None, reduced_spectra=None)


def _generate_lstsq_matrix(
    illuminants: tuple[str, ...] = ALL_ILLUMINANT_KEYS,
):
    all_xyz, all_spectra_norm = _get_training_data(illuminants, True)

    matrix, _, _, _ = np.linalg.lstsq(all_xyz, all_spectra_norm, rcond=False)

    return matrix


@register_method("Simple Matrix")
def _generate_matrix_lut(resolution: int, cmfs: np.ndarray) -> np.ndarray:
    matrix = _generate_lstsq_matrix(("D65",))

    x = np.linspace(0, 1, resolution, dtype=DEFAULT_DTYPE)
    y = np.linspace(0, 1, resolution, dtype=DEFAULT_DTYPE)

    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")

    ones = np.ones((resolution, resolution), dtype=DEFAULT_DTYPE)
    result = np.stack([grid_x, grid_y, ones], axis=-1)

    xyz_lut = xyS_to_XYZ(result)

    lut = xyz_lut @ matrix

    # Enforce physical non-negativity
    np.maximum(lut, 0.0, out=lut)

    # Re-normalize reconstructed spectra to unit chromaticity sum S
    s_spectrum = np.sum(lut @ cmfs, axis=-1, keepdims=True)
    valid_s = s_spectrum > 1e-12
    np.divide(lut, s_spectrum, out=lut, where=valid_s)
    lut[~valid_s[..., 0]] = 0.0

    return lut


def _prepare_target_xyz(valid_xyz: np.ndarray) -> np.ndarray:
    """Soft gamut maps XYZ coordinates into sRGB bounds to preserve hue direction."""
    with colour.domain_range_scale("1"):
        rgb = colour.XYZ_to_RGB(valid_xyz, "sRGB")

    # Shift negative RGB components towards white, then scale peak to 1.0
    min_c = np.min(rgb, axis=-1, keepdims=True)
    rgb_shifted = rgb + np.maximum(0.0, -min_c)
    max_c = np.max(rgb_shifted, axis=-1, keepdims=True)
    max_c[max_c == 0.0] = 1.0

    with colour.domain_range_scale("1"):
        return colour.RGB_to_XYZ(rgb_shifted / max_c, "sRGB")


def _evaluate_xyz_to_sd(
    valid_xyz: np.ndarray,
    method: str,
    target_shape: colour.SpectralShape = SPECTRAL_SHAPE,
) -> np.ndarray:
    """
    Evaluates colour.XYZ_to_sd across vectorized inputs (e.g., Mallett 2019)
    or element-wise loops for non-vectorized methods (Smits 1999, Otsu 2018).
    """
    target_xyz = _prepare_target_xyz(valid_xyz)

    # Try vectorized evaluation (supported by Mallett 2019)
    try:
        with colour.domain_range_scale("1"):
            sd_result = colour.XYZ_to_sd(target_xyz, method=method)

        if hasattr(sd_result, "align") and target_shape is not None:
            sd_result = sd_result.copy()
            sd_result.align(target_shape)

        spec_values = (
            sd_result.values if hasattr(sd_result, "values") else np.asarray(sd_result)
        )

        if spec_values.ndim == 2 and spec_values.shape[0] == len(target_xyz):
            return np.nan_to_num(spec_values, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        pass  # Fall back to element-wise processing below

    # Point-by-point fallback loop for methods expecting single 1D vectors (Smits, Otsu)
    spec_list = []
    with colour.domain_range_scale("1"):
        for xyz in target_xyz:
            sd = colour.XYZ_to_sd(xyz, method=method)
            if hasattr(sd, "align") and target_shape is not None:
                sd = sd.copy()
                sd.align(target_shape)

            val = sd.values if hasattr(sd, "values") else np.asarray(sd)
            spec_list.append(val)

    spec_array = np.array(spec_list, dtype=DEFAULT_DTYPE)
    return np.nan_to_num(spec_array, nan=0.0, posinf=0.0, neginf=0.0)


def _generate_colour_sd_lut(
    resolution: int,
    cmfs: np.ndarray | colour.SpectralDistribution = None,
    method: str = "Mallett 2019",
) -> np.ndarray:
    """Generates a 2D xy chromaticity to spectrum LUT using colour.XYZ_to_sd."""
    # Build xy chromaticity grid (S = X + Y + Z = 1)
    x = np.linspace(0, 1, resolution, dtype=DEFAULT_DTYPE)
    y = np.linspace(0, 1, resolution, dtype=DEFAULT_DTYPE)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    grid_z = 1.0 - grid_x - grid_y

    valid_chroma = (grid_x >= 0.0) & (grid_y >= 1e-4) & (grid_z >= 0.0)

    xyz_flat = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    valid_flat = valid_chroma.reshape(-1)
    valid_xyz = xyz_flat[valid_flat]

    # Target spectral shape and wavelength count
    if hasattr(cmfs, "shape") and isinstance(cmfs.shape, colour.SpectralShape):
        target_shape = cmfs.shape
    elif hasattr(cmfs, "spectral_shape"):
        target_shape = cmfs.spectral_shape
    else:
        target_shape = SPECTRAL_SHAPE

    cmf_matrix = cmfs.values if hasattr(cmfs, "values") else cmfs
    num_wavelengths = (
        cmf_matrix.shape[0] if cmf_matrix is not None else target_shape.count
    )

    # Evaluate spectra
    lut_flat = np.zeros((resolution * resolution, num_wavelengths), dtype=DEFAULT_DTYPE)
    if len(valid_xyz) > 0:
        lut_flat[valid_flat] = _evaluate_xyz_to_sd(valid_xyz, method, target_shape)

    lut = lut_flat.reshape((resolution, resolution, num_wavelengths))
    np.maximum(lut, 0.0, out=lut)

    # Re-normalize reconstructed spectra to unit chromaticity sum S
    if cmf_matrix is not None:
        s_spectrum = np.sum(lut @ cmf_matrix, axis=-1, keepdims=True)
        valid_s = s_spectrum > 1e-12

        np.divide(lut, s_spectrum, out=lut, where=valid_s)

        # Equal-energy fallback inside locus to avoid zero-exposure gaps
        flat_spec = np.full(num_wavelengths, 1.0 / num_wavelengths, dtype=DEFAULT_DTYPE)
        lut[~valid_s[..., 0] & valid_chroma] = flat_spec
        lut[~valid_chroma] = 0.0

    return lut


@register_method("Mallett 2019")
def _generate_mallett2019_lut(resolution: int, cmfs: np.ndarray) -> np.ndarray:
    return _generate_colour_sd_lut(resolution, cmfs, method="Mallett 2019")


@register_method("Otsu 2018")
def _generate_otsu2018_lut(resolution: int, cmfs: np.ndarray) -> np.ndarray:
    return _generate_colour_sd_lut(resolution, cmfs, method="Otsu 2018")


@register_method("Smits 1999")
def _generate_smits1999_lut(resolution: int, cmfs: np.ndarray) -> np.ndarray:
    return _generate_colour_sd_lut(resolution, cmfs, method="Smits 1999")


@functools.lru_cache(maxsize=32)
def get_spectrum_lut(
    method: UpsampleMethod | str = "SFL upsampling",
    resolution: int = 33,
    illuminants: tuple[str, ...] | None = None,
) -> np.ndarray:
    """
    Computes or retrieves a cached 2D spectral lookup table.

    Args:
        method: Upsampling method registered in the registry.
        resolution: Grid resolution of the chromaticity LUT (resolution x resolution).
        illuminants: Tuple of illuminant keys for dataset preparation.
                     If None, defaults to ALL_ILLUMINANT_KEYS.

    Returns:
        np.ndarray: LUT array of shape (resolution, resolution, n_bins).
    """
    if method not in _METHOD_REGISTRY:
        available = list(_METHOD_REGISTRY.keys())
        raise ValueError(
            f"Unknown upsampling method '{method}'. Available: {available}"
        )

    handler = _METHOD_REGISTRY[method]

    return handler(resolution=resolution, cmfs=XYZ_CMFS)
