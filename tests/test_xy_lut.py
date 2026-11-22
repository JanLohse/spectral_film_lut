"""Component tests for spectral 2D chromaticity LUT generation."""

import numpy as np
import pytest

from spectral_film_lut.core.xy_lut import (
    XYZ_CMFS,
    UpsampleMethod,
    apply_2d_lut,
    get_spectrum_lut,
)

pytestmark = pytest.mark.component

METHODS = (
    "SFL upsampling",
    "Pure NNLS optimization",
    "Simple Matrix",
    "Mallett 2019",
    "Otsu 2018",
    "Smits 1999",
)


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_generate_valid_spectral_luts(method: UpsampleMethod) -> None:
    """All registered methods should generate finite, non-negative spectral LUTs."""
    lut = get_spectrum_lut(method=method, resolution=17)

    assert lut.shape == (17, 17, XYZ_CMFS.shape[0])
    assert np.all(np.isfinite(lut))
    assert np.all(lut >= -1e-7)


@pytest.mark.parametrize("method", METHODS)
def test_center_chromaticity_round_trip(method: UpsampleMethod) -> None:
    """Interior CIE XYZ values should survive spectral upsampling closely."""
    rng = np.random.default_rng(20261122)
    count = 128

    x = rng.uniform(0.28, 0.40, count)
    y = rng.uniform(0.28, 0.40, count)
    valid = x + y < 0.72
    x = x[valid]
    y = y[valid]
    sums = rng.uniform(0.1, 2.0, len(x))

    xyz = np.column_stack(
        (
            x * sums,
            y * sums,
            (1.0 - x - y) * sums,
        )
    ).astype(np.float32)

    lut = get_spectrum_lut(method=method, resolution=33)
    sampled_spectra = apply_2d_lut(xyz, lut)
    reconstructed_xyz = sampled_spectra @ XYZ_CMFS

    original_norm = xyz / xyz.sum(axis=-1, keepdims=True)
    reconstructed_norm = reconstructed_xyz / reconstructed_xyz.sum(axis=-1, keepdims=True)

    max_xy_error = np.max(np.abs(original_norm[..., :2] - reconstructed_norm[..., :2]))
    assert max_xy_error < 0.08, f"{method}: normalized xy error {max_xy_error:.4f}"


def test_unknown_method_is_rejected() -> None:
    """An unregistered upsampling method should raise a clear error."""
    with pytest.raises(ValueError, match="Unknown upsampling method"):
        get_spectrum_lut(method="unknown", resolution=17)
