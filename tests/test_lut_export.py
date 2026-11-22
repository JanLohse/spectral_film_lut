"""Component tests for 3D LUT generation and .cube export."""

from pathlib import Path

import numpy as np
import pytest

from spectral_film_lut.core import utils

pytestmark = pytest.mark.component


def parse_cube(path: Path) -> tuple[int, np.ndarray]:
    """Read the LUT size and RGB rows from a .cube file."""
    size = None
    values = []

    with path.open(encoding="utf-8") as cube_file:
        for line in cube_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("LUT_3D_SIZE"):
                size = int(line.split()[1])
            elif not line.startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX")):
                values.append([float(value) for value in line.split()])

    assert size is not None
    return size, np.asarray(values, dtype=np.float32)


@pytest.mark.parametrize("filter_mode", [None, "oversampling", "shift"])
def test_create_lut_returns_expected_grid_shape(
    monkeypatch: pytest.MonkeyPatch,
    filter_mode: str | None,
) -> None:
    """Create LUT tables of the requested size for every supported filter mode."""
    monkeypatch.setattr(
        utils,
        "film_conversion",
        lambda grid, negative_film, print_film, **kwargs: np.asarray(grid).copy(),
    )

    lut = utils.create_lut(
        negative_film=object(),
        lut_size=5,
        cube=False,
        filter_mode=filter_mode,
    )

    assert isinstance(lut, np.ndarray)
    assert lut.shape == (5, 5, 5, 3)
    assert np.all(np.isfinite(lut))


def test_create_lut_exports_identity_cube(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Export an identity LUT with the requested size and RGB values."""
    monkeypatch.setattr(
        utils,
        "film_conversion",
        lambda grid, negative_film, print_film, **kwargs: np.asarray(grid).copy(),
    )

    name = tmp_path / "identity"
    result = utils.create_lut(
        negative_film=object(),
        lut_size=3,
        name=str(name),
        cube=True,
    )

    cube_path = Path(result)
    assert cube_path == tmp_path / "identity.cube"
    assert cube_path.is_file()

    size, values = parse_cube(cube_path)
    assert size == 3
    assert values.shape == (27, 3)

    expected = utils.create_lut(
        negative_film=object(),
        lut_size=3,
        cube=False,
    )
    exported = utils.colour.io.read_LUT(str(cube_path)).table
    np.testing.assert_allclose(exported, expected, rtol=1e-6, atol=1e-6)


def test_create_lut_rejects_unknown_filter_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject unsupported LUT filtering modes."""
    monkeypatch.setattr(
        utils,
        "film_conversion",
        lambda grid, negative_film, print_film, **kwargs: np.asarray(grid).copy(),
    )

    with pytest.raises(ValueError, match="Unsupported filter_mode"):
        utils.create_lut(
            negative_film=object(),
            lut_size=5,
            cube=False,
            filter_mode="unknown",
        )
