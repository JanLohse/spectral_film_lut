"""Smoke-test the application entry point in a headless environment."""

import pytest

from spectral_film_lut.__main__ import run


@pytest.mark.smoke
def test_application_smoke() -> None:
    """Start the application headlessly and confirm it exits cleanly."""
    with pytest.raises(SystemExit) as exit_info:
        run(exit_immediately=True)

    assert exit_info.value.code == 0
