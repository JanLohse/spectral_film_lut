"""
The main class for storing unprocessed film data.
"""

from dataclasses import dataclass


@dataclass
class FilmData:
    """Store data collected from a film datasheet."""

    # Required
    name: str
    manufacturer: str
    film_type: str
    stage: str
    medium: str
    density_measure: str
    sensiometric_curve: list[dict[float, float]]
    year: int | None = None
    iso: int | None = None
    alias: str | None = None
    comment: str | None = None
    exposure_kelvin: int = 5500
    projection_kelvin: int = 6500
    exposure_base: int = 10
    rms: float | None = None
    color_masking: float | None = None
    lad: list[float] | None = None
    d_min_adjustment: bool | None = None
    log_sensitivity: list[dict[float, float]] | None = None
    sensitivity: list[dict[float, float]] | None = None
    spectral_density: list[dict[float, float]] | None = None
    d_ref_sd: dict[float, float] | None = None
    d_min_sd: dict[float, float] | None = None
    rms_curve: list[dict[float, float]] | None = None
    rms_density: list[dict[float, float]] | None = None
    mtf: list[dict[float, float]] | None = None
