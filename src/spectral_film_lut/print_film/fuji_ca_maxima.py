"""Fuji Crystal Archive Maxima"""

from dataclasses import replace

from spectral_film_lut.print_film.fuji_ca_dpII import FUJI_CA_DPII

FUJI_CA_MAXIMA = replace(
    FUJI_CA_DPII,
    name="Fuji Crystal Archive Maxima",
    comment="No contrast curve available. Therefore just DPII with deeper blacks.",
    sensiometric_curve=[
        {x: y * 2.55 / 2.35 for x, y in FUJI_CA_DPII.sensiometric_curve[0].items()},
        {x: y * 2.55 / 2.35 for x, y in FUJI_CA_DPII.sensiometric_curve[1].items()},
        {x: y * 2.45 / 2.25 for x, y in FUJI_CA_DPII.sensiometric_curve[2].items()},
    ],
)
"""Fuji Crystal Archive Maxima"""
