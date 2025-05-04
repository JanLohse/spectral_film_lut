from spectral_film_lut.film_spectral import *
from spectral_film_lut.print_film.fuji_ca_dpII import FujiCrystalArchiveDPII


class FujiCrystalArchiveMaxima(FujiCrystalArchiveDPII):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # estimate d-max based on calibration data
        self.density_curve[0] *= 2.55 / 2.35
        self.density_curve[1] *= 2.55 / 2.35
        self.density_curve[2] *= 2.45 / 2.25
