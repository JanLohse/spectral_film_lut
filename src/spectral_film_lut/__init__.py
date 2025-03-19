from spectral_film_lut.gui import main
from spectral_film_lut.print_film.kodak_2383 import Kodak2383
from spectral_film_lut.print_film.kodak_2393 import Kodak2393
from spectral_film_lut.print_film.kodak_endura_premier import KodakEnduraPremier
from spectral_film_lut.negative_film.kodak_5207 import Kodak5207
from spectral_film_lut.negative_film.kodak_portra_400 import KodakPortra400
from spectral_film_lut.reversal_film.kodachrome_64 import Kodachrome64
from spectral_film_lut.reversal_film.fuji_instax_color import FujiInstaxColor
from spectral_film_lut.reversal_film.kodak_ektachrome_100d import KodakEktachrome100D
from spectral_film_lut.print_film.fuji_ca_dpII import FujiCrystalArchiveDPII

FILMSTOCKS = [Kodak2383, Kodak2393, KodakEnduraPremier, Kodak5207, KodakPortra400, Kodachrome64, FujiInstaxColor,
              KodakEktachrome100D, FujiCrystalArchiveDPII]
FILMSTOCKS = {film.__name__: film for film in FILMSTOCKS}
