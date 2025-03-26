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
from spectral_film_lut.negative_film.kodak_ektar_100 import KodakEktar100

NEGATIVE_FILM = [KodakPortra400, KodakEktar100, Kodak5207]
PRINT_FILM = [KodakEnduraPremier, Kodak2383, Kodak2393, FujiCrystalArchiveDPII]
REVERSAL_FILM = [KodakEktachrome100D, Kodachrome64, FujiInstaxColor]
FILMSTOCKS = NEGATIVE_FILM + PRINT_FILM + REVERSAL_FILM
NEGATIVE_FILM = {film.__name__: film for film in NEGATIVE_FILM}
PRINT_FILM = {film.__name__: film for film in PRINT_FILM}
REVERSAL_FILM = {film.__name__: film for film in REVERSAL_FILM}
FILMSTOCKS = {film.__name__: film for film in FILMSTOCKS}
