from spectral_film_lut.bw_negative_film.kodak_5222 import *
from spectral_film_lut.bw_negative_film.kodak_trix_400 import *
from spectral_film_lut.bw_print_film.kodak_2303 import *
from spectral_film_lut.bw_print_film.kodak_polymax_fine_art import *
from spectral_film_lut.gui import main
from spectral_film_lut.negative_film.fuji_eterna_500 import FujiEterna500
from spectral_film_lut.negative_film.fuji_eterna_500_vivid import FujiEterna500Vivid
from spectral_film_lut.negative_film.fuji_pro_400h import FujiPro400H
from spectral_film_lut.negative_film.kodak_5207 import Kodak5207
from spectral_film_lut.negative_film.kodak_ektar_100 import KodakEktar100
from spectral_film_lut.negative_film.kodak_portra_400 import KodakPortra400
from spectral_film_lut.print_film.fuji_3513di import Fuji3513DI
from spectral_film_lut.print_film.fuji_ca_dpII import FujiCrystalArchiveDPII
from spectral_film_lut.print_film.fuji_ca_maxima import FujiCrystalArchiveMaxima
from spectral_film_lut.print_film.fuji_ca_super_c import FujiCrystalArchiveSuperTypeC
from spectral_film_lut.print_film.fujiflex_new import FujiflexNew
from spectral_film_lut.print_film.fujiflex_old import FujiflexOld
from spectral_film_lut.print_film.kodak_2383 import Kodak2383
from spectral_film_lut.print_film.kodak_2393 import Kodak2393
from spectral_film_lut.print_film.kodak_duraflex_plus import KodakDuraflexPlus
from spectral_film_lut.print_film.kodak_endura_premier import KodakEnduraPremier
from spectral_film_lut.print_film.kodak_portra_endura import KodakPortraEndura
from spectral_film_lut.print_film.kodak_supra_endura import KodakSupraEndura
from spectral_film_lut.reversal_film.fuji_instax_color import FujiInstaxColor
from spectral_film_lut.reversal_film.fuji_provia_100f import FujiProvia100F
from spectral_film_lut.reversal_film.fuji_velvia_50 import FujiVelvia50
from spectral_film_lut.reversal_film.kodachrome_64 import Kodachrome64
from spectral_film_lut.reversal_film.kodak_ektachrome_100d import KodakEktachrome100D
from spectral_film_lut.reversal_film.kodak_ektachrome_e100 import KodakEktachromeE100

NEGATIVE_FILM = [KodakPortra400, KodakEktar100, Kodak5207, FujiPro400H, FujiEterna500, FujiEterna500Vivid, Kodak5222Dev4,
                 Kodak5222Dev5, Kodak5222Dev6, Kodak5222Dev9, Kodak5222Dev12, KodakTriX400Dev6, KodakTriX400Dev7,
                 KodakTriX400Dev9, KodakTriX400Dev11]
PRINT_FILM = [KodakEnduraPremier, KodakDuraflexPlus, KodakPortraEndura, KodakSupraEndura, Kodak2383, Kodak2393,
              FujiCrystalArchiveDPII, FujiCrystalArchiveMaxima, FujiCrystalArchiveSuperTypeC, FujiflexNew, FujiflexOld,
              Fuji3513DI, Kodak2303Dev2, Kodak2303Dev3, Kodak2303Dev5, Kodak2303Dev7, Kodak2303Dev9, KodakPolymax,
              KodakPolymaxGradeNeg1, KodakPolymaxGrade0, KodakPolymaxGrade1, KodakPolymaxGrade2, KodakPolymaxGrade3,
              KodakPolymaxGrade4, KodakPolymaxGrade5]
REVERSAL_FILM = [KodakEktachromeE100, KodakEktachrome100D, Kodachrome64, FujiVelvia50, FujiProvia100F, FujiInstaxColor]
FILMSTOCKS = NEGATIVE_FILM + PRINT_FILM + REVERSAL_FILM
NEGATIVE_FILM = {film.__name__: film for film in NEGATIVE_FILM}
PRINT_FILM = {film.__name__: film for film in PRINT_FILM}
REVERSAL_FILM = {film.__name__: film for film in REVERSAL_FILM}
FILMSTOCKS = {film.__name__: film for film in FILMSTOCKS}
