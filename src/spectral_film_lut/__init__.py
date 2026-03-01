"""
.. include:: ../../README.md
   :start-line: 2
"""

from pathlib import Path

from spectral_film_lut.bw_negative_film.kodak_5222 import (
    KODAK_5222,
    KODAK_5222_DEV_4,
    KODAK_5222_DEV_5,
    KODAK_5222_DEV_9,
    KODAK_5222_DEV_12,
)
from spectral_film_lut.bw_negative_film.kodak_trix_400 import (
    KODAK_TRI_X_400,
    KODAK_TRI_X_400_DEV_7,
    KODAK_TRI_X_400_DEV_9,
    KODAK_TRI_X_400_DEV_11,
)
from spectral_film_lut.bw_print_film.kodak_2302 import (
    KODAK_2302,
    KODAK_2302_DEV_2,
    KODAK_2302_DEV_3,
    KODAK_2302_DEV_5,
    KODAK_2302_DEV_7,
    KODAK_2302_DEV_9,
)
from spectral_film_lut.bw_print_film.kodak_polymax_fine_art import (
    KODAK_POLYMAX,
    KODAK_POLYMAX_GRADE_0,
    KODAK_POLYMAX_GRADE_1,
    KODAK_POLYMAX_GRADE_2,
    KODAK_POLYMAX_GRADE_3,
    KODAK_POLYMAX_GRADE_4,
    KODAK_POLYMAX_GRADE_5,
    KODAK_POLYMAX_GRADE_MINUS_1,
)
from spectral_film_lut.negative_film.agfa_vista_100 import AGFA_VISTA_100
from spectral_film_lut.negative_film.fuji_c200 import FUJI_C200
from spectral_film_lut.negative_film.fuji_eterna_500 import (
    FUJI_ETERNA_500,
)
from spectral_film_lut.negative_film.fuji_eterna_500_vivid import FUJI_ETERNA_500_VIVID
from spectral_film_lut.negative_film.fuji_natura_1600 import FUJI_NATURA_1600
from spectral_film_lut.negative_film.fuji_pro_160c import FUJI_PRO_160C
from spectral_film_lut.negative_film.fuji_pro_160s import FUJI_PRO_160S
from spectral_film_lut.negative_film.fuji_pro_400h import FUJI_PRO_400H
from spectral_film_lut.negative_film.fuji_superia_reala import FUJI_SUPERIA_REALA
from spectral_film_lut.negative_film.fuji_superia_xtra_400 import FUJI_SUPERIA_XTRA_400
from spectral_film_lut.negative_film.kodak_5203 import KODAK_5203
from spectral_film_lut.negative_film.kodak_5207 import KODAK_5207
from spectral_film_lut.negative_film.kodak_5213 import KODAK_5213
from spectral_film_lut.negative_film.kodak_5219 import KODAK_5219
from spectral_film_lut.negative_film.kodak_5247 import KODAK_5247
from spectral_film_lut.negative_film.kodak_5247_II import (
    KODAK_5247_II,
    KODAK_5247_II_ALT,
)
from spectral_film_lut.negative_film.kodak_5248 import KODAK_5248
from spectral_film_lut.negative_film.kodak_5250 import KODAK_5250
from spectral_film_lut.negative_film.kodak_5277 import KODAK_5277
from spectral_film_lut.negative_film.kodak_5293 import KODAK_5293
from spectral_film_lut.negative_film.kodak_aerocolor import (
    KODAK_AEROCOLOR,
    KODAK_AEROCOLOR_HIGH,
    KODAK_AEROCOLOR_LOW,
)
from spectral_film_lut.negative_film.kodak_ektar_100 import KODAK_EKTAR_100
from spectral_film_lut.negative_film.kodak_exr_5248 import KODAK_EXR_5248
from spectral_film_lut.negative_film.kodak_gold_200 import KODAK_GOLD_200
from spectral_film_lut.negative_film.kodak_portra_160 import KODAK_PORTRA_160
from spectral_film_lut.negative_film.kodak_portra_400 import KODAK_PORTRA_400
from spectral_film_lut.negative_film.kodak_portra_800 import (
    KODAK_PORTRA_800,
    KODAK_PORTRA_800_AT_1600,
    KODAK_PORTRA_800_AT_3200,
)
from spectral_film_lut.negative_film.kodak_ultramax_400 import KODAK_ULTRAMAX_400
from spectral_film_lut.negative_film.kodak_vericolor_iii import KODAK_VERICOLOR_III
from spectral_film_lut.print_film.fuji_3513di import FUJI_3513
from spectral_film_lut.print_film.fuji_3523XD import FUJI_3523
from spectral_film_lut.print_film.fuji_ca_dpII import FUJI_CA_DPII
from spectral_film_lut.print_film.fuji_ca_maxima import FUJI_CA_MAXIMA
from spectral_film_lut.print_film.fuji_ca_pdII import FUJI_CA_PRO_PDII
from spectral_film_lut.print_film.fuji_ca_super_c import FUJI_CA_SUPER_C
from spectral_film_lut.print_film.fujiflex_new import FUJIFLEX_NEW
from spectral_film_lut.print_film.fujiflex_old import FUJIFLEX_OLD
from spectral_film_lut.print_film.kodak_2383 import KODAK_2383
from spectral_film_lut.print_film.kodak_2393 import KODAK_2393
from spectral_film_lut.print_film.kodak_5381 import KODAK_5381
from spectral_film_lut.print_film.kodak_5383 import KODAK_5383
from spectral_film_lut.print_film.kodak_5384 import KODAK_5384
from spectral_film_lut.print_film.kodak_duraflex_plus import KODAK_DURAFLEX_PLUS
from spectral_film_lut.print_film.kodak_endura_premier import KODAK_ENDURA_PREMIER
from spectral_film_lut.print_film.kodak_exr_5386 import KODAK_EXR_5386
from spectral_film_lut.print_film.kodak_portra_endura import KODAK_PORTRA_ENDURA
from spectral_film_lut.print_film.kodak_supra_endura import KODAK_SUPRA_ENDURA
from spectral_film_lut.reversal_film.fuji_fp100c import FUJI_FP100C
from spectral_film_lut.reversal_film.fuji_instax_color import FUJI_INSTAX_COLOR
from spectral_film_lut.reversal_film.fuji_provia_100f import FUJI_PROVIA_100F
from spectral_film_lut.reversal_film.fuji_velvia_50 import FUJI_VELVIA_50
from spectral_film_lut.reversal_film.kodachrome_64 import KODACHROME_64
from spectral_film_lut.reversal_film.kodak_aerochrome_iii import KODAK_AEROCHROME_III
from spectral_film_lut.reversal_film.kodak_ektachrome_100d import KODAK_EKTACHROME_100D

try:
    from ._version import __version__
except ImportError:
    __version__ = ""

BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = str(BASE_DIR).replace("\\", "/")


NEGATIVE_FILM = [
    KODAK_5222,
    KODAK_5222,
    KODAK_5222_DEV_4,
    KODAK_5222_DEV_5,
    KODAK_5222_DEV_9,
    KODAK_5222_DEV_12,
    KODAK_TRI_X_400,
    KODAK_TRI_X_400_DEV_7,
    KODAK_TRI_X_400_DEV_9,
    KODAK_TRI_X_400_DEV_11,
    AGFA_VISTA_100,
    FUJI_C200,
    FUJI_ETERNA_500,
    FUJI_ETERNA_500_VIVID,
    FUJI_NATURA_1600,
    FUJI_PRO_160C,
    FUJI_PRO_160S,
    FUJI_PRO_400H,
    FUJI_SUPERIA_REALA,
    FUJI_SUPERIA_XTRA_400,
    KODAK_5203,
    KODAK_5207,
    KODAK_5213,
    KODAK_5219,
    KODAK_5247,
    KODAK_5247_II,
    KODAK_5247_II_ALT,
    KODAK_5248,
    KODAK_EXR_5248,
    KODAK_5250,
    KODAK_5277,
    KODAK_5293,
    KODAK_AEROCOLOR,
    KODAK_AEROCOLOR_LOW,
    KODAK_AEROCOLOR_HIGH,
    KODAK_EKTAR_100,
    KODAK_GOLD_200,
    KODAK_EXR_5248,
    KODAK_PORTRA_160,
    KODAK_PORTRA_400,
    KODAK_PORTRA_800,
    KODAK_PORTRA_800_AT_1600,
    KODAK_PORTRA_800_AT_3200,
    KODAK_ULTRAMAX_400,
    KODAK_VERICOLOR_III,
]
PRINT_FILM = [
    KODAK_2383,
    KODAK_2302,
    KODAK_2302_DEV_2,
    KODAK_2302_DEV_3,
    KODAK_2302_DEV_5,
    KODAK_2302_DEV_7,
    KODAK_2302_DEV_9,
    KODAK_5222_DEV_4,
    KODAK_5222_DEV_5,
    KODAK_5222_DEV_9,
    KODAK_5222_DEV_12,
    KODAK_POLYMAX,
    KODAK_POLYMAX_GRADE_MINUS_1,
    KODAK_POLYMAX_GRADE_0,
    KODAK_POLYMAX_GRADE_1,
    KODAK_POLYMAX_GRADE_2,
    KODAK_POLYMAX_GRADE_3,
    KODAK_POLYMAX_GRADE_4,
    KODAK_POLYMAX_GRADE_5,
    FUJI_3513,
    FUJI_3523,
    FUJI_CA_PRO_PDII,
    FUJI_CA_DPII,
    FUJI_CA_MAXIMA,
    FUJI_CA_SUPER_C,
    FUJIFLEX_NEW,
    FUJIFLEX_OLD,
    KODAK_2383,
    KODAK_2393,
    KODAK_5381,
    KODAK_5383,
    KODAK_5384,
    KODAK_DURAFLEX_PLUS,
    KODAK_ENDURA_PREMIER,
    KODAK_EXR_5386,
    KODAK_PORTRA_ENDURA,
    KODAK_SUPRA_ENDURA,
]
REVERSAL_PRINT = []
REVERSAL_FILM = [
    FUJI_FP100C,
    FUJI_INSTAX_COLOR,
    FUJI_PROVIA_100F,
    FUJI_VELVIA_50,
    KODACHROME_64,
    KODAK_AEROCHROME_III,
    KODAK_EKTACHROME_100D,
]
FILM_STOCKS = NEGATIVE_FILM + REVERSAL_FILM + PRINT_FILM + REVERSAL_PRINT
FILM_STOCKS = sorted(FILM_STOCKS, key=lambda x: x.name)
