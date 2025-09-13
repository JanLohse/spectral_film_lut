from spectral_film_lut.bw_negative_film.kodak_5222 import *
from spectral_film_lut.bw_negative_film.kodak_trix_400 import *
from spectral_film_lut.bw_print_film.kodak_2303 import *
from spectral_film_lut.bw_print_film.kodak_polymax_fine_art import *
from spectral_film_lut.negative_film.agfa_vista_100 import AgfaVista100
from spectral_film_lut.negative_film.fuji_c200 import FujiC200
from spectral_film_lut.negative_film.fuji_eterna_500 import FujiEterna500
from spectral_film_lut.negative_film.fuji_eterna_500_vivid import FujiEterna500Vivid
from spectral_film_lut.negative_film.fuji_natura_1600 import FujiNatura1600
from spectral_film_lut.negative_film.fuji_pro_160c import FujiPro160C
from spectral_film_lut.negative_film.fuji_pro_160s import FujiPro160S
from spectral_film_lut.negative_film.fuji_pro_400h import FujiPro400H
from spectral_film_lut.negative_film.fuji_superia_reala import FujiSuperiaReala
from spectral_film_lut.negative_film.fuji_superia_xtra_400 import FujiSuperiaXtra400
from spectral_film_lut.negative_film.kodak_5203 import Kodak5203
from spectral_film_lut.negative_film.kodak_5207 import Kodak5207
from spectral_film_lut.negative_film.kodak_5213 import Kodak5213
from spectral_film_lut.negative_film.kodak_5219 import Kodak5219
from spectral_film_lut.negative_film.kodak_5250 import Kodak5250
from spectral_film_lut.negative_film.kodak_5248 import Kodak5248
from spectral_film_lut.negative_film.kodak_5277 import Kodak5277
from spectral_film_lut.negative_film.kodak_5293 import Kodak5293
from spectral_film_lut.negative_film.kodak_exr_5248 import KodakEXR5248
from spectral_film_lut.negative_film.kodak_5247_II import Kodak5247II
from spectral_film_lut.negative_film.kodak_5247 import Kodak5247
from spectral_film_lut.negative_film.kodak_ektar_100 import KodakEktar100
from spectral_film_lut.negative_film.kodak_gold_200 import KodakGold200
from spectral_film_lut.negative_film.kodak_portra_160 import KodakPortra160
from spectral_film_lut.negative_film.kodak_portra_400 import KodakPortra400
from spectral_film_lut.negative_film.kodak_portra_800 import *
from spectral_film_lut.negative_film.kodak_vericolor_iii import KodakVericolorIII
from spectral_film_lut.negative_film.kodak_aerocolor import *
from spectral_film_lut.negative_film.kodak_ultramax_400 import KodakUltramax400
from spectral_film_lut.print_film.fuji_3513di import Fuji3513DI
from spectral_film_lut.print_film.fuji_ca_dpII import FujiCrystalArchiveDPII
from spectral_film_lut.print_film.fuji_ca_maxima import FujiCrystalArchiveMaxima
from spectral_film_lut.print_film.fuji_ca_super_c import FujiCrystalArchiveSuperTypeC
from spectral_film_lut.print_film.fujiflex_new import FujiflexNew
from spectral_film_lut.print_film.fujiflex_old import FujiflexOld
from spectral_film_lut.print_film.kodak_2383 import Kodak2383
from spectral_film_lut.print_film.kodak_2393 import Kodak2393
from spectral_film_lut.print_film.kodak_5383 import Kodak5383
from spectral_film_lut.print_film.kodak_5381 import Kodak5381
from spectral_film_lut.print_film.kodak_5384 import Kodak5384
from spectral_film_lut.reversal_film.fuji_fp100c import FujiFP100C
from spectral_film_lut.reversal_film.fuji_instax_color import FujiInstaxColor
from spectral_film_lut.reversal_film.kodak_aerochrome_iii import KodakAerochromeIII
from spectral_film_lut.reversal_film.technicolor_iv import *
from spectral_film_lut.reversal_print.ilfochrome_micrographic_p import IlfochromeMicrographicP
from spectral_film_lut.reversal_print.ilfochrome_micrographic_m import IlfochromeMicrographicM
from spectral_film_lut.reversal_print.kodak_dye_transfer_slide import KodakDyeTransferSlide
from spectral_film_lut.reversal_print.kodak_dye_transfer_kodachrome import KodakDyeTransferKodachrome
from spectral_film_lut.print_film.kodak_dye_transfer_negative import KodakDyeTransferNegative
from spectral_film_lut.print_film.technicolor_v import TechinicolorV
from spectral_film_lut.reversal_film.kodak_dye_transfer_separation import KodakDyeTransferSeparation
from spectral_film_lut.print_film.kodak_exr_5386 import KodakExr5386
from spectral_film_lut.print_film.kodak_duraflex_plus import KodakDuraflexPlus
from spectral_film_lut.print_film.kodak_endura_premier import KodakEnduraPremier
from spectral_film_lut.print_film.kodak_portra_endura import KodakPortraEndura
from spectral_film_lut.print_film.kodak_supra_endura import KodakSupraEndura
from spectral_film_lut.reversal_film.fuji_provia_100f import FujiProvia100F
from spectral_film_lut.reversal_film.fuji_velvia_50 import FujiVelvia50
from spectral_film_lut.reversal_film.kodachrome_64 import Kodachrome64
from spectral_film_lut.reversal_film.kodak_ektachrome_e100 import KodakEktachromeE100
from spectral_film_lut.reversal_print.kodak_ektachrome_radiance_iii import KodakEktachromeRadianceIIIPaper

NEGATIVE_FILM = [KodakEktar100, KodakPortra160, KodakPortra400, KodakPortra800, KodakPortra800At1600,
                 KodakPortra800At3200, KodakUltramax400, KodakGold200, KodakVericolorIII, Kodak5207, Kodak5219,
                 Kodak5277, KodakEXR5248, Kodak5293, Kodak5247II, Kodak5250, Kodak5248, Kodak5247, KodakAerocolor,
                 KodakAerocolorLow, KodakAerocolorHigh, FujiPro160S, FujiPro160C, FujiPro400H, FujiSuperiaReala,
                 FujiC200, FujiSuperiaXtra400, FujiNatura1600, FujiEterna500, FujiEterna500Vivid, AgfaVista100,
                 Kodak5222Dev4, Kodak5222Dev5, Kodak5222Dev6, Kodak5222Dev9, Kodak5222Dev12, KodakTriX400Dev6,
                 KodakTriX400Dev7, KodakTriX400Dev9, KodakTriX400Dev11]
PRINT_FILM = [KodakEnduraPremier, KodakDuraflexPlus, KodakPortraEndura, KodakSupraEndura, Kodak2383, Kodak2393,
              KodakExr5386, Kodak5384, Kodak5383, Kodak5381, FujiCrystalArchiveDPII, FujiCrystalArchiveMaxima,
              FujiCrystalArchiveSuperTypeC, FujiflexNew, FujiflexOld, Fuji3513DI, Kodak2303Dev2, Kodak2303Dev3,
              Kodak2303Dev5, Kodak2303Dev7, Kodak2303Dev9, KodakPolymax, KodakPolymaxGradeNeg1, KodakPolymaxGrade0,
              KodakPolymaxGrade1, KodakPolymaxGrade2, KodakPolymaxGrade3, KodakPolymaxGrade4, KodakPolymaxGrade5,
              KodakDyeTransferNegative, TechinicolorV, KodakDyeTransferKodachrome]
REVERSAL_PRINT = [KodakDyeTransferSlide, IlfochromeMicrographicP, IlfochromeMicrographicM,
                  KodakEktachromeRadianceIIIPaper]
REVERSAL_FILM = [KodakEktachromeE100, Kodachrome64, FujiVelvia50, FujiProvia100F, KodakDyeTransferSeparation,
                 TechnicolorIV, TechnicolorIValt1, TechnicolorIValt2, KodakAerochromeIII, FujiFP100C, FujiInstaxColor]
FILMSTOCKS = NEGATIVE_FILM + PRINT_FILM + REVERSAL_FILM
NEGATIVE_FILM = {film.__name__: film for film in NEGATIVE_FILM}
PRINT_FILM = {film.__name__: film for film in PRINT_FILM}
REVERSAL_PRINT = {film.__name__: film for film in REVERSAL_PRINT}
REVERSAL_FILM = {film.__name__: film for film in REVERSAL_FILM}
FILMSTOCKS = {film.__name__: film for film in FILMSTOCKS}
