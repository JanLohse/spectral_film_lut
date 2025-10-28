import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QProgressBar, QVBoxLayout, QWidget, QLabel

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
from spectral_film_lut.negative_film.kodak_5247 import Kodak5247
from spectral_film_lut.negative_film.kodak_5247_II import Kodak5247II
from spectral_film_lut.negative_film.kodak_5248 import Kodak5248
from spectral_film_lut.negative_film.kodak_5250 import Kodak5250
from spectral_film_lut.negative_film.kodak_5277 import Kodak5277
from spectral_film_lut.negative_film.kodak_5293 import Kodak5293
from spectral_film_lut.negative_film.kodak_aerocolor import *
from spectral_film_lut.negative_film.kodak_ektar_100 import KodakEktar100
from spectral_film_lut.negative_film.kodak_exr_5248 import KodakEXR5248
from spectral_film_lut.negative_film.kodak_gold_200 import KodakGold200
from spectral_film_lut.negative_film.kodak_portra_160 import KodakPortra160
from spectral_film_lut.negative_film.kodak_portra_400 import KodakPortra400
from spectral_film_lut.negative_film.kodak_portra_800 import *
from spectral_film_lut.negative_film.kodak_ultramax_400 import KodakUltramax400
from spectral_film_lut.negative_film.kodak_vericolor_iii import KodakVericolorIII
from spectral_film_lut.print_film.fuji_3513di import Fuji3513DI
from spectral_film_lut.print_film.fuji_ca_dpII import FujiCrystalArchiveDPII
from spectral_film_lut.print_film.fuji_ca_maxima import FujiCrystalArchiveMaxima
from spectral_film_lut.print_film.fuji_ca_super_c import FujiCrystalArchiveSuperTypeC
from spectral_film_lut.print_film.fujiflex_new import FujiflexNew
from spectral_film_lut.print_film.fujiflex_old import FujiflexOld
from spectral_film_lut.print_film.kodak_2383 import Kodak2383
from spectral_film_lut.print_film.kodak_2393 import Kodak2393
from spectral_film_lut.print_film.kodak_5381 import Kodak5381
from spectral_film_lut.print_film.kodak_5383 import Kodak5383
from spectral_film_lut.print_film.kodak_5384 import Kodak5384
from spectral_film_lut.print_film.kodak_duraflex_plus import KodakDuraflexPlus
from spectral_film_lut.print_film.kodak_dye_transfer_negative import KodakDyeTransferNegative
from spectral_film_lut.print_film.kodak_endura_premier import KodakEnduraPremier
from spectral_film_lut.print_film.kodak_exr_5386 import KodakExr5386
from spectral_film_lut.print_film.kodak_portra_endura import KodakPortraEndura
from spectral_film_lut.print_film.kodak_supra_endura import KodakSupraEndura
from spectral_film_lut.print_film.technicolor_v import TechinicolorV
from spectral_film_lut.reversal_film.fuji_fp100c import FujiFP100C
from spectral_film_lut.reversal_film.fuji_instax_color import FujiInstaxColor
from spectral_film_lut.reversal_film.fuji_provia_100f import FujiProvia100F
from spectral_film_lut.reversal_film.fuji_velvia_50 import FujiVelvia50
from spectral_film_lut.reversal_film.kodachrome_64 import Kodachrome64
from spectral_film_lut.reversal_film.kodak_aerochrome_iii import KodakAerochromeIII
from spectral_film_lut.reversal_film.kodak_dye_transfer_separation import KodakDyeTransferSeparation
from spectral_film_lut.reversal_film.kodak_ektachrome_e100 import KodakEktachromeE100
from spectral_film_lut.reversal_film.technicolor_iv import *
from spectral_film_lut.reversal_print.ilfochrome_micrographic_m import IlfochromeMicrographicM
from spectral_film_lut.reversal_print.ilfochrome_micrographic_p import IlfochromeMicrographicP
from spectral_film_lut.reversal_print.kodak_dye_transfer_kodachrome import KodakDyeTransferKodachrome
from spectral_film_lut.reversal_print.kodak_dye_transfer_slide import KodakDyeTransferSlide
from spectral_film_lut.reversal_print.kodak_ektachrome_radiance_iii import KodakEktachromeRadianceIIIPaper

NEGATIVE_FILM = [KodakEktar100, KodakPortra160, KodakPortra400, KodakPortra800, KodakPortra800At1600,
                 KodakPortra800At3200, KodakUltramax400, KodakGold200, KodakVericolorIII, Kodak5203, Kodak5207,
                 Kodak5213, Kodak5219, Kodak5277, KodakEXR5248, Kodak5293, Kodak5247II, Kodak5250, Kodak5248, Kodak5247,
                 KodakAerocolor, KodakAerocolorLow, KodakAerocolorHigh, FujiPro160S, FujiPro160C, FujiPro400H,
                 FujiSuperiaReala, FujiC200, FujiSuperiaXtra400, FujiNatura1600, FujiEterna500, FujiEterna500Vivid,
                 AgfaVista100, Kodak5222Dev4, Kodak5222Dev5, Kodak5222Dev6, Kodak5222Dev9, Kodak5222Dev12,
                 KodakTriX400Dev6, KodakTriX400Dev7, KodakTriX400Dev9, KodakTriX400Dev11]
PRINT_FILM = [KodakEnduraPremier, KodakDuraflexPlus, KodakPortraEndura, KodakSupraEndura, Kodak2383, Kodak2393,
              KodakExr5386, Kodak5384, Kodak5383, Kodak5381, FujiCrystalArchiveDPII, FujiCrystalArchiveMaxima,
              FujiCrystalArchiveSuperTypeC, FujiflexNew, FujiflexOld, Fuji3513DI, Kodak2303Dev2, Kodak2303Dev3,
              Kodak2303Dev5, Kodak2303Dev7, Kodak2303Dev9, KodakPolymax, KodakPolymaxGradeNeg1, KodakPolymaxGrade0,
              KodakPolymaxGrade1, KodakPolymaxGrade2, KodakPolymaxGrade3, KodakPolymaxGrade4, KodakPolymaxGrade5,
              KodakDyeTransferNegative, TechinicolorV, KodakDyeTransferKodachrome]
REVERSAL_PRINT = [KodakDyeTransferSlide, IlfochromeMicrographicP, IlfochromeMicrographicM,
                  KodakEktachromeRadianceIIIPaper, KodakAerochromeIII, FujiFP100C, FujiInstaxColor]
REVERSAL_FILM = [KodakEktachromeE100, Kodachrome64, FujiVelvia50, FujiProvia100F, KodakDyeTransferSeparation,
                 TechnicolorIV, TechnicolorIValt1, TechnicolorIValt2]
filmstocks = NEGATIVE_FILM + REVERSAL_FILM + PRINT_FILM + REVERSAL_PRINT


def load_filmstocks(progress_callback):
    result = []
    total = len(filmstocks)
    for i, film_cls in enumerate(filmstocks, start=1):
        instance = film_cls()
        if result and instance.stage == "print" and instance.density_measure == "status_a":
            instance.set_color_checker(negative=result[0])
        else:
            instance.set_color_checker()
        result.append(instance)
        progress_callback(i, total, film_cls.__name__)
    return {stock.__class__.__name__: stock for stock in result}


PROGRESS_BACKGROUND = PRESSED_COLOR
PROGRESS_COLOR = TEXT_PRIMARY


class SplashScreen(QWidget):
    def __init__(self, total_items, name):
        super().__init__()
        self.setWindowTitle(name)
        self.setFixedSize(400, 180)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.CoverWindow)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Background color
        self.setStyleSheet(f"background-color: {BACKGROUND_COLOR}; color: {TEXT_PRIMARY};")

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Heading label
        self.heading = QLabel(name)
        self.heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heading.setStyleSheet("font-weight: bold; font-size: 18px;")

        # Sub-label
        self.label = QLabel("Starting up...")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 13px;")

        # Progress bar
        self.total_items = total_items
        self.progress = QProgressBar()
        self.progress.setRange(0, total_items)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        self.update_style_sheet(0)

        layout.addWidget(self.heading)
        layout.addWidget(self.label)
        layout.addWidget(self.progress)

        self.show()
        QApplication.processEvents()

    def update_style_sheet(self, progress):
        # Style the progress bar
        progress_color = colour.convert((0.5, 0.08, progress), "Oklch", "Hexadecimal")
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border-radius: {BUTTON_RADIUS}px;
                text-align: center;
                background-color: {PROGRESS_BACKGROUND};
                height: 16px;  /* thicker bar */
                color: {TEXT_PRIMARY};
            }}
            QProgressBar::chunk {{
                background-color: {progress_color};
                border-radius: {BUTTON_RADIUS}px;
            }}
        """)

    def update(self, current, total, name):
        self.progress.setValue(current)
        self.update_style_sheet(current / self.total_items)
        self.label.setText(f"Loading {name} ({current}/{total})")
        QApplication.processEvents()


def load_ui(main_window, name="Spectral Film LUT"):
    app = QApplication(sys.argv)

    app.setStyleSheet(THEME)

    splash = SplashScreen(total_items=len(filmstocks), name=name)

    def update_progress(current, total, name):
        splash.update(current, total, name)

    loaded_filmstocks = load_filmstocks(update_progress)

    window = main_window(loaded_filmstocks)
    window.show()
    splash.close()

    sys.exit(app.exec())
