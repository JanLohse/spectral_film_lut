import ctypes
import sys
from ctypes import wintypes

import colour
from PyQt6.QtWidgets import QApplication, QWidget

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
from spectral_film_lut.css_theme import PRESSED_COLOR, TEXT_PRIMARY, THEME
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.negative_film.agfa_vista_100 import AGFA_VISTA_100
from spectral_film_lut.negative_film.kodak_5207 import KODAK_5207
from spectral_film_lut.print_film.kodak_2383 import KODAK_2383

NEGATIVE_FILM = [
    KODAK_5207,
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
]
PRINT_FILM = [KODAK_2383]
REVERSAL_PRINT = []
REVERSAL_FILM = []
filmstocks = NEGATIVE_FILM + REVERSAL_FILM + PRINT_FILM + REVERSAL_PRINT


def load_filmstocks(progress_callback, gray_value=None):
    result = []
    total = len(filmstocks)
    for i, film_stock in enumerate(filmstocks, start=1):
        instance = FilmSpectral(film_stock, gray_value=gray_value)
        if (
            result
            and instance.stage == "print"
            and instance.density_measure == "status_a"
        ):
            instance.set_color_checker(negative=result[0])
        else:
            instance.set_color_checker()
        result.append(instance)
        progress_callback(i, total, film_stock.name)
    return {stock.name: stock for stock in result}


PROGRESS_BACKGROUND = PRESSED_COLOR
PROGRESS_COLOR = TEXT_PRIMARY

DWMWA_USE_IMMERSIVE_DARK_MODE = 20


def set_dark_title_bar(hwnd):
    value = ctypes.c_int(1)
    ctypes.windll.dwmapi.DwmSetWindowAttribute(
        wintypes.HWND(hwnd),
        wintypes.DWORD(DWMWA_USE_IMMERSIVE_DARK_MODE),
        ctypes.byref(value),
        ctypes.sizeof(value),
    )


class DarkApp(QApplication):
    def notify(self, receiver, event):
        result = super().notify(receiver, event)
        if (
            event.type() == event.Type.Show
            and isinstance(receiver, QWidget)
            and receiver.isWindow()
        ):
            try:
                hwnd = int(receiver.winId())
                set_dark_title_bar(hwnd)
            except Exception:
                pass
        return result


def load_ui(main_window, splash_screen, app, gray_value=None):
    app.setStyleSheet(THEME)

    def update_progress(current, total, name):
        splash_screen.update(current, total, name, colour)

    splash_screen.set_total_items(len(filmstocks))

    loaded_filmstocks = load_filmstocks(update_progress, gray_value)

    window = main_window(loaded_filmstocks)

    window.show()
    splash_screen.close()

    sys.exit(app.exec())
