import ctypes
import sys
from ctypes import wintypes

import colour
from PyQt6.QtWidgets import QApplication, QWidget

from spectral_film_lut import FILM_STOCKS
from spectral_film_lut.css_theme import PRESSED_COLOR, TEXT_PRIMARY, THEME
from spectral_film_lut.film_spectral import FilmSpectral


def load_filmstocks(progress_callback, gray_value=None):
    result = []
    total = len(FILM_STOCKS)
    for i, film_stock in enumerate(FILM_STOCKS, start=1):
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

    splash_screen.set_total_items(len(FILM_STOCKS))

    loaded_filmstocks = load_filmstocks(update_progress, gray_value)

    window = main_window(loaded_filmstocks)

    window.show()
    splash_screen.close()

    sys.exit(app.exec())
