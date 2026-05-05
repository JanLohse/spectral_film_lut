"""
A helper function to load film stocks with a dynamic loading bar.
"""

import sys

import colour

from spectral_film_lut import FILM_STOCKS
from spectral_film_lut.css_theme import THEME
from spectral_film_lut.film_spectral import FilmSpectral


def load_filmstocks(progress_callback, gray_value=None):
    """
    Processes film stock data in parallel and reports the progress to the progress bar.
    """
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


def load_ui(
    main_window,
    splash_screen,
    app,
    gray_value: float | None = None,
    exit_immediately: bool = False,
):
    """
    Helper function for loading UI elements in *spectral_film_lut* and *raw2film*.
    """
    app.setStyleSheet(THEME)

    def update_progress(current, total, name):
        splash_screen.update(current, total, name, colour)

    splash_screen.set_total_items(len(FILM_STOCKS))

    loaded_filmstocks = load_filmstocks(update_progress, gray_value)

    window = main_window(loaded_filmstocks)

    window.show()
    splash_screen.close()

    if exit_immediately:
        from PyQt6.QtCore import QTimer

        QTimer.singleShot(1000, app.quit)

    sys.exit(app.exec())
