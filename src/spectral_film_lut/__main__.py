from spectral_film_lut.splash_screen import launch_splash_screen
from spectral_film_lut import __version__

def run():
    app, splash_screen = launch_splash_screen(f"Spectral Film LUT", __version__)

    from spectral_film_lut.gui import MainWindow
    from spectral_film_lut.film_loader import load_ui

    load_ui(MainWindow, splash_screen, app)


if __name__ == "__main__":
    run()
