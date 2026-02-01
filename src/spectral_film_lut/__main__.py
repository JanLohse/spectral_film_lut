from spectral_film_lut.splash_screen import launch_splash_screen


def run():
    app, splash_screen = launch_splash_screen("Spectral Film LUT")

    from spectral_film_lut.gui import MainWindow
    from spectral_film_lut.film_loader import load_ui

    load_ui(MainWindow, splash_screen, app)


if __name__ == "__main__":
    run()
