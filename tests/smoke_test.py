import os

from spectral_film_lut.__main__ import run

os.environ["QT_QPA_PLATFORM"] = "offscreen"
run(exit_immediately=True)
