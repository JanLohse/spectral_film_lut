import os
import time

import colour
import ffmpeg
import numpy as np
from colour import MultiSpectralDistributions, SpectralDistribution

from film_spectral import FilmSpectral
from negative_film.kodak_5207 import Kodak5207
from negative_film.kodak_portra_400 import KodakPortra400
from print_film.kodak_2383 import Kodak2383
from print_film.kodak_2393 import Kodak2393
from print_film.kodak_endura_premier import KodakEnduraPremier
from reversal_film.kodachrome_64 import Kodachrome64


def create_lut(negative_film, print_film=None, size=67, name="test", verbose=True, **kwargs):
    lut = colour.LUT3D(size=size, name="test")
    transform = FilmSpectral.generate_conversion(negative_film, print_film, **kwargs)
    start = time.time()
    lut.table = transform(lut.table)
    end = time.time()
    path = f"LUTs/{name}.cube"
    if not os.path.exists("LUTs"):
        os.makedirs("LUTs")
    colour.io.write_LUT(lut, path)
    if verbose:
        print(f"created {path} in {end - start:.2f} seconds")
    return path


if __name__ == '__main__':
    kodak5207 = Kodak5207()
    kodak2393 = Kodak2393()
    kodak2383 = Kodak2383()
    portra = KodakPortra400()
    kodachrome64 = Kodachrome64()
    endura = KodakEnduraPremier()
    luts = []
    luts.append(create_lut(kodak5207, kodak2383, name="2383"))
    luts.append(create_lut(kodak5207, kodak2393, name="2393"))
    luts.append(create_lut(kodachrome64, name="Kodachrome"))
    luts.append(create_lut(portra, endura, name="Portra"))
    src = "ARRI_Alexa_35_AWG4_LogC4.tif"
    for lut in luts:
        name = f"{src.split('.')[0]}_{lut.split('/')[-1].split('.')[0]}.jpg"
        if os.path.isfile(name):
            os.remove(name)
        ffmpeg.input(src).filter('lut3d', file=lut).output(name, loglevel="quiet").run()
