import os
import time

import colour
import ffmpeg

from film_spectral import FilmSpectral
from negative_film.kodak_5207 import Kodak5207
from print_film.kodak_2383 import Kodak2383
from print_film.kodak_2393 import Kodak2393
from reversal_film.kodachrome_64 import Kodachrome64


def create_lut(negative_film, print_film=None, size=33, name="test", verbose=True, **kwargs):
    lut = colour.LUT3D(size=size, name="test")
    transform = FilmSpectral.generate_conversion(negative_film, print_film, **kwargs)
    start = time.time()
    lut.table = transform(lut.table)
    end = time.time()
    path = f"LUTs/{name}.cube"
    colour.io.write_LUT(lut, path)
    if verbose:
        print(f"created {path} in {end - start:.2f} seconds")
    return path


if __name__ == '__main__':
    start = time.time()
    kodak5207 = Kodak5207()
    kodak2393 = Kodak2393()
    kodak2383 = Kodak2383()
    kodachrome64 = Kodachrome64()
    luts = ["LUTs/Filmbox_Full.cube", "LUTs/ARRI_LogC3_709_33.cube"]
    luts.append(create_lut(kodak5207, kodak2383, size=65, name="2383"))
    luts.append(create_lut(kodak5207, kodak2393, size=65, name="2393"))
    luts.append(create_lut(kodak5207, kodak2383, size=65, name="2383_matrix", print_matrix=True))
    luts.append(create_lut(kodak5207, kodak2393, size=65, name="2393_matrix", print_matrix=True))
    luts.append(create_lut(kodachrome64, size=33, name="Kodachrome"))
    src = "ARRI_ALEXA_Mini_LF_LogC3.tif"
    for lut in luts:
        name = f"{src.split('.')[0]}_{lut.split('/')[-1].split('.')[0]}.jpg"
        if os.path.isfile(name):
            os.remove(name)
        ffmpeg.input(src).filter('lut3d', file=lut).output(name, loglevel="quiet").run()
