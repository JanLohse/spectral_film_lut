import time

from negative_film.kodak_5207 import Kodak5207
from print_film.kodak_2383 import Kodak2383
from print_film.kodak_2393 import Kodak2393
from reversal_film.kodachrome_64 import Kodachrome64
from utility import *


def create_lut(negative_film, print_film=None, size=16, name="test", **kwargs):
    lut = colour.LUT3D(size=size, name="test")
    transform = FilmSpectral.generate_conversion(negative_film, print_film, projector_kelvin=5500, **kwargs)
    lut.table = transform(lut.table)
    colour.io.write_LUT(lut, f"LUTs/{name}.cube")

if __name__ == '__main__':
    start = time.time()
    kodak5207 = Kodak5207()
    kodak2393 = Kodak2393()
    kodak2383 = Kodak2383()
    kodachrome64 = Kodachrome64()
    print(f"init {time.time() - start:.2f}s")
    start = time.time()
    create_lut(kodak5207, kodak2383, size=33, name="2383")
    create_lut(kodak5207, kodak2393, size=33, name="2393")
    print(f"spectral {time.time() - start:.2f}s")
    start = time.time()
    create_lut(kodak5207, kodak2383, size=33, name="2383_matrix", print_matrix=True)
    create_lut(kodak5207, kodak2393, size=33, name="2393_matrix", print_matrix=True)
    print(f"matrix {time.time() - start:.2f}s")
    start = time.time()
    create_lut(kodachrome64, size=33, name="Kodachrome")
