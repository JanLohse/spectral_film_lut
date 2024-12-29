import time

from negative_film.kodak_5207 import Kodak5207
from print_film.kodak_2383 import Kodak2383
from print_film.kodak_2393 import Kodak2393
from reversal_film.kodachrome_64 import Kodachrome64
from utility import *

setups = {'5207-2383': ((Kodak5207(), Kodak2383()), (printer_light(4500, 300, g=0.2), kelvin_to_spectral(5500, 140))),
          '5207-2393': ((Kodak5207(), Kodak2393()), (printer_light(4300, 320, g=0.), kelvin_to_spectral(5500, 140))),
          'Kodachrome': ((Kodachrome64(),), (printer_light(7000, 140, g=0.),))}


def create_lut(setup, size=16):
    start = time.time()

    film_stocks, lights = setups[setup]

    rgb_gray = map_rgb(np.ones(3) * .391, film_stocks, lights)
    rgb_white = map_rgb(np.ones(3), film_stocks, lights)
    print(f"{rgb_gray=} {rgb_white=}")
    lut = colour.LUT3D(size=size, name=setup)

    convert = lambda x: map_rgb(x, film_stocks, lights)

    lut.table = np.apply_along_axis(convert, 3, lut.table)

    colour.io.write_LUT(lut, f"LUTs/{setup}.cube")
    print(f"{setup}.cube", end=" ")
    end = time.time()
    print(f"\nfinished in {end - start:.2f}s")


if __name__ == '__main__':
    for setup in setups:
        create_lut(setup, 8)
