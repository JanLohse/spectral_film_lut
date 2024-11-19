import time

from negative_film.kodak_5207 import Kodak5207
from print_film.kodak_2383 import Kodak2383
from reversal_film.kodachrome_64 import Kodachrome64
from utility import *

setups = {'5207-2383': ((Kodak5207(), Kodak2383()), (printer_light(3750, 310, g=.5), kelvin_to_spectral(5500, 150))),
          'Kodachrome': ((Kodachrome64(), ), (printer_light(6500, 130, g=.1, b=.2), ))}

if __name__ == '__main__':
    start = time.time()

    combo = 'Kodachrome'

    film_stocks, lights = setups[combo]

    rgb_gray = map_rgb(np.ones(3) * .391, film_stocks, lights)
    rgb_white = map_rgb(np.ones(3), film_stocks, lights)
    print(f"{rgb_gray=} {rgb_white=}")

    lut = colour.LUT3D(size=16, name=combo)

    convert = lambda x: map_rgb(x, film_stocks, lights)

    lut.table = np.apply_along_axis(convert, 3, lut.table)

    colour.io.write_LUT(lut, f"LUTs/{combo}.cube")
    print(f"{combo}.cube", end=" ")
    end = time.time()
    print(f"\nfinished in {end - start:.2f}s")
