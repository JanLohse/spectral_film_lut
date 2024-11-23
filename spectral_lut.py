import time
import sys

import colour

from negative_film.kodak_5207 import Kodak5207
from print_film.kodak_2383 import Kodak2383
from print_film.kodak_2393 import Kodak2393
from reversal_film.kodachrome_64 import Kodachrome64
from utility import *

setups = {'5207-2383': ((Kodak5207(), Kodak2383()), (printer_light(4530, 288, g=0.43), kelvin_to_spectral(5500, 140))),
          '5207-2393': ((Kodak5207(), Kodak2393()), (printer_light(4500, 297, g=0.38), kelvin_to_spectral(5500, 140))),
          'Kodachrome': ((Kodachrome64(),), (printer_light(5500, 140, g=0.3),))}


# kodak lad 1.09 1.06 1.03

def test_illuminants(film_stock):
    kodak = film_stock()

    illuminant_1 = colour.SDS_ILLUMINANTS['E'].align(colour.SPECTRAL_SHAPE_DEFAULT) / 100 * .18
    illuminant_2 = kelvin_to_spectral(5500, 18)
    illuminant_3 = arri_to_spectral(np.ones(3) * .391)
    log_exposure = kodak.spectral_to_log_exposure(illuminant_2)
    density = kodak.log_exposure_to_density(log_exposure)
    spectral_density = kodak.density_to_spectral_density(density)

    density_2 = kodak.log_exposure_to_density(np.ones(3) * kodak.log_H_ref)
    spectral_density_2 = kodak.density_to_spectral_density(density_2)
    print(f"{log_exposure=} {density=} {kodak.log_H_ref=}")
    colour.plotting.plot_multi_sds(
        (spectral_density, spectral_density_2, kodak.midscale_spectral_density, illuminant_1, illuminant_2))
    # colour.plotting.plot_multi_sds((kodak.yellow_sensitivity, kodak.magenta_sensitivity, kodak.cyan_sensitivity))
    # colour.plotting.plot_multi_sds((kodak.yellow_spectral_density, kodak.magenta_spectral_density, kodak.cyan_spectral_density))
    sys.exit()


def create_lut(setup, size=16):
    start = time.time()

    film_stocks, lights = setups[setup]

    rgb_gray = map_rgb(np.ones(3) * .391, film_stocks, lights)
    rgb_white = map_rgb(np.ones(3), film_stocks, lights)
    print(f"{rgb_gray=} {rgb_white=}")
    if len(film_stocks) == 2 and False:
        projection = film_stocks[0].spectral_to_projection(arri_to_spectral(np.ones(3) * .391), lights[0])
        log_exposure = film_stocks[1].spectral_to_log_exposure(projection)
        density = film_stocks[1].log_exposure_to_density(log_exposure)
        spectral_density = film_stocks[1].density_to_spectral_density(density)
        indices = [film_stocks[1].cyan_spectral_density.values.argmax(),
                   film_stocks[1].magenta_spectral_density.values.argmax(),
                   film_stocks[1].yellow_spectral_density.values.argmax()]
        # colour.plotting.plot_single_sd(film_stocks[1].spectral_to_projection(projection, lights[1]))
        print(f"LAD_density={spectral_density.values[indices]}")

    lut = colour.LUT3D(size=size, name=setup)

    convert = lambda x: map_rgb(x, film_stocks, lights)

    lut.table = np.apply_along_axis(convert, 3, lut.table)

    colour.io.write_LUT(lut, f"LUTs/{setup}.cube")
    print(f"{setup}.cube", end=" ")
    end = time.time()
    print(f"\nfinished in {end - start:.2f}s")


if __name__ == '__main__':
    for setup in setups:
        create_lut(setup, 16)
