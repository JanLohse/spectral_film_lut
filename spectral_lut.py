import time
import sys

from negative_film.kodak_5207 import Kodak5207
from print_film.kodak_2383 import Kodak2383
from intermediate_film.kodak_5242 import Kodak5242
from utility import *



def map_rgb(rgb, film_stocks, lights, rgb_to_spectral=arri_to_spectral, norm=1., gamma=2.2, linear=False):
    spectral = rgb_to_spectral(rgb)

    colour.plotting.plot_single_sd(spectral)

    for film_stock, light in zip(film_stocks, lights):
        spectral = film_stock.spectral_to_projection(spectral, light)

        colour.plotting.plot_single_sd(spectral)

    XYZ = colour.sd_to_XYZ(spectral, k=0.01) / norm

    sRGB = colour.XYZ_to_sRGB(XYZ)

    if linear:
        return sRGB ** 2.2

    return np.clip(sRGB, 0, 1)




if __name__ == '__main__':
    start = time.time()

    kodak_positive = Kodak5242('positive')

    # colour.plotting.plot_multi_sds((kodak_positive.cyan_spectral_density + kodak_positive.magenta_spectral_density + kodak_positive.yellow_spectral_density,
    #                                 kodak_positive.cyan_spectral_density * kodak_positive.target_density[0], kodak_positive.magenta_spectral_density * kodak_positive.target_density[1], kodak_positive.yellow_spectral_density * kodak_positive.target_density[2],
    #                                 kodak_positive.midscale_spectral_density, kodak_positive.minimum_spectral_density))

    # colour.plotting.plot_single_sd(kodak_positive.spectral_to_projection(kelvin_to_spectral(3200), kelvin_to_spectral(3200)))

    # colour.plotting.plot_multi_sds((kodak_positive.cyan_spectral_density * kodak_positive.target_density[0], kodak_positive.yellow_spectral_density * kodak_positive.target_density[1], kodak_positive.magenta_spectral_density * kodak_positive.target_density[2]))

    film_stocks = [Kodak5207(), Kodak5242('negative'), Kodak5242('positive'), Kodak2383()]
    lights = [3200, 3200, 3200, 3200]
    lights = [kelvin_to_spectral(light) for light in lights]

    rgb = map_rgb(np.array([1, 0, 0]) * .391, film_stocks, lights, linear=True)

    sys.exit()

    norm = max(max(rgb), 1)

    print(map_rgb(np.ones(3) * .391, film_stocks, lights, linear=False, norm=norm))

    lut = colour.LUT3D(size=8, name=f"Kodak Vision 250D")

    convert = lambda x: map_rgb(x, film_stocks, lights, norm=norm)

    lut.table = np.apply_along_axis(convert, 3, lut.table)

    colour.io.write_LUT(lut, "LUTs/Kodak_5207.cube")
    print(f"Kodak_5207.cube", end=" ")
    end = time.time()
    print(f"\nfinished in {end - start:.2f}s")
