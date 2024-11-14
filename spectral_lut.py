import time

import colour

spectral_shape = colour.SPECTRAL_SHAPE_DEFAULT

from utility import *

from negative_film.kodak_5207 import Kodak5207
from print_film.kodak_2383 import Kodak2383

if __name__ == '__main__':
    start = time.time()

    colour.plotting.plot_multi_sds((rgb_to_spectral(np.array([1, 1, 1]) * .35), kelvin_to_spectral(5000)))

    kodak_5207 = Kodak5207()
    kodak_2383 = Kodak2383()
    for print_light in [5000]:
        print_spectral = kelvin_to_spectral(print_light)
        for projection_light in [5400]:
            projection_spectral = kelvin_to_spectral(projection_light)
            rgb = arri_to_film_sRGB(np.ones(3), kodak_5207, kodak_2383, print_light, projection_light, linear=True)
            norm = max(rgb)

            lut = colour.LUT3D(size=16, name=f"{print_light}-{projection_light}")

            convert = lambda x: arri_to_film_sRGB(x, kodak_5207, kodak_2383, print_spectral, projection_spectral, norm)

            lut.table = np.apply_along_axis(convert, 3, lut.table)

            colour.io.write_LUT(lut, f"{print_light}-{projection_light}.cube")
            print(f"{print_light}-{projection_light}.cube", end=" ")
    end = time.time()
    print(f"\nfinished in {end - start:.2f}s")
