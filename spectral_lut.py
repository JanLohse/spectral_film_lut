import time

import colour

from negative_film.kodak_5207 import Kodak5207
from print_film.kodak_2383 import Kodak2383
from print_film.kodak_2393 import Kodak2393
from reversal_film.kodachrome_64 import Kodachrome64
from utility import *
from colour.plotting import plot_single_sd, plot_multi_sds, plot_single_cmfs, plot_multi_cmfs
from colour import SpectralDistribution, MultiSpectralDistributions
from matplotlib import pyplot as plt

def create_lut(film_stocks, size=16, name="test", verbose=False):
    lut = colour.LUT3D(size=size, name="test")
    transform = FilmSpectral.generate_conversion(film_stocks, projector_kelvin=5500, verbose=verbose)
    # lut.table = np.apply_along_axis(transform, 3, lut.table)
    lut.table = transform(lut.table)
    if verbose:
        print(lut.table)

    colour.io.write_LUT(lut, f"LUTs/{name}.cube")
    print(f"{name}.cube", end=" ")


if __name__ == '__main__':
    film_stocks = [Kodak5207(), Kodak2383()]
    for kodak in film_stocks:
        for x in []:
            plot_single_cmfs(MultiSpectralDistributions(x, colour.SPECTRAL_SHAPE_DEFAULT))
        plt.plot(kodak.red_log_exposure, kodak.red_density_curve, 'red')
        plt.plot(kodak.green_log_exposure, kodak.green_density_curve, 'green')
        plt.plot(kodak.blue_log_exposure, kodak.blue_density_curve, 'blue')
        plt.show()
    transform = FilmSpectral.generate_conversion(film_stocks, projector_kelvin=5500, verbose=True)
    transform(np.ones(3) * 1.)
    print("lol")
    transform(np.ones(3) * 0.391)
    print("lol")
    transform(np.ones(3) * 0.)

    create_lut(film_stocks, size=33, name="KodakVision", verbose=False)