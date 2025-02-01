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

def create_lut(negative_film, print_film=None, size=16, name="test", verbose=False):
    lut = colour.LUT3D(size=size, name="test")
    transform = FilmSpectral.generate_conversion_spectral(negative_film, print_film, projector_kelvin=5500, verbose=verbose)
    # lut.table = np.apply_along_axis(transform, 3, lut.table)
    lut.table = transform(lut.table)
    if verbose:
        print(lut.table)

    colour.io.write_LUT(lut, f"LUTs/{name}.cube")
    print(f"{name}.cube", end=" ")


if __name__ == '__main__':
    negative_film = Kodachrome64()
    print_film = None
    for kodak in [negative_film, print_film]:
        if kodak is None:
            continue
        for x in []:
            plot_single_cmfs(MultiSpectralDistributions(x, colour.SPECTRAL_SHAPE_DEFAULT))
        plt.plot(kodak.red_log_exposure, kodak.red_density_curve, 'red')
        plt.plot(kodak.green_log_exposure, kodak.green_density_curve, 'green')
        plt.plot(kodak.blue_log_exposure, kodak.blue_density_curve, 'blue')
        plt.show()
    transform = FilmSpectral.generate_conversion(negative_film, print_film, projector_kelvin=5500, verbose=True, input_colourspace="sRGB")
    transform_2 = FilmSpectral.generate_conversion_spectral(negative_film, print_film, projector_kelvin=5500, verbose=True, input_colourspace="sRGB")
    print('white')
    white = transform(np.ones(3) * 1.), transform_2(np.ones(3) * 1.)
    print("gray")
    gray = transform(np.ones(3) * 0.5), transform_2(np.ones(3) * 0.5)
    print("black")
    black = transform(np.ones(3) * 0.), transform_2(np.ones(3) * 0.)
    print("red")
    red = transform(np.array([1, 0, 0])), transform_2(np.array([1, 0, 0]))
    print("green")
    green = transform(np.array([0, 1, 0])), transform_2(np.array([0, 1, 0]))
    print("blue")
    blue = transform(np.array([0, 0, 1])), transform_2(np.array([0, 0, 1]))
    create_lut(negative_film, print_film, size=33, name="Kodachrome64_2", verbose=False)