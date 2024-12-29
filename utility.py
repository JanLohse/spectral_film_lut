import colour
import numpy as np
from scipy.ndimage import gaussian_filter

from film_spectral import FilmSpectral


def arri_to_spectral(rgb):
    arri_wcg_linear = colour.models.log_decoding_ARRILogC3(rgb)

    XYZ = colour.RGB_to_XYZ(arri_wcg_linear, 'ARRI Wide Gamut 3')

    spectral = colour.XYZ_to_sd(XYZ, method='Jakob 2019').align(colour.SPECTRAL_SHAPE_DEFAULT)

    return spectral


def rgb_to_spectral(rgb):
    XYZ = colour.sRGB_to_XYZ(rgb)

    spectral = colour.XYZ_to_sd(XYZ, method='Jakob 2019').align(colour.SPECTRAL_SHAPE_DEFAULT)

    return spectral


def normalize_spectral(spectral, target_flux=18):
    spectral *= target_flux / colour.sd_to_XYZ(spectral)[1]

    return spectral


def kelvin_to_spectral(kelvin, target_flux=100):
    spectral = colour.sd_blackbody(kelvin, colour.SPECTRAL_SHAPE_DEFAULT)
    spectral = normalize_spectral(spectral, target_flux)

    return spectral


def arri_to_film_sRGB(arri, negative: FilmSpectral, print_film: FilmSpectral, print_light, projection_light, norm=1.,
                      linear=False):
    spectral_input = arri_to_spectral(arri)

    printer_lights = negative.spectral_to_projection(spectral_input, print_light)

    projection = print_film.spectral_to_projection(printer_lights, projection_light) / norm

    XYZ = colour.sd_to_XYZ(projection, k=0.01)

    sRGB = colour.XYZ_to_sRGB(XYZ)

    if linear:
        return sRGB ** (1 / 2.2)

    return np.clip(sRGB, 0, 1)


def arri_to_sRGB(arri):
    arri_wcg_linear = colour.models.log_decoding_ARRILogC3(arri)

    XYZ = colour.RGB_to_XYZ(arri_wcg_linear, 'ARRI Wide Gamut 3')

    sRGB = colour.XYZ_to_sRGB(XYZ)

    sRGB = np.clip(sRGB, 0, 1)

    return sRGB


def color_filters(light, r=0, g=0, b=0, b_g_cut=515, g_r_cut=575):
    wavelengths = np.array([*colour.SPECTRAL_SHAPE_DEFAULT])
    transmittance = np.piecewise(wavelengths,
                                 [wavelengths < b_g_cut, (b_g_cut <= wavelengths) & (wavelengths <= g_r_cut),
                                  g_r_cut < wavelengths], [2 ** b, 2 ** g, 2 ** r])
    transmittance = gaussian_filter(transmittance, 5)
    transmittance = colour.SpectralDistribution(transmittance, colour.SPECTRAL_SHAPE_DEFAULT)
    filtered = light * transmittance

    return filtered


def printer_light(kelvin, lux=100, **kwargs):
    light = kelvin_to_spectral(kelvin, lux)
    filtered = color_filters(light, **kwargs)

    return filtered


def map_rgb(rgb, film_stocks, lights, rgb_to_spectral=arri_to_spectral):
    spectral = rgb_to_spectral(rgb)

    for film_stock, light in zip(film_stocks, lights):
        spectral = film_stock.spectral_to_projection(spectral, light)

    XYZ = colour.sd_to_XYZ(spectral, k=0.01)

    sRGB = colour.XYZ_to_sRGB(XYZ)

    return np.clip(sRGB, 0, 1)
