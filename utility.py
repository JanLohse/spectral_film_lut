import colour
import numpy as np

from film_spectral import FilmSpectral


def arri_to_spectral(rgb):
    arri_wcg_linear = colour.models.log_decoding_ARRILogC3(rgb)

    XYZ = colour.RGB_to_XYZ(arri_wcg_linear, 'ARRI Wide Gamut 3')

    spectral = colour.XYZ_to_sd(XYZ, method='Otsu 2018').align(colour.SPECTRAL_SHAPE_DEFAULT)

    return spectral


def rgb_to_spectral(rgb):
    XYZ = colour.sRGB_to_XYZ(rgb)

    spectral = colour.XYZ_to_sd(XYZ, method='Otsu 2018').align(colour.SPECTRAL_SHAPE_DEFAULT)

    return spectral


def normalize_spectral(spectral, target_flux=18):
    spectral *= target_flux / colour.sd_to_XYZ(spectral)[1]

    return spectral


def kelvin_to_spectral(kelvin):
    spectral = colour.sd_blackbody(kelvin, colour.SPECTRAL_SHAPE_DEFAULT)
    spectral = normalize_spectral(spectral)

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
