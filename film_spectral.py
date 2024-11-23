import math
from abc import ABC

import colour
import numpy as np

colour.SPECTRAL_SHAPE_DEFAULT = colour.SpectralShape(400, 720, 1)


def kelvin_to_spectral(kelvin, target_flux=100):
    spectral = colour.sd_blackbody(kelvin, colour.SPECTRAL_SHAPE_DEFAULT)
    spectral *= target_flux / colour.sd_to_XYZ(spectral)[1]

    return spectral


class FilmSpectral(ABC):
    def calibrate(self):
        # target exposure of middle gray in log lux-seconds
        # normally use iso value, if not provided use target density of 1.0 on the green channel
        if self.iso:
            self.log_H_ref = math.log10(12.5 / self.iso)
        else:
            self.log_H_ref = np.interp(1., self.green_density_curve, self.green_log_exposure)

        # extrapolate log_sensitivity to linear sensitivity
        self.yellow_log_sensitivity = colour.SpectralDistribution(self.yellow_log_sensitivity).align(colour.SPECTRAL_SHAPE_DEFAULT, extrapolator_kwargs={'method': 'linear'})
        self.magenta_log_sensitivity = colour.SpectralDistribution(self.magenta_log_sensitivity).align(colour.SPECTRAL_SHAPE_DEFAULT, extrapolator_kwargs={'method': 'linear'})
        self.cyan_log_sensitivity = colour.SpectralDistribution(self.cyan_log_sensitivity).align(colour.SPECTRAL_SHAPE_DEFAULT, extrapolator_kwargs={'method': 'linear'})
        self.yellow_sensitivity = colour.SpectralDistribution(10 ** self.yellow_log_sensitivity.values, colour.SPECTRAL_SHAPE_DEFAULT)
        self.magenta_sensitivity = colour.SpectralDistribution(10 ** self.magenta_log_sensitivity.values, colour.SPECTRAL_SHAPE_DEFAULT)
        self.cyan_sensitivity = colour.SpectralDistribution(10 ** self.cyan_log_sensitivity.values, colour.SPECTRAL_SHAPE_DEFAULT)

        # align spectral densities
        self.yellow_spectral_density.align(colour.SPECTRAL_SHAPE_DEFAULT, extrapolator_kwargs={'method': 'linear'})
        self.magenta_spectral_density.align(colour.SPECTRAL_SHAPE_DEFAULT, extrapolator_kwargs={'method': 'linear'})
        self.cyan_spectral_density.align(colour.SPECTRAL_SHAPE_DEFAULT, extrapolator_kwargs={'method': 'Linear'})
        self.midscale_spectral_density.align(colour.SPECTRAL_SHAPE_DEFAULT, extrapolator_kwargs={'method': 'linear'})

        self.yellow_spectral_density.values = np.clip(self.yellow_spectral_density.values, 0, None)
        self.magenta_spectral_density.values = np.clip(self.magenta_spectral_density.values, 0, None)
        self.cyan_spectral_density.values = np.clip(self.cyan_spectral_density.values, 0, None)
        self.midscale_spectral_density.values = np.clip(self.midscale_spectral_density.values, 0, None)

        # convert relative camera exposure to absolute exposure in log lux-seconds for characteristic curve
        if self.exposure_base != 10:
            self.red_log_exposure = np.log10(self.exposure_base ** self.red_log_exposure * 10 ** self.log_H_ref)
            self.green_log_exposure = np.log10(self.exposure_base ** self.green_log_exposure * 10 ** self.log_H_ref)
            self.blue_log_exposure = np.log10(self.exposure_base ** self.blue_log_exposure * 10 ** self.log_H_ref)

        target_illuminant = kelvin_to_spectral(self.target_illuminant_kelvin, 18)
        self.exposure_comp = np.ones(3)
        ref_exposure = 10 ** self.spectral_to_log_exposure(target_illuminant)
        self.exposure_comp = 10 ** self.log_H_ref / ref_exposure

        # peak normalize dye density curves
        self.cyan_spectral_density.normalise()
        self.magenta_spectral_density.normalise()
        self.yellow_spectral_density.normalise()

    def spectral_to_log_exposure(self, light_intensity):
        cyan_spectral_exposure = light_intensity * self.cyan_sensitivity
        magenta_spectral_exposure = light_intensity * self.magenta_sensitivity
        yellow_spectral_exposure = light_intensity * self.yellow_sensitivity

        cyan_effective_exposure = np.sum(cyan_spectral_exposure.values)
        magenta_effective_exposure = np.sum(magenta_spectral_exposure.values)
        yellow_effective_exposure = np.sum(yellow_spectral_exposure.values)

        effective_exposure = np.array(
            [cyan_effective_exposure, magenta_effective_exposure, yellow_effective_exposure]) * self.exposure_comp
        log_exposure = np.log10(np.clip(effective_exposure, 0.000000000000000000001, None))

        return log_exposure

    def log_exposure_to_density(self, log_exposure):
        red_density = np.interp(log_exposure[0], self.red_log_exposure, self.red_density_curve)
        green_density = np.interp(log_exposure[1], self.green_log_exposure, self.green_density_curve)
        blue_density = np.interp(log_exposure[2], self.blue_log_exposure, self.blue_density_curve)

        return np.array([red_density, green_density, blue_density])

    def density_to_spectral_density(self, density):
        spectral_density = self.cyan_spectral_density * density[0] + self.magenta_spectral_density * density[
            1] + self.yellow_spectral_density * density[2]

        return spectral_density

    def spectral_density_to_transmittance(self, spectral_density: colour.SpectralDistribution):
        spectral_density_series = spectral_density.to_series()
        transmittance_series = 10 ** -spectral_density_series
        transmittance = colour.SpectralDistribution(transmittance_series)

        return transmittance

    def spectral_to_transmittance(self, spectral):
        log_exposure = self.spectral_to_log_exposure(spectral)
        density = self.log_exposure_to_density(log_exposure)
        spectral_density = self.density_to_spectral_density(density)
        transmittance = self.spectral_density_to_transmittance(spectral_density)

        return transmittance

    def spectral_to_projection(self, spectral, illuminant):
        projection = self.spectral_to_transmittance(spectral) * illuminant

        return projection
