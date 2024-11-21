import math
from abc import ABC

import colour
import numpy as np

colour.SPECTRAL_SHAPE_DEFAULT = colour.SpectralShape(380, 760, 1)


class FilmSpectral(ABC):
    def calibrate(self):
        # target exposure of middle gray in log lux-seconds
        # normally use iso value, if not provided use target density of 1.0 on the green channel
        if self.iso:
            self.log_H_ref = math.log10(12.5 / self.iso)
        else:
            self.log_H_ref = np.interp(1., self.green_density_curve, self.green_log_exposure)

        # convert relative camera exposure to absolute exposure in log lux-seconds for characteristic curve
        if self.exposure_base != 10:
            self.red_log_exposure = np.log10(self.exposure_base ** self.red_log_exposure * 10 ** self.log_H_ref)
            self.green_log_exposure = np.log10(self.exposure_base ** self.green_log_exposure * 10 ** self.log_H_ref)
            self.blue_log_exposure = np.log10(self.exposure_base ** self.blue_log_exposure * 10 ** self.log_H_ref)

        # density values produced by flat exposure of intensity 1
        if self.density_type == 'd-min':
            red_sensitivity_density = np.min(self.red_density_curve) + self.ref_density
            green_sensitivity_density = np.min(self.green_density_curve) + self.ref_density
            blue_sensitivity_density = np.min(self.blue_density_curve) + self.ref_density
        elif self.density_type == 'e.n.d.':
            red_sensitivity_density = max(self.cyan_spectral_density.values)
            green_sensitivity_density = max(self.magenta_spectral_density.values)
            blue_sensitivity_density = max(self.yellow_spectral_density.values)
        elif self.density_type == 'absolute':
            red_sensitivity_density = self.ref_density
            green_sensitivity_density = self.ref_density
            blue_sensitivity_density = self.ref_density

        print(f"{red_sensitivity_density=} {green_sensitivity_density=} {blue_sensitivity_density=}")
        print(f"red_target_density={np.interp(self.log_H_ref, self.red_log_exposure, self.red_density_curve)} green_target_density={np.interp(self.log_H_ref, self.green_log_exposure, self.green_density_curve)} blue_target_density={np.interp(self.log_H_ref, self.blue_log_exposure, self.blue_density_curve)}")

        # exposure producing density equivalent to density specified in density curve
        self.red_sensitivity_exposure = 10 ** np.interp(red_sensitivity_density, self.red_density_curve,
                                                        self.red_log_exposure)
        self.green_sensitivity_exposure = 10 ** np.interp(green_sensitivity_density, self.green_density_curve,
                                                          self.green_log_exposure)
        self.blue_sensitivity_exposure = 10 ** np.interp(blue_sensitivity_density, self.blue_density_curve,
                                                         self.blue_log_exposure)

        print(f"red_sensitivity_exposure{math.log10(self.red_sensitivity_exposure)} green_sensitivity_exposure{math.log10(self.green_sensitivity_exposure)} blue_sensitivity_exposure{math.log10(self.blue_sensitivity_exposure)}")

        # compute exposure compensation, such that a middle gray exposure produces the target density
        exposure_compensation = 10 ** self.log_H_ref / self.green_sensitivity_exposure / np.sum(self.magenta_sensitivity.values) / .18

        self.red_sensitivity_exposure *= exposure_compensation
        self.green_sensitivity_exposure *= exposure_compensation
        self.blue_sensitivity_exposure *= exposure_compensation

        # peak normalize dye density curves
        self.cyan_spectral_density /= max(self.cyan_spectral_density.values)
        self.magenta_spectral_density /= max(self.magenta_spectral_density.values)
        self.yellow_spectral_density /= max(self.yellow_spectral_density.values)

    def spectral_to_log_exposure(self, light_intensity):
        cyan_spectral_exposure = light_intensity * self.cyan_sensitivity * self.red_sensitivity_exposure
        magenta_spectral_exposure = light_intensity * self.magenta_sensitivity * self.green_sensitivity_exposure
        yellow_spectral_exposure = light_intensity * self.yellow_sensitivity * self.blue_sensitivity_exposure

        cyan_effective_exposure = np.sum(cyan_spectral_exposure.values)
        magenta_effective_exposure = np.sum(magenta_spectral_exposure.values)
        yellow_effective_exposure = np.sum(yellow_spectral_exposure.values)

        effective_exposure = np.array([cyan_effective_exposure, magenta_effective_exposure, yellow_effective_exposure])
        log_exposure = np.log10(effective_exposure)

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
