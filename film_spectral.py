from abc import ABC

from utility import *


class FilmSpectral(ABC):
    def calibrate(self, whitebalace, type: {'negative_film', 'print_film'}):
        self.cyan_spectral_density /= max(self.cyan_spectral_density.values)
        self.magenta_spectral_density /= max(self.magenta_spectral_density.values)
        self.yellow_spectral_density /= max(self.yellow_spectral_density.values)

        if type == 'negative_film':
            coeffs = self.log_exposure_to_density(-1 * np.ones(3))

        elif type == 'print_film':
            A = np.stack([self.cyan_spectral_density.values, self.magenta_spectral_density.values,
                          self.yellow_spectral_density.values])
            coeffs = np.linalg.lstsq(A.T, self.midscale_spectral_density.values, rcond=None)[0]

        self.base_spectral_density = self.midscale_spectral_density - (
                self.cyan_spectral_density * coeffs[0] + self.magenta_spectral_density * coeffs[
            1] + self.yellow_spectral_density * coeffs[2])

        self.exposure_adjustment = np.zeros(3)
        neutral_spectral = kelvin_to_spectral(whitebalace)
        neutral_exposure = self.spectral_to_log_exposure(neutral_spectral)

        cyan_target_exposure = np.interp(coeffs[0], self.red_density_curve, self.red_log_exposure)
        magenta_target_exposure = np.interp(coeffs[1], self.green_density_curve, self.green_log_exposure)
        yellow_target_exposure = np.interp(coeffs[2], self.blue_density_curve, self.blue_log_exposure)

        self.exposure_adjustment = np.array(
            [cyan_target_exposure, magenta_target_exposure, yellow_target_exposure]) - neutral_exposure

        target_flux = 18
        self.amplification = 1
        test_projection = self.spectral_to_projection(neutral_spectral, neutral_spectral)
        self.amplification = target_flux / colour.sd_to_XYZ(test_projection)[1]

    def spectral_to_log_exposure(self, light_intensity):
        cyan_spectral_exposure = light_intensity * self.cyan_sensitivity
        magenta_spectral_exposure = light_intensity * self.magenta_sensitivity
        yellow_spectral_exposure = light_intensity * self.yellow_sensitivity

        cyan_effective_exposure = np.sum(cyan_spectral_exposure.values) * cyan_spectral_exposure.shape.interval
        magenta_effective_exposure = np.sum(magenta_spectral_exposure.values) * magenta_spectral_exposure.shape.interval
        yellow_effective_exposure = np.sum(yellow_spectral_exposure.values) * yellow_spectral_exposure.shape.interval

        effective_exposure = np.array([cyan_effective_exposure, magenta_effective_exposure, yellow_effective_exposure])
        log_exposure = np.emath.logn(self.exposure_base, effective_exposure) + self.exposure_adjustment

        return log_exposure

    def log_exposure_to_density(self, log_exposure):
        red_density = np.interp(log_exposure[0], self.red_log_exposure, self.red_density_curve)
        green_density = np.interp(log_exposure[1], self.green_log_exposure, self.green_density_curve)
        blue_density = np.interp(log_exposure[2], self.blue_log_exposure, self.blue_density_curve)

        return np.array([red_density, green_density, blue_density])

    def density_to_spectral_density(self, density):
        spectral_density = self.cyan_spectral_density * density[0] + self.magenta_spectral_density * density[
            1] + self.yellow_spectral_density * density[2] + self.base_spectral_density

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
        projection = self.spectral_to_transmittance(spectral) * illuminant * self.amplification

        return projection
