"""
The main class for handling all film data processing and rendering.
"""

import math
from functools import cache
from typing import Any

import colour.plotting
import numpy as np
import scipy
from colour import SpectralDistribution
from colour.hints import LiteralRGBColourspace
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator

from spectral_film_lut.color_processing import (
    CDD_TO_CID,
    COLORCHECKER_2005,
    CCT_to_XYZ,
)
from spectral_film_lut.color_space import LAD_NEGATIVE
from spectral_film_lut.config import DEFAULT_DTYPE, SPECTRAL_SHAPE
from spectral_film_lut.densiometry import (
    APD,
    DENSIOMETRY,
    PRINTER_LIGHT_APD,
    PRINTER_LIGHTS,
    adx16_decode,
    adx16_encode,
    construct_spectral_density,
    output_to_density,
)
from spectral_film_lut.film_data import FilmData
from spectral_film_lut.utils import (
    apd_to_xyz,
    film_conversion,
    log_clip,
    multi_channel_interp,
    smooth_roll_off,
)
from spectral_film_lut.xy_lut import (
    RAWTOACES_XYZ,
    XYZ_CMFS,
    UpsampleMethod,
    apply_2d_lut,
    get_spectrum_lut,
)


class FilmSpectral:
    """
    The main class that profiles a film stock from its raw data and provides functions
    for simulating the look of printed or scanned film.
    """

    def __init__(self, film_data: FilmData, gray_value=0.18):
        """Profiles a film stock from its raw data as reported in a datasheet."""
        # Copy variables from data.
        self.name = film_data.name
        self.color_masking = film_data.color_masking
        self.iso = film_data.iso
        self.lad = film_data.lad
        self.exposure_base = film_data.exposure_base
        self.density_measure = film_data.density_measure
        self.rms_curve = film_data.rms_curve
        self.rms_density = film_data.rms_density
        self.rms = film_data.rms
        self.mtf = film_data.mtf
        self.year = film_data.year
        self.stage = film_data.stage
        self.film_type = film_data.film_type
        self.medium = film_data.medium
        self.manufacturer = film_data.manufacturer
        self.exposure_kelvin = film_data.exposure_kelvin
        self.projection_kelvin = film_data.projection_kelvin
        self.alias = film_data.alias
        self.comment = film_data.comment
        self.gray_value = 0.18 if gray_value is None else gray_value
        self.log_sensitivity = film_data.log_sensitivity
        self.sensitivity = film_data.sensitivity

        # Initialize computable variables.
        self.XYZ_to_exp = None
        self.d_min = None
        self.d_ref = None
        self.d_max = None
        self.color = None
        self.resolution = None
        self.color_checker = None
        self.gamma = None
        self.log_H_ref = None
        self.H_ref = None

        self._cid_to_xyz = None

        # Basic conversion from film_data.
        if film_data.d_ref_sd is not None:
            self.d_ref_sd = colour.SpectralDistribution(film_data.d_ref_sd)
        else:
            self.d_ref_sd = None
        if film_data.d_min_sd is not None:
            self.d_min_sd = colour.SpectralDistribution(film_data.d_min_sd)
        else:
            self.d_min_sd = None
        if film_data.spectral_density is not None:
            self.spectral_density = [
                colour.SpectralDistribution(x) for x in film_data.spectral_density
            ]
        else:
            self.spectral_density = None
        if film_data.sensiometric_curve is not None:
            self.log_exposure = [
                np.array(list(curve.keys()), dtype=DEFAULT_DTYPE)
                for curve in film_data.sensiometric_curve
            ]
            self.density_curve = [
                np.array(list(curve.values()), dtype=DEFAULT_DTYPE)
                for curve in film_data.sensiometric_curve
            ]

        # target exposure of middle gray in log lux-seconds
        # normally use iso value, if not provided use target density of 1.0 on the green
        # channel
        if self.iso is not None:
            self.log_H_ref = np.ones(len(self.log_exposure)) * math.log10(12.5 / self.iso)
            self.H_ref = 10**self.log_H_ref
        elif self.density_measure == "bw":
            self.log_H_ref = np.array(
                [
                    np.interp(np.asarray(a), np.asarray(sorted_b), np.asarray(sorted_c))
                    for a, b, c in zip(self.lad, self.density_curve, self.log_exposure)
                    for sorted_b, sorted_c in [zip(*sorted(zip(b, c)))]
                ]
            )
            self.H_ref = 10**self.log_H_ref

        self.color = "BW" if self.density_measure == "bw" else "Color"

        if self.color_masking is None:
            if self.density_measure == "status_m":
                self.color_masking = 1.0
            else:
                self.color_masking = 0.0

        # extrapolate log_sensitivity to linear sensitivity
        if self.log_sensitivity is not None:
            self.log_sensitivity = np.stack(
                [
                    np.asarray(
                        colour.SpectralDistribution(x)
                        .align(SPECTRAL_SHAPE, extrapolator_kwargs={"method": "linear"})
                        .align(SPECTRAL_SHAPE)
                        .values
                    )
                    for x in self.log_sensitivity
                ]
            ).T
            self.sensitivity = 10**self.log_sensitivity

        # Convert relative camera exposure to absolute exposure in log lux-seconds for
        # characteristic curve.
        if self.exposure_base != 10:
            self.log_exposure = [
                np.log10(self.exposure_base**x * 10**y)
                for x, y in zip(self.log_exposure, self.log_H_ref)
            ]

        # Interpolate and process characteristic curve.
        if self.density_measure == "status_m" or self.density_measure == "bw":
            self.extend_characteristic_curve()
        log_H_min = min([x.min() for x in self.log_exposure])
        log_H_max = max([x.max() for x in self.log_exposure])
        x_new = np.linspace(log_H_min, log_H_max, 1024, dtype=DEFAULT_DTYPE)
        self.density_curve = [
            PchipInterpolator(x, y)(x_new).astype(DEFAULT_DTYPE)
            for x, y in zip(self.log_exposure, self.density_curve)
        ]
        self.log_exposure = x_new
        self.d_min = np.array([np.min(x) for x in self.density_curve])
        self.density_curve = [x - d for x, d in zip(self.density_curve, self.d_min)]
        self.sensiometric_curve = np.stack((self.log_exposure, *self.density_curve))
        if self.log_H_ref is not None:
            self.d_ref = self.log_exposure_to_density(self.log_H_ref).reshape(-1)
        self.d_max = np.array([np.max(x) for x in self.density_curve])

        # align spectral densities
        if self.density_measure == "bw":
            self.spectral_density = np.asarray(
                colour.colorimetry.sd_constant(1, SPECTRAL_SHAPE).values
            )
            self.d_min_sd = np.asarray(
                colour.colorimetry.sd_constant(self.d_min, SPECTRAL_SHAPE).values
            )
            self.d_ref_sd = self.spectral_density * self.d_ref + self.d_min
            self.spectral_density = self.spectral_density.reshape(-1, 1)
        else:
            if self.d_min_sd is not None:
                self.d_min_sd = self.gaussian_extrapolation(self.d_min_sd)
                self.d_min_sd = np.asarray(self.d_min_sd.values)
            else:
                self.d_min_sd = np.asarray(colour.sd_zeros(SPECTRAL_SHAPE).values)

            if self.d_ref_sd is not None:
                self.gaussian_extrapolation(self.d_ref_sd)
            if self.spectral_density is not None and self.density_measure != "absolute":
                if (
                    self.density_measure == "status_a"
                    and film_data.d_min_adjustment is None
                    and min([x.values.min() for x in self.spectral_density]) > 0.05
                ) or film_data.d_min_adjustment:
                    self.estimate_d_min_sd()
                self.spectral_density = np.stack(
                    [
                        np.asarray(self.gaussian_extrapolation(x).values)
                        for x in self.spectral_density
                    ]
                ).T
            elif self.density_measure != "absolute":
                self.spectral_density = construct_spectral_density(self.d_ref_sd - self.d_min_sd)

            self.spectral_density /= (
                self.spectral_density * DENSIOMETRY[self.density_measure]
            ).sum(axis=0)

            self._crosstalk_matrix = DENSIOMETRY[self.density_measure].T @ self.spectral_density
            self._crosstalk_matrix_inv = np.linalg.inv(self._crosstalk_matrix)

            density_curve = np.stack(self.density_curve).T

            density_curve @= self._crosstalk_matrix_inv.T
            self.density_curve = [
                density_curve[:, 0],
                density_curve[:, 1],
                density_curve[:, 2],
            ]

            self.d_min_sd = self.d_min_sd + self.spectral_density @ self._crosstalk_matrix_inv @ (
                self.d_min - DENSIOMETRY[self.density_measure].T @ self.d_min_sd
            )
            if self.H_ref is None:
                self.lad = self.compute_lad(self.gray_value)
                self.log_H_ref = np.array(
                    [
                        np.interp(np.asarray(a), np.asarray(sorted_b), np.asarray(sorted_c))
                        for a, b, c in zip(
                            self.lad,
                            self.density_curve,
                            [self.log_exposure] * len(self.lad),
                        )
                        for sorted_b, sorted_c in [zip(*sorted(zip(b, c)))]
                    ]
                )
                self.H_ref = 10**self.log_H_ref
            self.sensiometric_curve = np.stack(
                (self.log_exposure, *self.density_curve), dtype=DEFAULT_DTYPE
            )
            self.d_ref = self.log_exposure_to_density(self.log_H_ref).reshape(-1)
            self.d_ref_sd = self.spectral_density @ self.d_ref + self.d_min_sd

        self.d_max = np.array([np.max(x) for x in self.density_curve])

        if self.rms_curve is not None and self.rms_density is not None:
            rms_temp = [
                self.prepare_rms_data(a, b) for a, b in zip(self.rms_curve, self.rms_density)
            ]
            self.rms_curve = [x[0] for x in rms_temp]
            self.rms_density = [x[1] for x in rms_temp]
            n_channels = len(self.rms_density)
            density_min = min(x.min() for x in self.rms_density)
            density_max = max(x.max() for x in self.rms_density)
            density_common = np.linspace(density_min, density_max, 1024).astype(DEFAULT_DTYPE)
            rms_uniform = np.empty((n_channels, 1024), dtype=DEFAULT_DTYPE)
            for ch in range(n_channels):
                rms_uniform[ch] = np.interp(
                    density_common, self.rms_density[ch], self.rms_curve[ch]
                )
            self.grain_curve = np.stack((density_common, *rms_uniform))
            if n_channels == 3:
                rms_color_factors = np.array([0.26, 0.57, 0.17], dtype=DEFAULT_DTYPE)
                scaling = 1.2375
                rms_color_factors /= rms_color_factors.sum()
                ref_rms = (
                    np.sqrt(
                        np.sum(
                            multi_channel_interp(np.ones(3), self.grain_curve) ** 2
                            * rms_color_factors**2
                        )
                    )
                    / scaling
                )
            else:
                ref_rms = multi_channel_interp(np.ones(1), self.grain_curve)[0]
            if self.rms is not None:
                if self.rms > 1:
                    self.rms /= 1000
                factor = self.rms / ref_rms
                self.grain_curve[1:] *= factor
            else:
                self.rms = ref_rms
            self.rms = round(float(self.rms) * 10000) / 10

        if self.mtf is not None:
            mtf = self.mtf[0] if len(self.mtf) == 1 else self.mtf[1]
            self.resolution = round(
                np.interp(
                    0.5,
                    np.array(sorted(mtf.values())),
                    np.array(sorted(mtf.keys()))[::-1],
                )
            )

            mtf = []
            for mtf_dict in self.mtf:
                freqs = np.array(sorted(mtf_dict.keys()))
                vals = np.array([mtf_dict[f] for f in freqs])
                f_tail = freqs[-1] * 2
                freqs = np.append(freqs, f_tail)
                vals = np.append(vals, 0.0)

                # Interpolation axis in log space
                lowest_log = np.log1p(0)
                logf = np.log1p(freqs)
                logf = np.insert(logf, 0, lowest_log)
                vals = np.insert(vals, 0, 1.0)
                mtf.append((tuple(logf), tuple(vals)))
            self.mtf = list(mtf)

        for key, value in self.__dict__.items():
            if type(value) is np.ndarray and value.dtype is not DEFAULT_DTYPE:
                self.__dict__[key] = value.astype(DEFAULT_DTYPE)

        self.sensiometric_curve = np.stack(
            (self.log_exposure, *self.density_curve), dtype=DEFAULT_DTYPE
        )

        # compute gamma
        index = 2 if len(self.log_exposure) == 3 else 1
        density_curve = self.get_density_curve()
        log_exp_np = density_curve[0]
        density_np = density_curve[index]
        grad = np.abs(np.gradient(density_np, log_exp_np))
        window = 32
        kernel = np.ones(window) / window
        running_avg = np.convolve(grad, kernel, mode="valid")
        self.gamma = running_avg.max()

        if self.film_type == "positive":
            self._dye_inhibition_base = np.ones((3, 3), dtype=DEFAULT_DTYPE) * 0.2 / self.gamma
        else:
            self._dye_inhibition_base = np.ones((3, 3), dtype=DEFAULT_DTYPE) * 0.15 / self.gamma

    def set_color_checker(
        self,
        negative: "FilmSpectral | None" = None,
        print_stock: "FilmSpectral | None" = None,
    ):
        """
        Simulate the look of the 2005 ColorChecker photographed with the current film
        stock.

        Args:
            negative: When a negative film is provided assume current film is print
                film.
            print_stock: Use this film as the print film for the color checker if
                provided.
        """
        inversion = False
        if negative is None:
            negative = self
            if self.film_type == "negative":
                inversion = True

        elif print_stock is None:
            print_stock = self

        color_checker = film_conversion(
            COLORCHECKER_2005, negative, print_stock, inversion=inversion
        )
        color_checker *= 255
        self.color_checker = color_checker.astype(np.uint8)

    def extend_characteristic_curve(self, height: float = 3.0):
        """
        Extend the characteristic curve of the current film with a smooth rolloff.

        Args:
            height: Assumed height of the logistic curve used for extrapolation in
                density steps.
        """
        for i, (log_exposure, density_curve) in enumerate(
            zip(self.log_exposure, self.density_curve)
        ):
            dy_dx = np.gradient(density_curve, log_exposure)
            gamma = dy_dx.max()
            end_gamma = dy_dx[-4:].mean()
            stepsize = (log_exposure.max() - log_exposure.min()) / log_exposure.shape[0]

            def logistic_func(x):
                return height / (1 + np.exp(-4 * gamma / height * x))

            step_count = math.floor(1.5 * height / gamma / stepsize)
            logistic_func_x = np.linspace(0, step_count * stepsize, step_count)
            logistic_func_y = logistic_func(logistic_func_x)
            logistic_func_derivative = np.gradient(logistic_func_y, logistic_func_x)
            idx = np.abs(logistic_func_derivative - end_gamma).argmin()
            logistic_func_x = logistic_func_x[idx:]
            logistic_func_y = logistic_func_y[idx:]
            logistic_func_x += log_exposure[-1] - logistic_func_x[0]
            logistic_func_y += density_curve[-1] - logistic_func_y[0]
            self.log_exposure[i] = np.concatenate([log_exposure, logistic_func_x[1:]])
            self.density_curve[i] = np.concatenate([density_curve, logistic_func_y[1:]])

    def get_d_ref(self) -> np.ndarray:
        """
        Get the d_ref of the current film stock under specified color masking intensity.

        Returns:
            d_ref value for each channel.
        """

        return self.log_exposure_to_density(self.log_H_ref).reshape(-1)

    def estimate_d_min_sd(self):
        """
        Certain film stocks provide the minimum density for each layer, but they don't
        subtract the base density of the material, resulting in low saturation during
        emulation. To separate the layers more clearly we estimate the base density by
        subtracting the lower hull of the combined layers.
        """
        x_values = np.concatenate([x.wavelengths for x in self.spectral_density])
        y_values = np.concatenate([x.values for x in self.spectral_density])
        # Combine x and y into a single array of points
        points = np.column_stack((x_values, y_values))

        # Sort points by x_values (then y for stability)
        points = points[np.lexsort((points[:, 1], points[:, 0]))]

        # Function to compute the cross product of two vectors
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build the lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(tuple(p))
        lower = np.array(lower)[1:-1].T
        self.spectral_density = [
            SpectralDistribution(
                {
                    x: y
                    - scipy.interpolate.interp1d(lower[0], lower[1], fill_value="extrapolate")(x)
                    + 0.005
                    for x, y in zip(sd.wavelengths, sd.values)
                }
            )
            for sd in self.spectral_density
        ]
        if not self.d_min_sd.any() and lower.shape[1] > 1:
            self.d_min_sd = SpectralDistribution({x: y for x, y in lower.T})
            self.d_min_sd.align(SPECTRAL_SHAPE, interpolator=colour.LinearInterpolator)
            self.d_min_sd = np.asarray(self.d_min_sd.align(SPECTRAL_SHAPE).values)

    @staticmethod
    def prepare_rms_data(rms: dict, density: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Align the provided rms granularity and density data.

        Args:
            rms: RMS granularity data in relation to exposure.
            density: Density values for each channel in relation to exposure..

        Returns:
            Aligned rms and density data.
        """
        x = np.array(list(density.keys()), dtype=DEFAULT_DTYPE)
        fp = np.array(list(density.values()), dtype=DEFAULT_DTYPE)
        fp -= fp.min()
        density = np.interp(np.array(list(rms.keys()), dtype=DEFAULT_DTYPE), x, fp)
        rms = np.array(list(rms.values()), dtype=DEFAULT_DTYPE)
        sorting = density.argsort()
        density = density[sorting]
        rms = rms[sorting]
        return rms, density

    @staticmethod
    def gaussian_extrapolation(
        sd: colour.SpectralDistribution,
    ) -> colour.SpectralDistribution:
        """
        Extrapolate using a Gaussian distribution. Intended to be used for extrapolating
        spectral data.

        Args:
            sd: Spectral density data to extrapolate.

        Returns:
            Extrapolated spectral density.
        """

        def extrapolate(a_x, a_y, b_x, b_y, wavelengths, d_1=30, d_2=0.75):
            m = (a_y - b_y) / (a_x - b_x)
            if abs(a_y) < 0.001:
                a_y = a_y / abs(a_y) * 0.001
            d = d_1 * m / np.absolute(a_y) ** d_2
            a = a_y / max(np.exp(-(d**2)), 10**-10)
            c = a / m * -2 * d * np.exp(-(d**2))
            b = a_x - c * d

            def extrapolator(x):
                return a * np.exp(-((x - b) ** 2) / max(c**2, 0.0001))

            return extrapolator(wavelengths)

        sd.interpolate(SPECTRAL_SHAPE)

        def_wv = SPECTRAL_SHAPE.wavelengths
        wv_left = def_wv[def_wv < sd.wavelengths[0]]
        wv_right = def_wv[def_wv > sd.wavelengths[-1]]
        values_left = extrapolate(
            sd.wavelengths[0], sd.values[0], sd.wavelengths[1], sd.values[1], wv_left
        )
        values_right = extrapolate(
            sd.wavelengths[-1],
            sd.values[-1],
            sd.wavelengths[-2],
            sd.values[-2],
            wv_right,
        )
        sd.values, sd.wavelengths = (
            np.concatenate((values_left, sd.values, values_right)),
            np.concatenate((wv_left, sd.wavelengths, wv_right)),
        )
        sd.interpolate(SPECTRAL_SHAPE)

        return sd

    def log_exposure_to_density(
        self,
        log_exposure: np.ndarray,
        push_pull: float = 0.0,
        idealized_curve: bool = False,
        idealized_gamma: float = 3.0,
        d_min: float = 0.0,
        dye_inhibition: float | None = None,
        color_masking: float | None = None,
    ) -> np.ndarray:
        """
        Convert log_exposure to density values for current film stock.

        Args:
            log_exposure: Log exposure data to convert as array.
            push_pull: By how many stops to push/pull the negative to adjust contrast.
            idealized_curve: Replace the characteristic curve with an ideal gamma curve.
            idealized_gamma: The gamma of the idealized curve when it is used.
        """
        if idealized_curve:
            if self.film_type == "positive":
                idealized_gamma = -idealized_gamma
            return smooth_roll_off(
                log_exposure,
                self.log_H_ref,
                self.d_ref,
                idealized_gamma,
                d_min,
            )

        density = multi_channel_interp(
            log_exposure,
            self.get_density_curve(
                push_pull, dye_inhibition=dye_inhibition, color_masking=color_masking
            ),
        )

        return density

    @cache
    def apd_alignment_matrix(
        self, reference_negative: "FilmSpectral", max_delta: float = 0.05
    ) -> np.ndarray:
        """Compute matrix to make APD scan closer to proper print exposure."""
        # Get reference sensitivity computed like APD.
        scaled_sensitivity = self.sensitivity * PRINTER_LIGHT_APD.reshape(-1, 1)
        scaled_sensitivity /= scaled_sensitivity.sum(axis=0)

        # Get training transmittance spectra from reference negative.
        spectral_density = reference_negative.spectral_density
        m = 8
        values = np.linspace(0, 1, m, dtype=DEFAULT_DTYPE)
        grid = np.stack(np.meshgrid(values, values, values, indexing="ij"), axis=-1)
        points = grid.reshape(-1, 3) * 3
        training_spectra = 10 ** -(points @ spectral_density.T)

        # Measure with APD and stock based sensitivity.
        apd_measurements = training_spectra @ APD
        sensitivity_measurements = training_spectra @ scaled_sensitivity

        # Compute matrix with least squares to match measurements.
        M, residuals, rank, s = np.linalg.lstsq(
            apd_measurements, sensitivity_measurements, rcond=None
        )

        # Scale to preserve log_H_ref
        ref_linear = 10**self.log_H_ref
        current_mapped_ref = ref_linear @ M
        scale_factors = ref_linear / np.clip(current_mapped_ref, 1e-5, None)
        M = M * scale_factors

        # Bound to limit too extreme adjustments.
        eye = np.eye(3, dtype=DEFAULT_DTYPE)
        delta = (M - eye).max()
        scaling = min(max_delta, delta) / delta
        M = M * scaling + eye * (1 - scaling)

        return M

    def print_from_apd(
        self,
        apd_image: np.ndarray,
        idealized_curve: bool = False,
        idealized_gamma: float = 3.0,
        reference_negative: "FilmSpectral | None" = None,
        color_masking: float | None = None,
        d_min: float = 0.0,
    ) -> np.ndarray:
        """
        Converts independent APD intermediate densities into this print film's
        native layer activations, properly calibrated for mid-gray exposure.
        """

        apd_image = -(apd_image - self.log_H_ref - LAD_NEGATIVE)

        if self.density_measure == "bw":
            apd_image = np.ascontiguousarray(apd_image[..., 1:2])

        elif reference_negative is not None:
            M = self.apd_alignment_matrix(reference_negative)
            apd_image = np.log10(np.clip(10**apd_image @ M, 0.00001, None))

        return self.log_exposure_to_density(
            apd_image,
            idealized_curve=idealized_curve,
            idealized_gamma=idealized_gamma,
            d_min=d_min,
        )

    def get_density_curve(
        self,
        push_pull: float = 0.0,
        dye_inhibition: float | None = None,
        color_masking: float | None = None,
    ) -> np.ndarray:
        """
        Get characteristic density curve for current film stock.

        Args:
            push_pull: By how many stops to push/pull the negative to adjust contrast.

        Returns:
            The density curve.
        """
        sensiometric_curve = self.sensiometric_curve.copy()

        if self.density_measure != "bw" and color_masking:
            color_masking_matrix = self.get_color_masking_matrix(color_masking, dye_inhibition)
            color_masking_inv = np.linalg.inv(color_masking_matrix)
            sensiometric_curve[1:] = color_masking_inv @ sensiometric_curve[1:]

        if push_pull != 0:
            push_pull *= math.log10(2)
            sensiometric_curve[0] += push_pull
            log_exposure = sensiometric_curve[0]
            for i in range(len(sensiometric_curve) - 1):
                curve = sensiometric_curve[i + 1]
                d_ref = np.interp(self.log_H_ref[0] + push_pull, log_exposure, curve)
                d_post = np.interp(self.log_H_ref[0], log_exposure, curve)
                curve *= d_ref / d_post
                sensiometric_curve[i + 1] = curve

        return sensiometric_curve

    def compute_print_matrix(self, print_film: "FilmSpectral", **kwargs: Any) -> np.ndarray:
        """
        Computed matrix to convert from density of current film stock to log exposure of
        print film stock.

        Args:
            print_film: The film to print onto.
            **kwargs: Args passed to compute_printer_light.

        Returns:
            The printing matrix and the exposure for zero density.
        """
        printer_light = self.compute_printer_light(print_film, **kwargs)
        if print_film.density_measure == "absolute":
            print_sensitivity = print_film.sensitivity * printer_light
            peak_exposure = np.log10(np.sum(print_sensitivity, axis=0))
        else:
            # Compute max exposure produced by unfiltered printer light.
            peak_exposure = np.log10(print_film.sensitivity.T @ printer_light)
            # Compute density matrix from print film sensitivity under adjusted printer
            # light.
            print_sensitivity = (print_film.sensitivity.T * printer_light).T
        print_sensitivity /= np.sum(print_sensitivity, axis=0)
        density_matrix = print_sensitivity.T @ self.spectral_density
        density_base = print_sensitivity.T @ self.d_min_sd
        return density_matrix, peak_exposure - density_base

    def compute_printer_light(
        self,
        print_film: "FilmSpectral",
        red_light: float = 0.0,
        green_light: float = 0.0,
        blue_light: float = 0.0,
    ) -> np.ndarray:
        """
        Compute printer light needed to print onto target print film to generate neutral
        exposure.

        Args:
            print_film: Film stock to print onto.
            red_light: Red printer light offset.
            green_light: Green printer light offset.
            blue_light: Blue printer light offset.

        Returns:
            Printer light as spectral curve.
        """
        if print_film.density_measure == "bw":
            target_exp = 2**green_light * print_film.H_ref
            reduced_lights = 10**-self.d_ref_sd
            printer_light = np.ones_like(reduced_lights, DEFAULT_DTYPE)
            printer_light *= target_exp / (print_film.sensitivity.T @ reduced_lights)
        else:
            compensation = 2 ** np.array([red_light, green_light, blue_light], dtype=DEFAULT_DTYPE)
            target_exp = np.multiply(print_film.H_ref, compensation)
            # transmitted printer lights by middle gray negative
            reduced_lights = (PRINTER_LIGHTS.T * 10**-self.d_ref_sd).T
            light_factors = np.linalg.inv(print_film.sensitivity.T @ reduced_lights) @ target_exp
            printer_light = np.sum(PRINTER_LIGHTS * light_factors, axis=1)
        return printer_light

    def plot_data(self, film_b=None):
        """Plots the spectral density, sensitivity, and sensiometric curve."""
        wavelengths = SPECTRAL_SHAPE.wavelengths
        default_colors = ["r", "g", "b"]

        is_comparison = film_b is not None
        cols = 2 if is_comparison else 1

        fig, axes = plt.subplots(3, cols, figsize=(12 if cols == 2 else 8, 12), squeeze=False)

        def plot_film_data(film, ax_col):
            # Spectral Sensitivity
            num_curves = film.sensitivity.shape[1]
            colors = ["black"] if num_curves == 1 else default_colors
            for i, a in enumerate(film.sensitivity.T):
                color = colors[i] if i < len(colors) else None
                axes[0, ax_col].plot(wavelengths, a, color=color)
            axes[0, ax_col].set_title(f"{film.__class__.__name__} - Spectral Sensitivity")
            axes[0, ax_col].set_xlabel("Wavelength")
            axes[0, ax_col].set_ylabel("Sensitivity")

            # Density Curve
            num_curves = len(film.log_exposure)
            colors = ["black"] if num_curves == 1 else default_colors
            gamma_values = []

            for i, (log_exp, density) in enumerate(zip(*film.get_density_curve())):
                color = colors[i] if i < len(colors) else None
                axes[1, ax_col].plot(log_exp, density, color=color)

                # Compute gamma
                d_density_d_logH = np.gradient(density, log_exp)
                gamma_interp = scipy.interpolate.interp1d(
                    log_exp,
                    d_density_d_logH,
                    kind="linear",
                    fill_value="extrapolate",
                )
                log_H_val = film.log_H_ref[i]
                gamma = gamma_interp(log_H_val)
                gamma_values.append((color, gamma))

            # Draw vertical line(s)
            if np.allclose(film.log_H_ref, film.log_H_ref[0]):
                ref_val = film.log_H_ref[0]
                axes[1, ax_col].axvline(x=ref_val, color="black", linestyle="--", linewidth=1)
            else:
                for i in range(len(film.log_H_ref)):
                    color = colors[i] if i < len(colors) else None
                    axes[1, ax_col].axvline(
                        x=film.log_H_ref[i],
                        color=color,
                        linestyle="--",
                        linewidth=1,
                    )

            axes[1, ax_col].set_title(f"{film.__class__.__name__} - Density Curve")
            axes[1, ax_col].set_xlabel("Log Exposure")
            axes[1, ax_col].set_ylabel("Density")

            # Add gamma annotations in top-right
            text_lines = [
                f"{color.upper() if color else 'Channel'} γ = {gamma:.2f}"
                for color, gamma in gamma_values
            ]
            text = "\n".join(text_lines)
            axes[1, ax_col].text(
                0.98,
                0.95,
                text,
                transform=axes[1, ax_col].transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

            # Spectral Density
            num_curves = film.spectral_density.shape[1]
            colors = ["black"] if num_curves == 1 else default_colors
            for i, x in enumerate(film.spectral_density.T):
                color = colors[i] if i < len(colors) else None
                axes[2, ax_col].plot(wavelengths, x, color=color)
            axes[2, ax_col].plot(wavelengths, film.d_min_sd, "--", color="black")
            axes[2, ax_col].plot(wavelengths, film.d_ref_sd, color="black")
            axes[2, ax_col].set_title(f"{film.__class__.__name__} - Spectral Density")
            axes[2, ax_col].set_xlabel("Wavelength")
            axes[2, ax_col].set_ylabel("Density")

        # Plot film_a in the first column
        plot_film_data(self, 0)

        # Plot film_b in the second column if provided
        if is_comparison:
            plot_film_data(film_b, 1, None)

        plt.tight_layout()
        plt.show()

    def get_grain_curve(
        self,
        scale: float = 1.0,
        std_div: float = 1.0,
        adx: bool = True,
        bw_grain: bool = False,
        apd_intermediate: bool = False,
    ) -> np.ndarray:
        """Compute grain intensity 1D LUT."""
        # scale = max(image.shape) / max(frame_width, frame_height) in pixels per mm,
        # default for 3840 / 24mm
        # std_div is of the sampled gaussian noise to be applied, default is 0.1 to stay
        # in [0, 1] range
        std_factor = np.array((math.sqrt(math.pi) * 0.024 * scale / std_div,), dtype=DEFAULT_DTYPE)
        if adx and self.density_measure != "bw":
            adx_density_scale = np.array([1.00, 0.92, 0.95], dtype=DEFAULT_DTYPE) * (
                8000.0 / 65535.0
            )
            std_factor = std_factor * adx_density_scale
        elif adx:
            std_factor *= 8000.0 / 65535.0
        elif self.density_measure != "bw":
            std_factor = std_factor.repeat(3)
        grain_curve = self.grain_curve.copy()
        if apd_intermediate:
            d_ref = self.log_exposure_to_density(self.log_H_ref)
            if len(d_ref) == 3:
                d_ref = d_ref[1]
            offset = d_ref - LAD_NEGATIVE[1]
            grain_curve[0] -= offset
        grain_curve[1:] *= std_factor.reshape(-1, 1)

        if bw_grain and self.density_measure != "bw":
            grain_curve[1:] /= 1.5

        return grain_curve

    def grain_transform(
        self,
        image: np.ndarray,
        scale: float = 1.0,
        std_div: float = 1.0,
        adx: bool = True,
        bw_grain: bool = False,
        apd_intermediate: bool = False,
    ) -> np.ndarray:
        """
        Compute a grain intensity map from an image containing density values.

        Args:
            image: The image containing density values.
            scale: The scaling of the intensity of the grain. Can either be the max
                density for an ADX encoded LUT or the size scalling in pixels per mm.
            std_div: The std_div of the gaussian noise for which the intensity map is
                to be used.
            adx: Whether
            bw_grain:

        Returns:

        """
        # scale = max(image.shape) / max(frame_width, frame_height) in pixels per mm,
        # default for 3840 / 24mm
        # std_div is of the sampled gaussian noise to be applied, default is 0.1 to stay
        # in [0, 1] range
        grain_curve = self.get_grain_curve(scale, std_div, adx, bw_grain, apd_intermediate)
        if self.density_measure == "bw" and image.shape[-1] == 3:
            image = np.ascontiguousarray(image[..., 1:2])
        noise_factors = multi_channel_interp(image, grain_curve)

        return noise_factors

    def get_input_lut(
        self,
        exposure_kelvin: int | float = 6500,
        tint: float = 0.0,
        exp_comp: float = 0.0,
        upsampling_method: UpsampleMethod | str = "SFL upsampling",
    ) -> np.ndarray:
        """
        Compute a 2D LUT for use with [`apply_2d_lut`][] that converts from scene linear
        CIE XYZ the per layer exposure.

        Args:
            exposure_kelvin: The scene WB in kelvin.
            tint: The tint adjustment on the green -- red/purple axis.
            exp_comp: Exposure compensation in stops.
            upsampling_method: Method for reconstructing a spectral distribution from
                CIE XYZ values.

        Returns:
            The 2D LUT of shape (n, n, 3).
        """
        exp_comp = 2**exp_comp
        gray_XYZ = CCT_to_XYZ(exposure_kelvin, 0.18, tint)
        reference_XYZ = CCT_to_XYZ(6504, 0.18)
        spectrum_lut = get_spectrum_lut(upsampling_method)
        gray_spectral = apply_2d_lut(gray_XYZ, get_spectrum_lut())
        reference_spectral = apply_2d_lut(reference_XYZ, get_spectrum_lut())
        corrected_sensitivity = self.sensitivity * (reference_spectral / gray_spectral)[:, None]
        spectral_input_lut = spectrum_lut @ corrected_sensitivity
        if self.density_measure == "bw":
            ref_exp = apply_2d_lut(reference_XYZ, spectral_input_lut)
        else:
            ref_exp = apply_2d_lut(gray_XYZ, spectral_input_lut)
        correction_factors = self.H_ref / ref_exp
        spectral_input_lut *= correction_factors * exp_comp
        return spectral_input_lut

    def compute_lad(self, luminance=0.1):
        """Find the Lab Aim Density for a neutral gray."""
        projection_light = apply_2d_lut(CCT_to_XYZ(6504), get_spectrum_lut())
        d_min_sd = self.d_min_sd
        output_mat = (XYZ_CMFS.T * projection_light * 10**-d_min_sd).T
        lad = output_to_density(CCT_to_XYZ(6504, luminance), self.spectral_density, output_mat)
        return lad

    def layer_activation_to_apd_matrix(
        self, color_masking: None | float = None, dye_inhibition: None | float = None
    ):
        """
        Get the matrix that converts from layer activation to ACES Printing Density for
        ADX encoding.
        """
        if self.density_measure == "bw":
            return np.ones((1, 1), dtype=DEFAULT_DTYPE)
        return APD.T @ self.get_spectral_density(color_masking, dye_inhibition)

    def apd_to_layer_activation_matrix(
        self, color_masking: None | float = None, dye_inhibition: None | float = None
    ) -> np.ndarray:
        """
        Get the matrix that converts from ACES Printing Density to layer activation for
        ADX decoding.
        """
        if self.density_measure == "bw":
            return np.ones((1, 1), dtype=DEFAULT_DTYPE)
        return np.linalg.inv(self.layer_activation_to_apd_matrix(color_masking, dye_inhibition))

    def adx_encoding(
        self,
        image: np.ndarray,
        scaling: float = 1.0,
        apd_intermediate: bool = False,
        color_masking: None | float = None,
        dye_inhibition: None | float = None,
    ) -> np.ndarray:
        """
        Encode layer activation in absolute densities as ADX16 data in the [0, 1] range.

        Args:
            image: The image (or LUT) containing layer activations.
            scaling: Linear scaling applied to the values.
                For 1.0 it is ADX16-like and for 4.0 it is ADX10-like.

        Returns:
            The image (or LUT) encoded in ADX.
        """
        if not apd_intermediate:
            image @= self.layer_activation_to_apd_matrix(color_masking, dye_inhibition).T

        image = adx16_encode(image, scaling=scaling)

        return image

    def adx_decoding(
        self,
        image: np.ndarray,
        scaling: float = 1.0,
        apd_intermediate: bool = False,
        color_masking: None | float = None,
        dye_inhibition: None | float = None,
    ) -> np.ndarray:
        """
        Decode layer activation from ADX16 data in the [0, 1] range to absolute
        densities.

        Args:
            image: The image (or LUT) encoded in ADX.
            scaling: Linear scaling applied to the endoced values.
                For 1.0 it is ADX16-like and for 4.0 it is ADX10-like.

        Returns:
            The image (or LUT) represented as absolute layer activations.
        """
        if self.density_measure == "bw" and not apd_intermediate:
            image = image[..., 0][..., np.newaxis]  # reduce dim

        image = adx16_decode(image, scaling=scaling)

        if not apd_intermediate:
            image @= self.apd_to_layer_activation_matrix(color_masking, dye_inhibition).T

        return image

    def input_transform(
        self,
        image: np.ndarray,
        colorspace: LiteralRGBColourspace | None = None,
        exp_comp: float = 0.0,
        exp_kelvin: int | float = 6500,
        tint: float = 0.0,
        dye_inhibition: None | float = None,
        color_masking: None | float = None,
        push_pull: float = 0.0,
        upsampling_method: UpsampleMethod | str = "barycentric_nnls_all",
    ) -> np.ndarray:
        """
        Transform from scene referred image data to the per layer activation in absolute
        densities.
        """
        if colorspace is not None:
            image = colour.RGB_to_XYZ(image, colorspace, apply_cctf_decoding=True)

        input_lut = self.get_input_lut(exp_kelvin, tint, exp_comp, upsampling_method)

        image = apply_2d_lut(np.clip(image, 0, None), input_lut)

        log_clip(image)

        if dye_inhibition is not None and dye_inhibition != 0 and self.density_measure != "bw":
            density_curve = self.get_density_curve(
                push_pull, dye_inhibition=dye_inhibition, color_masking=color_masking
            )
            dye_inhibition_matrix = self.get_dye_inhibition_matrix(dye_inhibition)

            d_init = multi_channel_interp(image, density_curve)
            if self.film_type == "positive":
                inh_source = np.clip(self.d_max - d_init, 0.0, None)
            else:
                inh_source = np.clip(d_init, 0.0, None)

            image -= inh_source @ dye_inhibition_matrix.T

            density_curve = self._adjust_density_curve(density_curve, dye_inhibition_matrix)

            image = multi_channel_interp(image, density_curve)
        else:
            image = self.log_exposure_to_density(
                image, push_pull, color_masking=color_masking, dye_inhibition=dye_inhibition
            )

        return image

    def _adjust_density_curve(
        self, density_curve: np.ndarray, dye_inhibition_matrix: np.ndarray
    ) -> np.ndarray:
        orig_log_H = density_curve[0]  # shape (N,)
        density_values = density_curve[1:]  # shape (3, N)

        if self.film_type == "positive":
            inh_source = np.clip(self.d_max[:, None] - density_values, 0.0, None)
        else:
            inh_source = np.clip(density_values, 0.0, None)

        shift = dye_inhibition_matrix @ inh_source  # shape (3, N)

        # Prevent folding along the original exposure progression:
        # require d(orig_log_H - scale * shift) >= min_step
        base_step = np.diff(orig_log_H)
        min_allowed_step = np.maximum(base_step * 0.15, 1e-5)  # keep at least 15% of original step

        scale = 1.0
        for ch in range(density_values.shape[0]):
            d_shift = np.diff(shift[ch])
            # Where shift decreases with increasing log_H, (base_step - s * d_shift) can shrink
            # or flip
            folding_mask = (base_step - d_shift) < min_allowed_step
            if np.any(folding_mask):
                # Largest s in [0, 1] such that base_step - s * d_shift >= min_allowed_step
                denom = np.where(d_shift > 1e-8, d_shift, np.nan)
                max_s_per_step = (base_step - min_allowed_step) / denom
                valid_s = max_s_per_step[np.isfinite(max_s_per_step)]
                if valid_s.size > 0:
                    scale = min(scale, float(np.clip(np.min(valid_s), 0.0, 1.0)))

        eff_shift = shift * scale
        log_H_new = orig_log_H[None, :] - eff_shift

        x_min = float(log_H_new.min())
        x_max = float(log_H_new.max())
        x_new = np.linspace(x_min, x_max, density_curve.shape[1], dtype=DEFAULT_DTYPE)

        adjusted_densities = np.empty_like(density_values)
        for ch in range(density_values.shape[0]):
            x_ch = log_H_new[ch]
            y_ch = density_values[ch]

            order = np.argsort(x_ch)
            x_sorted = x_ch[order]
            y_sorted = y_ch[order]

            keep = np.concatenate(([True], np.diff(x_sorted) > 1e-6))
            x_sorted = x_sorted[keep]
            y_sorted = y_sorted[keep]

            interp = PchipInterpolator(x_sorted, y_sorted, extrapolate=True)
            adjusted_densities[ch] = interp(x_new).astype(DEFAULT_DTYPE)

        return np.vstack((x_new[None, :], adjusted_densities)).astype(DEFAULT_DTYPE)

    def _estimate_adjusted_gamma(
        self, adjusted_curve: np.ndarray, fallback: float = 0.6
    ) -> np.ndarray:
        """Estimate per-channel d(density)/d(log_H) around log_H_ref on the adjusted curve."""
        x = adjusted_curve[0]
        densities = adjusted_curve[1:]  # (3, N)
        ref_x = float(np.asarray(self.log_H_ref)[0])

        slopes = np.empty(densities.shape[0], dtype=DEFAULT_DTYPE)
        for ch in range(densities.shape[0]):
            dy_dx = np.gradient(densities[ch], x)
            if np.all(np.isfinite(dy_dx)):
                slopes[ch] = float(np.interp(ref_x, x, dy_dx))
            else:
                slopes[ch] = float(self.gamma) if self.gamma else fallback

        return np.clip(np.abs(slopes), 0.1, None).astype(DEFAULT_DTYPE)

    def get_effective_inhibition_matrix(
        self,
        dye_inhibition: float | None,
        push_pull: float = 0.0,
        color_masking: float | None = 0.0,
        adx_gamma: float = 0.6,
    ) -> np.ndarray:
        """
        Recompute the effective DIR + curve-adjustment matrix statelessly from
        current parameters and the base curve initialized in __init__.
        """
        eye = np.eye(3, dtype=DEFAULT_DTYPE)
        if dye_inhibition is None or dye_inhibition == 0 or self.density_measure == "bw":
            return eye

        inh = self.get_dye_inhibition_matrix(dye_inhibition)
        base_curve = self.get_density_curve(
            push_pull=push_pull,
            color_masking=color_masking,
        )
        adjusted_curve = self._adjust_density_curve(base_curve, inh)
        eff_gamma = self._estimate_adjusted_gamma(adjusted_curve, fallback=adx_gamma)

        gamma_ratio = np.clip(eff_gamma / adx_gamma, 0.25, 4.0)[:, None]
        return (eye - (inh * gamma_ratio) * adx_gamma).astype(DEFAULT_DTYPE)

    def get_dye_inhibition_matrix(self, dye_inhibition: float) -> np.ndarray:
        return (self._dye_inhibition_base * float(dye_inhibition)).astype(DEFAULT_DTYPE)

    def get_color_masking_matrix(
        self,
        color_masking: float | None = None,
        dye_inhibition: float | None = None,
        push_pull: float = 0.0,
    ) -> np.ndarray:
        mask_amount = 0.0 if color_masking is None else float(color_masking)
        eye = np.eye(3, dtype=DEFAULT_DTYPE)
        if self.density_measure == "bw" or mask_amount == 0.0:
            return eye

        # Unmix base spectral crosstalk without inverting the DIR saturation matrix,
        # which otherwise over-compensates and reduces saturation when combined with DIR.
        base_unmix = np.linalg.inv(CDD_TO_CID @ self._crosstalk_matrix)

        # Optional subtle coupling so DIR doesn't get wiped out when color_masking is high
        if dye_inhibition is not None and dye_inhibition != 0:
            inh = self.get_dye_inhibition_matrix(dye_inhibition)
            # Keep diagonal at 1 and attenuate off-diagonal over-correction slightly
            damping = 1.0 / (1.0 + 0.5 * abs(float(dye_inhibition)))
            off_diag = (inh - np.diag(np.diag(inh))) * damping
            unmix_target = base_unmix @ (eye - off_diag * 0.25)
        else:
            unmix_target = base_unmix

        return (eye * (1.0 - mask_amount) + unmix_target * mask_amount).astype(DEFAULT_DTYPE)

    def get_spectral_density(
        self, color_masking: float | None = None, dye_inhibition: float | None = None
    ):
        if self.density_measure == "bw" or not color_masking:
            return self.spectral_density.copy()

        color_masking_matrix = self.get_color_masking_matrix(color_masking, dye_inhibition)

        spectral_density = self.spectral_density @ color_masking_matrix

        return spectral_density

    def print_to(
        self,
        image: np.ndarray,
        print_film: "FilmSpectral",
        red_light: float = 0.0,
        green_light: float = 0.0,
        blue_light: float = 0.0,
        idealized_curve: bool = False,
        idealized_gamma: float = 3.0,
        d_min: float = 0.0,
        color_masking: float | None = None,
        dye_inhibition: float | None = None,
    ) -> np.ndarray:
        """
        Print to another film stock.

        Args:
            image: The image (or LUT) containing layer activations in absolute
                densities.
            print_film: The film stock to print onto.
            red_light: Offset of the red printer light from neutral.
            green_light: Offset of the green printer light from neutral.
            blue_light: Offset of the blue printer light from neutral.
            idealized_curve: Replace the characteristic curve of the print stock with
                an ideal gamma curve.
            idealized_gamma: The gamma of the idealized curve when it is used.

        Returns:
            The resulting layer activations of the print stock.
        """
        return self.printing_process(
            image,
            self,
            print_film,
            red_light,
            green_light,
            blue_light,
            idealized_curve,
            idealized_gamma,
            d_min,
            color_masking=color_masking,
            dye_inhibition=dye_inhibition,
        )

    @staticmethod
    def printing_process(
        image: np.ndarray,
        negative_film: "FilmSpectral",
        print_film: "FilmSpectral",
        red_light: float = 0.0,
        green_light: float = 0.0,
        blue_light: float = 0.0,
        idealized_curve: bool = False,
        idealized_gamma: float = 3.0,
        d_min: float = 1.0,
        color_masking: float | None = None,
        dye_inhibition: float | None = None,
    ) -> np.ndarray:
        """
        Print from one film stock onto another.

        Args:
            image: The image (or LUT) containing layer activations in absolute
                densities.
            negative_film: The film stock from which to print.
            print_film: The film stock to print onto.
            red_light: Offset of the red printer light from neutral.
            green_light: Offset of the green printer light from neutral.
            blue_light: Offset of the blue printer light from neutral.
            idealized_curve: Replace the characteristic curve of the print stock with
                an ideal gamma curve.
            idealized_gamma: The gamma of the idealized curve when it is used.


        Returns:
            The resulting layer activations of the print stock.
        """
        if negative_film.density_measure == print_film.density_measure == "bw":
            image = -image + print_film.log_H_ref + negative_film.d_ref + green_light
        else:
            printer_light = negative_film.compute_printer_light(
                print_film, red_light, green_light, blue_light
            )
            printing_mat = (
                print_film.sensitivity.T * printer_light * 10**-negative_film.d_min_sd
            ).T

            density_neg = negative_film.get_spectral_density(color_masking, dye_inhibition)
            image = np.log10(
                np.clip(
                    10 ** -(image @ density_neg.T) @ printing_mat,
                    0.00001,
                    None,
                )
            )

        image = print_film.log_exposure_to_density(
            image,
            idealized_curve=idealized_curve,
            idealized_gamma=idealized_gamma,
            d_min=d_min,
        )

        return image

    def scan_with_apd(
        self,
        image: np.ndarray,
        red_light: float = 0.0,
        green_light: float = 0.0,
        blue_light: float = 0.0,
        dye_inhibition: None | float = None,
        color_masking: None | float = None,
    ) -> np.ndarray:
        """
        Print from one film stock onto another.

        Args:
            image: The image (or LUT) containing layer activations in absolute
                densities.
            red_light: Offset of the red printer light from neutral.
            green_light: Offset of the green printer light from neutral.
            blue_light: Offset of the blue printer light from neutral.


        Returns:
            The resulting layer activations of the print stock.
        """
        if self.density_measure == "bw":
            compensation = 0.3 * green_light  # log10(2) = 0.3
            reference = self.d_ref

            image = image + (compensation + LAD_NEGATIVE - reference)
            return image
        else:
            compensation = 0.3 * np.array(
                [red_light, green_light, blue_light], dtype=DEFAULT_DTYPE
            )  # log10(2) = 0.3
            density_neg = self.get_spectral_density(color_masking, dye_inhibition)
            printing_mat = (APD.T * 10**-self.d_min_sd).T

            d_ref = self.log_exposure_to_density(
                self.log_H_ref, color_masking=color_masking, dye_inhibition=dye_inhibition
            )
            reference = -np.log10(
                np.clip(10 ** -(d_ref @ density_neg.T) @ printing_mat, 0.00001, None)
            )

            image = -np.log10(
                np.clip(10 ** -(image @ density_neg.T) @ printing_mat, 0.00001, None)
            ) + (LAD_NEGATIVE - reference - compensation)
        return image

    def project(
        self,
        image: np.ndarray,
        projector_kelvin: int | float = 6500,
        white_comp: bool = False,
        white_balance: int = 0,
        color_masking: float | None = None,
        dye_inhibition: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Get the scene referred output from projection or viewing under an illuminant.

        Args:
            image: The image (or LUT) containing layer activations in absolute
                densities.
            projector_kelvin: The white balance in kelvin of the projection lamp.
            white_comp: Whether to adjust the output brightness that it clips at exactly
                1.
            white_balance: White balance mode for slide projection:
                0 disabled, 1 based on mid gray, 2 based on peak white.

        Returns:
            The projected image in linear CIE XYZ color space.
        """
        if self.density_measure == "bw":
            image = 10**-image
            out_gray = 10**-self.d_ref

            if not 6500 <= projector_kelvin <= 6505:
                adjust = np.asarray(CCT_to_XYZ(projector_kelvin))
                image = image * adjust

        else:
            projection_light = apply_2d_lut(
                CCT_to_XYZ(projector_kelvin),
                get_spectrum_lut(),
            )

            density_mat = self.spectral_density

            output_mat = (XYZ_CMFS.T * projection_light * 10**-self.d_min_sd).T

            mid_gray = 10 ** -(self.get_d_ref() @ density_mat.T) @ output_mat

            if white_balance:
                balance_source = mid_gray
                if white_balance == 1:
                    balance_source = output_mat.sum(axis=0)

                mid_gray_sd = apply_2d_lut(
                    balance_source / balance_source[1],
                    get_spectrum_lut(),
                )

                output_mat = output_mat * (projection_light / mid_gray_sd)[:, None]

            if white_comp:
                peak_rgb = colour.XYZ_to_RGB(output_mat.sum(axis=0), "sRGB")
                peak = peak_rgb.max()
                output_mat *= 1 / peak

            out_gray = 10 ** -(self.get_d_ref() @ density_mat.T) @ output_mat

            image = 10 ** -np.clip(image @ density_mat.T, 0, None) @ output_mat

        return image, out_gray

    @property
    def cid_to_xyz(self):
        """A per negative film stock optimized inversion matrix for use with apd_to_xyz.
        The matrix assumes row vectors and therefore does not need to be transposed.
        """
        if self._cid_to_xyz is None:
            self._cid_to_xyz = self._solve_cid_to_xyz()

        return self._cid_to_xyz

    def _solve_cid_to_xyz(self, inversion_gamma: float = 3.5):
        training_samples = RAWTOACES_XYZ / RAWTOACES_XYZ.sum(axis=1, keepdims=True) * 0.18
        activation = self.input_transform(training_samples)
        apd = self.scan_with_apd(activation)
        exp_to_xyz = np.eye(3, dtype=DEFAULT_DTYPE)
        uncalibrated_xyz = apd_to_xyz(apd, inversion_gamma=inversion_gamma, exp_to_xyz=exp_to_xyz)

        cid_to_xyz, _, _, _ = np.linalg.lstsq(uncalibrated_xyz, training_samples, rcond=None)
        cid_to_xyz /= cid_to_xyz.sum(axis=0)

        return cid_to_xyz
