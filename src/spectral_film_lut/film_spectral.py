"""
The main class for handling all film data procesing and rendering.
"""

import math

import colour.plotting
import numpy as np
import scipy
from colour import SpectralDistribution
from colour.hints import LiteralRGBColourspace
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator

from spectral_film_lut.color_processing import (
    COLORCHECKER_2005,
    CCT_to_xy,
    CCT_to_XYZ,
)
from spectral_film_lut.config import DEFAULT_DTYPE, SPECTRAL_SHAPE
from spectral_film_lut.densiometry import (
    APD,
    DENSIOMETRY,
    PRINTER_LIGHTS,
    STATUS_A,
    adx16_decode,
    adx16_encode,
    construct_spectral_density,
    output_to_density,
)
from spectral_film_lut.film_data import FilmData
from spectral_film_lut.utils import film_conversion, multi_channel_interp
from spectral_film_lut.xy_lut import SPECTRUM_LUT, XYZ_CMFS, apply_2d_lut


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
        self.spectral_density_pure = None
        self.density_curve_pure = None
        self.d_min = None
        self.d_ref = None
        self.d_max = None
        self.color = None
        self.resolution = None
        self.color_checker = None
        self.gamma = None
        self.log_H_ref = None
        self.H_ref = None

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
        else:
            self.log_exposure = None
            self.density_curve = None

        # target exposure of middle gray in log lux-seconds
        # normally use iso value, if not provided use target density of 1.0 on the green
        # channel
        if self.iso is not None:
            self.log_H_ref = np.ones(len(self.log_exposure)) * math.log10(
                12.5 / self.iso
            )
            self.H_ref = 10**self.log_H_ref
        elif self.density_measure == "absolute" or self.density_measure == "bw":
            if self.density_measure == "absolute":
                self.lad = np.linalg.inv(STATUS_A.T @ self.spectral_density) @ np.array(
                    self.lad
                )
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
                self.color_masking = 1
            else:
                self.color_masking = 0

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
                self.spectral_density = construct_spectral_density(
                    self.d_ref_sd - self.d_min_sd
                )

            self.spectral_density /= (
                self.spectral_density * DENSIOMETRY[self.density_measure]
            ).sum(axis=0)

            status_matrix = np.linalg.inv(
                DENSIOMETRY[self.density_measure].T @ self.spectral_density
            )
            self.spectral_density_pure = self.spectral_density @ status_matrix
            density_curve = np.stack(self.density_curve).T
            density_curve @= status_matrix.T
            self.density_curve_pure = self.density_curve
            self.density_curve = [
                density_curve[:, 0],
                density_curve[:, 1],
                density_curve[:, 2],
            ]

            self.d_min_sd = self.d_min_sd + self.spectral_density @ status_matrix @ (
                self.d_min - DENSIOMETRY[self.density_measure].T @ self.d_min_sd
            )
            if self.H_ref is None:
                self.lad = self.compute_lad(self.gray_value)
                self.log_H_ref = np.array(
                    [
                        np.interp(
                            np.asarray(a), np.asarray(sorted_b), np.asarray(sorted_c)
                        )
                        for a, b, c in zip(
                            self.lad,
                            self.density_curve,
                            [self.log_exposure] * len(self.lad),
                        )
                        for sorted_b, sorted_c in [zip(*sorted(zip(b, c)))]
                    ]
                )
                self.H_ref = 10**self.log_H_ref
            self.d_ref = self.log_exposure_to_density(self.log_H_ref, 0.0).reshape(-1)
            self.d_ref_sd = self.spectral_density @ self.d_ref + self.d_min_sd

        self.d_max = np.array([np.max(x) for x in self.density_curve])

        if self.rms_curve is not None and self.rms_density is not None:
            rms_temp = [
                self.prepare_rms_data(a, b)
                for a, b in zip(self.rms_curve, self.rms_density)
            ]
            self.rms_curve = [x[0] for x in rms_temp]
            self.rms_density = [x[1] for x in rms_temp]
            if len(self.rms_density) == 3:
                rms_color_factors = np.array([0.26, 0.57, 0.17], dtype=DEFAULT_DTYPE)
                scaling = 1.2375
                rms_color_factors /= rms_color_factors.sum()
                ref_rms = (
                    np.sqrt(
                        np.sum(
                            multi_channel_interp(
                                np.ones(3), self.rms_density, self.rms_curve
                            )
                            ** 2
                            * rms_color_factors**2
                        )
                    )
                    / scaling
                )
            else:
                ref_rms = np.interp(
                    np.asarray(1), self.rms_density[0], self.rms_curve[0]
                )
            if self.rms is not None:
                if self.rms > 1:
                    self.rms /= 1000
                factor = self.rms / ref_rms
                self.rms_curve = [x * factor for x in self.rms_curve]
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

        # compute gamma
        index = 1 if len(self.log_exposure) == 3 else 0
        log_exp_np, density_np = self.get_density_curve()
        log_exp_np = log_exp_np[index]
        density_np = density_np[index]
        d_density_d_logH = np.gradient(density_np, log_exp_np)
        log_H_val = self.log_H_ref[index]
        gamma_interp = scipy.interpolate.interp1d(
            log_exp_np, d_density_d_logH, kind="linear", fill_value="extrapolate"
        )
        self.gamma = abs(gamma_interp(log_H_val))

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

    def extend_characteristic_curve(self, height=3):
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

    def get_d_ref(self, color_masking: float | None = None):
        """
        Get the d_ref of the current film stock under specified color masking intensity.

        Args:
            color_masking: Color masking factor. If None use default value for current
                film. Safe values are in the range [0, 1], but higher values can be used
                to get a highly saturated look.

        Returns:
            np.array: d_ref value for each channel.
        """
        if color_masking is None:
            color_masking = self.color_masking

        return self.log_exposure_to_density(self.log_H_ref, color_masking).reshape(-1)

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
                    - scipy.interpolate.interp1d(
                        lower[0], lower[1], fill_value="extrapolate"
                    )(x)
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
    def prepare_rms_data(rms, density):
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
    def gaussian_extrapolation(sd):
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
        color_masking: None | float = None,
        push_pull: float = 0.0,
    ) -> np.ndarray:
        """
        Convert log_exposure to density values for current film stock.

        Args:
            log_exposure: Log exposure data to convert as array.
            color_masking: Color Masking factor in range [0, 1].
            push_pull: By how many stops to push/pull the negative to adjust contrast.
        """
        density = multi_channel_interp(
            log_exposure,
            *self.get_density_curve(color_masking, push_pull),
        )

        return density

    def get_density_curve(
        self, color_masking: None | float = None, push_pull: float = 0.0
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Get characteristic density curve for current film stock.

        Args:
            color_masking: Color Masking factor in range [0, 1]. If None use default for
                current film stock.
            push_pull: By how many stops to push/pull the negative to adjust contrast.

        Returns:
            The density curve.
        """
        if color_masking is None:
            color_masking = self.color_masking
        if self.density_curve_pure is None:
            density_curve = [self.density_curve[0].copy()]
        else:
            density_curve = [
                a * color_masking + b * (1 - color_masking)
                for a, b in zip(self.density_curve_pure, self.density_curve)
            ]

        log_exposure = self.log_exposure

        if push_pull != 0:
            push_pull *= math.log10(2)
            log_exposure = log_exposure + push_pull
            for curve in density_curve:
                d_ref = np.interp(self.log_H_ref[0] + push_pull, log_exposure, curve)
                d_post = np.interp(self.log_H_ref[0], log_exposure, curve)
                curve *= d_ref / d_post

        log_exposure = [log_exposure] * len(density_curve)

        return log_exposure, density_curve

    def get_spectral_density(self, color_masking=None):
        """
        Get spectral density for current film stock.

        Args:
            color_masking: Color Masking factor in range [0, 1]. If None use default for
            current film stock.

        Returns:
            Spectral density.
        """
        if color_masking is None:
            color_masking = self.color_masking
        if self.spectral_density_pure is None:
            return self.spectral_density
        return self.spectral_density_pure * color_masking + self.spectral_density * (
            1 - color_masking
        )

    def compute_print_matrix(self, print_film, **kwargs):
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
        self, print_film, red_light=0.0, green_light=0.0, blue_light=0.0, **kwargs
    ):
        """
        Compute printer light needed to print onto target print film to generate neutral
        exposure.

        Args:
            print_film: Film stock to print onto.
            red_light: Red printer light offset.
            green_light: Green printer light offset.
            blue_light: Blue printer light offset.
            **kwargs: Not used.

        Returns:
            Printer light as spectral curve.
        """
        compensation = 2 ** np.array(
            [red_light, green_light, blue_light], dtype=DEFAULT_DTYPE
        )
        # transmitted printer lights by middle gray negative
        reduced_lights = (PRINTER_LIGHTS.T * 10**-self.d_ref_sd).T

        target_exp = np.multiply(print_film.H_ref, compensation)
        # adjust printer lights to produce neutral exposure with middle gray negative
        if print_film.density_measure == "bw":
            light_factors = (
                (print_film.sensitivity.T @ reduced_lights) ** -1 * target_exp
            ).min()
        elif print_film.density_measure == "absolute":
            black_body = np.asarray(colour.sd_blackbody(10000, SPECTRAL_SHAPE).values)
            lights = black_body[:, np.newaxis] * (
                target_exp
                / (print_film.sensitivity.T @ (black_body * 10**-self.d_ref_sd))
            )
            return lights
        else:
            light_factors = (
                np.linalg.inv(print_film.sensitivity.T @ reduced_lights) @ target_exp
            )
        printer_light = np.sum(PRINTER_LIGHTS * light_factors, axis=1)
        return printer_light

    def compute_projection_light(
        self,
        projector_kelvin: int | float = 5500,
        reference_kelvin: int | float = 6504,
        white_comp: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes a projection light of the specified temperature whose intensity is
        scaled so that minimum density of the current film will produce the specified
        white point in linear rec. 709 on the maximum color channel. Also gives scaled
        XYZ cmfs for use in conjunction with that light.

        Args:
            projector_kelvin: The light temperature of the projection lamp.
            reference_kelvin: The reference temperature for the XYZ cmfs calibration.
                Should be left unchanged under normal circumstances.
            white_comp: Whether to scale the output to clip at 1.0 in sRGB gamut.

        Returns:
            A tuple (projector_light, xyz_cmfs).
        """
        reference_light = np.asarray(
            colour.sd_blackbody(reference_kelvin)
            .align(SPECTRAL_SHAPE)
            .normalise()
            .values
        )
        projector_light = np.asarray(
            colour.sd_blackbody(projector_kelvin)
            .align(SPECTRAL_SHAPE)
            .normalise()
            .values
        )
        reference_white = np.asarray(
            colour.xyY_to_XYZ([*CCT_to_xy(reference_kelvin), 1.0])
        )
        xyz_cmfs = XYZ_CMFS * (reference_white / (XYZ_CMFS.T @ reference_light))
        if white_comp:
            peak_rgb = colour.XYZ_to_RGB(
                xyz_cmfs.T @ (projector_light * 10**-self.d_min_sd), "sRGB"
            )
            peak = peak_rgb.max()
            projector_light *= 1 / peak
        return projector_light, xyz_cmfs

    def plot_data(self, film_b=None, color_masking=None):
        """Plots the spectral density, sensitivity, and sensiometric curve."""
        wavelengths = SPECTRAL_SHAPE.wavelengths
        default_colors = ["r", "g", "b"]

        is_comparison = film_b is not None
        cols = 2 if is_comparison else 1

        fig, axes = plt.subplots(
            3, cols, figsize=(12 if cols == 2 else 8, 12), squeeze=False
        )

        def plot_film_data(film, ax_col, color_masking=None):
            # Spectral Sensitivity
            num_curves = film.sensitivity.shape[1]
            colors = ["black"] if num_curves == 1 else default_colors
            for i, a in enumerate(film.sensitivity.T):
                color = colors[i] if i < len(colors) else None
                axes[0, ax_col].plot(wavelengths, a, color=color)
            axes[0, ax_col].set_title(
                f"{film.__class__.__name__} - Spectral Sensitivity"
            )
            axes[0, ax_col].set_xlabel("Wavelength")
            axes[0, ax_col].set_ylabel("Sensitivity")

            # Density Curve
            num_curves = len(film.log_exposure)
            colors = ["black"] if num_curves == 1 else default_colors
            gamma_values = []

            for i, (log_exp, density) in enumerate(
                zip(*film.get_density_curve(color_masking))
            ):
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
                axes[1, ax_col].axvline(
                    x=ref_val, color="black", linestyle="--", linewidth=1
                )
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
            for i, x in enumerate(film.get_spectral_density(color_masking).T):
                color = colors[i] if i < len(colors) else None
                axes[2, ax_col].plot(wavelengths, x, color=color)
            axes[2, ax_col].plot(wavelengths, film.d_min_sd, "--", color="black")
            axes[2, ax_col].plot(wavelengths, film.d_ref_sd, color="black")
            axes[2, ax_col].set_title(f"{film.__class__.__name__} - Spectral Density")
            axes[2, ax_col].set_xlabel("Wavelength")
            axes[2, ax_col].set_ylabel("Density")

        # Plot film_a in the first column
        plot_film_data(self, 0, color_masking)

        # Plot film_b in the second column if provided
        if is_comparison:
            plot_film_data(film_b, 1, None)

        plt.tight_layout()
        plt.show()

    def grain_transform(self, rgb, scale=1.0, std_div=1.0):
        """Encoding for the grain intensity LUT."""
        # TODO: fix for BW film
        # scale = max(image.shape) / max(frame_width, frame_height) in pixels per mm,
        # default for 3840 / 24mm
        # std_div is of the sampled gaussian noise to be applied, default is 0.1 to stay
        # in [0, 1] range
        adx_density_scale = np.array([1.00, 0.92, 0.95], dtype=DEFAULT_DTYPE) * (
            8000.0 / 65535.0
        )
        std_factor = math.sqrt(math.pi) * 0.024 * scale * adx_density_scale / std_div
        xps = [rms_density for rms_density in self.rms_density]
        fps = [rms * std_factor[i] for i, rms in enumerate(self.rms_curve)]
        noise_factors = multi_channel_interp(rgb, xps, fps)

        return noise_factors

    def get_input_lut(self, exposure_kelvin=6500, tint=0.0, exp_comp=0.0) -> np.ndarray:
        """
        Compute a 2D LUT for use with [`apply_2d_lut`][] that converts from scene linear
        CIE XYZ the per layer exposure.

        Args:
            exposure_kelvin: The scene WB in kelvin.
            tint: The tint adjustment on the green -- red/purple axis.
            exp_comp: Exposure compensation in stops.

        Returns:
            The 2D LUT of shape (n, n, 3).
        """
        exp_comp = 2**exp_comp
        gray_XYZ = CCT_to_XYZ(exposure_kelvin, 0.18, tint)
        reference_XYZ = CCT_to_XYZ(6504, 0.18)
        gray_spectral = apply_2d_lut(gray_XYZ, SPECTRUM_LUT)
        reference_spectral = apply_2d_lut(reference_XYZ, SPECTRUM_LUT)
        corrected_sensitivity = (
            self.sensitivity * (reference_spectral / gray_spectral)[:, None]
        )
        spectral_input_lut = SPECTRUM_LUT @ corrected_sensitivity
        if self.density_measure == "bw":
            ref_exp = apply_2d_lut(reference_XYZ, spectral_input_lut)
        else:
            ref_exp = apply_2d_lut(gray_XYZ, spectral_input_lut)
        correction_factors = self.H_ref / ref_exp
        spectral_input_lut *= correction_factors * exp_comp
        return spectral_input_lut

    def compute_lad(self, luminance=0.1):
        """Find the Lab Aim Density for a neutral gray."""
        projection_light, xyz_cmfs = self.compute_projection_light(
            projector_kelvin=6504
        )
        d_min_sd = self.d_min_sd
        density_mat = self.get_spectral_density()
        output_mat = (xyz_cmfs.T * projection_light * 10**-d_min_sd).T
        lad = output_to_density(CCT_to_XYZ(6504, luminance), density_mat, output_mat)
        return lad

    def layer_activation_to_apd_matrix(self, color_masking: None | float = None):
        """
        Get the matrix that converts from layer activation to ACES Printing Density for
        ADX encoding.
        """
        if self.density_measure == "bw":
            return np.ones((1, 1), dtype=DEFAULT_DTYPE)
        return APD.T @ self.get_spectral_density(color_masking)

    def apd_to_layer_activation_matrix(self, color_masking: None | float = None):
        """
        Get the matrix that converts from ACES Printing Density to layer activation for
        ADX decoding.
        """
        if self.density_measure == "bw":
            return np.ones((1, 1), dtype=DEFAULT_DTYPE)
        return np.linalg.inv(self.layer_activation_to_apd_matrix(color_masking))

    def adx_encoding(self, image, scaling=1.0, color_masking: None | float = None):
        """
        Encode layer activation in absolute densities as ADX16 data in the [0, 1] range.

        Args:
            image: The image (or LUT) containing layer activations.
            scaling: Linear scaling applied to the values.
                For 1.0 it is ADX16-like and for 4.0 it is ADX10-like.
            color_masking: The color masking assumed for the layer activations.

        Returns:
            The image (or LUT) encoded in ADX.
        """
        image @= self.layer_activation_to_apd_matrix(color_masking).T

        image = adx16_encode(image, scaling=scaling)

        return image

    def adx_decoding(self, image, scaling=1.0, color_masking: None | float = None):
        """
        Decode layer activation from ADX16 data in the [0, 1] range to absolute
        densities.

        Args:
            image: The image (or LUT) encoded in ADX.
            scaling: Linear scaling applied to the endoced values.
                For 1.0 it is ADX16-like and for 4.0 it is ADX10-like.
            color_masking: The color masking assumed for the layer activations.

        Returns:
            The image (or LUT) represented as absolute layer activations.
        """
        if self.density_measure == "bw":
            image = image[..., 0][..., np.newaxis]  # reduce dim

        image = adx16_decode(image, scaling=scaling)

        image @= self.apd_to_layer_activation_matrix(color_masking).T

        return image

    def input_transform(
        self,
        image,
        colorspace: LiteralRGBColourspace | None = None,
        exp_comp=0.0,
        exp_kelvin=6500,
        tint=0.0,
        color_masking: None | float = None,
        push_pull: float = 0.0,
    ):
        """
        Transform from scene referred image data to the per layer activation in absolute
        densities.

        Args:
            image: The image (or LUT) in a scene referred color space.
            colorspace: What color space the input is encoded in.
            exp_comp: The exposure adjustment.
            exp_kelvin: The scene white balance in kelving.
            tint: The tint adjustment on the green -- red/purple axis.
            color_masking: How strong the color mask is on the negative film. 0 for no
                mask. 1 for a perfectly optimized mask (no pollution of other layers).
                >1 for increased saturation. Most film stocks don't provide exact data.
                For films with orange mask can be close to 1, for other (e.g. slide
                film) should be set quite low.
            push_pull: By how many stops to push/pull the negative to adjust contrast.

        Returns:
            The resulting layer activation as densities.
        """
        if colorspace is not None:
            image = colour.RGB_to_XYZ(image, colorspace, apply_cctf_decoding=True)

        input_lut = self.get_input_lut(exp_kelvin, tint, exp_comp)

        image = apply_2d_lut(np.clip(image, 0, None), input_lut)

        image = np.log10(np.clip(image, 10**-16, None))

        image = self.log_exposure_to_density(image, color_masking, push_pull)

        return image

    def print_to(
        self,
        image,
        print_film,
        color_masking: None | float = None,
        red_light=0.0,
        green_light=0.0,
        blue_light=0.0,
    ):
        """
        Print to another film stock.

        Args:
            image: The image (or LUT) containing layer activations in absolute
                densities.
            negative_film: The film stock from which to print.
            print_film: The film stock to print onto.
            color_masking: How strong the color mask is on the negative film. 0 for no
                mask. 1 for a perfectly optimized mask (no pollution of other layers).
                >1 for increased saturation. Most film stocks don't provide exact data.
                For films with orange mask can be close to 1, for other (e.g. slide
                film) should be set quite low.
            red_light: Offset of the red printer light from neutral.
            green_light: Offset of the green printer light from neutral.
            blue_light: Offset of the blue printer light from neutral.

        Returns:
            The resulting layer activations of the print stock.
        """
        return self.printing_process(
            image, self, print_film, color_masking, red_light, green_light, blue_light
        )

    @staticmethod
    def printing_process(
        image,
        negative_film: "FilmSpectral",
        print_film: "FilmSpectral",
        color_masking: None | float = None,
        red_light=0.0,
        green_light=0.0,
        blue_light=0.0,
    ):
        """
        Print from one film stock onto another.

        Args:
            image: The image (or LUT) containing layer activations in absolute
                densities.
            negative_film: The film stock from which to print.
            print_film: The film stock to print onto.
            color_masking: How strong the color mask is on the negative film. 0 for no
                mask. 1 for a perfectly optimized mask (no pollution of other layers).
                >1 for increased saturation. Most film stocks don't provide exact data.
                For films with orange mask can be close to 1, for other (e.g. slide
                film) should be set quite low.
            red_light: Offset of the red printer light from neutral.
            green_light: Offset of the green printer light from neutral.
            blue_light: Offset of the blue printer light from neutral.

        Returns:
            The resulting layer activations of the print stock.
        """
        if negative_film.density_measure == print_film.density_measure == "bw":
            image = -image + print_film.log_H_ref + negative_film.d_ref + green_light
        else:
            density_neg = negative_film.get_spectral_density(color_masking)
            printer_light = negative_film.compute_printer_light(
                print_film, red_light, green_light, blue_light
            )
            printing_mat = (
                print_film.sensitivity.T * printer_light * 10**-negative_film.d_min_sd
            ).T
            printing_mat = printing_mat.reshape(-1, 3, printing_mat.shape[-1]).sum(
                axis=1
            )
            density_neg = density_neg.reshape(-1, 3, density_neg.shape[-1]).mean(axis=1)
            image = np.log10(
                np.clip(10 ** -(image @ density_neg.T) @ printing_mat, 0.00001, None)
            )

        image = print_film.log_exposure_to_density(image)

        return image

    def project(
        self,
        image: np.ndarray,
        projector_kelvin: int | float = 6500,
        color_masking: None | float = None,
        white_comp: bool = False,
        white_balance: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Get the scene referred output from projection or viewing under an illuminant.

        Args:
            image: The image (or LUT) containing layer activations in absolute
                densities.
            projector_kelvin: The white balance in kelvin of the projection lamp.
            color_masking: How strong the color mask is on the negative film. 0 for no
                mask. 1 for a perfectly optimized mask (no pollution of other layers).
                >1 for increased saturation. Most film stocks don't provide exact data.
                For films with orange mask can be close to 1, for other (e.g. slide
                film) should be set quite low.
            white_comp: Whether to adjust the output brightness that it clips at exactly
                1.
            white_balance: Whether to adjust

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
            projection_light, xyz_cmfs = self.compute_projection_light(
                projector_kelvin=projector_kelvin, white_comp=white_comp
            )
            d_min_sd = self.d_min_sd

            density_mat = self.get_spectral_density(color_masking)

            output_mat = (xyz_cmfs.T * projection_light * 10**-d_min_sd).T
            output_mat = output_mat.reshape(-1, 3, 3).sum(axis=1)
            density_mat = density_mat.reshape(-1, 3, 3).mean(axis=1)

            mid_gray = (
                10 ** -(self.get_d_ref(color_masking) @ density_mat.T) @ output_mat
            )
            if white_balance:
                out_gray = np.asarray(CCT_to_XYZ(projector_kelvin, mid_gray[1]))
                output_mat = np.asarray(
                    colour.chromatic_adaptation(output_mat, mid_gray, out_gray)
                )
            else:
                out_gray = mid_gray

            image = 10 ** -(image @ density_mat.T) @ output_mat

        return image, out_gray
