"""
The main class for handling all film data procesing and rendering.
"""

import math
import time

import colour.plotting
import numpy as np
import scipy
from colour import SpectralDistribution
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator

from spectral_film_lut import densiometry
from spectral_film_lut.color_processing import (
    COLORCHECKER_2005,
    CCT_to_xy,
    CCT_to_XYZ,
    output_color_transform,
    shadow_compensation,
)
from spectral_film_lut.color_space import GAMMA_FUNCTIONS
from spectral_film_lut.densiometry import (
    DENSIOMETRY,
    adx16_decode,
    adx16_encode,
    output_to_density,
)
from spectral_film_lut.film_data import FilmData
from spectral_film_lut.utils import (
    DEFAULT_DTYPE,
    SPECTRAL_SHAPE,
    construct_spectral_density,
    multi_channel_interp,
)
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
                self.lad = np.linalg.inv(
                    densiometry.STATUS_A.T @ self.spectral_density
                ) @ np.array(self.lad)
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
        x_new = np.linspace(log_H_min, log_H_max, 100, dtype=DEFAULT_DTYPE)
        self.density_curve = [
            PchipInterpolator(x, y)(x_new)
            for x, y in zip(self.log_exposure, self.density_curve)
        ]
        self.log_exposure = [x_new] * len(self.log_exposure)
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
                            self.lad, self.density_curve, self.log_exposure
                        )
                        for sorted_b, sorted_c in [zip(*sorted(zip(b, c)))]
                    ]
                )
                self.H_ref = 10**self.log_H_ref
            self.d_ref = self.log_exposure_to_density(self.log_H_ref).reshape(-1)
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
        log_exp_np = self.log_exposure[index]
        density_np = self.get_density_curve()[index]
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
        if negative is None:
            negative = self
        elif print_stock is None:
            print_stock = self
        self.color_checker = (
            self.generate_conversion(negative, print_stock, input_colourspace=None)(
                COLORCHECKER_2005
            )
            * 255
        ).astype(np.uint8)

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

    def log_exposure_to_density(self, log_exposure, color_masking=0):
        """
        Convert log_exposure to density values for current film stock.

        Args:
            log_exposure: Log exposure data to convert as array.
            color_masking: Color Masking factor in range [0, 1].
            gray. Deactivated for -4 or lower.
        """
        log_exposure_curve = self.log_exposure
        density = multi_channel_interp(
            log_exposure, log_exposure_curve, self.get_density_curve(color_masking)
        )

        return density

    def get_density_curve(self, color_masking=None):
        """
        Get characteristic density curve for current film stock.

        Args:
            color_masking: Color Masking factor in range [0, 1]. If None use default for
            current film stock.

        Returns:
            The density curve.
        """
        if color_masking is None:
            color_masking = self.color_masking
        if self.density_curve_pure is None:
            return self.density_curve
        return [
            a * color_masking + b * (1 - color_masking)
            for a, b in zip(self.density_curve_pure, self.density_curve)
        ]

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
        self, print_film, red_light=0, green_light=0, blue_light=0, **kwargs
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
        reduced_lights = (densiometry.PRINTER_LIGHTS.T * 10**-self.d_ref_sd).T

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
        printer_light = np.sum(densiometry.PRINTER_LIGHTS * light_factors, axis=1)
        return printer_light

    def compute_projection_light(
        self, projector_kelvin=5500, reference_kelvin=6504, white_comp=True
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
                zip(film.log_exposure, film.get_density_curve(color_masking))
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

    def get_input_lut(self, exposure_kelvin, tint, exp_comp):
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

    @staticmethod
    def generate_conversion(
        negative_film,
        print_film=None,
        input_colourspace: None | str = "ARRI Wide Gamut 4",
        measure_time=False,
        output_gamut="Rec. 709",
        gamma_func="Gamma 2.4",
        projector_kelvin=6500,
        exp_comp=0,
        white_comp=True,
        mode="full",
        exposure_kelvin=5500,
        halation_func=None,
        shadow_comp=0,
        color_masking=None,
        tint=0,
        sat_adjust=1,
        adx=True,
        adx_scaling=1.0,
        **kwargs,
    ):
        """The main function that performs the film simulation."""
        pipeline = []

        if color_masking is None:
            color_masking = negative_film.color_masking

        def add(func, name):
            pipeline.append((func, name))

        def add_output_transform():
            add(
                lambda x: output_color_transform(x, output_gamut, sat_adjust),
                "output_color_processing",
            )
            if shadow_comp:
                add(
                    lambda x: shadow_compensation(x, shadow_comp),
                    "shadow_comp",
                )

            add(GAMMA_FUNCTIONS[gamma_func], "output_gamut")

        if mode == "negative" or mode == "full":
            if input_colourspace is not None:
                add(
                    lambda x: np.asarray(
                        colour.RGB_to_XYZ(
                            x, input_colourspace, apply_cctf_decoding=True
                        )
                    ),
                    "input",
                )

            input_lut = negative_film.get_input_lut(exposure_kelvin, tint, exp_comp)
            add(
                lambda x: apply_2d_lut(np.clip(x, 0, None), input_lut),
                "input transform",
            )

            if halation_func is not None:
                add(lambda x: halation_func(x), "halation")

            add(lambda x: np.log10(np.clip(x, 10**-16, None)), "log exposure")

            add(
                lambda x: negative_film.log_exposure_to_density(x, color_masking),
                "characteristic curve",
            )

        if mode == "negative":
            if adx:
                layer_activation_to_apd = (
                    densiometry.APD.T
                    @ negative_film.get_spectral_density(color_masking)
                )
                add(lambda x: x @ layer_activation_to_apd.T, "encode APD")
            add(lambda x: adx16_encode(x, scaling=adx_scaling), "scale density")
        elif mode == "print":
            if negative_film.density_measure == "bw":
                add(lambda x: x[..., 0][..., np.newaxis], "reduce dim")
            add(lambda x: adx16_decode(x, scaling=adx_scaling), "scale density")
            if adx:
                layer_activation_to_apd = (
                    densiometry.APD.T
                    @ negative_film.get_spectral_density(color_masking)
                )
                apd_to_layer_activation = np.linalg.inv(layer_activation_to_apd)
                add(lambda x: x @ apd_to_layer_activation.T, "decode APD")

        if mode == "print" or mode == "full":
            if print_film is not None:
                if (
                    negative_film.density_measure == "bw"
                    and print_film.density_measure == "bw"
                ):
                    printer_light = kwargs.get("green_light", 0)
                    add(
                        lambda x: (
                            -x
                            + (
                                print_film.log_H_ref
                                + negative_film.d_ref
                                + printer_light
                            )
                        ),
                        "printing",
                    )
                else:
                    density_neg = negative_film.get_spectral_density(color_masking)

                    printer_light = negative_film.compute_printer_light(
                        print_film, **kwargs
                    )
                    if print_film.density_measure == "absolute":
                        printing_mat = (
                            print_film.sensitivity
                            * printer_light
                            * 10 ** -negative_film.d_min_sd[:, np.newaxis]
                        )
                    else:
                        printing_mat = (
                            print_film.sensitivity.T
                            * printer_light
                            * 10**-negative_film.d_min_sd
                        ).T
                    printing_mat = printing_mat.reshape(
                        -1, 3, printing_mat.shape[-1]
                    ).sum(axis=1)
                    density_neg = density_neg.reshape(
                        -1, 3, density_neg.shape[-1]
                    ).mean(axis=1)
                    add(
                        lambda x: np.log10(
                            np.clip(
                                10 ** -(x @ density_neg.T) @ printing_mat, 0.00001, None
                            )
                        ),
                        "printing",
                    )

                add(
                    lambda x: print_film.log_exposure_to_density(x),
                    "characteristic curve print",
                )
                output_film = print_film
            else:
                output_film = negative_film

            if output_film.density_measure == "bw":
                add(
                    lambda x: 1 / 10**-output_film.d_min * 10**-x,
                    "projection",
                )
                if not 6500 <= projector_kelvin <= 6505:
                    wb = np.asarray(CCT_to_XYZ(projector_kelvin))
                    add(lambda x: x * wb, "projection color")
                else:
                    add(lambda x: x.repeat(3, axis=-1), "repeat axis")
            else:
                if print_film is None:
                    output_kelvin = projector_kelvin
                    if negative_film.density_measure == "status_m":
                        projector_kelvin = (
                            negative_film.projection_kelvin
                            if negative_film.projection_kelvin is not None
                            else 8500
                        )
                    # elif negative_film.density_measure == "status_a":
                    #     projector_kelvin = negative_film.projection_kelvin
                if output_film.density_measure == "status_a" and print_film is None:
                    white_balance, white_comp = white_comp, False
                projection_light, xyz_cmfs = output_film.compute_projection_light(
                    projector_kelvin=projector_kelvin, white_comp=white_comp
                )
                d_min_sd = output_film.d_min_sd
                if print_film is None:
                    density_mat = output_film.get_spectral_density(color_masking)
                else:
                    density_mat = output_film.get_spectral_density()
                output_mat = (xyz_cmfs.T * projection_light * 10**-d_min_sd).T
                output_mat = output_mat.reshape(-1, 3, 3).sum(axis=1)
                density_mat = density_mat.reshape(-1, 3, 3).mean(axis=1)

                add(lambda x: 10 ** -(x @ density_mat.T) @ output_mat, "output matrix")
                if (
                    output_film.density_measure == "status_a"
                    and print_film is None
                    and white_balance
                ):
                    mid_gray = pipeline[-1][0](output_film.get_d_ref(color_masking))
                    out_gray = np.asarray(CCT_to_XYZ(output_kelvin, mid_gray[1]))
                    output_mat = np.asarray(
                        colour.chromatic_adaptation(output_mat, mid_gray, out_gray)
                    )

            add_output_transform()

        if mode == "grain":
            add(lambda x: adx16_decode(x, scaling=adx_scaling), "scale density")
            if adx:
                layer_activation_to_apd = (
                    densiometry.APD.T
                    @ negative_film.get_spectral_density(color_masking)
                )
                apd_to_layer_activation = np.linalg.inv(layer_activation_to_apd)
                add(lambda x: x @ apd_to_layer_activation.T, "decode APD")
            add(
                lambda x: negative_film.grain_transform(
                    x, std_div=0.001, scale=adx_scaling
                ),
                "grain_map",
            )

        def convert(x):
            start = time.time()
            for transform, title in pipeline:
                x = transform(x)
                if measure_time:
                    end = time.time()
                    print(
                        f"{title:28} {end - start:.4f}s {x.dtype} {x.shape} {type(x)} "
                        f"{x.min()} {x.max()}"
                    )
                start = time.time()
            return x

        return convert

    def compute_lad(self, luminance=0.1):
        projection_light, xyz_cmfs = self.compute_projection_light(
            projector_kelvin=6504
        )
        d_min_sd = self.d_min_sd
        density_mat = self.get_spectral_density()
        output_mat = (xyz_cmfs.T * projection_light * 10**-d_min_sd).T
        lad = output_to_density(CCT_to_XYZ(6504, luminance), density_mat, output_mat)
        return lad
