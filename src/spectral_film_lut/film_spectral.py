from colour import SpectralDistribution, MultiSpectralDistributions

from spectral_film_lut.utils import *
from spectral_film_lut.densiometry import DENSIOMETRY
from spectral_film_lut import densiometry

default_dtype = np.float32
colour.utilities.set_default_float_dtype(default_dtype)


class FilmSpectral:
    def __init__(self):
        self.iso = None
        self.lad = None
        self.density_curve = None
        self.log_exposure = None
        self.log_H_ref = None
        self.H_ref = None
        self.log_sensitivity = None
        self.sensitivity = None
        self.exposure_base = 10
        self.d_min = None
        self.d_ref = None
        self.d_max = None
        self.density_measure = None
        self.spectral_density = None
        self.d_min_sd = None
        self.d_ref_sd = None
        self.XYZ_to_exp = None
        self.rms_curve = None
        self.rms_density = None
        self.rms = None
        self.exposure_kelvin = 5500
        self.projection_kelvin = 6500

    def calibrate(self):
        # target exposure of middle gray in log lux-seconds
        # normally use iso value, if not provided use target density of 1.0 on the green channel
        if self.iso is not None:
            self.log_H_ref = xp.ones(len(self.log_exposure)) * math.log10(12.5 / self.iso)
        elif self.lad is not None:
            if self.density_measure == 'absolute':
                self.lad = xp.linalg.inv(densiometry.status_a.T @ self.spectral_density) @ xp.array(self.lad)
            self.log_H_ref = xp.array([
                xp.interp(xp.asarray(a), xp.asarray(sorted_b), xp.asarray(sorted_c))
                for a, b, c in zip(self.lad, self.density_curve, self.log_exposure)
                for sorted_b, sorted_c in [zip(*sorted(zip(b, c)))]
            ])
        self.H_ref = 10 ** self.log_H_ref

        # extrapolate log_sensitivity to linear sensitivity
        if self.log_sensitivity is not None:
            self.log_sensitivity = xp.stack(
                [xp.asarray(colour.SpectralDistribution(x).align(spectral_shape, extrapolator_kwargs={
                    'method': 'linear'}).align(spectral_shape).values) for x in self.log_sensitivity]).T
            self.sensitivity = 10 ** self.log_sensitivity

        # convert relative camera exposure to absolute exposure in log lux-seconds for characteristic curve
        if self.exposure_base != 10:
            self.log_exposure = [xp.log10(self.exposure_base ** x * 10 ** y) for x, y in
                                 zip(self.log_exposure, self.log_H_ref)]

        self.d_min = xp.array([xp.min(x) for x in self.density_curve])
        self.density_curve = [x - d for x, d in zip(self.density_curve, self.d_min)]
        self.d_ref = self.log_exposure_to_density(self.log_H_ref).reshape(-1)
        self.d_max = xp.array([xp.max(x) for x in self.density_curve])

        # align spectral densities
        if self.density_measure == 'bw':
            self.spectral_density = xp.asarray(colour.colorimetry.sd_constant(1, spectral_shape).values)
            self.d_min_sd = xp.asarray(colour.colorimetry.sd_constant(to_numpy(self.d_min), spectral_shape).values)
            self.d_ref_sd = self.spectral_density * self.d_ref + self.d_min
            self.spectral_density = self.spectral_density.reshape(-1, 1)
        else:
            if self.d_min_sd is not None:
                self.d_min_sd = self.gaussian_extrapolation(self.d_min_sd)
                self.d_min_sd = xp.asarray(self.d_min_sd.values)
            else:
                self.d_min_sd = xp.asarray(colour.sd_zeros(spectral_shape).values)

            if self.d_ref_sd is not None:
                self.gaussian_extrapolation(self.d_ref_sd)
            if self.spectral_density is not None and self.density_measure != 'absolute':
                self.spectral_density = xp.stack(
                    [xp.asarray(self.gaussian_extrapolation(x).values) for x in self.spectral_density]).T
            elif self.density_measure != 'absolute':
                self.spectral_density = construct_spectral_density(self.d_ref_sd - to_numpy(self.d_min_sd))

            if self.density_measure != 'absolute':
                self.spectral_density @= xp.linalg.inv(DENSIOMETRY[self.density_measure].T @ self.spectral_density)
            self.d_min_sd = self.d_min_sd + self.spectral_density @ (
                    self.d_min - DENSIOMETRY[self.density_measure].T @ self.d_min_sd)
            self.d_ref_sd = self.spectral_density @ self.d_ref + self.d_min_sd

        self.XYZ_to_exp = self.sensitivity.T @ densiometry.xyz_dual

        if self.rms_curve is not None and self.rms_density is not None:
            rms_temp = [self.prepare_rms_data(a, b) for a, b in zip(self.rms_curve, self.rms_density)]
            self.rms_curve = [x[0] for x in rms_temp]
            self.rms_density = [x[1] for x in rms_temp]
            if self.rms is not None:
                if len(self.rms_density) == 3:
                    rms_color_factors = xp.array([0.26, 0.57, 0.17], dtype=xp.float32)
                    scaling = 1.2375
                    rms_color_factors /= rms_color_factors.sum()
                    ref_rms = xp.sqrt(xp.sum((multi_channel_interp(xp.ones(3), self.rms_density, self.rms_curve) ** 2 * rms_color_factors ** 2))) / scaling
                else:
                    ref_rms = xp.interp(xp.asarray(1), self.rms_density[0], self.rms_curve[0])
                if self.rms > 1:
                    self.rms /= 1000
                factor = self.rms / ref_rms
                self.rms_curve = [x * factor for x in self.rms_curve]

        for key, value in self.__dict__.items():
            if type(value) is xp.ndarray and value.dtype is not default_dtype:
                self.__dict__[key] = value.astype(default_dtype)

    @staticmethod
    def prepare_rms_data(rms, density):
        x = xp.array(list(density.keys()), dtype=default_dtype)
        fp = xp.array(list(density.values()), dtype=default_dtype)
        fp -= fp.min()
        density = xp.interp(xp.array(list(rms.keys()), dtype=default_dtype), x, fp)
        rms = xp.array(list(rms.values()), dtype=default_dtype)
        sorting = density.argsort()
        density = density[sorting]
        rms = rms[sorting]
        return rms, density

    @staticmethod
    def gaussian_extrapolation(sd):
        def extrapolate(a_x, a_y, b_x, b_y, wavelengths, d_1=30, d_2=0.75):
            m = (a_y - b_y) / (a_x - b_x)
            d = d_1 * m / np.absolute(a_y) ** d_2
            a = a_y / np.exp(-d ** 2)
            c = to_numpy(a / m * -2 * d * np.exp(-d ** 2))
            b = to_numpy(a_x - c * d)
            extrapolator = lambda x: a * np.exp(- (x - b) ** 2 / c ** 2)
            return extrapolator(wavelengths)

        sd.interpolate(spectral_shape)

        def_wv = spectral_shape.wavelengths
        wv_left = def_wv[def_wv < sd.wavelengths[0]]
        wv_right = def_wv[def_wv > sd.wavelengths[-1]]
        values_left = extrapolate(sd.wavelengths[0], sd.values[0], sd.wavelengths[1], sd.values[1], wv_left)
        values_right = extrapolate(sd.wavelengths[-1], sd.values[-1], sd.wavelengths[-2], sd.values[-2], wv_right)
        sd.values, sd.wavelengths = np.concatenate((values_left, sd.values, values_right)), np.concatenate(
            (wv_left, sd.wavelengths, wv_right))
        sd.interpolate(spectral_shape)

        return sd

    def log_exposure_to_density(self, log_exposure, pre_flash=-4):
        if pre_flash > -4:
            log_exposure_curve = [
                xp.log10(xp.clip((10 ** x - y * 2 ** pre_flash) / (1 - 1 * 2 ** pre_flash), 10 ** -16, None)) for x, y
                in zip(self.log_exposure, self.H_ref)]
        else:
            log_exposure_curve = self.log_exposure
        extrapolate = self.density_measure != "status_a"
        density = multi_channel_interp(log_exposure, log_exposure_curve, self.density_curve, right_extrapolate=extrapolate)

        return density

    def compute_print_matrix(self, print_film, **kwargs):
        printer_light = self.compute_printer_light(print_film, **kwargs)
        if print_film.density_measure == 'absolute':
            print_sensitivity = print_film.sensitivity * printer_light
            peak_exposure = xp.log10(xp.sum(print_sensitivity, axis=0))
        else:
            # compute max exposure produced by unfiltered printer light
            peak_exposure = xp.log10(print_film.sensitivity.T @ printer_light)
            # compute density matrix from print film sensitivity under adjusted printer light
            print_sensitivity = (print_film.sensitivity.T * printer_light).T
        print_sensitivity /= xp.sum(print_sensitivity, axis=0)
        density_matrix = print_sensitivity.T @ self.spectral_density
        density_base = print_sensitivity.T @ self.d_min_sd
        return density_matrix, peak_exposure - density_base

    def compute_printer_light(self, print_film, red_light=0, green_light=0, blue_light=0, **kwargs):
        compensation = 2 ** xp.array([red_light, green_light, blue_light], dtype=default_dtype)
        # transmitted printer lights by middle gray negative
        reduced_lights = (densiometry.printer_lights.T * 10 ** -self.d_ref_sd).T

        target_exp = xp.multiply(print_film.H_ref, compensation)
        # adjust printer lights to produce neutral exposure with middle gray negative
        if print_film.density_measure == 'bw':
            light_factors = ((print_film.sensitivity.T @ reduced_lights) ** -1 * target_exp).min()
        elif print_film.density_measure == 'absolute':
            black_body = xp.asarray(colour.sd_blackbody(10000, spectral_shape).values)
            lights = black_body[:, xp.newaxis] * (
                        target_exp / (print_film.sensitivity.T @ (black_body * 10 ** -self.d_ref_sd)))
            return lights
        else:
            light_factors = xp.linalg.inv(print_film.sensitivity.T @ reduced_lights) @ target_exp
        printer_light = xp.sum(densiometry.printer_lights * light_factors, axis=1)
        return printer_light

    def compute_projection_light(self, projector_kelvin=5500, reference_kelvin=6504, white_point=1.):
        reference_light = xp.asarray(
            colour.sd_blackbody(reference_kelvin).align(spectral_shape).normalise().values)
        projector_light = xp.asarray(
            colour.sd_blackbody(projector_kelvin).align(spectral_shape).normalise().values)
        reference_white = xp.asarray(colour.xyY_to_XYZ([*colour.CCT_to_xy(reference_kelvin), 1.]))
        xyz_cmfs = densiometry.xyz_cmfs * (reference_white / (densiometry.xyz_cmfs.T @ reference_light))
        peak_xyz = colour.XYZ_to_RGB(to_numpy(xyz_cmfs.T @ (projector_light * 10 ** -self.d_min_sd)), "sRGB")
        projector_light /= xp.max(peak_xyz) / white_point
        return projector_light, xyz_cmfs

    @staticmethod
    def generate_conversion(negative_film, print_film=None, input_colourspace="ARRI Wide Gamut 4", measure_time=False,
                            output_colourspace="sRGB", projector_kelvin=6500, matrix_method=False, exp_comp=0,
                            white_point=1., mode='full', exposure_kelvin=5500, d_buffer=0.5, gamma=1,
                            halation_func=None, pre_flash_neg=-4, pre_flash_print=-4, gamut_compression=0.2,
                            output_transform=None, black_offset=0, black_pivot=0.18, photo_inversion=True, **kwargs):
        pipeline = []

        def add(func, name):
            pipeline.append((func, name))

        def add_output_transform():
            if output_colourspace is not None and output_transform is None:
                add(lambda x: colour.XYZ_to_RGB(to_numpy(x), output_colourspace, apply_cctf_encoding=True), "output")
            elif output_transform is not None:
                add(output_transform, "output")

        def add_black_offset(compute_factor=False):
            if compute_factor:
                offset = black_offset * max(pipeline[-1][0](output_film.d_max).min(), 0.0005)
            else:
                offset = black_offset * 0.0005
            func = lambda x: np.clip(
                np.where(x >= black_pivot, x, black_pivot * ((x - offset) / (black_pivot - offset)) ** (
                                                      (black_pivot - offset) / black_pivot))
                , 0, None)
            add(func,"black_offset")

        if mode == 'negative' or mode == 'full':
            if gamma != 1:
                add(lambda x: x ** gamma, "gamma")

            if input_colourspace is not None:
                add(lambda x: xp.asarray(colour.RGB_to_XYZ(x, input_colourspace, apply_cctf_decoding=True)), "input")
            elif cuda_available:
                add(lambda x: xp.asarray(x), "cast to cuda")

            exp_comp = 2 ** exp_comp
            gray = xp.asarray(negative_film.CCT_to_XYZ(exposure_kelvin, 0.18))
            ref_exp = negative_film.XYZ_to_exp @ gray
            correction_factors = negative_film.H_ref / ref_exp
            if negative_film.density_measure == 'bw':
                wb_factors = (xp.asarray(negative_film.CCT_to_XYZ(negative_film.exposure_kelvin, 0.18)) / gray)
                correction_factors = ref_exp / (
                        negative_film.XYZ_to_exp @ wb_factors) / .18 * correction_factors * wb_factors.reshape(-1, 1)
            XYZ_to_exp = (negative_film.XYZ_to_exp.T * correction_factors).T * exp_comp

            if gamut_compression and negative_film.density_measure != 'bw':
                XYZ_to_exp, compression_inv = FilmSpectral.gamut_compression_matrices(XYZ_to_exp, gamut_compression)
            add(lambda x: x @ XYZ_to_exp.T, "linear exposure")

            if gamut_compression and negative_film.density_measure != 'bw':
                add(lambda x: xp.clip(x, 0, None) @ compression_inv, "gamut_compression_inv")

            if halation_func is not None:
                add(lambda x: halation_func(x), "halation")
            if pre_flash_neg > -4:
                add(lambda x: (x + negative_film.H_ref * 2 ** pre_flash_neg) * (1 - 2 ** pre_flash_neg), "pre-flash")

            add(lambda x: xp.log10(xp.clip(x, 10 ** -16, None)), "log exposure")

            add(negative_film.log_exposure_to_density, "characteristic curve")

        density_scale = (negative_film.d_max.max() + d_buffer) + 2
        if mode == 'negative':
            add(lambda x: (x + d_buffer / 2) / density_scale, 'scale density')
        elif mode == 'print':
            if cuda_available:
                add(lambda x: xp.asarray(x), "cast to cuda")
            if negative_film.density_measure == 'bw':
                add(lambda x: x[..., 0][..., xp.newaxis], "reduce dim")
            add(lambda x: x * density_scale - d_buffer / 2, 'scale density')

        if mode == 'print' or mode == 'full':
            if print_film is not None:
                if negative_film.density_measure == 'bw' and print_film.density_measure == 'bw':
                    printer_light = kwargs.get('green_light', 0)
                    add(lambda x: -x + (print_film.log_H_ref + negative_film.d_ref + printer_light), "printing")
                elif matrix_method:
                    density_matrix, peak_exposure = negative_film.compute_print_matrix(print_film, **kwargs)
                    add(lambda x: peak_exposure - x @ density_matrix.T, "printing matrix")
                else:
                    density_neg = negative_film.spectral_density.T

                    printer_light = negative_film.compute_printer_light(print_film, **kwargs)
                    if print_film.density_measure == 'absolute':
                        printing_mat = print_film.sensitivity * printer_light * 10 ** -negative_film.d_min_sd[:,
                                                                                       xp.newaxis]
                    else:
                        printing_mat = (print_film.sensitivity.T * printer_light * 10 ** -negative_film.d_min_sd).T
                    add(lambda x: xp.log10(xp.clip(10 ** -(x @ density_neg) @ printing_mat, 0.00001, None)), "printing")

                add(lambda x: print_film.log_exposure_to_density(x, pre_flash_print), "characteristic curve print")
                output_film = print_film
            else:
                output_film = negative_film

            if output_film.density_measure == 'bw':
                add(lambda x: white_point / 10 ** -output_film.d_min * 10 ** -x, "projection")
                if print_film is None:
                    adjustment = 1 / pipeline[-1][0](0)
                    target_gray = 0.18
                    gray = pipeline[-1][0](negative_film.d_ref) * adjustment * target_gray
                    output_gamma = 2
                    add(lambda x: (gray / (x * adjustment)) ** output_gamma * target_gray ** (1 - output_gamma),
                        "invert")
                    add(lambda x: x / (x + 1), "roll-off")
                add_black_offset(print_film is not None)
                if not 6500 <= projector_kelvin <= 6505:
                    wb = xp.asarray(negative_film.CCT_to_XYZ(projector_kelvin))
                    add(lambda x: x * wb, "projection color")
                    add_output_transform()
                elif output_colourspace is not None and output_transform is None:
                    add(lambda x: colour.models.RGB_COLOURSPACES[output_colourspace].cctf_encoding(
                        to_numpy(x).repeat(3, axis=-1)), "output")
                elif output_transform is not None:
                    add(lambda x: output_transform(x, apply_matrix=False).repeat(3, axis=-1), "output")
            elif print_film is not None or negative_film.density_measure == "status_a" or photo_inversion:
                if print_film is None and negative_film.density_measure == "status_m":
                    output_kelvin = projector_kelvin
                    projector_kelvin = negative_film.projection_kelvin if negative_film.projection_kelvin is not None else 8500
                projection_light, xyz_cmfs = output_film.compute_projection_light(projector_kelvin=projector_kelvin,
                                                                                  white_point=white_point)
                d_min_sd = output_film.d_min_sd
                density_mat = output_film.spectral_density
                output_mat = (xyz_cmfs.T * projection_light * 10 ** -d_min_sd).T
                if matrix_method:
                    density_mat = density_mat.reshape(9, 9, 3).mean(axis=1)
                    output_mat = output_mat.reshape(9, 9, 3).sum(axis=1)
                add(lambda x: 10 ** -(x @ density_mat.T) @ output_mat, "output matrix")

                if print_film is None and negative_film.density_measure == "status_m":
                    FilmSpectral.add_photographic_inversion(add, negative_film, output_kelvin, pipeline)

                add_black_offset(True)
                add_output_transform()
            else:
                FilmSpectral.add_status_inversion(add, negative_film, add_black_offset, add_output_transform)

            if output_transform is None:
                add(lambda x: xp.clip(x, 0, 1), "clipping")


        def convert(x):
            start = time.time()
            for transform, title in pipeline:
                x = transform(x)
                if measure_time:
                    end = time.time()
                    print(f"{title:28} {end - start:.4f}s {x.dtype} {x.shape} {type(x)}")
                start = time.time()
            return x

        if mode == 'print' or mode == 'negative':
            return convert, density_scale
        else:
            return convert, 0

    @staticmethod
    def CCT_to_XYZ(CCT, Y=1.):
        xy = colour.CCT_to_xy(CCT)
        xyY = (xy[0], xy[1], Y)
        XYZ = colour.xyY_to_XYZ(xyY)
        return XYZ

    @staticmethod
    def linear_gamut_compression(rgb, gamut_compression=0):
        A = xp.identity(3, dtype=default_dtype) * (1 - gamut_compression) + gamut_compression / 3
        A_inv = xp.linalg.inv(A)
        rgb = xp.clip(rgb @ A_inv, 0, None) @ A
        return rgb

    @staticmethod
    def gamut_compression_matrices(matrix, gamut_compression=0.):
        A = xp.identity(3, dtype=default_dtype) * (1 - gamut_compression) + gamut_compression / 3
        A_inv = xp.linalg.inv(A)
        return matrix @ A_inv, A

    @staticmethod
    def add_photographic_inversion(add, negative_film, projector_kelvin, pipeline):
        XYZ_to_AP1 = xp.asarray(colour.RGB_COLOURSPACES["ACEScg"].matrix_XYZ_to_RGB)
        AP1_to_XYZ = xp.linalg.inv(XYZ_to_AP1)
        white = xp.asarray(negative_film.CCT_to_XYZ(projector_kelvin)) @ XYZ_to_AP1.T

        black = pipeline[-1][0](xp.zeros(3))
        gray = pipeline[-1][0](negative_film.d_ref)
        d_bright = negative_film.log_exposure_to_density(negative_film.log_H_ref + 0.5)
        light_gray = pipeline[-1][0](d_bright)

        adjustment = 1 / black
        gray = (gray * adjustment) @ XYZ_to_AP1.T
        light_gray = (light_gray * adjustment) @ XYZ_to_AP1.T
        reference_gamma = gray[..., 1] / light_gray[..., 1]
        gamma_adjustment = light_gray / gray * reference_gamma
        target_gray = 0.18 * white
        output_gamma = 4
        gray = target_gray * gray ** gamma_adjustment
        add(lambda x: (gray / (
                (x * adjustment) @ XYZ_to_AP1.T) ** gamma_adjustment) ** output_gamma * target_gray ** (
                              1 - output_gamma), "invert")
        add(lambda x: (x / (x + 1)) @ AP1_to_XYZ.T, "rolloff")

    @staticmethod
    def add_status_inversion(add, negative_film, add_black_offset, add_output_transform):
        status_m_to_apd = negative_film.densiometry["apd"].T @ negative_film.spectral_density
        gray = 10 ** -negative_film.d_ref @ status_m_to_apd.T
        target_gray = 0.18
        output_gamma = 4
        sRGB_to_XYZ = xp.linalg.inv(xp.asarray(colour.RGB_COLOURSPACES["sRGB"].matrix_XYZ_to_RGB))
        add(lambda x: 10 ** -x, "project")
        add(lambda x: (gray * target_gray / (x @ status_m_to_apd.T)) ** output_gamma * target_gray ** (
                1 - output_gamma), "invert")
        add(lambda x: (x / (x + 1)) @ sRGB_to_XYZ.T, "rolloff")
        add_black_offset()
        add_output_transform()