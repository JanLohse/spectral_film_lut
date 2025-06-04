from spectral_film_lut.bw_negative_film.kodak_5222 import *
from spectral_film_lut.wratten_filters import WRATTEN


class TechinicolorV(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.lad = [1.] * 3
        self.density_measure = 'absolute'

        sensitivity = Kodak5222().sensitivity
        filters = xp.stack([WRATTEN["29"], WRATTEN["99"], WRATTEN["98"]])
        self.sensitivity = sensitivity * filters.T

        # spectral dye density
        red_sd = {399.9648: 0.3509, 415.1483: 0.3383, 430.9935: 0.3022, 447.8416: 0.2446, 464.8902: 0.2102,
                  479.3314: 0.2026, 494.9760: 0.2126, 511.6235: 0.2372, 527.6693: 0.2882, 544.3168: 0.3758,
                  559.5603: 0.4831, 575.6061: 0.6270, 592.2536: 0.7903, 607.6976: 0.9055, 624.5457: 0.9883,
                  639.9898: 1.0661, 655.6344: 1.0282, 671.8807: 0.8137, 688.1271: 0.5674, 695.7488: 0.4673,
                  704.5740: 0.3625, 720.2186: 0.2451}
        green_sd = {399.6124: 0.4752, 414.9477: 0.4392, 431.3946: 0.4147, 448.2427: 0.4420, 464.6896: 0.5233,
                    479.5320: 0.6511, 494.7755: 0.8176, 511.2224: 0.9847, 528.0704: 1.0878, 544.3168: 1.0824,
                    559.5603: 1.0232, 576.2078: 0.9013, 591.8524: 0.6861, 607.8982: 0.4772, 624.7463: 0.3480,
                    640.1903: 0.2757, 656.0355: 0.2306, 671.8807: 0.1974, 687.9265: 0.1720, 704.1729: 0.1402,
                    720.4192: 0.1041}
        blue_sd = {399.6124: 1.1087, 414.7471: 1.1486, 431.1941: 1.1052, 448.3430: 0.9894, 464.2216: 0.8704,
                   479.5320: 0.7268, 487.1537: 0.6350, 495.9789: 0.5427, 511.4230: 0.3839, 544.7179: 0.2331,
                   559.5603: 0.2031, 576.0072: 0.1616, 591.8524: 0.1236, 607.6976: 0.1045, 625.3480: 0.1010,
                   640.1502: 0.1020, 653.8292: 0.1059, 672.0813: 0.1069, 688.7288: 0.1024, 706.3123: 0.1030,
                   713.1986: 0.1041}
        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]
        self.spectral_density = xp.stack(
            [xp.asarray(self.gaussian_extrapolation(x).values) for x in self.spectral_density]).T
        density_measurements = xp.sum(densiometry.status_a * self.spectral_density, axis=0)
        density_measurements /= density_measurements[0]

        # sensiometry curve from kodak matrix film 4150
        curve = {-0.8901: 0.0337, -0.8083: 0.0450, -0.7327: 0.0582, -0.6203: 0.0795, -0.5184: 0.1129, -0.4309: 0.1500,
                 -0.3650: 0.1855, -0.2970: 0.2272, -0.2282: 0.2761, -0.1474: 0.3376, -0.0246: 0.4576, 0.0716: 0.5770,
                 0.1933: 0.7584, 0.3935: 1.1285, 0.5697: 1.4631, 0.6975: 1.7012, 0.8412: 1.9662, 0.9197: 2.0985,
                 1.0265: 2.2539, 1.1249: 2.3620, 1.2107: 2.4380, 1.2977: 2.4977, 1.3779: 2.5387, 1.4553: 2.5663,
                 1.5962: 2.6006, 1.7229: 2.6213, 1.8457: 2.6357, 1.9573: 2.6437}
        log_exposure = xp.array(list(curve.keys()), dtype=default_dtype)
        density_curve = xp.array(list(curve.values()), dtype=default_dtype)

        density_curve_max = density_curve.max()

        # boost contrast
        density_curve = (1 - (1 - density_curve / density_curve_max) ** 10) * density_curve_max

        self.log_exposure = [log_exposure] * 3
        self.density_curve = [density_curve] * 3

        self.calibrate()
