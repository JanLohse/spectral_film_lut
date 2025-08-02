from spectral_film_lut.bw_negative_film.kodak_5222 import *
from spectral_film_lut.wratten_filters import WRATTEN

from matplotlib import pyplot as plt


class KodakDyeTransferSlide(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.lad = [0.95] * 3
        self.density_measure = 'absolute'
        self.manufacturer = 'Kodak'
        self.stage = 'print'
        self.type = 'positive'
        self.medium = 'photo'
        self.year = 1946

        separation_neg = Kodak5222Dev6()
        sensitivity = separation_neg.sensitivity
        filters = xp.stack([WRATTEN["29"], WRATTEN["61"], WRATTEN["47"]])
        self.sensitivity = sensitivity * filters.T

        # spectral dye density
        red_sd = {399.9333: 0.1744, 416.2330: 0.1563, 437.2710: 0.1569, 456.4138: 0.1708, 477.4518: 0.1968,
                  501.3328: 0.2447, 523.2237: 0.3199, 552.1273: 0.4851, 567.8584: 0.5867, 582.1681: 0.6796,
                  593.6348: 0.7644, 598.3731: 0.7951, 613.9147: 0.8621, 629.7406: 0.9218, 639.9753: 0.9580,
                  646.8932: 0.9775, 652.3896: 0.9830, 661.4871: 0.9660, 672.2905: 0.9171, 685.8420: 0.8424,
                  699.9621: 0.7477}
        green_sd = {399.7438: 0.1771, 406.7564: 0.1627, 412.8214: 0.1549, 420.2132: 0.1617, 430.0688: 0.1893,
                    441.2512: 0.2356, 461.6259: 0.3725, 481.3372: 0.5275, 504.7444: 0.7084, 517.4430: 0.7860,
                    527.2986: 0.8156, 537.3438: 0.8292, 546.8204: 0.8308, 556.8656: 0.8072, 563.4992: 0.7768,
                    571.0805: 0.7025, 589.4651: 0.4383, 603.5852: 0.2532, 617.1367: 0.1425, 642.3444: 0.0635,
                    661.8662: 0.0441, 683.0938: 0.0354, 699.1092: 0.0273}
        blue_sd = {400.1228: 0.9301, 405.8088: 0.9509, 412.2529: 0.9664, 417.4650: 0.9665, 424.1934: 0.9508,
                   433.0066: 0.9144, 442.0093: 0.8622, 451.9598: 0.7793, 460.9625: 0.6925, 471.1972: 0.5850,
                   483.7063: 0.4364, 495.0782: 0.3210, 506.8292: 0.2136, 519.7174: 0.1395, 536.0171: 0.0908,
                   548.3367: 0.0752, 563.3097: 0.0757, 595.9091: 0.0700, 630.7830: 0.0545, 655.2326: 0.0466,
                   679.1136: 0.0340, 693.7075: 0.0318, 699.7726: 0.0306}
        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]
        self.spectral_density = xp.stack(
            [xp.asarray(self.gaussian_extrapolation(x).values) for x in self.spectral_density]).T
        density_measurements = xp.sum(densiometry.status_a * self.spectral_density, axis=0)
        density_measurements /= density_measurements[0]

        # sensiometry curve from kodak matrix film 4150
        curve = {-0.8901: 0.0337, -0.8083: 0.0450, -0.7327: 0.0582, -0.6203: 0.0795, -0.5184: 0.1129,
                 -0.4309: 0.1500, -0.3650: 0.1855, -0.2970: 0.2272, -0.2282: 0.2761, -0.1474: 0.3376,
                 -0.0246: 0.4576, 0.0716: 0.5770, 0.1933: 0.7584, 0.3935: 1.1285, 0.5697: 1.4631, 0.6975: 1.7012,
                 0.8412: 1.9662, 0.9197: 2.0985, 1.0265: 2.2539, 1.1249: 2.3620, 1.2107: 2.4380, 1.2977: 2.4977,
                 1.3779: 2.5387, 1.4553: 2.5663, 1.5962: 2.6006, 1.7229: 2.6213, 1.8457: 2.6357, 1.9573: 2.6437}
        log_exposure_matrix = xp.array(list(curve.keys()), dtype=default_dtype)
        density_curve_matrix = xp.array(list(curve.values()), dtype=default_dtype)
        log_H_ref_mat = xp.interp(xp.asarray(self.lad[0]), density_curve_matrix, log_exposure_matrix)
        separation_curve = separation_neg.density_curve[0]
        separation_exposure = separation_neg.log_exposure[0]
        slope = (separation_curve[-1] - separation_curve[-2]) / (separation_exposure[-1] - separation_exposure[-2])
        separation_curve = xp.append(separation_curve, separation_curve[-1] + slope * 1)
        separation_exposure = xp.append(separation_exposure, separation_exposure[-1] + 1)
        density_curve = xp.interp(log_H_ref_mat - separation_curve + separation_neg.d_ref,
                                  log_exposure_matrix, density_curve_matrix)

        # TODO: highlight_mask

        self.log_exposure = [separation_exposure] * 3
        self.density_curve = [density_curve * scale for scale in density_measurements]

        self.calibrate()
