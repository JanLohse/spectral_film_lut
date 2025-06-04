import colour.plotting

from spectral_film_lut.bw_negative_film.kodak_5222 import *
from spectral_film_lut.wratten_filters import WRATTEN


class TechnicolorIV(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.lad = [1.] * 3
        self.density_measure = 'absolute'

        # taken from ilford ortho plus 80
        ortho_sensitivity = {356.6403: 0.1040, 360.7890: 0.2099, 364.2056: 0.3072, 367.8662: 0.4068, 371.7709: 0.5015,
                             375.9196: 0.5944, 380.8004: 0.6859, 387.1455: 0.7813, 395.9309: 0.8758, 406.1807: 0.9422,
                             416.4304: 0.9727, 426.6802: 0.9797, 436.9299: 0.9758, 447.1796: 0.9728, 457.4294: 0.9702,
                             467.6791: 0.9593, 477.9288: 0.9366, 488.1786: 0.8977, 498.4283: 0.8412, 508.6781: 0.8114,
                             518.9278: 0.8470, 529.1775: 0.9255, 539.4273: 0.9766, 549.1889: 0.9458, 555.7780: 0.8428,
                             559.0726: 0.7268, 560.9029: 0.6284, 562.3672: 0.5268, 563.0993: 0.4335, 563.6920: 0.3232,
                             564.4008: 0.2031}
        ortho_sensitivity = xp.asarray(colour.SpectralDistribution(ortho_sensitivity).align(spectral_shape,
                                                                                            extrapolator_kwargs={
                                                                                                'method': 'linear'}).align(
            spectral_shape).values)
        ortho_sensitivity = 10 ** ortho_sensitivity

        separation_neg = Kodak5222Dev9()
        sensitivity = separation_neg.sensitivity[:, 0]
        self.sensitivity = xp.stack([sensitivity * WRATTEN["32"] * WRATTEN["25"], sensitivity * WRATTEN["58"],
                                     ortho_sensitivity * WRATTEN["32"]]).T

        # spectral dye density
        red_sd = {414.9694: 0.1901, 431.0856: 0.1655, 448.5015: 0.1454, 465.0510: 0.1310, 494.2508: 0.1129,
                  511.4067: 0.1194, 528.0428: 0.1609, 544.9388: 0.2413, 559.4954: 0.3350, 575.8716: 0.4743,
                  582.3700: 0.5387, 590.9480: 0.6395, 607.3242: 0.7847, 624.2202: 0.9174, 639.5566: 1.0523,
                  655.6728: 1.0025, 671.5291: 0.6444, 688.7717: 0.3226, 704.0214: 0.1384, 719.6177: 0.0609}
        green_sd = {401.4526: 0.2299, 415.4893: 0.2310, 431.8654: 0.2299, 448.7615: 0.2684, 464.8777: 0.3274,
                    479.6942: 0.4203, 494.5107: 0.5355, 512.0133: 0.6519, 527.7829: 0.7170, 544.7655: 0.7077,
                    559.7554: 0.6517, 575.8716: 0.5468, 591.9878: 0.3904, 607.8440: 0.2632, 624.7401: 0.1993,
                    639.5566: 0.1682, 671.7890: 0.0847, 687.9052: 0.0414, 704.0214: 0.0172, 715.1988: 0.0089}
        blue_sd = {400.3058: 0.9943, 414.7095: 1.0864, 431.3456: 1.0853, 448.2416: 0.9716, 465.7875: 0.8061,
                   479.1743: 0.6362, 494.7706: 0.4412, 511.9266: 0.2852, 527.7829: 0.2120, 543.8991: 0.1812,
                   559.7554: 0.1625, 576.9980: 0.1297, 591.9878: 0.0918, 607.8440: 0.0690, 624.4801: 0.0707,
                   640.0765: 0.0907, 655.3262: 0.0912, 671.5291: 0.0541, 687: 0.028, 703.9755: 0.0019}
        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]
        self.spectral_density = xp.stack(
            [xp.asarray(self.gaussian_extrapolation(x).values) for x in self.spectral_density]).T

        # sensiometry curve from kodak matrix film 4150
        curve = {-0.8901: 0.0337, -0.8083: 0.0450, -0.7327: 0.0582, -0.6203: 0.0795, -0.5184: 0.1129, -0.4309: 0.1500,
                 -0.3650: 0.1855, -0.2970: 0.2272, -0.2282: 0.2761, -0.1474: 0.3376, -0.0246: 0.4576, 0.0716: 0.5770,
                 0.1933: 0.7584, 0.3935: 1.1285, 0.5697: 1.4631, 0.6975: 1.7012, 0.8412: 1.9662, 0.9197: 2.0985,
                 1.0265: 2.2539, 1.1249: 2.3620, 1.2107: 2.4380, 1.2977: 2.4977, 1.3779: 2.5387, 1.4553: 2.5663,
                 1.5962: 2.6006, 1.7229: 2.6213, 1.8457: 2.6357, 1.9573: 2.6437}
        log_exposure_matrix = xp.array(list(curve.keys()), dtype=default_dtype)
        density_curve_matrix = xp.array(list(curve.values()), dtype=default_dtype)
        log_H_ref_mat = xp.interp(xp.asarray(self.lad[0]), density_curve_matrix, log_exposure_matrix)
        separation_curve = separation_neg.density_curve[0]
        separation_exposure = separation_neg.log_exposure[0]
        slope = (separation_curve[-1] - separation_curve[-2]) / (separation_exposure[-1] - separation_exposure[-2])
        separation_curve = xp.append(separation_curve, separation_curve[-1] + slope * 1)
        separation_exposure = xp.append(separation_exposure, separation_exposure[-1] + 1)
        density_curve = xp.interp(log_H_ref_mat - separation_curve + separation_neg.d_ref, log_exposure_matrix,
                                  density_curve_matrix)

        self.log_exposure = [separation_exposure] * 3
        self.density_curve = [density_curve] * 3

        self.calibrate()

TechnicolorIV()