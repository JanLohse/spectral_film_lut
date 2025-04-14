from spectral_film_lut.film_spectral import *


class FujiInstaxColor(FilmSpectral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.iso = 800
        self.density_measure = "status_a"
        self.exposure_kelvin = 5500
        self.projection_kelvin = 6500

        # spectral sensitivity
        red_log_sensitivity = {575.0794: -0.1667, 579.2737: 0.0402, 586.5731: 0.3226, 594.1176: 0.5849,
                               605.2887: 0.9429, 617.7090: 1.2453, 630.0803: 1.4517, 635.9988: 1.5322, 640.0989: 1.5548,
                               643.9370: 1.5647, 648.5133: 1.5190, 652.2972: 1.4228, 657.0477: 1.2180, 660.7671: 0.9956,
                               665.6480: 0.5459, 672.5241: 0.0001, 677.4540: -0.3537}
        green_log_sensitivity = {488.6896: -0.2027, 502.9967: 0.2915, 494.9410: 0.0293, 512.3326: 0.5587,
                                 525.5246: 0.8712, 539.7169: 1.1406, 544.6055: 1.2060, 551.2906: 1.2865,
                                 558.2479: 1.3998, 563.6631: 1.4955, 566.2287: 1.5156, 569.2925: 1.5104,
                                 572.3356: 1.4647, 574.3154: 1.3384, 578.0167: 1.0806, 584.8902: 0.5298,
                                 593.7073: -0.2182}
        blue_log_sensitivity = {383.3905: -0.2378, 394.8275: 0.6404, 399.8116: 0.8927, 406.8012: 1.0691,
                                411.6873: 1.1294, 417.8381: 1.1645, 425.5053: 1.1666, 432.9040: 1.1435,
                                437.7591: 1.1433, 442.6271: 1.1683, 456.7354: 1.2736, 461.3350: 1.2734,
                                464.3961: 1.2631, 469.7739: 1.2856, 474.6457: 1.3182, 477.9651: 1.3129,
                                483.8126: 1.2546, 490.3389: 1.0244, 495.0468: 0.7363, 499.1999: 0.3624,
                                501.6055: 0.0693, 505.2913: -0.2187, 509.7566: -0.4816}
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry
        red_log_exposure = np.array(
            [-3.6357, -3.4299, -3.2372, -3.0392, -2.8703, -2.7383, -2.6301, -2.5377, -2.4427, -2.3529, -2.2130, -2.0969,
             -1.8937, -1.8092, -1.7221, -1.6112, -1.5136, -1.4133, -1.2179, -0.7956])
        red_density_curve = np.array(
            [2.3146, 2.3082, 2.2915, 2.2477, 2.1859, 2.0675, 1.9181, 1.7199, 1.4727, 1.2461, 0.9372, 0.7286, 0.4583,
             0.3862, 0.3231, 0.2639, 0.2317, 0.2150, 0.2111, 0.2124])
        green_log_exposure = np.array(
            [-3.6331, -3.3824, -3.2187, -3.0498, -2.9323, -2.8413, -2.7541, -2.6670, -2.5879, -2.4572, -2.3595, -2.2619,
             -2.1589, -2.0692, -1.9702, -1.8778, -1.7947, -1.7089, -1.6192, -1.5004, -1.3922, -0.7996])
        green_density_curve = np.array(
            [2.3455, 2.3378, 2.3249, 2.2850, 2.2309, 2.1666, 2.0855, 1.9799, 1.8357, 1.5191, 1.2616, 1.0350, 0.8419,
             0.6810, 0.5433, 0.4428, 0.3746, 0.3167, 0.2703, 0.2266, 0.2137, 0.2098])
        blue_log_exposure = np.array(
            [-3.6146, -3.3969, -3.1606, -2.9917, -2.8597, -2.7238, -2.6261, -2.5390, -2.4691, -2.3820, -2.3147, -2.2210,
             -2.1312, -2.0164, -1.8778, -1.7564, -1.6561, -1.5373, -1.4291, -1.2417, -0.8009])
        blue_density_curve = np.array(
            [2.0881, 2.0881, 2.0752, 2.0430, 1.9902, 1.9014, 1.8113, 1.6954, 1.5525, 1.3182, 1.1547, 0.9578, 0.7904,
             0.6038, 0.4441, 0.3476, 0.2858, 0.2356, 0.2163, 0.2124, 0.2111])
        self.log_exposure = [red_log_exposure, green_log_exposure, blue_log_exposure]
        self.density_curve = [red_density_curve, green_density_curve, blue_density_curve]

        self.exposure_base = 10

        # spectral dye density
        red_sd = {480.8194: 0.1784, 495.1898: 0.1853, 513.8533: 0.2131, 528.6684: 0.2560, 548.2313: 0.3454,
                  565.8516: 0.4615, 583.4296: 0.6218, 595.0717: 0.7252, 616.4947: 0.8739, 629.1242: 0.9471,
                  636.0408: 0.9807, 639.6251: 0.9912, 643.4534: 0.9969, 648.0173: 0.9853, 652.3561: 0.9585,
                  657.9297: 0.8933, 662.7912: 0.8212, 671.3205: 0.6723, 681.3231: 0.4862, 688.3791: 0.3745,
                  694.6760: 0.3047, 701.4455: 0.2418, 710.8530: 0.1767}
        green_sd = {440.9068: 0.3206, 457.8584: 0.3844, 471.2090: 0.4553, 488.3334: 0.5889, 504.9517: 0.7503,
                    524.9069: 0.9304, 530.8651: 0.9641, 537.3114: 0.9885, 545.6891: 0.9977, 552.8866: 0.9883,
                    558.4234: 0.9616, 568.8073: 0.8778, 576.5575: 0.7917, 584.0704: 0.7032, 588.9385: 0.6241,
                    595.0315: 0.5171, 602.5801: 0.3915, 609.8556: 0.3007, 616.3788: 0.2448, 624.5681: 0.2006,
                    630.3390: 0.1796}
        blue_sd = {385.8034: 0.3117, 392.6654: 0.4024, 405.3886: 0.6279, 412.1904: 0.7813, 418.3425: 0.8626,
                   427.3800: 0.9335, 438.1160: 0.9823, 451.0398: 0.9984, 466.3921: 0.9809, 476.2544: 0.9413,
                   485.4312: 0.8668, 493.9137: 0.7667, 507.7748: 0.5550, 520.6373: 0.3851, 527.1738: 0.3153,
                   534.6577: 0.2571, 546.9240: 0.2093, 558.4513: 0.1825}
        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]

        self.calibrate()
