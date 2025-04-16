from spectral_film_lut.film_spectral import *


class FujiCrystalArchiveSuperTypeC(FilmSpectral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lad = [0.8, 0.8, 0.8]
        self.density_measure = 'status_a'
        self.exposure_kelvin = None
        self.projection_kelvin = 6500

        # spectral sensitivity
        red_log_sensitivity = {590.1800: -0.5984, 608.9201: -0.4395, 623.6898: -0.2960, 634.6480: -0.2153,
                               642.9063: -0.1960, 652.2763: -0.1798, 660.2170: -0.1444, 668.3166: -0.0637,
                               681.7364: 0.1210, 692.7740: 0.2927, 699.1265: 0.3702, 703.7321: 0.4040, 707.7025: 0.4169,
                               713.4992: 0.3815, 717.2313: 0.3089, 721.5987: 0.2040, 725.2515: 0.0887}
        green_log_sensitivity = {437.3213: -0.1306, 446.3737: -0.0355, 458.2054: 0.1492, 467.2578: 0.3073,
                                 472.6575: 0.3694, 480.9158: 0.4347, 484.6480: 0.4895, 493.7004: 0.7081,
                                 502.1175: 0.8927, 506.1673: 0.9565, 510.3759: 0.9895, 516.6490: 0.9798,
                                 522.4457: 0.9710, 527.9248: 0.9935, 535.3097: 1.1556, 546.5855: 1.5992,
                                 548.5707: 1.6581, 551.0323: 1.6919, 554.2880: 1.6669, 556.6702: 1.5887,
                                 558.9730: 1.3815, 563.1816: 0.8089, 567.4696: 0.2847, 572.5516: -0.1589,
                                 577.4749: -0.4331, 583.6686: -0.6298}
        blue_log_sensitivity = {376.5749: 0.2040, 385.3097: 0.6718, 394.1239: 0.9460, 401.3499: 1.0855,
                                406.9878: 1.1411, 411.1170: 1.1573, 415.0873: 1.1355, 419.0577: 1.1073,
                                423.4251: 1.1153, 429.6983: 1.2161, 441.2917: 1.4016, 457.0143: 1.6395,
                                463.0492: 1.7887, 470.6723: 2.0065, 475.9926: 2.1315, 479.1689: 2.1726,
                                482.8216: 2.1500, 487.6654: 1.9621, 493.3827: 1.5145, 497.5913: 1.0589,
                                502.9910: 0.5024, 507.1996: 0.1677, 510.6935: 0.0065, 513.5521: -0.1484,
                                516.7284: -0.2839, 521.1752: -0.4040, 527.5278: -0.4952}
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry
        red_curve = {-0.2482: 0.0895, -0.0158: 0.1000, 0.1548: 0.1281, 0.3465: 0.1857, 0.4813: 0.2488, 0.6000: 0.3254,
                     0.7685: 0.4868, 0.9342: 0.7493, 1.0360: 1.0013, 1.1112: 1.2884, 1.1828: 1.6274, 1.2340: 1.8710,
                     1.3113: 2.1427, 1.4131: 2.3869, 1.5458: 2.5772, 1.6568: 2.6803, 1.7642: 2.7484, 1.8906: 2.8011,
                     2.0092: 2.8334, 2.1237: 2.8502, 2.1995: 2.8558, 2.2761: 2.8600}
        green_curve = {-0.2496: 0.1155, -0.0959: 0.1211, 0.0509: 0.1351, 0.1815: 0.1597, 0.3282: 0.2025, 0.4167: 0.2383,
                       0.4912: 0.2748, 0.5649: 0.3218, 0.6267: 0.3661, 0.7418: 0.4664, 0.8422: 0.5830, 0.9349: 0.7479,
                       1.0388: 1.0041, 1.1098: 1.2638, 1.1659: 1.5165, 1.2060: 1.7061, 1.2467: 1.8745, 1.2979: 2.0465,
                       1.3492: 2.1834, 1.4222: 2.3364, 1.5444: 2.4978, 1.6462: 2.5926, 1.7635: 2.6649, 1.8976: 2.7168,
                       2.0275: 2.7498, 2.1785: 2.7702, 2.2810: 2.7737}
        blue_curve = {-0.2489: 0.1176, -0.0650: 0.1225, 0.1036: 0.1443, 0.2370: 0.1737, 0.3634: 0.2166, 0.5213: 0.2910,
                      0.6512: 0.3794, 0.7903: 0.5156, 0.8837: 0.6391, 0.9813: 0.8392, 1.0571: 1.0498, 1.1048: 1.2358,
                      1.1561: 1.4358, 1.2165: 1.6639, 1.2699: 1.8359, 1.3309: 1.9889, 1.3906: 2.0977, 1.4637: 2.2016,
                      1.5683: 2.3104, 1.6568: 2.3785, 1.7803: 2.4417, 1.9004: 2.4859, 2.0219: 2.5161, 2.1714: 2.5364,
                      2.3224: 2.5442}
        red_log_exposure = xp.array(list(red_curve.keys()), dtype=default_dtype)
        red_density_curve = xp.array(list(red_curve.values()), dtype=default_dtype)
        green_log_exposure = xp.array(list(green_curve.keys()), dtype=default_dtype)
        green_density_curve = xp.array(list(green_curve.values()), dtype=default_dtype)
        blue_log_exposure = xp.array(list(blue_curve.keys()), dtype=default_dtype)
        blue_density_curve = xp.array(list(blue_curve.values()), dtype=default_dtype)
        self.log_exposure = [red_log_exposure, green_log_exposure, blue_log_exposure]
        self.density_curve = [red_density_curve, green_density_curve, blue_density_curve]

        self.exposure_base = 10

        # spectral dye density
        red_sd = {377.5238: 0.0540, 412.3810: 0.0454, 440.9524: 0.0422, 473.9683: 0.0383, 507.3016: 0.0438,
                  527.1746: 0.0693, 541.0794: 0.1080, 551.8730: 0.1543, 559.6825: 0.2061, 573.5238: 0.3233,
                  587.5556: 0.5073, 595.8095: 0.6409, 605.3968: 0.7812, 613.3968: 0.8799, 620.5079: 0.9374,
                  625.1429: 0.9693, 632.3810: 0.9927, 641.2698: 0.9984, 652.0635: 0.9815, 657.2698: 0.9457,
                  666.9206: 0.8626, 675.6825: 0.7636, 684.6984: 0.6486, 694.5397: 0.5367, 705.8413: 0.4281,
                  714.3492: 0.3722, 720.7619: 0.3393}
        green_sd = {377.9048: 0.0246, 402.6032: 0.0358, 423.6825: 0.0470, 442.1587: 0.0690, 458.3492: 0.1042,
                    470.9206: 0.1591, 478.2222: 0.2019, 485.9048: 0.2604, 496.1270: 0.3661, 506.4127: 0.5077,
                    516.0635: 0.6623, 522.2857: 0.7581, 527.4921: 0.8361, 533.5238: 0.9128, 538.7302: 0.9633,
                    542.5397: 0.9872, 547.1746: 0.9990, 550.6032: 0.9952, 554.2222: 0.9728, 558.1587: 0.9281,
                    565.6508: 0.7955, 573.4603: 0.6102, 580.3175: 0.4537, 588.6349: 0.3051, 598.4127: 0.1812,
                    603.8730: 0.1361, 610.0952: 0.0939, 619.0476: 0.0645, 630.1587: 0.0406, 658.0952: 0.0166,
                    689.2063: 0.0160, 723.4921: 0.0137}
        blue_sd = {378.6032: 0.0942, 389.7778: 0.2093, 398.8571: 0.3466, 407.4286: 0.5080, 414.9206: 0.6789,
                   420.3810: 0.7939, 427.3016: 0.8914, 433.0159: 0.9489, 439.0476: 0.9840, 446.3492: 0.9994,
                   452.8889: 0.9914, 459.0476: 0.9585, 466.6667: 0.8770, 475.5556: 0.7173, 481.8413: 0.5831,
                   490.6667: 0.3962, 501.7143: 0.2252, 509.2063: 0.1454, 515.6190: 0.1010, 526.9841: 0.0565,
                   540.0000: 0.0335, 558.4127: 0.0246, 590.1587: 0.0195, 652.0635: 0.0176, 723.3016: 0.0157}
        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]

        self.calibrate()
