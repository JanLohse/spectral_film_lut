from spectral_film_lut.film_spectral import *


class KodakEktachromeE100(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.iso = 100
        self.density_measure = "status_a"
        self.projection_kelvin = 6000

        # spectral sensitivity
        red_log_sensitivity = {554.0856: -1.0220, 564.3580: -0.9702, 573.3463: -0.6981, 588.6381: 0.4380,
                               599.3774: 0.6926, 631.5953: 1.1752, 641.6342: 1.3438, 647.1206: 1.4066, 652.1401: 1.3945,
                               656.4591: 1.2226, 661.2451: 0.7135, 669.5331: 0.0083, 684.5914: -0.9967}
        green_log_sensitivity = {472.8405: -0.5647, 483.1128: -0.0413, 493.8521: 0.3939, 500.0389: 0.5295,
                                 530.1556: 0.9934, 543.4630: 1.1873, 548.8327: 1.1961, 555.2529: 1.1510,
                                 564.5914: 1.2083, 568.3268: 1.1829, 573.4630: 1.2182, 576.9650: 1.1598,
                                 581.8677: 0.6915, 590.5058: -0.0028, 600.3113: -0.6088, 604.5136: -1.0022}
        blue_log_sensitivity = {394.7471: 0.5262, 398.0156: 0.8127, 404.4358: 1.0937, 410.9728: 1.3008,
                                419.6109: 1.4375, 427.0817: 1.5047, 430.3502: 1.4848, 434.4358: 1.3747,
                                439.1051: 1.3339, 446.1089: 1.3328, 456.3813: 1.3736, 467.9377: 1.2479,
                                474.9416: 0.7631, 481.7121: 0.3609, 487.1984: 0.1515, 498.4047: -0.1041,
                                505.2918: -0.2782, 514.0467: -0.5372, 519.6498: -0.7686}
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry
        red_curve = {-2.8175: 3.2439, -2.5863: 3.2304, -2.4000: 3.1746, -2.2233: 3.0688, -2.0533: 2.8937,
                     -1.8870: 2.6330, -1.6692: 2.2222, -1.4488: 1.7605, -1.3185: 1.4946, -1.1439: 1.1784,
                     -0.9768: 0.9689, -0.7688: 0.7039, -0.5909: 0.5172, -0.3712: 0.3347, -0.2383: 0.2539,
                     -0.0905: 0.1799, 0.0048: 0.1487}
        green_curve = {-2.8221: 3.6043, -2.6442: 3.5917, -2.4562: 3.5345, -2.3393: 3.4672, -2.2307: 3.3789,
                       -2.1446: 3.2805, -2.0511: 3.1602, -1.9149: 2.9011, -1.8197: 2.7144, -1.7345: 2.5125,
                       -1.6058: 2.2139, -1.5006: 1.9657, -1.3485: 1.6503, -1.2199: 1.3727, -1.1088: 1.1775,
                       -0.9526: 0.9302, -0.8473: 0.7973, -0.7228: 0.6451, -0.5783: 0.5046, -0.4179: 0.3590,
                       -0.2433: 0.2531, -0.0921: 0.1765, 0.0031: 0.1437}
        blue_curve = {-2.8204: 3.7591, -2.6567: 3.7322, -2.5030: 3.6767, -2.4061: 3.6186, -2.2917: 3.5135,
                      -2.1973: 3.3999, -2.0745: 3.2023, -1.9132: 2.8658, -1.8080: 2.6101, -1.7111: 2.3796,
                      -1.5966: 2.0911, -1.5022: 1.8732, -1.3995: 1.6503, -1.2784: 1.4021, -1.1422: 1.1649,
                      -0.9860: 0.9226, -0.8256: 0.7182, -0.7078: 0.5828, -0.5975: 0.4701, -0.4722: 0.3658,
                      -0.3411: 0.2766, -0.2342: 0.2228, -0.1005: 0.1782, 0.0064: 0.1521}
        red_log_exposure = xp.array(list(red_curve.keys()), dtype=default_dtype)
        red_density_curve = xp.array(list(red_curve.values()), dtype=default_dtype)
        green_log_exposure = xp.array(list(green_curve.keys()), dtype=default_dtype)
        green_density_curve = xp.array(list(green_curve.values()), dtype=default_dtype)
        blue_log_exposure = xp.array(list(blue_curve.keys()), dtype=default_dtype)
        blue_density_curve = xp.array(list(blue_curve.values()), dtype=default_dtype)
        self.log_exposure = [red_log_exposure, green_log_exposure, blue_log_exposure]
        self.density_curve = [red_density_curve, green_density_curve, blue_density_curve]

        # spectral dye density
        red_sd = {400.3102: 0.1455, 418.3040: 0.0948, 439.4002: 0.0578, 459.8759: 0.0422, 481.2823: 0.0358,
                  507.9628: 0.0582, 534.9535: 0.1194, 549.8449: 0.1813, 563.8056: 0.2593, 579.7518: 0.3799,
                  597.3113: 0.5560, 612.0786: 0.7056, 624.6122: 0.8097, 638.2627: 0.8955, 650.2999: 0.9369,
                  662.3992: 0.9410, 675.8014: 0.9030, 685.3568: 0.8507, 699.9380: 0.7448}
        green_sd = {399.6898: 0.0731, 420.4757: 0.1119, 428.9142: 0.1351, 439.5243: 0.1429, 443.7435: 0.1313,
                    455.5946: 0.1381, 468.0662: 0.1840, 481.5305: 0.2668, 499.5243: 0.4451, 514.6019: 0.6530,
                    528.4385: 0.8172, 534.9535: 0.8780, 540.8480: 0.9142, 547.4871: 0.9313, 555.2430: 0.9220,
                    562.6887: 0.8806, 569.5140: 0.8127, 580.3102: 0.6716, 591.1686: 0.4884, 599.1727: 0.3638,
                    610.8997: 0.2366, 621.5098: 0.1530, 636.7115: 0.0843, 655.6360: 0.0388, 675.4912: 0.0172,
                    699.6898: 0.0090}
        blue_sd = {400.0000: 0.5056, 418.5522: 0.7466, 425.8118: 0.8250, 434.3744: 0.8877, 442.5026: 0.9213,
                   451.8097: 0.9045, 456.4633: 0.8862, 460.1861: 0.8582, 472.8438: 0.6993, 488.4178: 0.4739,
                   501.5719: 0.2910, 511.8097: 0.1817, 525.0879: 0.0881, 539.7311: 0.0299, 562.4405: 0.0019,
                   699.5657: 0.0067}
        midscale_sd = {400.0000: 0.5056, 418.5522: 0.7466, 425.8118: 0.8250, 434.3744: 0.8877, 442.5026: 0.9213,
                       451.8097: 0.9045, 456.4633: 0.8862, 460.1861: 0.8582, 472.8438: 0.6993, 488.4178: 0.4739,
                       501.5719: 0.2910, 511.8097: 0.1817, 525.0879: 0.0881, 539.7311: 0.0299, 562.4405: 0.0019,
                       699.5657: 0.0067}

        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]
        self.d_ref_sd = colour.SpectralDistribution(midscale_sd)

        red_mtf = {2.4954: 0.9471, 7.1631: 0.8595, 9.9589: 0.7953, 13.1782: 0.7204, 15.1843: 0.6930, 16.8730: 0.6930,
                   19.7971: 0.6666, 24.3246: 0.5830, 32.4541: 0.4151, 46.4797: 0.2411, 57.3925: 0.1684, 67.3387: 0.1230,
                   77.7175: 0.0829}
        green_mtf = {2.4954: 0.9471, 9.0215: 0.8884, 11.9378: 0.8397, 14.3334: 0.8141, 18.1416: 0.7969, 22.8484: 0.7317,
                     29.8385: 0.6014, 41.7591: 0.4299, 54.5347: 0.3226, 66.2383: 0.2416, 78.6190: 0.1816}
        blue_mtf = {2.4954: 0.9563, 7.9074: 0.9657, 11.6463: 0.9164, 14.0528: 0.8562, 18.2015: 0.8612, 20.6977: 0.8300,
                    24.1649: 0.7547, 32.4007: 0.6653, 53.2911: 0.4258, 78.6190: 0.2866}
        self.mtf = [red_mtf, green_mtf, blue_mtf]

        red_rms = {-2.9756: 0.0307, -2.1453: 0.0279, -1.8942: 0.0311, -1.6395: 0.0298, -1.3500: 0.0238, -1.0779: 0.0169,
                   -0.8372: 0.0117, -0.6244: 0.0079, -0.4326: 0.0044, -0.2477: 0.0028, -0.1151: 0.0022, -0.0056: 0.0021}
        green_rms = {-2.5221: 0.0330, -2.1070: 0.0336, -1.7198: 0.0292, -1.4114: 0.0202, -1.0709: 0.0116,
                     -0.8233: 0.0074, -0.5895: 0.0041, -0.3837: 0.0026, -0.1744: 0.0019, -0.0042: 0.0017}
        blue_rms = {-2.0505: 0.0532, -1.9263: 0.0585, -1.8495: 0.0600, -1.7581: 0.0585, -1.5907: 0.0430,
                    -1.3465: 0.0300, -1.1233: 0.0219, -0.9767: 0.0162, -0.8372: 0.0112, -0.6872: 0.0071,
                    -0.5965: 0.0051, -0.4940: 0.0038, -0.3558: 0.0029, -0.2058: 0.0023, -0.0907: 0.0020, 0.0014: 0.0018}
        red_rms_density = {-2.9491: 3.2977, -2.7506: 3.2900, -2.5555: 3.2266, -2.3883: 3.1281, -2.2141: 2.9803,
                           -2.0058: 2.7100, -1.8630: 2.4538, -1.6811: 2.0336, -1.4652: 1.5739, -1.2123: 1.1667,
                           -1.0242: 0.9040, -0.7887: 0.6370, -0.6013: 0.4575, -0.4445: 0.3262, -0.2536: 0.2025,
                           -0.1170: 0.1401, -0.0063: 0.1182}
        green_rms_density = {-2.4998: 3.5549, -2.4127: 3.5002, -2.3117: 3.3962, -2.1514: 3.1499, -1.9633: 2.7559,
                             -1.6658: 2.0883, -1.4018: 1.5202, -1.1503: 1.0704, -0.8486: 0.6436, -0.5978: 0.3918,
                             -0.3504: 0.2047, -0.1414: 0.0974, -0.0028: 0.0591}
        blue_rms_density = {-2.0532: 2.8501, -1.8414: 2.2908, -1.5836: 1.7129, -1.3370: 1.2477, -1.0750: 0.8351,
                            -0.8723: 0.5735, -0.6953: 0.3864, -0.4814: 0.2080, -0.2703: 0.0919, -0.1101: 0.0394,
                            0.0000: 0.0296}
        self.rms_curve = [red_rms, green_rms, blue_rms]
        self.rms_density = [red_rms_density, green_rms_density, blue_rms_density]

        self.calibrate()