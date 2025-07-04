from spectral_film_lut.film_spectral import *


class FujiSuperiaXtra400(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.iso = 400
        self.density_measure = 'status_m'

        # spectral sensitivity
        red_log_sensitivity = {565.5787: 1.0349, 567.9504: 1.1481, 570.3221: 1.2672, 572.6938: 1.3842, 575.0655: 1.4891,
                               577.7336: 1.5895, 579.8089: 1.6669, 586.3310: 1.8816, 591.3708: 1.9871, 597.3001: 2.0842,
                               605.3045: 2.1787, 614.7913: 2.2353, 624.2781: 2.2597, 633.7648: 2.2736, 643.2516: 2.2738,
                               650.9596: 2.2174, 655.1100: 2.1147, 657.4817: 2.0112, 659.5570: 1.9017, 661.3357: 1.8024,
                               663.1145: 1.6932, 664.8933: 1.5850, 666.6720: 1.4698, 668.4508: 1.3516, 670.2296: 1.2315,
                               672.0083: 1.1133, 674.0836: 1.0014, 676.1588: 0.9237, }
        green_log_sensitivity = {464.1889: 0.7922, 466.5606: 0.9046, 468.9323: 1.0066, 471.6004: 1.1070,
                                 474.5651: 1.2095, 477.8261: 1.3089, 481.3837: 1.4112, 485.2377: 1.5098,
                                 488.2023: 1.5790, 501.2466: 1.8103, 509.5475: 1.9071, 518.1449: 1.9921,
                                 524.9635: 2.0930, 530.2998: 2.1913, 535.9326: 2.2895, 542.4547: 2.3872,
                                 550.7556: 2.4736, 560.2424: 2.4959, 567.6539: 2.4319, 565.2822: 1.0051,
                                 571.8044: 2.3270, 574.1761: 2.2204, 575.9549: 2.1171, 577.7336: 2.0149,
                                 579.8089: 1.8935, 585.1452: 1.5413, 587.2204: 1.4162, 588.9992: 1.3020,
                                 590.7779: 1.1868, 592.5567: 1.0726, 594.3355: 0.9604, 596.7072: 0.8508,
                                 598.7824: 0.7951, }
        blue_log_sensitivity = {401.9320: 2.0167, 404.3037: 2.1258, 407.5648: 2.2276, 414.3834: 2.3321,
                                423.8702: 2.3809, 433.3569: 2.3893, 442.8437: 2.3917, 452.3305: 2.4133,
                                460.9278: 2.4809, 469.5252: 2.5726, 477.5297: 2.5410, 482.2731: 2.4361,
                                484.9412: 2.3336, 487.0164: 2.2323, 488.7952: 2.1251, 490.5740: 2.0000,
                                492.2045: 1.8674, 495.3174: 1.5694, 497.0961: 1.4509, 498.8749: 1.3367,
                                500.9501: 1.2263, 503.3218: 1.1205, 505.6935: 1.0349, }
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry - characteristic curve
        red_curve = {-3.5450: 0.1397, -3.3811: 0.1518, -3.2171: 0.1744, -3.0531: 0.2059, -2.8892: 0.2469,
                     -2.7252: 0.2980, -2.5612: 0.3550, -2.3973: 0.4233, -2.2333: 0.5101, -2.0693: 0.6041,
                     -1.9054: 0.7011, -1.7414: 0.7995, -1.5774: 0.9027, -1.4135: 1.0039, -1.2495: 1.1004,
                     -1.0855: 1.2007, -0.9216: 1.3004, -0.7576: 1.4009, -0.5936: 1.5014, -0.4297: 1.6023,
                     -0.2657: 1.7035, -0.1018: 1.8053, 0.0622: 1.9077, 0.2262: 2.0106, 0.3199: 2.0693, }
        green_curve = {-3.5450: 0.4542, -3.3811: 0.4613, -3.2171: 0.4725, -3.0531: 0.4930, -2.8892: 0.5264,
                       -2.7252: 0.5799, -2.5612: 0.6546, -2.3973: 0.7508, -2.2333: 0.8547, -2.0693: 0.9611,
                       -1.9054: 1.0683, -1.7414: 1.1757, -1.5770: 1.2843, -1.4135: 1.3927, -1.2495: 1.5016,
                       -1.0855: 1.6101, -0.9216: 1.7199, -0.7576: 1.8299, -0.5936: 1.9401, -0.4297: 2.0505,
                       -0.2657: 2.1623, -0.1018: 2.2751, 0.0622: 2.3877, 0.2184: 2.4969, }
        blue_curve = {-3.5528: 0.7270, -3.3889: 0.7339, -3.2249: 0.7424, -3.0609: 0.7555, -2.8970: 0.7786,
                      -2.7330: 0.8222, -2.5690: 0.8944, -2.4051: 0.9926, -2.2411: 1.1026, -2.0771: 1.2180,
                      -1.9132: 1.3353, -1.7492: 1.4546, -1.5852: 1.5723, -1.4213: 1.6879, -1.2573: 1.8021,
                      -1.0934: 1.9159, -0.9294: 2.0287, -0.7654: 2.1415, -0.6015: 2.2535, -0.4375: 2.3657,
                      -0.2735: 2.4791, -0.1096: 2.5926, 0.0544: 2.7071, 0.1910: 2.8040, }
        red_log_exposure = xp.array(list(red_curve.keys()), dtype=default_dtype)
        red_density_curve = xp.array(list(red_curve.values()), dtype=default_dtype)
        green_log_exposure = xp.array(list(green_curve.keys()), dtype=default_dtype)
        green_density_curve = xp.array(list(green_curve.values()), dtype=default_dtype)
        blue_log_exposure = xp.array(list(blue_curve.keys()), dtype=default_dtype)
        blue_density_curve = xp.array(list(blue_curve.values()), dtype=default_dtype)
        self.log_exposure = [red_log_exposure, green_log_exposure, blue_log_exposure]
        self.density_curve = [red_density_curve, green_density_curve, blue_density_curve]

        # spectral dye density
        midscale_sd = {405.7563: 1.5294, 415.5171: 1.5944, 422.7681: 1.7090, 428.6246: 1.8181, 436.9910: 1.9126,
                       447.0308: 1.9539, 457.0705: 1.9364, 466.5525: 1.8580, 474.6400: 1.7514, 482.1698: 1.6456,
                       491.0940: 1.5528, 501.1338: 1.5230, 511.1735: 1.5612, 521.2132: 1.6262, 531.2530: 1.6860,
                       541.2927: 1.7246, 551.3324: 1.7286, 560.8144: 1.6573, 568.9020: 1.5515, 575.8740: 1.4439,
                       582.5672: 1.3347, 589.2603: 1.2257, 597.6268: 1.1197, 607.6665: 1.0912, 617.7062: 1.1617,
                       627.7460: 1.2629, 637.7857: 1.3663, 647.8254: 1.4687, 657.8652: 1.5636, 667.9049: 1.6467,
                       677.9446: 1.7132, 687.9844: 1.7571, 698.0241: 1.7706, 708.0638: 1.7537, 715.5937: 1.7227, }
        minimum_sd = {404.6408: 0.8787, 414.6805: 0.8440, 424.7202: 0.8553, 434.7600: 0.8710, 444.7997: 0.8659,
                      454.8394: 0.8448, 464.8792: 0.8170, 474.9189: 0.7850, 484.9586: 0.7515, 494.9984: 0.7178,
                      505.0381: 0.6888, 515.0778: 0.6670, 525.1176: 0.6451, 535.1573: 0.6198, 545.1970: 0.5933,
                      555.2368: 0.5648, 565.2765: 0.5297, 575.3162: 0.4638, 585.3560: 0.3877, 595.3957: 0.3158,
                      605.4355: 0.2630, 615.4752: 0.2441, 625.5149: 0.2386, 635.5547: 0.2464, 645.5944: 0.2583,
                      655.6341: 0.2721, 665.6739: 0.2813, 675.7136: 0.2740, 685.7533: 0.2608, 695.7931: 0.2460,
                      705.8328: 0.2311, 714.7570: 0.2157, }
        self.d_ref_sd = colour.SpectralDistribution(midscale_sd)
        self.d_min_sd = colour.SpectralDistribution(minimum_sd)

        self.mtf = [{2.8527: 1.1931, 3.5525: 1.1931, 4.4240: 1.1931, 5.5092: 1.1885, 6.8607: 1.1803, 8.5438: 1.1642,
                     10.6397: 1.1408, 13.2497: 1.1032, 16.5000: 1.0480, 20.5477: 0.9763, 25.5883: 0.8867,
                     31.8655: 0.7876, 39.6825: 0.6856, 49.4172: 0.5810, 60.5830: 0.4792, 72.3571: 0.3868,
                     84.1916: 0.3108, 95.9361: 0.2520, }]

        # copied from kodak 5207
        red_rms = {0.0161: 0.0049, 0.1713: 0.0050, 0.3747: 0.0051, 0.5514: 0.0050, 0.7388: 0.0054, 0.9261: 0.0055,
                   1.0546: 0.0062, 1.1563: 0.0074, 1.2580: 0.0085, 1.3651: 0.0096, 1.4775: 0.0100, 1.5685: 0.0099,
                   1.7238: 0.0088, 1.8308: 0.0084, 1.9165: 0.0082, 2.0664: 0.0077, 2.2698: 0.0070, 2.3769: 0.0066,
                   2.5589: 0.0061, 2.9443: 0.0056, 3.3191: 0.0053, 3.4850: 0.0054, 3.6724: 0.0050, 3.8330: 0.0048,
                   4.0739: 0.0046, 4.2666: 0.0045, 4.4968: 0.0046, 4.6306: 0.0043, 4.8233: 0.0042, 5.0000: 0.0043}
        green_rms = {0.0000: 0.0051, 0.1392: 0.0056, 0.2730: 0.0059, 0.4711: 0.0057, 0.7334: 0.0055, 0.9315: 0.0063,
                     1.1510: 0.0079, 1.2794: 0.0089, 1.4186: 0.0103, 1.5418: 0.0107, 1.6435: 0.0103, 1.8201: 0.0092,
                     1.9486: 0.0080, 2.1734: 0.0074, 2.3448: 0.0068, 2.5321: 0.0062, 2.7730: 0.0063, 3.0835: 0.0065,
                     3.3779: 0.0069, 3.5278: 0.0071, 3.7420: 0.0066, 4.0739: 0.0061, 4.3148: 0.0058, 4.5343: 0.0053,
                     4.7430: 0.0051, 5.0054: 0.0046}
        blue_rms = {0.0000: 0.0112, 0.0857: 0.0114, 0.1606: 0.0123, 0.2409: 0.0126, 0.3158: 0.0123, 0.3961: 0.0118,
                    0.5139: 0.0115, 0.6531: 0.0116, 0.8030: 0.0125, 0.9315: 0.0140, 1.0493: 0.0154, 1.1938: 0.0165,
                    1.3169: 0.0164, 1.4507: 0.0156, 1.5578: 0.0155, 1.6863: 0.0157, 1.7934: 0.0153, 1.9245: 0.0140,
                    2.0610: 0.0129, 2.2645: 0.0124, 2.4732: 0.0127, 2.6552: 0.0128, 2.8747: 0.0135, 3.0246: 0.0138,
                    3.2976: 0.0133, 3.6403: 0.0121, 4.1649: 0.0101, 4.4700: 0.0088, 4.8073: 0.0069, 5.0054: 0.0060}
        red_rms_density = {0.0000: 0.2286, 0.7369: 0.2286, 1.0380: 0.2429, 1.2520: 0.2714, 1.4659: 0.3429,
                           1.7829: 0.4976, 3.7837: 1.5857, 4.0967: 1.7381, 4.4176: 1.8762, 4.7385: 1.9714,
                           5.0000: 2.0238}
        green_rms_density = {0.0079: 0.5952, 0.5784: 0.5952, 0.9509: 0.6048, 1.1648: 0.6333, 1.3312: 0.6714,
                             1.5610: 0.7762, 1.9810: 1.0524, 2.9160: 1.6429, 3.8629: 2.2500, 4.1125: 2.4000,
                             4.3502: 2.5190, 4.5761: 2.6071, 4.7425: 2.6524, 4.8732: 2.6810, 4.9960: 2.7000}
        blue_rms_density = {0.0000: 1.0071, 0.6418: 1.0119, 0.9350: 1.0310, 1.1648: 1.0810, 1.4342: 1.1762,
                            1.7868: 1.3452, 2.2464: 1.6452, 2.9279: 2.0452, 3.8788: 2.6119, 4.2353: 2.7881,
                            4.5840: 2.9190, 4.8732: 2.9929}
        self.rms_curve = [red_rms, green_rms, blue_rms]
        self.rms_density = [red_rms_density, green_rms_density, blue_rms_density]
        self.rms = 4

        self.calibrate()
