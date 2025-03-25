from spectral_film_lut.film_spectral import *


class KodakPortra400(FilmSpectral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.iso = 400
        # self.log_H_ref = 1.44
        self.density_measure = 'status_m'
        self.exposure_kelvin = 5500
        self.projection_kelvin = None

        # spectral sensitivity
        self.red_log_sensitivity = {489.8305: 0.3816, 510.5932: 0.6779, 518.6441: 0.7632, 525.4237: 0.8081,
                                    529.6610: 0.8171, 534.7458: 0.8036, 538.9831: 0.7856, 544.9153: 0.7767,
                                    550.4237: 0.8036, 561.0169: 1.0415, 569.4915: 1.2750, 577.5424: 1.5937,
                                    582.2034: 1.7778, 588.9831: 1.9843, 594.4915: 2.0875, 603.3898: 2.2088,
                                    611.8644: 2.3075, 619.4915: 2.3704, 629.6610: 2.4018, 634.7458: 2.3973,
                                    640.6780: 2.4198, 648.3051: 2.5859, 650.4237: 2.6083, 655.0847: 2.5903,
                                    658.8983: 2.5365, 662.2881: 2.3614, 666.9492: 1.9933, 676.6949: 1.0415,
                                    679.6610: 0.7048}
        self.green_log_sensitivity = {390.2542: 1.2391, 396.1864: 1.4366, 399.5763: 1.5264, 403.8136: 1.5578,
                                      411.8644: 1.5219, 417.7966: 1.4725, 432.2034: 1.3737, 442.3729: 1.3244,
                                      452.5424: 1.3199, 461.0169: 1.3423, 467.7966: 1.3603, 473.3051: 1.4456,
                                      486.4407: 1.7957, 493.6441: 1.9439, 502.5424: 2.0606, 516.5254: 2.1728,
                                      530.0847: 2.3075, 540.6780: 2.4512, 544.9153: 2.5006, 550.0000: 2.5365,
                                      557.6271: 2.4871, 566.5254: 2.3973, 571.1864: 2.3389, 576.6949: 2.2716,
                                      579.6610: 2.1998, 583.8983: 1.9933, 591.9492: 1.4097, 597.8814: 0.8530,
                                      600.0000: 0.6644}
        self.blue_log_sensitivity = {380.0847: 1.7329, 393.2203: 2.3255, 398.3051: 2.5140, 402.5424: 2.5859,
                                     406.3559: 2.5993, 411.8644: 2.5948, 416.9492: 2.5634, 421.1864: 2.5499,
                                     427.9661: 2.5589, 436.8644: 2.5589, 443.2203: 2.5365, 447.8814: 2.5095,
                                     454.2373: 2.5095, 461.4407: 2.5365, 465.6780: 2.5769, 468.6441: 2.5859,
                                     471.6102: 2.5589, 481.7797: 2.2088, 486.0169: 1.9125, 496.6102: 1.3378,
                                     508.0508: 0.8530, 519.9153: 0.2783}

        # sensiometry - characteristic curve
        self.red_log_exposure = np.array(
            [-3.4479, -3.0878, -2.8889, -2.7791, -2.6626, -2.5048, -1.9835, -0.9616, -0.1043, 0.5645])
        self.red_density_curve = np.array(
            [0.2225, 0.2335, 0.2418, 0.2555, 0.2775, 0.3379, 0.6016, 1.1484, 1.6319, 2.0192])
        self.green_log_exposure = np.array(
            [-3.4376, -3.1907, -2.9060, -2.7723, -2.6317, -2.3642, -1.7641, -0.8073, 0.0604, 0.5610])
        self.green_density_curve = np.array(
            [0.6538, 0.6538, 0.6676, 0.6841, 0.7253, 0.8489, 1.1868, 1.7170, 2.1951, 2.4643])
        self.blue_log_exposure = np.array(
            [-3.4273, -3.1735, -3.0226, -2.8992, -2.7620, -2.5151, -1.8258, -1.0233, -0.3443, 0.5610])
        self.blue_density_curve = np.array(
            [0.8709, 0.8791, 0.8874, 0.9148, 0.9588, 1.0907, 1.5220, 2.0302, 2.4670, 3.0549, ])

        self.exposure_base = 10

        # spectral dye density
        midscale_spectral_density = {400.0000: 1.3267, 401.6349: 1.3199, 404.9046: 1.3370, 414.7139: 1.4973,
                                     426.5668: 1.7326, 431.4714: 1.8076, 436.3760: 1.8520, 441.2807: 1.8776,
                                     447.8202: 1.8895, 454.3597: 1.8861, 461.7166: 1.8554, 468.4605: 1.8042,
                                     477.0436: 1.7156, 485.2180: 1.6337, 488.6921: 1.6030, 492.7793: 1.5859,
                                     498.9101: 1.5859, 504.2234: 1.6098, 509.1281: 1.6405, 513.2153: 1.6473,
                                     520.5722: 1.6371, 526.7030: 1.6201, 530.3815: 1.6149, 536.1035: 1.6218,
                                     542.2343: 1.6201, 547.1390: 1.6030, 551.6349: 1.5655, 558.9918: 1.4632,
                                     565.9401: 1.3199, 575.7493: 1.0914, 586.7847: 0.8799, 590.4632: 0.8254,
                                     596.5940: 0.7674, 602.7248: 0.7384, 610.0817: 0.7384, 622.1390: 0.7930,
                                     642.3706: 0.9447, 666.8937: 1.1460, 675.8856: 1.2040, 684.0599: 1.2449,
                                     689.7820: 1.2653, 696.7302: 1.2705, 700.0000: 1.2705}
        minimum_spectral_density = {400.0000: 0.6924, 404.4959: 0.6685, 411.0354: 0.6753, 419.6185: 0.7367,
                                    429.8365: 0.7930, 438.0109: 0.8151, 445.3678: 0.8186, 453.1335: 0.8117,
                                    464.9864: 0.7913, 480.1090: 0.7640, 488.2834: 0.7452, 497.6839: 0.7538,
                                    501.9755: 0.7640, 506.6757: 0.7844, 509.5368: 0.7879, 513.0109: 0.7793,
                                    518.5286: 0.7503, 523.2289: 0.7128, 527.9292: 0.6736, 531.1989: 0.6514,
                                    537.3297: 0.6310, 547.5477: 0.6071, 557.3569: 0.5866, 566.3488: 0.5457,
                                    575.3406: 0.4741, 585.3542: 0.3649, 592.0981: 0.2984, 600.0681: 0.2370,
                                    611.3079: 0.2012, 632.9700: 0.1859, 646.0490: 0.1893, 662.8065: 0.2012,
                                    676.2943: 0.2115, 697.5477: 0.2183, 700.0000: 0.2183}

        self.red_mtf = {2.4722: 1.0533, 5.4540: 1.0779, 10.0285: 1.0594, 15.3692: 0.9942, 20.3136: 0.8962,
                        28.5837: 0.6715, 43.5573: 0.4562, 48.8095: 0.4065, 54.8510: 0.3370, 59.2323: 0.2942,
                        68.6808: 0.2489, 79.6366: 0.2142}
        self.green_mtf = {2.4863: 1.0533, 2.8502: 1.0594, 5.0076: 1.0841, 7.1270: 1.1223, 9.4198: 1.1619,
                          10.7988: 1.1754, 15.8129: 1.1585, 20.7815: 1.0999, 26.3192: 0.9885, 30.0863: 0.8808,
                          35.4864: 0.7893, 51.3753: 0.6018, 60.5965: 0.4946, 80.0912: 0.3859}
        self.blue_mtf = {2.5076: 1.0564, 5.3922: 1.0810, 8.4542: 1.0936, 10.2595: 1.1159, 16.1771: 1.0873,
                         22.6339: 1.0058, 33.8102: 0.8148, 55.3213: 0.6053, 80.0912: 0.4615}

        # TODO: replace with accurate data
        self.red_rms = {0.0161: 0.0049, 0.1713: 0.0050, 0.3747: 0.0051, 0.5514: 0.0050, 0.7388: 0.0054, 0.9261: 0.0055,
                        1.0546: 0.0062, 1.1563: 0.0074, 1.2580: 0.0085, 1.3651: 0.0096, 1.4775: 0.0100, 1.5685: 0.0099,
                        1.7238: 0.0088, 1.8308: 0.0084, 1.9165: 0.0082, 2.0664: 0.0077, 2.2698: 0.0070, 2.3769: 0.0066,
                        2.5589: 0.0061, 2.9443: 0.0056, 3.3191: 0.0053, 3.4850: 0.0054, 3.6724: 0.0050, 3.8330: 0.0048,
                        4.0739: 0.0046, 4.2666: 0.0045, 4.4968: 0.0046, 4.6306: 0.0043, 4.8233: 0.0042, 5.0000: 0.0043}
        self.green_rms = {0.0000: 0.0051, 0.1392: 0.0056, 0.2730: 0.0059, 0.4711: 0.0057, 0.7334: 0.0055,
                          0.9315: 0.0063, 1.1510: 0.0079, 1.2794: 0.0089, 1.4186: 0.0103, 1.5418: 0.0107,
                          1.6435: 0.0103, 1.8201: 0.0092, 1.9486: 0.0080, 2.1734: 0.0074, 2.3448: 0.0068,
                          2.5321: 0.0062, 2.7730: 0.0063, 3.0835: 0.0065, 3.3779: 0.0069, 3.5278: 0.0071,
                          3.7420: 0.0066, 4.0739: 0.0061, 4.3148: 0.0058, 4.5343: 0.0053, 4.7430: 0.0051,
                          5.0054: 0.0046}
        self.blue_rms = {0.0000: 0.0112, 0.0857: 0.0114, 0.1606: 0.0123, 0.2409: 0.0126, 0.3158: 0.0123, 0.3961: 0.0118,
                         0.5139: 0.0115, 0.6531: 0.0116, 0.8030: 0.0125, 0.9315: 0.0140, 1.0493: 0.0154, 1.1938: 0.0165,
                         1.3169: 0.0164, 1.4507: 0.0156, 1.5578: 0.0155, 1.6863: 0.0157, 1.7934: 0.0153, 1.9245: 0.0140,
                         2.0610: 0.0129, 2.2645: 0.0124, 2.4732: 0.0127, 2.6552: 0.0128, 2.8747: 0.0135, 3.0246: 0.0138,
                         3.2976: 0.0133, 3.6403: 0.0121, 4.1649: 0.0101, 4.4700: 0.0088, 4.8073: 0.0069, 5.0054: 0.0060}
        self.red_rms_density = {0.0000: 0.2286, 0.7369: 0.2286, 1.0380: 0.2429, 1.2520: 0.2714, 1.4659: 0.3429,
                                1.7829: 0.4976, 3.7837: 1.5857, 4.0967: 1.7381, 4.4176: 1.8762, 4.7385: 1.9714,
                                5.0000: 2.0238}
        self.green_rms_density = {0.0079: 0.5952, 0.5784: 0.5952, 0.9509: 0.6048, 1.1648: 0.6333, 1.3312: 0.6714,
                                  1.5610: 0.7762, 1.9810: 1.0524, 2.9160: 1.6429, 3.8629: 2.2500, 4.1125: 2.4000,
                                  4.3502: 2.5190, 4.5761: 2.6071, 4.7425: 2.6524, 4.8732: 2.6810, 4.9960: 2.7000}
        self.blue_rms_density = {0.0000: 1.0071, 0.6418: 1.0119, 0.9350: 1.0310, 1.1648: 1.0810, 1.4342: 1.1762,
                                 1.7868: 1.3452, 2.2464: 1.6452, 2.9279: 2.0452, 3.8788: 2.6119, 4.2353: 2.7881,
                                 4.5840: 2.9190, 4.8732: 2.9929}

        self.d_ref_sd = colour.SpectralDistribution(midscale_spectral_density)
        self.d_min_sd = colour.SpectralDistribution(minimum_spectral_density)

        self.calibrate()
