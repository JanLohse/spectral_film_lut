from spectral_film_lut.film_spectral import *


class FujiPro400H(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.iso = 400
        self.density_measure = 'status_m'

        # spectral sensitivity
        red_log_sensitivity = {555.7648: -0.7374, 560.9472: -0.5653, 566.2461: -0.3681, 569.0994: -0.2548,
                               574.4565: 0.0120, 578.6491: 0.2218, 584.2391: 0.4676, 589.9457: 0.6984, 594.6040: 0.8573,
                               598.2143: 0.9562, 600.7764: 1.0060, 604.9107: 1.0492, 609.2780: 1.0629, 614.2275: 1.0492,
                               618.0124: 1.0426, 622.9620: 1.0510, 627.7368: 1.0588, 633.1522: 1.0474, 636.8207: 1.0150,
                               639.9651: 0.9544, 642.1196: 0.8933, 645.3222: 0.7404, 650.7958: 0.4167, 654.0567: 0.2098,
                               657.0264: -0.0060, 660.6366: -0.2728, 663.7228: -0.5162}
        green_log_sensitivity = {451.1840: -0.7704, 461.6654: -0.6235, 468.0707: -0.5276, 473.0202: -0.4317,
                                 477.5621: -0.2668, 484.7244: 0.0390, 488.2764: 0.2188, 492.7601: 0.4077,
                                 497.0109: 0.5965, 500.7376: 0.7194, 506.1530: 0.8393, 517.3331: 1.0312,
                                 526.9410: 1.1781, 535.9084: 1.2692, 545.1669: 1.3285, 552.6203: 1.3513,
                                 560.2484: 1.3106, 565.3727: 1.2152, 569.2158: 1.0941, 573.4084: 0.8933,
                                 577.4845: 0.6295, 581.7352: 0.3267, 584.7632: 0.0839, 588.9557: -0.2938,
                                 591.7508: -0.5306, 594.0800: -0.7182}
        blue_log_sensitivity = {384.5109: -0.7374, 391.1491: -0.1559, 396.5062: 0.4317, 400.1165: 0.7854,
                                403.5520: 1.0372, 405.8812: 1.1781, 408.3851: 1.2530, 411.5295: 1.3010,
                                416.5373: 1.3399, 421.7780: 1.3489, 431.0365: 1.3375, 438.0241: 1.3213,
                                444.4876: 1.3070, 450.3106: 1.3082, 456.1335: 1.3357, 460.9666: 1.3999,
                                464.2275: 1.4598, 466.9061: 1.5048, 469.2352: 1.5228, 471.5644: 1.5078,
                                474.8835: 1.4341, 476.4557: 1.3699, 481.0559: 1.0552, 483.7927: 0.8513,
                                488.1017: 0.5336, 492.5272: 0.2758, 497.5932: 0.0330, 505.4542: -0.2878,
                                509.8214: -0.4556, 515.0039: -0.6223}
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry - characteristic curve
        red_curve = {-3.5869: 0.1759, -3.4304: 0.1735, -3.2557: 0.1753, -3.0563: 0.1771, -2.8804: 0.1829,
                     -2.6957: 0.1958, -2.5491: 0.2087, -2.4553: 0.2222, -2.2706: 0.2761, -2.0642: 0.3658,
                     -1.8748: 0.4631, -1.2005: 0.8618, -0.9067: 1.0400, -0.4323: 1.3338, -0.0723: 1.5419,
                     0.1746: 1.6826, 0.3851: 1.7951, 0.5692: 1.8872, 0.6712: 1.9358}
        green_curve = {-3.5811: 0.6918, -3.2644: 0.6859, -3.0064: 0.6859, -2.7719: 0.6918, -2.6658: 0.7000,
                       -2.5497: 0.7141, -2.4324: 0.7422, -2.2841: 0.7932, -2.0337: 0.9099, -1.6264: 1.1267,
                       -1.1437: 1.4116, -0.6758: 1.6754, -0.2888: 1.8872, 0.0004: 2.0420, 0.2425: 2.1658,
                       0.4543: 2.2720, 0.6632: 2.3677}
        blue_curve = {-3.5811: 0.9515, -3.1589: 0.9466, -2.8845: 0.9578, -2.7262: 0.9698, -2.5608: 0.9902,
                      -2.4609: 1.0155, -2.3638: 1.0479, -2.1801: 1.1232, -1.7115: 1.3687, -1.3773: 1.5488,
                      -0.9952: 1.7528, -0.5766: 1.9737, -0.0664: 2.2389, 0.2713: 2.4113, 0.4894: 2.5211, 0.6710: 2.6083}
        red_log_exposure = xp.array(list(red_curve.keys()), dtype=default_dtype)
        red_density_curve = xp.array(list(red_curve.values()), dtype=default_dtype)
        green_log_exposure = xp.array(list(green_curve.keys()), dtype=default_dtype)
        green_density_curve = xp.array(list(green_curve.values()), dtype=default_dtype)
        blue_log_exposure = xp.array(list(blue_curve.keys()), dtype=default_dtype)
        blue_density_curve = xp.array(list(blue_curve.values()), dtype=default_dtype)
        self.log_exposure = [red_log_exposure, green_log_exposure, blue_log_exposure]
        self.density_curve = [red_density_curve, green_density_curve, blue_density_curve]

        # spectral dye density
        midscale_sd = {400.6356: 1.9806, 402.7542: 1.8826, 404.8729: 1.7869, 406.6737: 1.7237, 408.7394: 1.6693,
                       411.4936: 1.6410, 416.6314: 1.6619, 423.1462: 1.7411, 431.0911: 1.8369, 438.7712: 1.8900,
                       447.7754: 1.9109, 457.9979: 1.8905, 466.3136: 1.8352, 474.3644: 1.7455, 482.8390: 1.6463,
                       493.3263: 1.5562, 500.2119: 1.5170, 507.0975: 1.5065, 516.6314: 1.5279, 525.3178: 1.5653,
                       533.6864: 1.5879, 540.9958: 1.5914, 549.5763: 1.5570, 560.0636: 1.4582, 573.0932: 1.2832,
                       583.6864: 1.1013, 591.8432: 0.9402, 600.7415: 0.8044, 605.3496: 0.7618, 610.3814: 0.7387,
                       617.7966: 0.7409, 628.3898: 0.7900, 657.7860: 0.9859, 672.4576: 1.0656, 687.1822: 1.1013,
                       699.8941: 1.1056, 709.8517: 1.0821, 717.2669: 1.0503}
        minimum_sd = {400.1059: 1.4386, 402.0127: 1.3450, 404.6081: 1.2319, 406.8326: 1.1457, 408.5805: 1.0969,
                      412.2352: 1.0595, 417.4258: 1.0303, 427.3835: 1.0334, 435.0636: 1.0338, 445.8686: 1.0160,
                      477.4364: 0.9098, 499.1525: 0.8244, 523.5169: 0.7644, 543.6441: 0.7117, 549.2055: 0.6921,
                      561.3877: 0.6725, 568.0085: 0.6573, 572.9873: 0.6320, 579.8199: 0.5607, 594.1208: 0.3656,
                      600.2119: 0.2956, 608.8983: 0.2272, 617.0021: 0.2041, 626.8008: 0.1985, 646.3983: 0.1994,
                      670.7627: 0.1880, 715.7309: 0.1602}
        self.d_ref_sd = colour.SpectralDistribution(midscale_sd)
        self.d_min_sd = colour.SpectralDistribution(minimum_sd)

        self.mtf = [{2.3879: 1.1535, 4.3448: 1.1468, 6.7160: 1.1259, 10.2450: 1.0666, 17.6993: 0.9057, 25.7326: 0.7490,
                     39.4027: 0.5470, 53.2758: 0.3995, 70.6880: 0.2764, 88.0507: 0.1936, 103.9405: 0.1372}]

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
