from spectral_film_lut.film_spectral import *


class KodakUltramax400(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.iso = 400
        self.density_measure = 'status_m'

        # spectral sensitivity
        red_log_sensitivity = {487.6141: 0.2352, 494.1950: 0.3520, 504.2095: 0.4907, 515.1043: 0.6582, 524.5245: 0.7858,
                               536.5419: 0.8244, 545.8697: 0.8644, 555.9986: 1.0746, 564.5824: 1.2233, 571.1634: 1.3240,
                               574.3108: 1.4600, 577.4582: 1.5967, 580.6056: 1.7204, 588.4741: 1.9717, 595.9611: 2.0859,
                               609.2184: 2.2313, 621.2357: 2.3445, 631.2502: 2.4563, 639.8340: 2.5931, 649.9916: 2.6855,
                               659.2907: 2.6474, 663.8688: 2.5053, 667.0162: 2.3567, 669.8774: 2.2140, 673.0248: 2.0889,
                               678.4613: 1.8388, 681.6087: 1.7060, 685.0422: 1.5795, 688.1896: 1.4583, 690.7075: 1.3221,
                               692.1954: 1.2015, 693.9122: 1.0713, 695.6289: 0.8975, 697.3457: 0.7457, 699.0625: 0.5990,
                               700.0925: 0.4685, }
        green_log_sensitivity = {390.6168: 0.8417, 394.0503: 0.9675, 401.9188: 1.1163, 413.5070: 1.0869,
                                 423.5875: 1.0197, 437.5417: 0.9105, 450.4175: 0.8939, 461.5764: 0.9461,
                                 470.6689: 0.9938, 477.0989: 1.1452, 479.6025: 1.2811, 481.6054: 1.4252,
                                 483.4039: 1.5607, 485.8973: 1.7181, 492.4782: 1.9378, 499.9176: 2.0863,
                                 510.5043: 2.2105, 520.2326: 2.3350, 529.3887: 2.4676, 538.5448: 2.5894,
                                 549.6157: 2.6678, 560.2905: 2.6114, 570.0189: 2.4591, 576.5998: 2.3513,
                                 580.0333: 2.2067, 586.3281: 1.7044, 587.9632: 1.5391, 589.0940: 1.3877,
                                 590.2385: 1.2582, 591.4784: 1.1169, 593.1135: 0.9198, 594.3397: 0.7794,
                                 595.8112: 0.6310, 597.3441: 0.4687, }
        blue_log_sensitivity = {366.8682: 1.2945, 371.4462: 1.4383, 374.8797: 1.5754, 378.5994: 1.7069,
                                382.8913: 1.8328, 391.2526: 2.1797, 393.4780: 2.3212, 396.3393: 2.4399,
                                412.3625: 2.6647, 424.3798: 2.6414, 436.3972: 2.6484, 445.6677: 2.6238,
                                460.4319: 2.5963, 472.8382: 2.6894, 479.1447: 2.4317, 481.3601: 2.2880,
                                483.5674: 2.1565, 494.9580: 1.5021, 505.9262: 0.7958, 515.3685: 0.3951,
                                518.8020: 0.2554, 521.9494: 0.1401, }
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry - characteristic curve
        red_curve = {-3.3735: 0.2856, -3.2430: 0.2856, -3.1124: 0.2886, -2.9818: 0.3016, -2.8512: 0.3256,
                     -2.7206: 0.3610, -2.5900: 0.4073, -2.4595: 0.4548, -2.3289: 0.5119, -2.1983: 0.5733,
                     -2.0677: 0.6378, -1.9371: 0.7046, -1.8065: 0.7700, -1.6759: 0.8343, -1.5454: 0.9008,
                     -1.4148: 0.9683, -1.2842: 1.0380, -1.1536: 1.1053, -1.0230: 1.1738, -0.8924: 1.2414,
                     -0.7619: 1.3089, -0.6313: 1.3768, -0.5007: 1.4456, -0.3701: 1.5140, -0.2395: 1.5807,
                     -0.1089: 1.6456, 0.0216: 1.7084, 0.1522: 1.7693, 0.2828: 1.8300, 0.4134: 1.8820, 0.5191: 1.9163, }
        green_curve = {-3.3735: 0.6928, -3.2430: 0.6936, -3.1124: 0.6993, -2.9818: 0.7154, -2.8512: 0.7452,
                       -2.7206: 0.7865, -2.5900: 0.8382, -2.4595: 0.8973, -2.3289: 0.9628, -2.1983: 1.0342,
                       -2.0677: 1.1063, -1.9371: 1.1794, -1.8065: 1.2513, -1.6759: 1.3201, -1.5454: 1.3890,
                       -1.4148: 1.4582, -1.2842: 1.5281, -1.1536: 1.5987, -1.0230: 1.6695, -0.8924: 1.7402,
                       -0.7619: 1.8111, -0.6313: 1.8821, -0.5007: 1.9539, -0.3701: 2.0243, -0.2395: 2.0927,
                       -0.1089: 2.1561, 0.0216: 2.2171, 0.1522: 2.2757, 0.2828: 2.3328, 0.4134: 2.3852,
                       0.5191: 2.4226, }
        blue_curve = {-3.3735: 0.9782, -3.2430: 0.9791, -3.1124: 0.9846, -2.9818: 1.0003, -2.8512: 1.0305,
                      -2.7206: 1.0725, -2.5900: 1.1257, -2.4595: 1.1875, -2.3289: 1.2558, -2.1983: 1.3276,
                      -2.0677: 1.4023, -1.9371: 1.4786, -1.8065: 1.5543, -1.6759: 1.6295, -1.5454: 1.7069,
                      -1.4148: 1.7854, -1.2842: 1.8644, -1.1536: 1.9445, -1.0230: 2.0244, -0.8924: 2.1031,
                      -0.7619: 2.1815, -0.6313: 2.2607, -0.5007: 2.3397, -0.3701: 2.4173, -0.2395: 2.4953,
                      -0.1089: 2.5765, 0.0216: 2.6577, 0.1522: 2.7378, 0.2828: 2.8172, 0.4134: 2.8923, 0.5222: 2.9473, }
        red_log_exposure = xp.array(list(red_curve.keys()), dtype=default_dtype)
        red_density_curve = xp.array(list(red_curve.values()), dtype=default_dtype)
        green_log_exposure = xp.array(list(green_curve.keys()), dtype=default_dtype)
        green_density_curve = xp.array(list(green_curve.values()), dtype=default_dtype)
        blue_log_exposure = xp.array(list(blue_curve.keys()), dtype=default_dtype)
        blue_density_curve = xp.array(list(blue_curve.values()), dtype=default_dtype)
        self.log_exposure = [red_log_exposure, green_log_exposure, blue_log_exposure]
        self.density_curve = [red_density_curve, green_density_curve, blue_density_curve]

        # spectral dye density
        midscale_sd = {408.0789: 1.3713, 412.8083: 1.4453, 416.4999: 1.5107, 420.1575: 1.5803, 424.9209: 1.6557,
                       429.7524: 1.7201, 436.0554: 1.7675, 444.2658: 1.7872, 452.4763: 1.7702, 460.1555: 1.7316,
                       466.7853: 1.6721, 472.3376: 1.6036, 478.5908: 1.5248, 484.5095: 1.4574, 491.9381: 1.3996,
                       500.0737: 1.3894, 508.3590: 1.4127, 516.1618: 1.4402, 523.5168: 1.4769, 531.8443: 1.5046,
                       540.0548: 1.5226, 548.0079: 1.4929, 554.0430: 1.4289, 558.7213: 1.3599, 563.0254: 1.2790,
                       566.7681: 1.2064, 569.9493: 1.1395, 572.9435: 1.0748, 576.3119: 1.0048, 580.0545: 0.9314,
                       583.7972: 0.8656, 588.4221: 0.7914, 594.7913: 0.7253, 603.2357: 0.6859, 611.4462: 0.6937,
                       619.7598: 0.7301, 627.5490: 0.7807, 635.3213: 0.8403, 644.1822: 0.9144, 651.4675: 0.9740,
                       659.0872: 1.0369, 666.7722: 1.0913, 674.9972: 1.1416, 683.2820: 1.1797, 691.6983: 1.1986,
                       697.3872: 1.1971, }
        minimum_sd = {408.4532: 0.8116, 416.1256: 0.8707, 423.7981: 0.9290, 431.6577: 0.9645, 439.5173: 0.9762,
                      447.3769: 0.9636, 455.2365: 0.9316, 463.0961: 0.8872, 470.9557: 0.8328, 478.8153: 0.7749,
                      486.6749: 0.7201, 494.5345: 0.6765, 502.3942: 0.6559, 510.2538: 0.6444, 518.1134: 0.6259,
                      525.9730: 0.6076, 533.8326: 0.5998, 541.6922: 0.5946, 549.5518: 0.5876, 557.4114: 0.5736,
                      565.2710: 0.5471, 573.1306: 0.4961, 580.0545: 0.4321, 585.8557: 0.3673, 592.2182: 0.3003,
                      599.7035: 0.2445, 607.5631: 0.2130, 615.4227: 0.2015, 623.2823: 0.1990, 631.1981: 0.2045,
                      639.0016: 0.2135, 646.8612: 0.2238, 654.7208: 0.2350, 662.5804: 0.2460, 670.4400: 0.2563,
                      678.2996: 0.2652, 686.1592: 0.2710, 691.9603: 0.2739, }
        self.d_ref_sd = colour.SpectralDistribution(midscale_sd)
        self.d_min_sd = colour.SpectralDistribution(minimum_sd)

        red_mtf = {2.5343: 1.0256, 3.1724: 1.0063, 4.2554: 0.9912, 5.4826: 0.9875, 6.8238: 1.0140, 8.4443: 1.0191,
                   9.6959: 0.9887, 12.3489: 0.9962, 14.7626: 1.0025, 17.7294: 0.9353, 21.9650: 0.8070, 29.6336: 0.5999,
                   38.4000: 0.4516, 45.9052: 0.3531, 50.9191: 0.2978, 57.7967: 0.2659, 67.7534: 0.2105, 80.2530: 0.1615}
        green_mtf = {2.5343: 1.0584, 3.0647: 1.0438, 4.0405: 1.0386, 5.7081: 1.0692, 7.4824: 1.1288, 9.7519: 1.1489,
                     14.3435: 1.1708, 19.6885: 1.1203, 26.5623: 0.9813, 34.6188: 0.8434, 47.2463: 0.7113,
                     53.3198: 0.6430, 65.4523: 0.5111, 79.5169: 0.4047}
        blue_mtf = {2.5489: 1.0191, 3.8808: 0.9875, 4.9713: 0.9862, 6.2952: 1.0000, 8.0641: 1.0127, 9.8082: 1.0544,
                    12.4920: 1.0412, 17.9554: 0.9690, 24.5048: 0.8595, 29.4635: 0.7529, 37.5255: 0.6512,
                    50.0469: 0.5562, 59.4851: 0.4750, 79.3339: 0.3645}
        self.mtf = [red_mtf, green_mtf, blue_mtf]

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
        # guessed value:
        self.rms = 4.6

        self.calibrate()
