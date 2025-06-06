from spectral_film_lut.film_spectral import *


class KodakEXR5248(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.iso = 100
        self.density_measure = 'status_m'
        self.exposure_kelvin = 3200

        # spectral sensitivity
        red_log_sensitivity = {417.0772: -0.9838, 418.2266: -0.9140, 420.5255: -0.8599, 424.9589: -0.8931,
                               430.3777: -1.0083, 435.6322: -1.1148, 440.3941: -1.1707, 448.6043: -1.1934,
                               454.8440: -1.2196, 460.0985: -1.3051, 463.7110: -1.4064, 466.9951: -1.5111,
                               469.9507: -1.5635, 473.5632: -1.5216, 477.8325: -1.4308, 483.2512: -1.3383,
                               489.9836: -1.2684, 499.5074: -1.1899, 507.7176: -1.0432, 514.1215: -0.9158,
                               522.4959: -0.8127, 531.0345: -0.7569, 541.7077: -0.7167, 551.3957: -0.5805,
                               561.0837: -0.2959, 568.8013: -0.0061, 575.8621: 0.2802, 583.4154: 0.4863,
                               592.7750: 0.6381, 601.4778: 0.7708, 608.0460: 0.8512, 613.4647: 0.8459, 619.7044: 0.8232,
                               625.9442: 0.8669, 631.5271: 0.9612, 636.6174: 1.0450, 643.0213: 1.0904, 648.7685: 1.0816,
                               654.5156: 1.0083, 659.1133: 0.8704, 668.1445: 0.3466, 675.8621: -0.2296,
                               680.4598: -0.5613, 684.7291: -0.8651, 686.8637: -0.9856}
        green_log_sensitivity = {400.3284: 0.0602, 405.2545: 0.0969, 411.4943: 0.0812, 417.5698: -0.0044,
                                 423.9737: -0.1021, 432.5123: -0.1598, 441.3793: -0.1632, 448.1117: -0.1615,
                                 453.8588: -0.1283, 459.1133: -0.1004, 464.6962: -0.1353, 469.2939: -0.1667,
                                 473.2348: -0.1353, 477.9967: 0.0358, 483.5796: 0.2278, 490.6404: 0.3920,
                                 500.1642: 0.5351, 507.5534: 0.6783, 517.5698: 0.9192, 527.7504: 1.1846,
                                 537.7668: 1.3645, 544.0066: 1.4553, 549.0969: 1.5094, 555.5008: 1.4972,
                                 562.3974: 1.3872, 567.8161: 1.2161, 571.9212: 0.9262, 574.8768: 0.5701,
                                 579.9672: 0.0218, 587.0279: -0.4810, 592.2824: -0.7813, 596.0591: -0.9926}
        blue_log_sensitivity = {401.3136: 1.2230, 406.2397: 1.3208, 411.8227: 1.3505, 418.7192: 1.3243,
                                423.3169: 1.3243, 431.5271: 1.4011, 441.3793: 1.5007, 449.5895: 1.5827,
                                456.4860: 1.6910, 461.7406: 1.8132, 465.3530: 1.9005, 468.8013: 1.9616,
                                474.2200: 1.9459, 479.3103: 1.7992, 485.2217: 1.4727, 491.9540: 0.8808,
                                497.8654: 0.3605, 503.6125: -0.0550, 512.3153: -0.5020, 520.3612: -0.8407,
                                524.4663: -0.9873}
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry - characteristic curve
        red_curve = {-2.8094: 0.1970, -2.6799: 0.1981, -2.5794: 0.2037, -2.4903: 0.2172, -2.3728: 0.2449,
                     -2.2590: 0.2869, -2.0486: 0.3868, -1.6940: 0.5652, -1.2826: 0.7752, -0.8750: 0.9861,
                     -0.5281: 1.1732, -0.3442: 1.2731, -0.1376: 1.3880, 0.0008: 1.4515, 0.0918: 1.4879}
        green_curve = {-2.8116: 0.5882, -2.6173: 0.6016, -2.4619: 0.6191, -2.3728: 0.6373, -2.2893: 0.6666,
                       -2.0618: 0.7856, -1.6732: 1.0091, -1.2713: 1.2453, -0.7708: 1.5434, -0.4636: 1.7257,
                       -0.2779: 1.8367, -0.0674: 1.9469, 0.0871: 2.0175}
        blue_curve = {-2.8126: 1.0376, -2.6420: 1.0400, -2.4998: 1.0543, -2.4012: 1.0725, -2.3310: 1.0931,
                      -2.1623: 1.1756, -1.8381: 1.3571, -1.5689: 1.5125, -1.2201: 1.7139, -0.7357: 1.9961,
                      -0.4399: 2.1705, -0.1878: 2.3211, -0.0181: 2.4154, 0.0861: 2.4678}
        red_log_exposure = xp.array(list(red_curve.keys()), dtype=default_dtype)
        red_density_curve = xp.array(list(red_curve.values()), dtype=default_dtype)
        green_log_exposure = xp.array(list(green_curve.keys()), dtype=default_dtype)
        green_density_curve = xp.array(list(green_curve.values()), dtype=default_dtype)
        blue_log_exposure = xp.array(list(blue_curve.keys()), dtype=default_dtype)
        blue_density_curve = xp.array(list(blue_curve.values()), dtype=default_dtype)
        self.log_exposure = [red_log_exposure, green_log_exposure, blue_log_exposure]
        self.density_curve = [red_density_curve, green_density_curve, blue_density_curve]

        # spectral dye density
        red_sd = {350.8584: 0.3467, 361.5880: 0.3081, 385.6223: 0.2865, 407.0815: 0.2009, 434.9785: 0.1002,
                  482.6180: 0.0324, 522.9614: 0.0324, 559.0129: 0.1378, 579.1845: 0.2413, 604.7210: 0.4709,
                  635.1931: 0.7419, 656.6524: 0.8944, 669.3133: 0.9640, 681.7597: 0.9997, 693.3476: 0.9997,
                  704.0773: 0.9621, 722.9614: 0.8210, 746.9957: 0.5537, 772.9614: 0.2921, 794.6352: 0.1519,
                  800.6438: 0.1265}
        green_sd = {350.4292: 0.1350, 380.4721: 0.0456, 402.3605: -0.0015, 418.6695: -0.0240, 427.2532: -0.0127,
                    432.4034: -0.0146, 439.6996: -0.0353, 448.2833: -0.0372, 462.0172: 0.0155, 471.2446: 0.0870,
                    486.9099: 0.2733, 498.7124: 0.4596, 515.2361: 0.7400, 523.6052: 0.8624, 531.3305: 0.9508,
                    536.6953: 0.9922, 542.2747: 1.0054, 550.0000: 0.9790, 558.5837: 0.8868, 578.1116: 0.5838,
                    601.5021: 0.3034, 621.8884: 0.1980, 646.1373: 0.1566, 690.3433: 0.1463, 723.3906: 0.1162,
                    753.4335: 0.0738, 776.1803: 0.0456, 800.6438: 0.0268}
        blue_sd = {350.0000: 0.2902, 362.8755: 0.3213, 375.7511: 0.3467, 389.0558: 0.4088, 400.4292: 0.5029,
                   412.8755: 0.6723, 421.4592: 0.8021, 430.9013: 0.9169, 438.1974: 0.9762, 446.9957: 1.0073,
                   457.9399: 0.9649, 469.7425: 0.8134, 481.7597: 0.6102, 495.4936: 0.4107, 512.8755: 0.2451,
                   540.1288: 0.1397, 567.5966: 0.0851, 599.3562: 0.0578, 634.5494: 0.0672, 683.4764: 0.0804,
                   720.8155: 0.0682, 761.5880: 0.0362, 786.9099: 0.0211, 801.0730: 0.0174}
        midscale_sd = {400.2146: 1.2914, 407.5107: 1.3385, 415.2361: 1.4439, 421.4592: 1.5455, 427.2532: 1.6340,
                       433.6910: 1.7036, 438.4120: 1.7281, 447.8541: 1.7299, 455.1502: 1.6998, 462.8755: 1.6396,
                       470.6009: 1.5662, 479.1845: 1.4834, 486.9099: 1.4062, 495.0644: 1.3620, 501.0730: 1.3526,
                       506.2232: 1.3620, 514.3777: 1.3592, 521.0300: 1.3319, 527.6824: 1.2914, 531.7597: 1.2679,
                       545.7082: 1.2199, 560.7296: 1.1522, 568.8841: 1.0976, 575.9657: 1.0186, 583.6910: 0.9151,
                       589.4850: 0.8360, 595.4936: 0.7683, 602.3605: 0.7287, 607.0815: 0.7165, 614.3777: 0.7175,
                       622.5322: 0.7400, 628.5408: 0.7626, 637.7682: 0.8031, 643.5622: 0.8247, 652.1459: 0.8492,
                       657.7253: 0.8567, 670.3863: 0.8944, 681.7597: 0.9038, 690.7725: 0.9019, 697.2103: 0.8831}
        minimum_sd = {399.3562: 0.8040, 407.5107: 0.8134, 411.8026: 0.8322, 416.5236: 0.8718, 422.7468: 0.9245,
                      429.3991: 0.9715, 438.8412: 1.0035, 451.7167: 0.9922, 464.5923: 0.9452, 474.8927: 0.9132,
                      482.1888: 0.8962, 489.9142: 0.8548, 502.3605: 0.8210, 509.8712: 0.8059, 518.2403: 0.7532,
                      526.3948: 0.6478, 536.6953: 0.5688, 547.8541: 0.5293, 553.8627: 0.5264, 561.1588: 0.5340,
                      571.4592: 0.5246, 581.3305: 0.4672, 593.1330: 0.3326, 604.9356: 0.2357, 619.0987: 0.1952,
                      639.2704: 0.1849, 654.7210: 0.1867, 669.7425: 0.1849, 680.0429: 0.1849, 690.9871: 0.1736}
        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]
        self.d_ref_sd = colour.SpectralDistribution(midscale_sd)
        self.d_min_sd = colour.SpectralDistribution(minimum_sd)

        self.mtf = [{2.6755: 1.0000, 5.5916: 1.0023, 9.2993: 0.9746, 15.2667: 0.8958, 23.2925: 0.7856, 28.6470: 0.6923,
                     33.4561: 0.5987, 40.4436: 0.4693, 56.8525: 0.2951, 86.5532: 0.1479, 117.2922: 0.0835},
                    {2.8175: 1.0023, 4.7878: 1.0406, 7.5611: 1.0678, 12.6291: 1.0854, 18.2968: 1.0703, 24.7415: 0.9746,
                     35.8451: 0.8252, 51.4861: 0.6810, 72.8446: 0.5338, 89.0127: 0.4313, 108.3015: 0.3028,
                     163.1129: 0.1167, 196.7553: 0.0691},
                    {2.6525: 1.0023, 3.6808: 0.9700, 4.7878: 0.9791, 7.1954: 1.0333, 10.6747: 1.1008, 18.0617: 1.1754,
                     26.5082: 1.1590, 38.2395: 1.0931, 59.6133: 0.9213, 75.8897: 0.7783, 109.9480: 0.5313,
                     137.8716: 0.3882, 162.0616: 0.2951, 182.0656: 0.2249, 193.8090: 0.1895}]

        red_rms = {0.0081: 0.0074, 0.5625: 0.0078, 1.2141: 0.0083, 1.4852: 0.0084, 1.9061: 0.0075, 2.2541: 0.0065,
                   2.7357: 0.0054, 3.1364: 0.0047, 3.6301: 0.0044, 4.0753: 0.0044, 4.5326: 0.0044, 4.9939: 0.0044}
        green_rms = {0.0121: 0.0086, 0.6151: 0.0093, 1.0643: 0.0103, 1.3395: 0.0115, 1.4852: 0.0117, 1.7159: 0.0104,
                     1.9547: 0.0083, 2.1773: 0.0068, 2.4444: 0.0061, 2.6993: 0.0061, 3.0554: 0.0065, 3.3144: 0.0063,
                     3.6463: 0.0060, 4.0550: 0.0059, 4.5245: 0.0059, 4.9980: 0.0063}
        blue_rms = {0.0202: 0.0111, 0.3399: 0.0091, 0.5868: 0.0094, 0.7770: 0.0113, 1.0279: 0.0157, 1.2546: 0.0190,
                    1.5217: 0.0201, 1.8495: 0.0180, 2.1327: 0.0153, 2.4605: 0.0134, 2.7398: 0.0130, 3.0231: 0.0129,
                    3.2821: 0.0122, 3.5208: 0.0115, 3.9417: 0.0113, 4.5892: 0.0112, 4.9960: 0.0112}
        red_rms_density = {0.0182: 0.2203, 0.4518: 0.2284, 0.7840: 0.2382, 0.9542: 0.2495, 1.1933: 0.2835,
                           1.5904: 0.3791, 2.4048: 0.6221, 3.4704: 0.9332, 4.1917: 1.1373, 5.0041: 1.3625}
        green_rms_density = {0.0122: 0.6237, 0.5166: 0.6189, 0.9988: 0.6529, 1.4972: 0.7647, 2.0199: 0.9883,
                             3.5069: 1.5326, 4.0174: 1.7173, 4.4854: 1.8566, 5.0020: 2.0608}
        blue_rms_density = {0.0162: 1.0239, 0.6179: 1.0304, 0.9947: 1.0595, 1.3169: 1.1260, 1.7545: 1.2847,
                            2.7087: 1.6671, 3.6244: 2.0381, 4.4246: 2.3248, 5.0000: 2.5095}
        self.rms_curve = [red_rms, green_rms, blue_rms]
        self.rms_density = [red_rms_density, green_rms_density, blue_rms_density]

        self.calibrate()
