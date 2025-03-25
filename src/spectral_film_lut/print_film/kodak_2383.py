from spectral_film_lut.film_spectral import *


class Kodak2383(FilmSpectral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lad = [1.09, 1.06, 1.03]
        self.density_measure = 'status_a'
        self.exposure_kelvin = 3200
        self.projection_kelvin = 5500

        # spectral sensitivity
        self.red_log_sensitivity = {583.6181: -3.0050, 587.4146: -2.5975, 588.8383: -2.4818, 591.2111: -2.4365,
                                    594.5330: -2.3912, 599.2787: -2.3711, 604.4989: -2.2956, 614.1800: -2.1316,
                                    624.4305: -1.9484, 629.6507: -1.8629, 638.6674: -1.7874, 647.6841: -1.7723,
                                    656.7008: -1.7019, 666.6667: -1.4956, 675.2088: -1.3094, 684.4153: -1.1454,
                                    689.9203: -1.0931, 693.7168: -1.0830, 700.3607: -1.1384, 707.4791: -1.3497,
                                    713.2688: -1.5479, 717.4450: -1.7371, 722.1906: -2.1296, 725.9871: -2.5623,
                                    727.8853: -2.8189, 729.7836: -2.9950}

        self.green_log_sensitivity = {448.8421: -2.9950, 449.3166: -1.7572, 451.6894: -1.6969, 458.1435: -1.6264,
                                      463.4586: -1.5499, 470.5771: -1.4392, 477.9803: -1.3517, 485.5733: -1.1937,
                                      492.9765: -1.0730, 504.3660: -0.9623, 510.2506: -0.9260, 516.7046: -0.9119,
                                      521.9248: -0.9099, 529.5178: -0.8465, 538.5345: -0.6302, 542.3311: -0.4692,
                                      544.7039: -0.3082, 546.6021: -0.2025, 549.4495: -0.1723, 552.2969: -0.2679,
                                      556.7578: -0.5255, 558.1815: -0.6453, 564.4457: -0.9824, 569.1913: -1.2289,
                                      576.0251: -1.5862, 581.2453: -1.8679, 584.0926: -2.0843, 586.4655: -2.4616,
                                      588.5535: -2.7384, 589.7874: -2.9950}

        self.blue_log_sensitivity = {360.5733: 0.0340, 370.0645: -0.0164, 377.6576: -0.0818, 385.7251: -0.1874,
                                     391.8945: -0.2780, 398.0638: -0.3535, 400.4366: -0.3585, 406.6059: -0.3082,
                                     411.3516: -0.2579, 416.5718: -0.1824, 421.7920: -0.0969, 425.1139: -0.0063,
                                     428.9104: 0.0591, 433.6560: 0.1195, 439.8254: 0.1748, 445.9947: 0.2956,
                                     451.2149: 0.4013, 455.9605: 0.5220, 460.2316: 0.5975, 468.2043: 0.6347,
                                     473.0448: 0.4616, 481.5869: -0.0969, 489.6545: -0.9673, 500.5695: -2.0239,
                                     509.8709: -2.9970}

        # sensiometry
        self.red_log_exposure = np.array(
            [-0.7105, -0.3752, -0.0105, 0.1993, 0.3651, 0.4860, 0.5996, 0.6803, 0.7664, 0.8937, 1.0293, 1.1292, 1.2602,
             1.3930, 1.5030, 1.6386, 1.7815, 1.9455, 2.1333, 2.2900])
        self.red_density_curve = np.array(
            [0.0446, 0.0481, 0.0847, 0.1499, 0.2575, 0.3811, 0.5688, 0.7713, 1.0414, 1.5621, 2.1801, 2.6265, 3.1243,
             3.5191, 3.7526, 3.9448, 4.0593, 4.1096, 4.1279, 4.1371])
        self.green_log_exposure = np.array(
            [-0.7106, -0.4229, -0.1665, 0.0946, 0.3107, 0.2146, 0.4454, 0.5571, 0.6670, 0.8145, 0.9821, 1.1745, 1.2936,
             1.3879, 1.5006, 1.5977, 1.7140, 1.8587, 2.0181, 2.2901])
        self.green_density_curve = np.array(
            [0.0481, 0.0584, 0.0813, 0.1339, 0.2782, 0.1980, 0.4522, 0.6582, 0.9329, 1.4366, 2.1005, 2.8903, 3.3024,
             3.5714, 3.8175, 3.9686, 4.0888, 4.1712, 4.2067, 4.2090])
        self.blue_log_exposure = np.array(
            [-0.7069, -0.4724, -0.2077, -0.0016, 0.1907, 0.3575, 0.5342, 0.6826, 0.8173, 0.9253, 1.0591, 1.1763, 1.2817,
             1.4026, 1.5519, 1.7717, 1.6581, 2.0044, 2.2901])
        self.blue_density_curve = np.array(
            [0.0652, 0.0733, 0.0893, 0.1374, 0.2289, 0.3720, 0.6181, 0.9787, 1.4423, 1.8945, 2.5011, 3.0334, 3.4054,
             3.7088, 3.9160, 4.0705, 4.0076, 4.1346, 4.1518])

        self.exposure_base = 10

        # spectral dye density
        red_spectral_density = {349.1999: 0.2560, 356.9782: 0.2550, 366.8778: 0.2688, 378.5452: 0.2866,
                                385.8285: 0.2945, 393.6775: 0.2929, 400.8193: 0.2837, 406.4763: 0.2639,
                                411.7796: 0.2382, 419.2044: 0.1977, 428.0433: 0.1463, 438.2965: 0.1028,
                                450.6710: 0.0643, 463.0456: 0.0406, 477.8950: 0.0307, 493.8051: 0.0303,
                                525.2718: 0.0613, 512.1901: 0.0396, 539.7676: 0.1097, 554.2635: 0.1799,
                                572.6485: 0.3212, 589.9729: 0.5030, 601.9231: 0.6483, 617.1968: 0.8271,
                                629.3592: 0.9517, 637.4911: 1.0109, 645.4815: 1.0574, 652.5526: 1.0793,
                                658.2096: 1.0870, 666.5535: 1.0833, 674.5439: 1.0613, 681.5444: 1.0228,
                                694.5553: 0.9230, 702.6872: 0.8410, 710.5361: 0.7511, 720.0822: 0.6364,
                                727.2241: 0.5534, 734.5074: 0.4704, 742.0735: 0.3904, 749.3568: 0.3281}
        green_spectral_density = {349.5534: 0.0791, 361.2209: 0.0564, 373.2418: 0.0396, 386.6770: 0.0425,
                                  404.0014: 0.0663, 416.8709: 0.0769, 422.0328: 0.0831, 427.4069: 0.0943,
                                  430.5182: 0.0947, 436.5287: 0.0761, 444.6605: 0.0682, 458.8029: 0.1038,
                                  471.5309: 0.1769, 482.8448: 0.2797, 495.5729: 0.4566, 501.6541: 0.5404,
                                  508.6545: 0.6473, 516.1500: 0.7202, 525.2718: 0.8064, 531.2822: 0.8499,
                                  537.9291: 0.8758, 543.3032: 0.8775, 551.8593: 0.8422, 559.3547: 0.7708,
                                  569.8201: 0.6117, 585.0231: 0.3647, 596.6905: 0.2283, 608.0043: 0.1453,
                                  616.1362: 0.1028, 627.0965: 0.0718, 644.4208: 0.0416, 666.2707: 0.0238,
                                  682.2515: 0.0169, 710.5361: 0.0121}
        blue_spectral_density = {349.5534: 0.2688, 358.6752: 0.2352, 365.4636: 0.2056, 370.0598: 0.1898,
                                 377.4845: 0.1987, 384.9092: 0.2372, 393.7482: 0.3242, 402.5871: 0.4240,
                                 412.1332: 0.5524, 418.4972: 0.6364, 427.0534: 0.7333, 432.2153: 0.7714,
                                 436.7408: 0.7951, 441.1957: 0.8086, 445.7212: 0.8119, 453.4288: 0.7985,
                                 450.1760: 0.8072, 459.5100: 0.7659, 465.5205: 0.7135, 474.9251: 0.5949,
                                 481.9963: 0.4951, 492.8858: 0.3449, 503.0683: 0.2313, 514.8772: 0.1364,
                                 528.2416: 0.0791, 552.8493: 0.0396, 587.4980: 0.0188, 650.0777: 0.0119,
                                 748.1547: 0.0089}
        midscale_spectral_density = {349.4624: 1.1599, 358.7459: 1.0050, 364.7564: 0.9022, 370.4134: 0.8212,
                                     376.4238: 0.7511, 379.6059: 0.7372, 383.4950: 0.7343, 387.0306: 0.7422,
                                     391.9804: 0.7698, 401.5265: 0.8627, 408.2441: 0.9270, 418.1437: 1.0218,
                                     425.5684: 1.0712, 427.4069: 1.0841, 432.8517: 1.0943, 437.9429: 1.0841,
                                     450.6710: 1.0406, 459.0857: 1.0030, 470.1167: 0.9388, 478.2485: 0.8924,
                                     483.1276: 0.8726, 487.0875: 0.8647, 491.6838: 0.8647, 497.2700: 0.8771,
                                     501.0884: 0.8916, 508.7252: 0.9349, 518.9077: 0.9922, 528.6659: 1.0515,
                                     537.4342: 1.0989, 544.6467: 1.1238, 550.1622: 1.1250, 556.6677: 1.1021,
                                     567.5573: 1.0189, 576.1841: 0.9437, 579.5783: 0.9177, 583.1138: 0.8999,
                                     587.6394: 0.8847, 594.0034: 0.8819, 600.5796: 0.9003, 606.1658: 0.9299,
                                     614.7926: 0.9912, 628.7935: 1.0930, 638.9053: 1.1562, 645.4815: 1.1849,
                                     652.6233: 1.2005, 660.4723: 1.2023, 668.3213: 1.1888, 673.1297: 1.1750,
                                     683.4536: 1.1088, 701.7679: 0.9319, 720.2943: 0.7105, 748.6497: 0.4020}

        self.red_sd = colour.SpectralDistribution(red_spectral_density)
        self.green_sd = colour.SpectralDistribution(green_spectral_density)
        self.blue_sd = colour.SpectralDistribution(blue_spectral_density)
        self.d_ref_sd = colour.SpectralDistribution(midscale_spectral_density)

        self.calibrate()
