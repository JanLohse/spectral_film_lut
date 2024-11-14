import colour
import numpy as np

from film_spectral import FilmSpectral


class Kodak2383(FilmSpectral):
    def __init__(self):
        super().__init__()
        # spectral sensitivity
        yellow_log_sensitivity = {360.5733: 0.0340, 370.0645: -0.0164, 377.6576: -0.0818, 385.7251: -0.1874,
                                  391.8945: -0.2780, 398.0638: -0.3535, 400.4366: -0.3585, 406.6059: -0.3082,
                                  411.3516: -0.2579, 416.5718: -0.1824, 421.7920: -0.0969, 425.1139: -0.0063,
                                  428.9104: 0.0591, 433.6560: 0.1195, 439.8254: 0.1748, 445.9947: 0.2956,
                                  451.2149: 0.4013, 455.9605: 0.5220, 460.2316: 0.5975, 468.2043: 0.6347,
                                  473.0448: 0.4616, 481.5869: -0.0969, 489.6545: -0.9673, 500.5695: -2.0239,
                                  509.8709: -2.9970}
        self.yellow_sensitivity = colour.SpectralDistribution(
            {x: 10 ** y for x, y in yellow_log_sensitivity.items()}).align(colour.SPECTRAL_SHAPE_DEFAULT)

        magenta_log_sensitivity = {448.8421: -2.9950, 449.3166: -1.7572, 451.6894: -1.6969, 458.1435: -1.6264,
                                   463.4586: -1.5499, 470.5771: -1.4392, 477.9803: -1.3517, 485.5733: -1.1937,
                                   492.9765: -1.0730, 504.3660: -0.9623, 510.2506: -0.9260, 516.7046: -0.9119,
                                   521.9248: -0.9099, 529.5178: -0.8465, 538.5345: -0.6302, 542.3311: -0.4692,
                                   544.7039: -0.3082, 546.6021: -0.2025, 549.4495: -0.1723, 552.2969: -0.2679,
                                   556.7578: -0.5255, 558.1815: -0.6453, 564.4457: -0.9824, 569.1913: -1.2289,
                                   576.0251: -1.5862, 581.2453: -1.8679, 584.0926: -2.0843, 586.4655: -2.4616,
                                   588.5535: -2.7384, 589.7874: -2.9950}
        self.magenta_sensitivity = colour.SpectralDistribution(
            {x: 10 ** y for x, y in magenta_log_sensitivity.items()}).align(colour.SPECTRAL_SHAPE_DEFAULT)

        cyan_log_sensitivity = {583.6181: -3.0050, 587.4146: -2.5975, 588.8383: -2.4818, 591.2111: -2.4365,
                                594.5330: -2.3912, 599.2787: -2.3711, 604.4989: -2.2956, 614.1800: -2.1316,
                                624.4305: -1.9484, 629.6507: -1.8629, 638.6674: -1.7874, 647.6841: -1.7723,
                                656.7008: -1.7019, 666.6667: -1.4956, 675.2088: -1.3094, 684.4153: -1.1454,
                                689.9203: -1.0931, 693.7168: -1.0830, 700.3607: -1.1384, 707.4791: -1.3497,
                                713.2688: -1.5479, 717.4450: -1.7371, 722.1906: -2.1296, 725.9871: -2.5623,
                                727.8853: -2.8189, 729.7836: -2.9950}
        self.cyan_sensitivity = colour.SpectralDistribution(
            {x: 10 ** y for x, y in cyan_log_sensitivity.items()}).align(colour.SPECTRAL_SHAPE_DEFAULT)

        self.sensitivity_exposure_time = 1 / 50

        # sensiometry
        self.red_log_exposure = np.array(
            [-0.3927, -0.0777, 0.1694, 0.4214, 0.5911, 0.7191, 0.8013, 0.8865, 0.9609, 1.0836, 1.2176, 1.3855, 1.4992,
             1.5945, 1.6827, 1.7803, 1.9054, 2.0388, 2.1996, 2.3580, 2.5801])
        self.red_density_curve = np.array(
            [0.0513, 0.0579, 0.0775, 0.1458, 0.2385, 0.3532, 0.4684, 0.6408, 0.8309, 1.2789, 1.8761, 2.6385, 3.1013,
             3.3955, 3.6064, 3.7746, 3.9231, 4.0307, 4.0960, 4.1133, 4.1186])
        self.green_log_exposure = np.array(
            [-0.7006, -0.4439, -0.2474, -0.0747, 0.0861, 0.2439, 0.3469, 0.4350, 0.5291, 0.6649, 0.7602, 0.9222, 1.0746,
             1.2116, 1.3242, 1.4641, 1.5653, 1.6582, 1.7744, 1.8994, 2.0334, 2.1668])
        self.green_density_curve = np.array(
            [0.0543, 0.0680, 0.0757, 0.1036, 0.1500, 0.2510, 0.3603, 0.4833, 0.6598, 0.9967, 1.3057, 1.9296, 2.5832,
             3.0526, 3.3854, 3.6914, 3.8393, 3.9439, 4.0182, 4.0717, 4.0895, 4.0966])
        self.blue_log_exposure = np.array(
            [-0.9990, -0.7966, -0.5951, -0.4193, -0.2692, -0.1320, -0.0334, 0.0798, 0.1887, 0.3011, 0.4134, 0.5523,
             0.7298, 0.8542, 0.9365, 1.0180, 1.1106, 1.2238, 1.3301, 1.4459, 1.5574, 1.6603, 1.7803, 1.9090, 1.9879])
        self.blue_density_curve = np.array(
            [0.1089, 0.1175, 0.1260, 0.1491, 0.1876, 0.2544, 0.3305, 0.4512, 0.6103, 0.8491, 1.1571, 1.6970, 2.5355,
             3.0575, 3.3612, 3.5623, 3.7377, 3.8738, 3.9688, 4.0244, 4.0543, 4.0800, 4.0980, 4.1099, 4.1125])

        self.exposure_base = 10

        self.sensiometry_exposure_time = 1 / 500

        # spectral dye density
        yellow_spectral_density = {349.5534: 0.2688, 358.6752: 0.2352, 365.4636: 0.2056, 370.0598: 0.1898,
                                   377.4845: 0.1987, 384.9092: 0.2372, 393.7482: 0.3242, 402.5871: 0.4240,
                                   412.1332: 0.5524, 418.4972: 0.6364, 427.0534: 0.7333, 432.2153: 0.7714,
                                   436.7408: 0.7951, 441.1957: 0.8086, 445.7212: 0.8119, 453.4288: 0.7985,
                                   450.1760: 0.8072, 459.5100: 0.7659, 465.5205: 0.7135, 474.9251: 0.5949,
                                   481.9963: 0.4951, 492.8858: 0.3449, 503.0683: 0.2313, 514.8772: 0.1364,
                                   528.2416: 0.0791, 552.8493: 0.0396, 587.4980: 0.0188, 650.0777: 0.0119,
                                   748.1547: 0.0089}
        magenta_spectral_density = {349.5534: 0.0791, 361.2209: 0.0564, 373.2418: 0.0396, 386.6770: 0.0425,
                                    404.0014: 0.0663, 416.8709: 0.0769, 422.0328: 0.0831, 427.4069: 0.0943,
                                    430.5182: 0.0947, 436.5287: 0.0761, 444.6605: 0.0682, 458.8029: 0.1038,
                                    471.5309: 0.1769, 482.8448: 0.2797, 495.5729: 0.4566, 501.6541: 0.5404,
                                    508.6545: 0.6473, 516.1500: 0.7202, 525.2718: 0.8064, 531.2822: 0.8499,
                                    537.9291: 0.8758, 543.3032: 0.8775, 551.8593: 0.8422, 559.3547: 0.7708,
                                    569.8201: 0.6117, 585.0231: 0.3647, 596.6905: 0.2283, 608.0043: 0.1453,
                                    616.1362: 0.1028, 627.0965: 0.0718, 644.4208: 0.0416, 666.2707: 0.0238,
                                    682.2515: 0.0169, 710.5361: 0.0121}
        cyan_spectral_density = {349.1999: 0.2560, 356.9782: 0.2550, 366.8778: 0.2688, 378.5452: 0.2866,
                                 385.8285: 0.2945, 393.6775: 0.2929, 400.8193: 0.2837, 406.4763: 0.2639,
                                 411.7796: 0.2382, 419.2044: 0.1977, 428.0433: 0.1463, 438.2965: 0.1028,
                                 450.6710: 0.0643, 463.0456: 0.0406, 477.8950: 0.0307, 493.8051: 0.0303,
                                 525.2718: 0.0613, 512.1901: 0.0396, 539.7676: 0.1097, 554.2635: 0.1799,
                                 572.6485: 0.3212, 589.9729: 0.5030, 601.9231: 0.6483, 617.1968: 0.8271,
                                 629.3592: 0.9517, 637.4911: 1.0109, 645.4815: 1.0574, 652.5526: 1.0793,
                                 658.2096: 1.0870, 666.5535: 1.0833, 674.5439: 1.0613, 681.5444: 1.0228,
                                 694.5553: 0.9230, 702.6872: 0.8410, 710.5361: 0.7511, 720.0822: 0.6364,
                                 727.2241: 0.5534, 734.5074: 0.4704, 742.0735: 0.3904, 749.3568: 0.3281}
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

        self.yellow_spectral_density = colour.SpectralDistribution(yellow_spectral_density).align(
            colour.SPECTRAL_SHAPE_DEFAULT)
        self.magenta_spectral_density = colour.SpectralDistribution(magenta_spectral_density).align(
            colour.SPECTRAL_SHAPE_DEFAULT)
        self.cyan_spectral_density = colour.SpectralDistribution(cyan_spectral_density).align(
            colour.SPECTRAL_SHAPE_DEFAULT)
        self.midscale_spectral_density = colour.SpectralDistribution(midscale_spectral_density).align(
            colour.SPECTRAL_SHAPE_DEFAULT)

        self.calibrate(3200, 'print_film')