import time
from abc import ABC

import colour
import numpy as np

spectral_shape = colour.SPECTRAL_SHAPE_DEFAULT


class FilmSpectral(ABC):
    def calibrate(self, whitebalace, type: {'negative', 'print'}):
        self.cyan_spectral_density /= max(self.cyan_spectral_density.values)
        self.magenta_spectral_density /= max(self.magenta_spectral_density.values)
        self.yellow_spectral_density /= max(self.yellow_spectral_density.values)

        if type == 'negative':
            coeffs = self.log_exposure_to_density(-1 * np.ones(3))

        elif type == 'print':
            A = np.stack([self.cyan_spectral_density.values, self.magenta_spectral_density.values,
                          self.yellow_spectral_density.values])
            coeffs = np.linalg.lstsq(A.T, self.midscale_spectral_density.values, rcond=None)[0]

        self.base_spectral_density = self.midscale_spectral_density - (
                self.cyan_spectral_density * coeffs[0] + self.magenta_spectral_density * coeffs[
            1] + self.yellow_spectral_density * coeffs[2])

        self.exposure_adjustment = np.zeros(3)
        neutral_spectral = kelvin_to_spectral(whitebalace)
        neutral_exposure = self.spectral_to_log_exposure(neutral_spectral)

        cyan_target_exposure = np.interp(coeffs[0], self.red_density_curve, self.red_log_exposure)
        magenta_target_exposure = np.interp(coeffs[1], self.green_density_curve, self.green_log_exposure)
        yellow_target_exposure = np.interp(coeffs[2], self.blue_density_curve, self.blue_log_exposure)

        self.exposure_adjustment = np.array(
            [cyan_target_exposure, magenta_target_exposure, yellow_target_exposure]) - neutral_exposure

        target_flux = 18
        self.amplification = 1
        test_projection = self.spectral_to_projection(neutral_spectral, neutral_spectral)
        self.amplification = target_flux / colour.sd_to_XYZ(test_projection)[1]

    def spectral_to_log_exposure(self, light_intensity):
        cyan_spectral_exposure = light_intensity * self.cyan_sensitivity
        magenta_spectral_exposure = light_intensity * self.magenta_sensitivity
        yellow_spectral_exposure = light_intensity * self.yellow_sensitivity

        cyan_effective_exposure = np.sum(cyan_spectral_exposure.values) * cyan_spectral_exposure.shape.interval
        magenta_effective_exposure = np.sum(magenta_spectral_exposure.values) * magenta_spectral_exposure.shape.interval
        yellow_effective_exposure = np.sum(yellow_spectral_exposure.values) * yellow_spectral_exposure.shape.interval

        effective_exposure = np.array([cyan_effective_exposure, magenta_effective_exposure, yellow_effective_exposure])
        log_exposure = np.emath.logn(self.exposure_base, effective_exposure) + self.exposure_adjustment

        return log_exposure

    def log_exposure_to_density(self, log_exposure):
        red_density = np.interp(log_exposure[0], self.red_log_exposure, self.red_density_curve)
        green_density = np.interp(log_exposure[1], self.green_log_exposure, self.green_density_curve)
        blue_density = np.interp(log_exposure[2], self.blue_log_exposure, self.blue_density_curve)

        return np.array([red_density, green_density, blue_density])

    def density_to_spectral_density(self, density):
        spectral_density = self.cyan_spectral_density * density[0] + self.magenta_spectral_density * density[
            1] + self.yellow_spectral_density * density[2] + self.base_spectral_density

        return spectral_density

    def spectral_density_to_transmittance(self, spectral_density: colour.SpectralDistribution):
        spectral_density_series = spectral_density.to_series()
        transmittance_series = 10 ** -spectral_density_series
        transmittance = colour.SpectralDistribution(transmittance_series)

        return transmittance

    def spectral_to_transmittance(self, spectral):
        log_exposure = self.spectral_to_log_exposure(spectral)
        density = self.log_exposure_to_density(log_exposure)
        spectral_density = self.density_to_spectral_density(density)
        transmittance = self.spectral_density_to_transmittance(spectral_density)

        return transmittance

    def spectral_to_projection(self, spectral, illuminant):
        projection = self.spectral_to_transmittance(spectral) * illuminant * self.amplification

        return projection


class Kodak_5207(FilmSpectral):
    def __init__(self):
        super().__init__()
        # spectral sensitivity
        yellow_log_sensitivity = {359.4238: 1.5366, 367.7450: 1.6467, 381.7870: 2.0484, 386.9877: 2.1200,
                                  397.4932: 2.3368, 403.3785: 2.3941, 418.4270: 2.3516, 437.9900: 2.4259,
                                  449.7779: 2.3596, 460.5627: 2.4259, 467.5853: 2.4896, 472.3506: 2.4259,
                                  477.6176: 2.2348, 483.6370: 1.8978, 488.6531: 1.4704, 499.1871: 0.9980,
                                  506.4605: 0.7485, 519.7533: 0.0930}

        self.yellow_sensitivity = colour.SpectralDistribution(
            {x: 10 ** y for x, y in yellow_log_sensitivity.items()}).align(spectral_shape)
        magenta_log_sensitivity = {399.8672: 0.6424, 439.9964: 0.3451, 449.0255: 0.3876, 460.0611: 0.5468,
                                   467.0837: 0.5734, 472.0998: 0.6583, 491.1612: 1.5341, 496.6790: 1.7491,
                                   501.1935: 1.8526, 528.5316: 2.1818, 548.5962: 2.4127, 550.6026: 2.4121,
                                   566.1527: 2.2455, 575.9342: 2.1181, 579.6963: 2.0066, 586.7190: 1.5554,
                                   593.7416: 0.9502, 599.2593: 0.3398}
        self.magenta_sensitivity = colour.SpectralDistribution(
            {x: 10 ** y for x, y in magenta_log_sensitivity.items()}).align(spectral_shape)

        cyan_log_sensitivity = {530.0364: 0.0532, 535.5542: 0.0346, 542.0752: 0.0346, 549.5994: 0.0930,
                                558.8793: 0.3000, 564.6479: 0.4061, 568.7611: 0.4932, 581.2012: 1.2369,
                                587.2206: 1.5129, 591.7351: 1.6881, 597.7545: 1.8181, 610.5457: 2.0570,
                                620.3272: 2.1950, 628.6038: 2.2109, 637.6329: 2.2746, 646.4112: 2.3649,
                                651.2267: 2.3845, 655.6911: 2.3835, 658.1490: 2.3654, 662.7137: 2.2640,
                                669.9871: 2.0013, 676.5081: 1.7252, 680.2702: 1.5660, 686.2896: 1.3563,
                                689.2993: 1.1891, 694.2653: 0.7379, 699.6828: 0.1657}
        self.cyan_sensitivity = colour.SpectralDistribution(
            {x: 10 ** y for x, y in cyan_log_sensitivity.items()}).align(spectral_shape)

        self.sensitivity_exposure_time = 1 / 50

        # sensiometry
        self.red_log_exposure = np.array(
            [-7.9924, -7.4827, -6.8785, -6.3687, -5.7456, -5.0658, -4.4616, -3.8952, -3.3665, -2.8190, -1.9882, -0.7043,
             1.0517, 2.6377, 4.0160, 4.9601, 6.0175, 6.7538, 7.3014, 7.9887])
        self.red_density_curve = np.array(
            [0.1763, 0.1798, 0.1876, 0.1940, 0.2194, 0.2584, 0.3008, 0.3532, 0.4155, 0.4913, 0.6074, 0.7907, 1.0413,
             1.2685, 1.4611, 1.5779, 1.6975, 1.7619, 1.8037, 1.8447])
        self.green_log_exposure = np.array(
            [-8.0038, -7.2938, -6.6330, -6.2176, -5.5568, -4.7071, -4.0463, -3.6120, -2.5735, -1.1385, 0.1454, 1.9580,
             3.4307, 4.5825, 5.5832, 6.4517, 7.2070, 7.9887])
        self.green_density_curve = np.array(
            [0.5762, 0.5840, 0.5904, 0.5996, 0.6336, 0.6881, 0.7539, 0.8063, 0.9740, 1.2112, 1.4271, 1.7308, 1.9778,
             2.1576, 2.3034, 2.4132, 2.4882, 2.5547])
        self.blue_log_exposure = np.array(
            [-8.0113, -7.2561, -6.7274, -6.2743, -5.3679, -4.5183, -3.7819, -3.0456, -2.1392, -0.7420, 0.6929, 2.4678,
             4.1482, 5.3944, 6.1874, 6.7916, 7.6413, 7.9962])
        self.blue_density_curve = np.array(
            [0.8452, 0.8487, 0.8530, 0.8558, 0.9018, 0.9585, 1.0328, 1.1461, 1.3025, 1.5248, 1.7548, 2.0408, 2.3077,
             2.4832, 2.5866, 2.6432, 2.7098, 2.7331])

        self.exposure_base = 2
        self.sensiometry_exposure_time = 1 / 50

        # spectral dye density
        yellow_spectral_density = {399.2913: 0.4979, 408.9852: 0.6483, 418.6790: 0.7920, 426.6346: 0.8937,
                                   433.7211: 0.9625, 438.6683: 0.9906, 442.6795: 1.0040, 445.7548: 1.0060,
                                   449.7660: 1.0006, 453.2424: 0.9859, 456.4514: 0.9619, 459.9947: 0.9334,
                                   462.7357: 0.8990, 466.3458: 0.8539, 470.4907: 0.7870, 477.8446: 0.6650,
                                   486.2682: 0.5246, 490.2126: 0.4544, 499.9064: 0.3174, 507.7283: 0.2305,
                                   519.9626: 0.1436, 538.0131: 0.0774, 552.0524: 0.0536, 567.4288: 0.0292,
                                   586.1479: 0.0142, 602.8613: 0.0058, 625.9259: 0.0018, 661.6927: -0.0035,
                                   674.3950: -0.0042, 698.4624: -0.0035, 725.5382: -0.0005, 753.6168: 0.0035,
                                   784.0353: 0.0052, 790.0521: 0.0038}
        magenta_spectral_density = {399.6256: 0.0032, 416.0048: -0.0069, 420.3503: -0.0059, 425.1638: 0.0025,
                                    428.6402: 0.0052, 433.7211: -0.0042, 440.4065: -0.0186, 451.1031: -0.0102,
                                    462.3345: 0.0540, 471.7609: 0.1355, 479.6497: 0.2298, 486.5356: 0.3341,
                                    495.7615: 0.4845, 503.0485: 0.6115, 511.2716: 0.7386, 518.3581: 0.8372,
                                    525.8457: 0.9258, 530.4586: 0.9702, 534.5367: 0.9933, 537.4783: 1.0003,
                                    541.1552: 0.9959, 544.1637: 0.9836, 547.1052: 0.9645, 549.5120: 0.9385,
                                    551.7850: 0.9161, 555.5288: 0.8599, 560.6097: 0.7780, 566.4928: 0.6690,
                                    570.5041: 0.5865, 577.8580: 0.4611, 584.9445: 0.3575, 594.5715: 0.2572,
                                    601.9922: 0.2037, 611.1512: 0.1566, 623.0512: 0.1195, 650.9961: 0.0851,
                                    688.5680: 0.0690, 716.8472: 0.0580, 756.6252: 0.0359, 790.3864: 0.0179,
                                    799.6122: 0.0132}
        cyan_spectral_density = {399.6256: 0.2251, 410.6565: 0.1757, 423.8267: 0.1235, 437.2643: 0.0814,
                                 453.5767: 0.0500, 473.4991: 0.0349, 491.6834: 0.0292, 511.0710: 0.0326,
                                 530.9266: 0.0570, 548.5092: 0.1111, 566.7603: 0.1817, 578.9277: 0.2472,
                                 594.7052: 0.3889, 608.7445: 0.5280, 622.0484: 0.6583, 636.0209: 0.7800,
                                 649.6590: 0.8763, 661.4253: 0.9438, 667.8433: 0.9719, 673.6596: 0.9903,
                                 681.3478: 1.0020, 687.2309: 1.0013, 696.9916: 0.9812, 702.6073: 0.9535,
                                 709.4932: 0.9117, 720.0562: 0.8255, 729.2151: 0.7369, 737.9730: 0.6443,
                                 746.9983: 0.5407, 753.6168: 0.4678, 761.6393: 0.3876, 771.0657: 0.3040,
                                 777.0825: 0.2572, 785.3724: 0.2004, 792.6594: 0.1566, 800.1471: 0.1305}
        midscale_spectral_density = {399.9599: 1.1113, 413.6649: 1.3185, 422.0885: 1.4385, 431.4481: 1.5355,
                                     435.9273: 1.5615, 440.2728: 1.5732, 445.7548: 1.5706, 450.1671: 1.5545,
                                     453.6435: 1.5335, 458.1896: 1.4994, 464.2733: 1.4699, 470.8250: 1.4399,
                                     477.9783: 1.3997, 482.6581: 1.3536, 487.0705: 1.2961, 490.9480: 1.2637,
                                     495.8283: 1.2453, 500.7087: 1.2443, 504.7867: 1.2664, 508.5974: 1.2888,
                                     513.5446: 1.2888, 519.6951: 1.2721, 528.6536: 1.2527, 536.4755: 1.2543,
                                     543.2946: 1.2446, 549.4451: 1.2136, 558.8047: 1.1434, 566.3591: 1.0665,
                                     575.5850: 0.9705, 580.3984: 0.9110, 584.8777: 0.8408, 590.4934: 0.7590,
                                     596.0423: 0.6978, 601.8585: 0.6583, 609.6136: 0.6480, 618.2377: 0.6751,
                                     629.6697: 0.7081, 638.9624: 0.7232, 648.2551: 0.7536, 659.0186: 0.7894,
                                     670.1832: 0.8158, 682.5511: 0.8308, 695.1197: 0.8158, 706.0168: 0.7824,
                                     720.6578: 0.7045, 729.7500: 0.6426, 737.3713: 0.5815, 748.6696: 0.4899,
                                     760.8370: 0.3969, 771.6005: 0.3217, 781.6286: 0.2632, 789.6510: 0.2181,
                                     799.8797: 0.1813}
        minimum_spectral_density = {399.6256: 0.6567, 410.9239: 0.7389, 419.2138: 0.8054, 429.2419: 0.8522,
                                    437.6655: 0.8876, 444.8857: 0.8883, 452.5739: 0.8649, 458.9250: 0.8348,
                                    469.6216: 0.8298, 477.7778: 0.8171, 481.3879: 0.7991, 484.5969: 0.7673,
                                    488.0064: 0.7285, 493.0205: 0.6971, 498.5025: 0.6777, 503.1822: 0.6811,
                                    509.0654: 0.6918, 513.7452: 0.6744, 517.4221: 0.6443, 524.9766: 0.5881,
                                    531.7957: 0.5514, 546.3030: 0.5313, 559.4063: 0.5260, 569.2339: 0.5290,
                                    575.6518: 0.5149, 580.5322: 0.4882, 585.4793: 0.4314, 594.1035: 0.3318,
                                    601.4574: 0.2636, 610.6832: 0.2341, 624.2546: 0.2311, 638.9624: 0.2087,
                                    659.3529: 0.2094, 674.3950: 0.2124, 696.9247: 0.2097, 721.1927: 0.1950,
                                    740.9146: 0.1723, 756.9595: 0.1476, 779.0213: 0.1182, 799.6791: 0.0914}

        self.yellow_spectral_density = colour.SpectralDistribution(yellow_spectral_density).align(spectral_shape)
        self.magenta_spectral_density = colour.SpectralDistribution(magenta_spectral_density).align(spectral_shape)
        self.cyan_spectral_density = colour.SpectralDistribution(cyan_spectral_density).align(spectral_shape)
        self.midscale_spectral_density = colour.SpectralDistribution(midscale_spectral_density).align(spectral_shape)
        self.minimum_spectral_density = colour.SpectralDistribution(minimum_spectral_density).align(spectral_shape)

        self.calibrate(5500, 'negative')


class Kodak_2383(FilmSpectral):
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
            {x: 10 ** y for x, y in yellow_log_sensitivity.items()}).align(spectral_shape)

        magenta_log_sensitivity = {448.8421: -2.9950, 449.3166: -1.7572, 451.6894: -1.6969, 458.1435: -1.6264,
                                   463.4586: -1.5499, 470.5771: -1.4392, 477.9803: -1.3517, 485.5733: -1.1937,
                                   492.9765: -1.0730, 504.3660: -0.9623, 510.2506: -0.9260, 516.7046: -0.9119,
                                   521.9248: -0.9099, 529.5178: -0.8465, 538.5345: -0.6302, 542.3311: -0.4692,
                                   544.7039: -0.3082, 546.6021: -0.2025, 549.4495: -0.1723, 552.2969: -0.2679,
                                   556.7578: -0.5255, 558.1815: -0.6453, 564.4457: -0.9824, 569.1913: -1.2289,
                                   576.0251: -1.5862, 581.2453: -1.8679, 584.0926: -2.0843, 586.4655: -2.4616,
                                   588.5535: -2.7384, 589.7874: -2.9950}
        self.magenta_sensitivity = colour.SpectralDistribution(
            {x: 10 ** y for x, y in magenta_log_sensitivity.items()}).align(spectral_shape)

        cyan_log_sensitivity = {583.6181: -3.0050, 587.4146: -2.5975, 588.8383: -2.4818, 591.2111: -2.4365,
                                594.5330: -2.3912, 599.2787: -2.3711, 604.4989: -2.2956, 614.1800: -2.1316,
                                624.4305: -1.9484, 629.6507: -1.8629, 638.6674: -1.7874, 647.6841: -1.7723,
                                656.7008: -1.7019, 666.6667: -1.4956, 675.2088: -1.3094, 684.4153: -1.1454,
                                689.9203: -1.0931, 693.7168: -1.0830, 700.3607: -1.1384, 707.4791: -1.3497,
                                713.2688: -1.5479, 717.4450: -1.7371, 722.1906: -2.1296, 725.9871: -2.5623,
                                727.8853: -2.8189, 729.7836: -2.9950}
        self.cyan_sensitivity = colour.SpectralDistribution(
            {x: 10 ** y for x, y in cyan_log_sensitivity.items()}).align(spectral_shape)

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

        self.yellow_spectral_density = colour.SpectralDistribution(yellow_spectral_density).align(spectral_shape).align(
            spectral_shape)
        self.magenta_spectral_density = colour.SpectralDistribution(magenta_spectral_density).align(
            spectral_shape).align(spectral_shape)
        self.cyan_spectral_density = colour.SpectralDistribution(cyan_spectral_density).align(spectral_shape).align(
            spectral_shape)
        self.midscale_spectral_density = colour.SpectralDistribution(midscale_spectral_density).align(
            spectral_shape).align(spectral_shape)

        self.calibrate(3200, 'print')


def arri_to_spectral(rgb):
    arri_wcg_linear = colour.models.log_decoding_ARRILogC3(rgb)

    XYZ = colour.RGB_to_XYZ(arri_wcg_linear, 'ARRI Wide Gamut 3')

    # TODO: better gammut limiting
    # rec2020 = colour.XYZ_to_RGB(XYZ, 'sRGB')
    # rec2020 = np.clip(rec2020, 0, None)

    # XYZ = colour.RGB_to_XYZ(rec2020, 'sRGB')

    spectral = colour.XYZ_to_sd(XYZ, method='Otsu 2018').align(spectral_shape)

    return spectral


def normalize_spectral(spectral, target_flux=18):
    spectral *= target_flux / colour.sd_to_XYZ(spectral)[1]

    return spectral


def kelvin_to_spectral(kelvin):
    spectral = colour.sd_blackbody(kelvin, spectral_shape)
    spectral = normalize_spectral(spectral)

    return spectral


def arri_to_film_sRGB(arri, negative: FilmSpectral, print_film: FilmSpectral, print_light, projection_light, norm=1.,
                      linear=False):
    spectral_input = arri_to_spectral(arri)

    printer_lights = negative.spectral_to_projection(spectral_input, print_light)

    projection = print_film.spectral_to_projection(printer_lights, projection_light) / norm

    XYZ = colour.sd_to_XYZ(projection, k=0.01)

    sRGB = colour.XYZ_to_sRGB(XYZ)

    if linear:
        return sRGB ** (1 / 2.2)

    return np.clip(sRGB, 0, 1)


def arri_to_sRGB(arri):
    arri_wcg_linear = colour.models.log_decoding_ARRILogC3(arri)

    XYZ = colour.RGB_to_XYZ(arri_wcg_linear, 'ARRI Wide Gamut 3')

    sRGB = colour.XYZ_to_sRGB(XYZ)

    sRGB = np.clip(sRGB, 0, 1)

    return sRGB


if __name__ == '__main__':
    start = time.time()

    kodak_5207 = Kodak_5207()
    kodak_2383 = Kodak_2383()
    for print_light in [5000]:
        print_spectral = kelvin_to_spectral(print_light)
        for projection_light in [5400]:
            projection_spectral = kelvin_to_spectral(projection_light)
            rgb = arri_to_film_sRGB(np.ones(3), kodak_5207, kodak_2383, print_light, projection_light,
                                           linear=True)
            norm = max(rgb)

            lut = colour.LUT3D(size=16, name=f"{print_light}-{projection_light}")

            convert = lambda x: arri_to_film_sRGB(x, kodak_5207, kodak_2383, print_spectral, projection_spectral, norm)

            lut.table = np.apply_along_axis(convert, 3, lut.table)

            colour.io.write_LUT(lut, f"{print_light}-{projection_light}.cube")
            print(f"{print_light}-{projection_light}.cube", end=" ")
    end = time.time()
    print(f"\nfinished in {end - start:.2f}s")
