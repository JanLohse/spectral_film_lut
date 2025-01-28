import colour
import numpy as np
from rdkit.sping.colors import green

from film_spectral import FilmSpectral


class Kodak5207(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.iso = 250
        self.density_measure = 'status_m'
        self.white_xy = [0.3127, 0.329]
        self.white_sd = colour.SDS_ILLUMINANTS['D65']

        # spectral sensitivity
        self.red_log_sensitivity = {530.0364: 0.0532, 535.5542: 0.0346, 542.0752: 0.0346, 549.5994: 0.0930,
                                    558.8793: 0.3000, 564.6479: 0.4061, 568.7611: 0.4932, 581.2012: 1.2369,
                                    587.2206: 1.5129, 591.7351: 1.6881, 597.7545: 1.8181, 610.5457: 2.0570,
                                    620.3272: 2.1950, 628.6038: 2.2109, 637.6329: 2.2746, 646.4112: 2.3649,
                                    651.2267: 2.3845, 655.6911: 2.3835, 658.1490: 2.3654, 662.7137: 2.2640,
                                    669.9871: 2.0013, 676.5081: 1.7252, 680.2702: 1.5660, 686.2896: 1.3563,
                                    689.2993: 1.1891, 694.2653: 0.7379, 699.6828: 0.1657}
        self.green_log_sensitivity = {399.8672: 0.6424, 439.9964: 0.3451, 449.0255: 0.3876, 460.0611: 0.5468,
                                      467.0837: 0.5734, 472.0998: 0.6583, 491.1612: 1.5341, 496.6790: 1.7491,
                                      501.1935: 1.8526, 528.5316: 2.1818, 548.5962: 2.4127, 550.6026: 2.4121,
                                      566.1527: 2.2455, 575.9342: 2.1181, 579.6963: 2.0066, 586.7190: 1.5554,
                                      593.7416: 0.9502, 599.2593: 0.3398}
        self.blue_log_sensitivity = {359.4238: 1.5366, 367.7450: 1.6467, 381.7870: 2.0484, 386.9877: 2.1200,
                                     397.4932: 2.3368, 403.3785: 2.3941, 418.4270: 2.3516, 437.9900: 2.4259,
                                     449.7779: 2.3596, 460.5627: 2.4259, 467.5853: 2.4896, 472.3506: 2.4259,
                                     477.6176: 2.2348, 483.6370: 1.8978, 488.6531: 1.4704, 499.1871: 0.9980,
                                     506.4605: 0.7485, 519.7533: 0.0930}

        # sensiometry - characteristic curve
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

        # spectral dye density
        red_spectral_density = {399.6256: 0.2251, 410.6565: 0.1757, 423.8267: 0.1235, 437.2643: 0.0814,
                                 453.5767: 0.0500, 473.4991: 0.0349, 491.6834: 0.0292, 511.0710: 0.0326,
                                 530.9266: 0.0570, 548.5092: 0.1111, 566.7603: 0.1817, 578.9277: 0.2472,
                                 594.7052: 0.3889, 608.7445: 0.5280, 622.0484: 0.6583, 636.0209: 0.7800,
                                 649.6590: 0.8763, 661.4253: 0.9438, 667.8433: 0.9719, 673.6596: 0.9903,
                                 681.3478: 1.0020, 687.2309: 1.0013, 696.9916: 0.9812, 702.6073: 0.9535,
                                 709.4932: 0.9117, 720.0562: 0.8255, 729.2151: 0.7369, 737.9730: 0.6443,
                                 746.9983: 0.5407, 753.6168: 0.4678, 761.6393: 0.3876, 771.0657: 0.3040,
                                 777.0825: 0.2572, 785.3724: 0.2004, 792.6594: 0.1566, 800.1471: 0.1305}
        green_spectral_density = {399.6256: 0.0032, 416.0048: -0.0069, 420.3503: -0.0059, 425.1638: 0.0025,
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
        blue_spectral_density = {399.2913: 0.4979, 408.9852: 0.6483, 418.6790: 0.7920, 426.6346: 0.8937,
                                   433.7211: 0.9625, 438.6683: 0.9906, 442.6795: 1.0040, 445.7548: 1.0060,
                                   449.7660: 1.0006, 453.2424: 0.9859, 456.4514: 0.9619, 459.9947: 0.9334,
                                   462.7357: 0.8990, 466.3458: 0.8539, 470.4907: 0.7870, 477.8446: 0.6650,
                                   486.2682: 0.5246, 490.2126: 0.4544, 499.9064: 0.3174, 507.7283: 0.2305,
                                   519.9626: 0.1436, 538.0131: 0.0774, 552.0524: 0.0536, 567.4288: 0.0292,
                                   586.1479: 0.0142, 602.8613: 0.0058, 625.9259: 0.0018, 661.6927: -0.0035,
                                   674.3950: -0.0042, 698.4624: -0.0035, 725.5382: -0.0005, 753.6168: 0.0035,
                                   784.0353: 0.0052, 790.0521: 0.0038}
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

        self.red_spectral_density = colour.SpectralDistribution(red_spectral_density)
        self.green_spectral_density = colour.SpectralDistribution(green_spectral_density)
        self.blue_spectral_density = colour.SpectralDistribution(blue_spectral_density)
        self.midscale_spectral_density = colour.SpectralDistribution(midscale_spectral_density)
        self.minimum_spectral_density = colour.SpectralDistribution(minimum_spectral_density)

        self.calibrate()
