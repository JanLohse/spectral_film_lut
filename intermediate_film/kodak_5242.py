import colour
import numpy as np

from film_spectral import FilmSpectral
from utility import kelvin_to_spectral


class Kodak5242(FilmSpectral):
    def __init__(self, type: {'positive', 'negative'}):
        super().__init__()
        # spectral sensitivity
        yellow_log_sensitivity = {402.0385: 0.0268, 408.0804: 0.0214, 420.0915: 0.0284, 431.7386: 0.0561,
                                  441.2019: 0.0815, 449.9372: 0.1500, 458.1630: 0.2502, 462.7490: 0.2563,
                                  467.6991: 0.1893, 470.9020: 0.0815, 492.0124: -1.2569, 495.7977: -1.5110,
                                  511.5941: -2.3419}

        self.yellow_sensitivity = colour.SpectralDistribution(
            {x: 10 ** y for x, y in yellow_log_sensitivity.items()}).align(colour.SPECTRAL_SHAPE_DEFAULT)
        magenta_log_sensitivity = {461.9483: -2.1293, 481.0204: -1.7482, 490.1198: -1.5402, 495.7977: -1.4363,
                                   502.2036: -1.3585, 509.0463: -1.3015, 519.0919: -1.2315, 528.9192: -1.1013,
                                   536.7082: -0.9742, 541.5126: -0.8934, 546.1715: -0.8048, 548.9377: -0.7779,
                                   550.5391: -0.7702, 553.7421: -0.7964, 556.8723: -0.8557, 563.7877: -1.1129,
                                   568.1554: -1.3285, 572.0135: -1.5595, 576.4540: -1.9099, 581.8408: -2.3889}
        self.magenta_sensitivity = colour.SpectralDistribution(
            {x: 10 ** y for x, y in magenta_log_sensitivity.items()}).align(colour.SPECTRAL_SHAPE_DEFAULT)

        cyan_log_sensitivity = {582.3503: -2.6183, 591.0857: -2.4181, 598.3651: -2.2911, 604.1159: -2.2179,
                                612.5600: -2.1232, 626.8277: -2.0100, 641.6050: -1.8798, 653.2521: -1.7713,
                                659.7308: -1.7243, 668.3206: -1.6712, 674.1441: -1.6173, 680.6956: -1.5349,
                                683.5346: -1.4971, 686.9559: -1.4902, 691.2508: -1.5626, 696.9288: -1.8637,
                                702.4612: -2.2256}
        self.cyan_sensitivity = colour.SpectralDistribution(
            {x: 10 ** y for x, y in cyan_log_sensitivity.items()}).align(colour.SPECTRAL_SHAPE_DEFAULT)

        self.sensitivity_exposure_time = 1 / 50

        # sensiometry
        self.red_log_exposure = np.array(
            [-0.5024, -0.3699, -0.2388, -0.1442, -0.0050, 0.1099, 0.2181, 0.5736, 0.9480, 1.2791, 1.5454, 1.7887,
             1.8874, 1.9928, 2.1374, 2.2685, 2.3821, 2.4970, 2.5632, ])
        self.red_density_curve = np.array(
            [0.0878, 0.1068, 0.1324, 0.1581, 0.2216, 0.2959, 0.3838, 0.7081, 1.0709, 1.3851, 1.6419, 1.8682, 1.9527,
             2.0338, 2.1284, 2.2014, 2.2581, 2.3088, 2.3324])
        self.green_log_exposure = np.array(
            [-0.4956, -0.2645, -0.0712, 0.0599, 0.1654, 0.6371, 1.0088, 1.5400, 1.8252, 1.9820, 2.1415, 2.2956, 2.4375,
             2.5537])
        self.green_density_curve = np.array(
            [0.5777, 0.6081, 0.6568, 0.7264, 0.8041, 1.2500, 1.6081, 2.1149, 2.3919, 2.5203, 2.6520, 2.7568, 2.8412,
             2.8953, ])
        self.blue_log_exposure = np.array(
            [-0.4956, -0.2064, -0.0793, 0.0194, 0.0870, 0.2100, 0.3600, 0.6709, 1.0047, 1.3643, 1.8049, 1.9509, 2.1117,
             2.2523, 2.4023, ])
        self.blue_density_curve = np.array(
            [0.6446, 0.6980, 0.7351, 0.7757, 0.8176, 0.9020, 1.0405, 1.3514, 1.6892, 2.0541, 2.4899, 2.6385, 2.7703,
             2.8851, 2.9865])

        self.exposure_base = 10
        self.sensiometry_exposure_time = 1 / 50

        # spectral dye density
        if type == 'positive':
            yellow_spectral_density = {400.3030: 0.6215, 412.5730: 0.7774, 425.1461: 0.8997, 434.6895: 0.9652,
                                       443.1725: 1.0010, 452.4886: 1.0113, 460.2900: 1.0021, 468.9245: 0.9409,
                                       486.0420: 0.7216, 499.0695: 0.5565, 511.0366: 0.4656, 522.2463: 0.4537,
                                       529.6689: 0.4710, 535.6525: 0.4445, 549.2101: 0.3465, 564.8128: 0.3129,
                                       576.5527: 0.2685, 588.4441: 0.1938, 600.9414: 0.1256, 627.6780: 0.0855,
                                       685.9987: 0.0693, 729.1712: 0.0531, 744.3194: 0.0552, 750.3787: 0.0666}
            magenta_spectral_density = {399.6970: 0.1965, 427.7213: 0.2669, 448.5501: 0.2609, 465.9706: 0.3183,
                                        477.7862: 0.3995, 495.0552: 0.5863, 506.2649: 0.7352, 517.7018: 0.8759,
                                        524.2155: 0.9360, 533.0015: 0.9977, 538.3034: 1.0134, 543.9840: 0.9842,
                                        549.9675: 0.9111, 557.6931: 0.8402, 566.4791: 0.7243, 585.6416: 0.4401,
                                        594.3519: 0.3291, 603.8195: 0.2425, 614.6505: 0.1862, 627.9810: 0.1554,
                                        655.3235: 0.1332, 691.4521: 0.1272, 721.9758: 0.1072, 740.9111: 0.0920,
                                        750.2272: 0.0937}
            cyan_spectral_density = {400.1515: 0.4022, 419.3897: 0.3540, 440.9760: 0.3118, 469.7576: 0.2544,
                                     498.1606: 0.2344, 514.8236: 0.2588, 521.2616: 0.2804, 526.5635: 0.3183,
                                     530.7293: 0.3281, 539.8182: 0.3216, 545.4988: 0.3048, 551.1794: 0.3021,
                                     560.6470: 0.3275, 587.5352: 0.4320, 597.3815: 0.4829, 615.1807: 0.6090,
                                     632.9799: 0.7519, 648.2796: 0.8656, 664.7154: 0.9615, 675.2435: 1.0015,
                                     684.0294: 1.0123, 695.8451: 0.9939, 709.0998: 0.9263, 718.7189: 0.8489,
                                     733.6399: 0.6940, 749.8485: 0.5219}
            midscale_spectral_density = {399.3183: 1.2603, 408.4830: 1.4092, 423.7827: 1.6896, 430.9024: 1.7941,
                                         436.6587: 1.8341, 441.4304: 1.8552, 449.4590: 1.8444, 456.2757: 1.7962,
                                         469.7576: 1.6663, 482.7851: 1.5445, 491.4953: 1.5023, 499.4482: 1.5055,
                                         508.9158: 1.5624, 515.4296: 1.6311, 519.8225: 1.7096, 525.2759: 1.8341,
                                         529.0630: 1.9045, 532.2441: 1.9213, 536.2584: 1.8964, 541.1816: 1.8265,
                                         548.5285: 1.6961, 553.4516: 1.6409, 561.1015: 1.5580, 570.0390: 1.4227,
                                         582.6120: 1.1980, 590.9435: 1.0492, 599.1993: 0.9544, 607.4551: 0.9284,
                                         616.4683: 0.9425, 633.0556: 1.0416, 651.3850: 1.1764, 666.9877: 1.2706,
                                         678.1974: 1.3123, 691.6793: 1.3117, 685.2413: 1.3198, 702.0558: 1.2727,
                                         716.0680: 1.1520, 727.8078: 1.0167, 749.7728: 0.7666}
            minimum_spectral_density = {398.9396: 0.5565, 413.4819: 0.6366, 430.6752: 0.7113, 442.2636: 0.7346,
                                        452.1099: 0.7189, 463.6983: 0.6897, 476.9530: 0.6735, 495.1309: 0.6810,
                                        512.7786: 0.7298, 519.3681: 0.7882, 524.6700: 0.8759, 528.8357: 0.9295,
                                        531.8654: 0.9306, 535.5010: 0.9008, 541.7118: 0.7812, 547.9983: 0.6724,
                                        553.0729: 0.6377, 564.6613: 0.6161, 569.3573: 0.5987, 577.2344: 0.5267,
                                        585.5659: 0.4049, 595.1850: 0.2588, 604.5769: 0.1667, 615.1807: 0.1186,
                                        631.4651: 0.0882, 692.8154: 0.0861, 728.4138: 0.0763, 739.7749: 0.0801,
                                        749.7728: 0.1012}
            self.target_density = np.array([1.15, 1.60, 1.70])

        elif type == 'negative':
            yellow_spectral_density = {399.6815: 0.5145, 414.8364: 0.7468, 425.8181: 0.8868, 435.1526: 0.9691,
                                       445.6951: 0.9966, 455.7984: 0.9472, 461.9482: 0.8868, 472.4907: 0.7330,
                                       490.5008: 0.4310, 505.1065: 0.2416, 517.0767: 0.1263, 529.8155: 0.0522,
                                       552.7674: 0.0071, 587.3600: 0.0192, 660.3888: 0.0181, 677.4105: 0.0049}
            magenta_spectral_density = {400.6699: 0.0192, 429.2225: 0.0280, 442.9497: 0.0181, 453.9315: 0.0423,
                                        463.3758: 0.0939, 470.9532: 0.1565, 476.9932: 0.2196, 507.5225: 0.6809,
                                        516.6374: 0.8264, 525.9719: 0.9291, 534.5377: 0.9790, 539.0402: 0.9862,
                                        544.7507: 0.9691, 554.3049: 0.8923, 559.4663: 0.8253, 578.5746: 0.5024,
                                        594.1687: 0.2883, 601.5265: 0.2114, 615.9126: 0.1224, 635.1307: 0.0686,
                                        664.2324: 0.0357, 698.2759: 0.0165, 717.4940: 0.0060}
            cyan_spectral_density = {399.5717: 0.2169, 420.4371: 0.1054, 431.0894: 0.0686, 460.8500: 0.0099,
                                     490.7204: -0.0093, 507.4127: -0.0126, 521.6890: -0.0253, 535.7457: -0.0176,
                                     563.9688: 0.0906, 577.3666: 0.1702, 594.8276: 0.3459, 606.5781: 0.4766,
                                     631.9460: 0.7056, 655.6666: 0.8758, 668.9545: 0.9527, 679.0578: 0.9862,
                                     685.3174: 0.9955, 694.7617: 0.9845, 706.1827: 0.9384, 716.3958: 0.8538,
                                     733.4175: 0.6507, 749.8902: 0.4365}
            midscale_spectral_density = {400.1208: 1.0570, 418.1309: 1.3590, 427.0261: 1.4880, 431.6385: 1.5374,
                                         436.7999: 1.5638, 442.4006: 1.5726, 448.4406: 1.5611, 453.1627: 1.5369,
                                         459.2027: 1.4913, 473.9183: 1.3469, 482.0448: 1.2838, 491.0499: 1.2503,
                                         501.7022: 1.2898, 514.8803: 1.4002, 521.1399: 1.4403, 532.2315: 1.4688,
                                         538.9304: 1.4677, 551.1201: 1.4068, 560.5645: 1.2887, 586.7011: 0.8236,
                                         590.6545: 0.7676, 595.3767: 0.7226, 608.1155: 0.6776, 615.3635: 0.6836,
                                         625.2471: 0.7166, 633.1540: 0.7660, 661.3771: 0.9669, 670.2723: 1.0131,
                                         677.6301: 1.0339, 685.6468: 1.0350, 696.0795: 1.0070, 716.1761: 0.8813,
                                         729.2445: 0.7561, 749.8902: 0.5107}
            minimum_spectral_density = {400.2306: 0.5282, 419.1193: 0.6128, 439.8748: 0.6540, 455.5787: 0.6463,
                                        479.1895: 0.6375, 489.0731: 0.6452, 500.3844: 0.6671, 507.6323: 0.6776,
                                        515.0999: 0.6644, 534.8671: 0.5859, 545.0802: 0.5535, 555.8423: 0.5376,
                                        565.8357: 0.5255, 574.9506: 0.4777, 589.2269: 0.3322, 602.7345: 0.2196,
                                        595.0472: 0.2734, 619.7562: 0.1620, 636.2289: 0.1433, 659.2906: 0.1367,
                                        680.1559: 0.1290, 725.1812: 0.1175, 749.8902: 0.1181, }
            self.target_density = np.array([1.00, 1.45, 1.55])

        self.yellow_spectral_density = colour.SpectralDistribution(yellow_spectral_density).align(
            colour.SPECTRAL_SHAPE_DEFAULT)
        self.magenta_spectral_density = colour.SpectralDistribution(magenta_spectral_density).align(
            colour.SPECTRAL_SHAPE_DEFAULT)
        self.cyan_spectral_density = colour.SpectralDistribution(cyan_spectral_density).align(
            colour.SPECTRAL_SHAPE_DEFAULT)
        self.midscale_spectral_density = colour.SpectralDistribution(midscale_spectral_density).align(
            colour.SPECTRAL_SHAPE_DEFAULT)
        self.minimum_spectral_density = colour.SpectralDistribution(minimum_spectral_density).align(
            colour.SPECTRAL_SHAPE_DEFAULT)

        if type == 'negative':
            self.calibrate(kelvin_to_spectral(3200), kelvin_to_spectral(3200), 'intermediate_film')
        elif type == 'positive':
            self.calibrate(kelvin_to_spectral(3200), kelvin_to_spectral(3200), 'print_film')