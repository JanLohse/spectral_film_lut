from spectral_film_lut.film_spectral import *


class FujiEterna500Vivid(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.iso = 500
        self.density_measure = 'status_m'
        self.exposure_kelvin = 3200
        self.projection_kelvin = 5500

        # spectral sensitivity
        red_log_sensitivity = {584.6949: 0.6932, 589.0259: 0.8656, 595.4765: 1.0114, 600.9873: 1.0802, 604.7466: 1.1067,
                               613.9740: 1.1081, 624.0557: 1.1204, 635.1627: 1.1040, 641.2716: 1.1408, 647.1669: 1.1319,
                               652.9340: 1.0809, 658.0603: 0.9440, 661.6487: 0.7635, 664.8954: 0.5966}
        green_log_sensitivity = {491.8396: 0.8656, 497.4785: 1.0407, 507.2185: 1.2539, 517.0440: 1.3969,
                                 527.0830: 1.5535, 539.4716: 1.7511, 544.8115: 1.8192, 551.4330: 1.8498,
                                 560.1904: 1.8260, 568.1362: 1.7824, 573.2198: 1.7238, 576.8937: 1.6148,
                                 582.4044: 1.3424, 586.4628: 1.0563, 591.2046: 0.7171}
        blue_log_sensitivity = {392.5173: 1.0883, 398.2417: 1.5093, 403.1544: 1.7633, 408.0671: 1.9118,
                                415.5857: 2.0031, 424.3431: 2.0337, 436.9453: 2.0112, 447.2834: 1.9799,
                                455.8700: 1.9772, 462.1497: 2.0712, 468.0450: 2.1978, 470.9072: 2.2687,
                                474.0684: 2.2796, 476.8024: 2.2346, 478.3830: 2.1767, 481.1598: 2.0848,
                                489.0628: 1.5161, 495.4707: 1.1108, 503.0320: 0.7253}
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry - characteristic curve
        red_curve = {-6.2116: 0.2655, -5.6207: 0.2655, -5.0079: 0.2742, -4.1325: 0.2982, -3.3520: 0.3460,
                     -2.5204: 0.4353, -1.5939: 0.5550, -0.5653: 0.7051, 0.3319: 0.8422, 1.1343: 0.9684, 1.7252: 1.0598,
                     2.3161: 1.1295, 3.1112: 1.2165, 3.9647: 1.3123, 4.7526: 1.3993, 5.5988: 1.4995, 6.3210: 1.5822}
        green_curve = {-6.2407: 0.5375, -5.4237: 0.5484, -4.6213: 0.5680, -3.9939: 0.6028, -3.3374: 0.6649,
                       -2.3380: 0.8205, -1.3204: 1.0000, -0.1021: 1.2318, 0.8827: 1.4091, 2.2030: 1.6366,
                       3.2681: 1.7965, 4.0559: 1.9140, 5.0590: 2.0577, 6.2590: 2.2209}
        blue_curve = {-6.2407: 0.6725, -5.1903: 0.6834, -4.4170: 0.7040, -3.9210: 0.7334, -3.2134: 0.7943,
                      -2.3818: 0.8945, -1.6523: 1.0054, -0.7331: 1.1839, 0.1167: 1.3460, 0.7842: 1.4733, 1.4371: 1.5898,
                      2.1228: 1.7095, 2.9252: 1.8379, 3.6000: 1.9325, 4.3112: 2.0337, 4.9714: 2.1197, 5.5149: 2.1839,
                      6.0365: 2.2437, 6.2480: 2.2644}
        red_log_exposure = xp.array(list(red_curve.keys()), dtype=default_dtype)
        red_density_curve = xp.array(list(red_curve.values()), dtype=default_dtype)
        green_log_exposure = xp.array(list(green_curve.keys()), dtype=default_dtype)
        green_density_curve = xp.array(list(green_curve.values()), dtype=default_dtype)
        blue_log_exposure = xp.array(list(blue_curve.keys()), dtype=default_dtype)
        blue_density_curve = xp.array(list(blue_curve.values()), dtype=default_dtype)
        self.log_exposure = [red_log_exposure, green_log_exposure, blue_log_exposure]
        self.density_curve = [red_density_curve, green_density_curve, blue_density_curve]

        self.exposure_base = 2

        # spectral dye density
        midscale_sd = {412.4857: 1.2681, 418.5576: 1.2853, 426.7674: 1.3644, 434.0365: 1.4196, 441.4766: 1.4474,
                       449.6009: 1.4421, 458.2383: 1.4117, 467.6454: 1.3542, 476.0262: 1.2862, 481.5849: 1.2429,
                       487.4857: 1.2072, 492.7879: 1.1935, 501.4253: 1.2129, 508.9510: 1.2580, 513.4835: 1.2831,
                       524.0878: 1.2915, 536.8301: 1.3154, 543.5006: 1.3216, 551.1117: 1.2951, 561.8871: 1.1820,
                       571.6363: 1.0155, 579.2474: 0.8860, 586.0889: 0.7933, 592.5029: 0.7314, 601.9954: 0.6833,
                       609.6066: 0.6692, 623.5462: 0.7032, 641.1631: 0.7959, 652.8791: 0.8507, 666.0490: 0.8816,
                       681.1003: 0.8746, 692.5599: 0.8485, 698.6317: 0.8264}
        minimum_sd = {410.6899: 0.8640, 414.6237: 0.8441, 423.7814: 0.8137, 432.0957: 0.8112, 441.3008: 0.8020,
                      458.2264: 0.7499, 480.2000: 0.6931, 492.0776: 0.6732, 507.5185: 0.6891, 513.8137: 0.6811,
                      522.0686: 0.6425, 528.3044: 0.6137, 546.7147: 0.5790, 559.1861: 0.5551, 573.3205: 0.4870,
                      586.9798: 0.4149, 598.9761: 0.3664, 614.4170: 0.3388, 632.2335: 0.3358, 644.1111: 0.3315,
                      663.1153: 0.2882, 683.6041: 0.2189, 699.5795: 0.1750}
        self.d_ref_sd = colour.SpectralDistribution(midscale_sd)
        self.d_min_sd = colour.SpectralDistribution(minimum_sd)

        self.mtf = [{1.9915: 1.0074, 4.9217: 0.9951, 7.5584: 0.9526, 9.8776: 0.9016, 12.1630: 0.8373, 16.0988: 0.7191,
                     20.7544: 0.6045, 27.6106: 0.4806, 37.1082: 0.3703, 37.1713: 0.3709, 47.2737: 0.2934}]

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
        self.rms = 3.5

        self.calibrate()
