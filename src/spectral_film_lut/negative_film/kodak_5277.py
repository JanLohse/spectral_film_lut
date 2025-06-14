from spectral_film_lut.film_spectral import *


class Kodak5277(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.iso = 320
        self.density_measure = 'status_m'
        self.exposure_kelvin = 3200

        # spectral sensitivity
        red_log_sensitivity = {503.0808: 0.4796, 509.0161: 0.5984, 516.2231: 0.7089, 525.5499: 0.7572, 534.8767: 0.7573,
                               543.3557: 0.7960, 549.5332: 0.8815, 556.0740: 1.0022, 562.0093: 1.1229, 567.0966: 1.2178,
                               570.9121: 1.3178, 573.8798: 1.4197, 576.4234: 1.5189, 578.9671: 1.6212, 583.6305: 1.7301,
                               587.4460: 1.8630, 595.9249: 1.9873, 604.4038: 2.0884, 613.7307: 2.1336, 623.0575: 2.1697,
                               631.1124: 2.2654, 638.3195: 2.3653, 646.7984: 2.4605, 654.3352: 2.4771, 659.9407: 2.3917,
                               662.4844: 2.2936, 664.0389: 2.1838, 664.6041: 2.0950, 666.2347: 1.9918, 667.4304: 1.8062,
                               668.4196: 1.6866, 669.2675: 1.5851, 670.1154: 1.4858, 670.9633: 1.3888, 671.9525: 1.2677,
                               673.3657: 1.1459, 675.5561: 0.9896, 676.4746: 0.8338, 678.1704: 0.6984,
                               679.4422: 0.5992, }
        green_log_sensitivity = {444.1524: 0.7676, 453.4792: 0.8488, 462.8060: 0.9348, 470.4370: 1.0018,
                                 475.1004: 1.1124, 478.0680: 1.2407, 480.1877: 1.3595, 481.8835: 1.4633,
                                 483.5793: 1.5671, 485.6990: 1.6859, 488.2427: 1.8077, 490.7864: 1.9109,
                                 494.6019: 2.0047, 500.4900: 2.1088, 507.7442: 2.2053, 514.5274: 2.3032,
                                 520.4626: 2.4051, 526.8218: 2.5039, 534.4528: 2.6072, 543.3557: 2.6693,
                                 552.6825: 2.6361, 562.0093: 2.5795, 570.0642: 2.5254, 575.1516: 2.4222,
                                 578.1192: 2.3071, 580.2389: 2.1920, 581.5108: 2.0702, 583.8425: 1.9738,
                                 584.2664: 1.7368, 586.7394: 1.8566, 587.4460: 1.5535, 589.1418: 1.4080,
                                 590.8376: 1.2749, 592.5334: 1.1594, 598.2566: 0.8406, 599.5991: 0.7323,
                                 601.4362: 0.6172, 606.0996: 0.3803, 608.6433: 0.2495, }
        blue_log_sensitivity = {372.0816: 2.4389, 376.7450: 2.5447, 382.6802: 2.6506, 389.4634: 2.7487,
                                396.0648: 2.8570, 400.9099: 2.9625, 408.5409: 3.0433, 417.8677: 3.0607,
                                427.1945: 3.0382, 436.5213: 2.9989, 445.8481: 2.9623, 455.1749: 2.9220,
                                464.5018: 2.9327, 472.5567: 2.9072, 477.2201: 2.7869, 480.1877: 2.6816,
                                484.0033: 2.5706, 488.2427: 2.4596, 492.4822: 2.3468, 497.0608: 2.2304,
                                500.6431: 2.1192, 506.7267: 1.9032, 509.4400: 1.7957, 511.9837: 1.6814,
                                514.5274: 1.5610, 516.6471: 1.4407, 518.3429: 1.3392, 520.0386: 1.2286,
                                521.3105: 1.1203, 522.3703: 0.9892, 524.7020: 0.8722, 526.0895: 0.7465,
                                528.5176: 0.6059, 529.3655: 0.5292, }
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry - characteristic curve
        red_curve = {-3.8117: 0.1220, -3.6436: 0.1332, -3.4756: 0.1466, -3.3076: 0.1642, -3.1396: 0.1843,
                     -2.9716: 0.2130, -2.8036: 0.2542, -2.6356: 0.3085, -2.4675: 0.3703, -2.2995: 0.4446,
                     -2.1315: 0.5068, -1.9635: 0.5800, -1.7955: 0.6552, -1.6275: 0.7322, -1.4595: 0.8104,
                     -1.2914: 0.8909, -1.1234: 0.9722, -0.9554: 1.0418, -0.7874: 1.1265, -0.6194: 1.1987,
                     -0.4514: 1.2657, -0.2834: 1.3261, -0.1153: 1.3773, -0.0193: 1.4022, }
        green_curve = {-3.8037: 0.5631, -3.6356: 0.5737, -3.4676: 0.5874, -3.2996: 0.6049, -3.1316: 0.6255,
                       -2.9636: 0.6565, -2.7956: 0.7022, -2.6276: 0.7614, -2.4595: 0.8296, -2.2915: 0.9066,
                       -2.1235: 0.9885, -1.9555: 1.0601, -1.7875: 1.1465, -1.6195: 1.2339, -1.4515: 1.3237,
                       -1.2834: 1.4153, -1.1154: 1.5053, -0.9474: 1.5961, -0.7794: 1.6834, -0.6114: 1.7638,
                       -0.4434: 1.8387, -0.2754: 1.9065, -0.1073: 1.9700, }
        blue_curve = {-3.8037: 0.8928, -3.6356: 0.9040, -3.4676: 0.9175, -3.2996: 0.9349, -3.1316: 0.9670,
                      -2.9636: 0.9913, -2.7956: 1.0302, -2.6276: 1.0928, -2.4595: 1.1599, -2.2915: 1.2336,
                      -2.1235: 1.3119, -1.9555: 1.3918, -1.7875: 1.4832, -1.6195: 1.5633, -1.4515: 1.6548,
                      -1.2834: 1.7451, -1.1154: 1.8369, -0.9474: 1.9310, -0.7794: 2.0117, -0.6114: 2.0919,
                      -0.4434: 2.1637, -0.2754: 2.2325, -0.1073: 2.2946, -0.0153: 2.3264, }
        red_log_exposure = xp.array(list(red_curve.keys()), dtype=default_dtype)
        red_density_curve = xp.array(list(red_curve.values()), dtype=default_dtype)
        green_log_exposure = xp.array(list(green_curve.keys()), dtype=default_dtype)
        green_density_curve = xp.array(list(green_curve.values()), dtype=default_dtype)
        blue_log_exposure = xp.array(list(blue_curve.keys()), dtype=default_dtype)
        blue_density_curve = xp.array(list(blue_curve.values()), dtype=default_dtype)
        self.log_exposure = [red_log_exposure, green_log_exposure, blue_log_exposure]
        self.density_curve = [red_density_curve, green_density_curve, blue_density_curve]

        # spectral dye density
        red_sd = {369.9627: 0.2827, 388.4482: 0.2428, 410.9151: 0.1455, 431.5192: 0.0576, 448.5258: 0.0156,
                  467.2423: 0.0013, 488.6056: -0.0019, 509.6958: 0.0107, 526.1646: 0.0349, 544.8041: 0.0958,
                  563.7825: 0.1675, 580.2345: 0.2582, 589.9394: 0.3361, 597.7957: 0.4145, 605.1899: 0.4942,
                  615.8190: 0.5972, 625.0617: 0.6781, 636.1530: 0.7720, 649.1955: 0.8659, 661.5704: 0.9382,
                  676.5688: 0.9885, 696.9868: 0.9716, 714.7160: 0.8689, 723.9588: 0.7871, 734.2798: 0.6788,
                  741.9820: 0.5966, 749.3762: 0.5150, 756.3083: 0.4446, 765.4189: 0.3533, 770.4804: 0.2926,
                  792.8170: 0.1553, }
        green_sd = {367.0359: 0.0956, 382.7485: 0.0651, 398.1530: 0.0406, 414.2438: 0.0175, 430.1944: 0.0047,
                    446.7613: -0.0204, 458.9061: 0.0176, 469.0138: 0.0931, 475.7918: 0.1691, 480.8753: 0.2301,
                    485.4967: 0.2971, 492.0986: 0.3747, 494.7394: 0.4452, 498.8986: 0.5146, 503.0579: 0.5825,
                    507.6792: 0.6559, 512.8887: 0.7325, 518.9553: 0.8242, 524.7782: 0.8934, 531.4462: 0.9572,
                    543.6547: 0.9965, 554.9711: 0.9388, 560.1316: 0.8676, 564.9840: 0.7971, 569.6054: 0.7330,
                    574.2268: 0.6644, 578.8481: 0.6003, 583.9316: 0.5345, 590.7096: 0.4585, 598.4427: 0.4035,
                    608.7945: 0.3395, 623.0591: 0.2957, 639.0182: 0.2741, 654.3023: 0.2716, 669.4267: 0.2756,
                    685.0553: 0.2702, 700.6839: 0.2591, 716.4105: 0.2356, 730.9421: 0.1991, 745.9872: 0.1607,
                    762.0079: 0.1172, }
        blue_sd = {367.6521: 0.2806, 378.2812: 0.2828, 388.9103: 0.3689, 395.2262: 0.4285, 401.8501: 0.5129,
                   408.7822: 0.6128, 412.9414: 0.6735, 418.0249: 0.7731, 425.4190: 0.8369, 434.1996: 0.9487,
                   443.5964: 0.9944, 458.6928: 0.9541, 469.3219: 0.8227, 476.2540: 0.7092, 480.4132: 0.6457,
                   485.4967: 0.5618, 489.6559: 0.4993, 496.1258: 0.4004, 503.3659: 0.3099, 512.7627: 0.2275,
                   534.0209: 0.1483, 546.2142: 0.1225, 566.3044: 0.0787, 577.4617: 0.0616, 601.1847: 0.0335,
                   613.5083: 0.0259, 638.1555: 0.0202, 650.9413: 0.0202, 676.0507: 0.0103, 688.8364: 0.0092,
                   713.9458: 0.0091, 726.7316: 0.0092, 751.8409: 0.0060, 764.1646: 0.0016, 787.2714: 0.0010,
                   795.1277: -0.0002, }
        midscale_sd = {368.5763: 1.1276, 383.3647: 1.1541, 398.1530: 1.1356, 409.2443: 1.1607, 415.2521: 1.2264,
                       419.4113: 1.2898, 423.5705: 1.3531, 429.1161: 1.4187, 439.7452: 1.4684, 453.6093: 1.4457,
                       464.7006: 1.3834, 473.4812: 1.3231, 481.3375: 1.2629, 492.4287: 1.2032, 507.2171: 1.2190,
                       518.3083: 1.1897, 529.3996: 1.1260, 544.1879: 1.1008, 557.1277: 1.0558, 566.8326: 0.9990,
                       574.2268: 0.9418, 580.2345: 0.8811, 585.3180: 0.8160, 589.9394: 0.7512, 595.0229: 0.6846,
                       603.3413: 0.6176, 616.2811: 0.5851, 631.0695: 0.5938, 645.8578: 0.6132, 660.6462: 0.6293,
                       675.4345: 0.6346, 690.2228: 0.6236, 705.0112: 0.5895, 719.3374: 0.5362, 731.8151: 0.4752,
                       742.4442: 0.4182, 752.6112: 0.3607, 763.2403: 0.3027, 775.7180: 0.2444, 789.5820: 0.1846, }
        minimum_sd = {367.6521: 0.8140, 382.4404: 0.8236, 393.5317: 0.7778, 404.6229: 0.7312, 416.6385: 0.7641,
                      425.8812: 0.8312, 437.8967: 0.8856, 452.6851: 0.8740, 467.4734: 0.8280, 481.7996: 0.7788,
                      496.1258: 0.7303, 510.9142: 0.7253, 520.6190: 0.6533, 525.7025: 0.5848, 535.8695: 0.5302,
                      550.6578: 0.5092, 565.4462: 0.5112, 578.3860: 0.4882, 586.2423: 0.4242, 593.2099: 0.3194,
                      589.0151: 0.3853, 603.8035: 0.2274, 616.2811: 0.1799, 631.0695: 0.1577, 645.8578: 0.1463,
                      660.6462: 0.1379, 675.4345: 0.1295, 690.2228: 0.1226, 705.0112: 0.1167, 719.7995: 0.1105,
                      734.5879: 0.1039, 749.3762: 0.0945, 760.4675: 0.0878, 793.2791: 0.0733, }
        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]
        self.d_ref_sd = colour.SpectralDistribution(midscale_sd)
        self.d_min_sd = colour.SpectralDistribution(minimum_sd)

        self.mtf = [{2.5429: 1.0143, 2.8615: 1.0067, 3.2200: 0.9961, 3.6235: 0.9945, 4.0776: 0.9881, 4.5885: 0.9790,
                     5.1634: 0.9642, 5.8104: 0.9596, 6.5384: 0.9504, 7.3577: 0.9449, 8.2796: 0.9388, 9.3171: 0.9299,
                     10.4845: 0.9205, 11.7982: 0.9108, 13.2766: 0.8944, 14.9401: 0.8732, 16.8122: 0.8435,
                     18.9187: 0.8100, 21.2893: 0.7729, 23.9568: 0.7281, 26.9587: 0.6868, 30.3366: 0.6390,
                     34.1378: 0.5940, 38.4153: 0.5481, 43.2288: 0.5041, 48.6454: 0.4628, 54.7407: 0.4156,
                     61.5997: 0.3742, 69.3182: 0.3351, 78.0039: 0.3001, },
                    {2.5703: 1.1010, 2.8924: 1.1158, 3.2548: 1.1320, 3.6626: 1.1502, 4.1215: 1.1669, 4.6380: 1.1863,
                     5.2191: 1.2080, 5.8731: 1.2288, 6.6090: 1.2533, 7.4371: 1.2789, 8.3690: 1.2924, 9.4176: 1.3034,
                     10.5977: 1.3155, 11.9255: 1.3311, 13.4198: 1.3353, 15.1013: 1.3353, 16.9936: 1.3240,
                     19.1229: 1.2933, 21.5190: 1.2493, 24.2153: 1.1908, 27.2495: 1.1206, 30.6639: 1.0372,
                     34.5061: 0.9735, 38.8298: 0.8961, 43.6952: 0.8195, 49.1702: 0.7464, 55.3313: 0.6765,
                     62.2644: 0.6126, 70.0662: 0.5530, 78.8455: 0.5033, },
                    {2.5429: 1.0999, 2.8615: 1.1146, 3.2200: 1.1314, 3.6235: 1.1490, 4.0776: 1.1638, 4.5885: 1.1838,
                     5.1634: 1.2054, 5.8104: 1.2268, 6.5384: 1.2513, 7.3577: 1.2837, 8.2796: 1.3106, 9.3171: 1.3353,
                     10.4845: 1.3605, 11.7982: 1.3817, 13.2766: 1.4017, 14.9401: 1.4129, 16.8122: 1.4062,
                     18.9187: 1.3831, 21.2893: 1.3453, 23.9568: 1.2830, 26.9587: 1.2151, 30.3366: 1.1368,
                     34.1378: 1.0517, 38.4153: 0.9751, 43.2288: 0.8911, 48.6454: 0.8178, 54.7407: 0.7422,
                     61.5997: 0.6803, 69.3182: 0.6130, 77.5864: 0.5575, }]

        red_rms = {0.1050: 0.0050, 0.2316: 0.0053, 0.3582: 0.0057, 0.4848: 0.0062, 0.6113: 0.0069, 0.7379: 0.0079,
                   0.8645: 0.0089, 0.9911: 0.0094, 1.1176: 0.0096, 1.2442: 0.0095, 1.3708: 0.0088, 1.4855: 0.0079,
                   1.6793: 0.0067, 1.8059: 0.0063, 1.9325: 0.0063, 2.0590: 0.0063, 2.1856: 0.0064, 2.3122: 0.0065,
                   2.4071: 0.0065, 2.6840: 0.0064, 2.8106: 0.0064, 2.9372: 0.0064, 3.0637: 0.0063, 3.1903: 0.0062,
                   3.3169: 0.0062, 3.4435: 0.0061, 3.5700: 0.0059, 3.6966: 0.0058, 3.8232: 0.0057, 3.9142: 0.0056, }
        green_rms = {0.2316: 0.0109, 0.3582: 0.0112, 0.4848: 0.0116, 0.6113: 0.0122, 0.7379: 0.0133, 0.8645: 0.0142,
                     0.9911: 0.0146, 1.1176: 0.0142, 1.2442: 0.0137, 1.5053: 0.0118, 1.6318: 0.0109, 1.7505: 0.0102,
                     1.9720: 0.0095, 2.0986: 0.0092, 2.2252: 0.0089, 2.3517: 0.0087, 2.4783: 0.0086, 2.5970: 0.0084,
                     2.8976: 0.0081, 3.0242: 0.0079, 3.1508: 0.0077, 3.2773: 0.0075, 3.4039: 0.0073, 3.5305: 0.0072,
                     3.6571: 0.0070, 3.7836: 0.0068, 3.8983: 0.0066, }
        blue_rms = {0.1446: 0.0196, 0.2712: 0.0205, 0.3977: 0.0215, 0.5243: 0.0226, 0.6509: 0.0238, 0.7775: 0.0252,
                    0.9040: 0.0268, 1.0306: 0.0284, 1.1572: 0.0298, 1.2838: 0.0309, 1.4103: 0.0318, 1.5369: 0.0321,
                    1.6635: 0.0325, 1.7901: 0.0325, 1.9166: 0.0325, 2.0432: 0.0324, 2.1483: 0.0320, 2.2964: 0.0306,
                    2.4229: 0.0290, 2.5416: 0.0275, 2.7394: 0.0247, 2.8660: 0.0230, 2.9925: 0.0215, 3.1191: 0.0203,
                    3.2457: 0.0193, 3.3549: 0.0185, 3.5305: 0.0173, 3.6571: 0.0165, 3.7836: 0.0156, 3.8983: 0.0149, }
        red_rms_density = {0.0914: 0.1580, 0.2179: 0.1579, 0.3443: 0.1579, 0.4708: 0.1590, 0.5972: 0.1634,
                           0.7236: 0.1688, 0.8501: 0.1812, 0.9765: 0.1973, 1.1030: 0.2197, 1.2294: 0.2466,
                           1.3558: 0.2773, 1.4823: 0.3140, 1.6087: 0.3594, 1.7352: 0.4094, 1.8616: 0.4653,
                           1.9880: 0.5227, 2.1145: 0.5811, 2.2409: 0.6405, 2.3674: 0.6998, 2.6756: 0.8454,
                           2.8178: 0.9130, 2.9442: 0.9719, 3.0707: 1.0308, 3.1971: 1.0882, 3.3236: 1.1474,
                           3.4500: 1.1942, 3.5764: 1.2478, 3.7029: 1.2958, 3.8293: 1.3414, 3.9400: 1.3795, }
        green_rms_density = {0.0993: 0.5716, 0.2258: 0.5776, 0.3522: 0.5842, 0.4787: 0.5903, 0.6051: 0.5970,
                             0.7315: 0.6046, 0.8580: 0.6174, 0.9844: 0.6431, 1.1109: 0.6724, 1.2373: 0.7044,
                             1.3653: 0.7430, 1.4902: 0.7833, 1.6272: 0.8370, 1.7431: 0.8882, 1.8648: 0.9447,
                             1.9959: 1.0063, 2.1224: 1.0687, 2.2488: 1.1321, 2.3753: 1.1986, 2.5017: 1.2651,
                             2.6281: 1.3352, 2.7546: 1.4014, 2.8810: 1.4699, 3.0075: 1.5377, 3.1339: 1.6031,
                             3.2603: 1.6672, 3.3868: 1.7279, 3.5132: 1.7861, 3.6397: 1.8416, 3.7661: 1.8959,
                             3.8925: 1.9470, 3.9676: 1.9754, }
        blue_rms_density = {0.0914: 0.9159, 0.2179: 0.9178, 0.3443: 0.9221, 0.4708: 0.9245, 0.5972: 0.9285,
                            0.7236: 0.9337, 0.8501: 0.9413, 0.9765: 0.9567, 1.1030: 0.9832, 1.2294: 1.0160,
                            1.3406: 1.0482, 1.4823: 1.0945, 1.6087: 1.1438, 1.7352: 1.1995, 1.8616: 1.2604,
                            1.9880: 1.3230, 2.1105: 1.3857, 2.2804: 1.4758, 2.4069: 1.5433, 2.5333: 1.6111,
                            2.6597: 1.6798, 2.7862: 1.7487, 2.9126: 1.8168, 3.0391: 1.8842, 3.1655: 1.9493,
                            3.2919: 2.0124, 3.4184: 2.0709, 3.5448: 2.1263, 3.6713: 2.1806, 3.7977: 2.2332,
                            3.9241: 2.2862, 3.9913: 2.3152, }
        self.rms_curve = [red_rms, green_rms, blue_rms]
        self.rms_density = [red_rms_density, green_rms_density, blue_rms_density]

        self.calibrate()
