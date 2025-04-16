from spectral_film_lut.film_spectral import *


class Fuji3513DI(FilmSpectral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lad = [1.09, 1.06, 1.03]
        self.density_measure = 'status_a'
        self.exposure_kelvin = 2854
        self.projection_kelvin = 6500

        # spectral sensitivity
        red_log_sensitivity = {361.2142: 1.7802, 370.3606: 1.7095, 379.7197: 1.5734, 394.2263: 1.3515,
                                    406.2655: 1.1230, 417.7942: 0.8662, 428.6422: 0.5152, 434.1726: 0.3275,
                                    441.8300: 0.1153, 630.0753: 0.1534, 639.6471: 0.2840, 649.2189: 0.3738,
                                    659.3863: 0.4385, 668.3200: 0.5734, 672.7017: 0.6213, 676.7432: 0.6931,
                                    680.6995: 0.7492, 686.6553: 0.7802, 693.0365: 0.7775, 698.4392: 0.7421,
                                    704.3099: 0.6649, 710.8613: 0.5392, 717.5403: 0.3743, 720.9011: 0.2617,
                                    722.9430: 0.1567}
        green_log_sensitivity = {360.7534: 1.4760, 366.2483: 1.4352, 371.8850: 1.3191, 376.2100: 1.2157,
                                      381.3859: 1.1074, 385.7818: 1.0017, 389.8587: 0.9051, 394.6446: 0.8090,
                                      400.3877: 0.7306, 406.4498: 0.6186, 409.6759: 0.5102, 412.9374: 0.4137,
                                      416.9079: 0.2409, 419.4958: 0.1439, 422.0129: 0.0800, 425.6289: 0.0351,
                                      429.7058: 0.0478, 437.1505: 0.1158, 443.5317: 0.2065, 447.8213: 0.3012,
                                      454.5215: 0.4196, 461.9662: 0.5692, 467.6739: 0.7029, 470.8645: 0.7777,
                                      475.5086: 0.8303, 481.1098: 0.8866, 489.6181: 0.9954, 497.4173: 1.1314,
                                      503.2668: 1.2175, 507.7336: 1.2565, 515.1429: 1.2647, 520.2833: 1.2647,
                                      524.8565: 1.2774, 533.1166: 1.4252, 538.0089: 1.5802, 542.5466: 1.7244,
                                      547.2261: 1.8296, 549.3532: 1.8568, 551.4803: 1.8545, 552.3665: 1.8251,
                                      555.3444: 1.6555, 561.3002: 1.1337, 566.3697: 0.6689, 569.9148: 0.3470,
                                      570.9784: 0.1634}
        blue_log_sensitivity = {361.2851: 2.1307, 367.5245: 2.0840, 373.4094: 1.9665, 382.0240: 1.7843,
                                     387.6962: 1.6800, 392.9430: 1.5331, 397.6225: 1.4352, 402.0539: 1.3889,
                                     408.7896: 1.4034, 415.5253: 1.4751, 428.0041: 1.6859, 436.4769: 1.8414,
                                     440.1284: 1.9117, 446.0487: 1.9611, 451.7918: 2.0604, 458.4920: 2.2649,
                                     463.7034: 2.4766, 466.8230: 2.6208, 468.6665: 2.7001, 470.8290: 2.7373,
                                     472.7788: 2.7296, 475.4377: 2.6729, 480.4008: 2.3261, 485.7539: 1.8455,
                                     488.5546: 1.5621, 492.8087: 1.1813, 497.4528: 0.8004, 503.3377: 0.4445,
                                     507.8400: 0.1589}
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry
        red_curve = {-0.5964: 0.0796, 0.0972: 0.0823, 0.5095: 0.0982, 0.7165: 0.1587, 0.8490: 0.2467, 0.9996: 0.3961,
                     1.1018: 0.5499, 1.1767: 0.6958, 1.2295: 0.8328, 1.2784: 1.0008, 1.4145: 1.5356, 1.4920: 1.8981,
                     1.5629: 2.2252, 1.6281: 2.5567, 1.7360: 3.0474, 1.8223: 3.3347, 1.9016: 3.5575, 1.9782: 3.7007,
                     2.0610: 3.8006, 2.1302: 3.8430, 2.1949: 3.8581}
        green_curve = {-0.6021: 0.0919, 0.1248: 0.1033, 0.3034: 0.1139, 0.4334: 0.1431, 0.5987: 0.2137, 0.7504: 0.3355,
                       0.8742: 0.5050, 1.1145: 1.0047, 1.2533: 1.4055, 1.3525: 1.7587, 1.4649: 2.1780, 1.6523: 2.9152,
                       1.7360: 3.2286, 1.8277: 3.5067, 1.9190: 3.7389, 1.9939: 3.9004, 2.0609: 3.9892}
        blue_curve = {0.0027: 0.1512, 0.2178: 0.1658, 0.4192: 0.1823, 0.5290: 0.2098, 0.6342: 0.2519, 0.8026: 0.3506,
                      0.9879: 0.5631, 1.1301: 0.8118, 1.2682: 1.1176, 1.3812: 1.4887, 1.5127: 2.0056, 1.6129: 2.4749,
                      1.7166: 3.0081, 1.8080: 3.4372, 1.8662: 3.6227, 1.9183: 3.7429, 1.9766: 3.8348, 2.0428: 3.8989,
                      2.0984: 3.9302, 2.1633: 3.9400}
        red_log_exposure = xp.array(list(red_curve.keys()), dtype=default_dtype)
        red_density_curve = xp.array(list(red_curve.values()), dtype=default_dtype)
        green_log_exposure = xp.array(list(green_curve.keys()), dtype=default_dtype)
        green_density_curve = xp.array(list(green_curve.values()), dtype=default_dtype)
        blue_log_exposure = xp.array(list(blue_curve.keys()), dtype=default_dtype)
        blue_density_curve = xp.array(list(blue_curve.values()), dtype=default_dtype)
        self.log_exposure = [red_log_exposure, green_log_exposure, blue_log_exposure]
        self.density_curve = [red_density_curve, green_density_curve, blue_density_curve]

        self.exposure_base = 10

        # spectral dye density
        midscale_sd = {401.0358: 0.9359, 404.4390: 0.9427, 407.6202: 0.9550, 410.3676: 0.9701,
                                     414.5726: 0.9968, 419.1010: 1.0335, 424.2116: 1.0756, 429.7751: 1.1036,
                                     435.1445: 1.1250, 440.4492: 1.1363, 447.8887: 1.1430, 455.0048: 1.1376,
                                     460.9564: 1.1243, 466.9728: 1.1036, 473.8947: 1.0629, 478.7466: 1.0218,
                                     484.6982: 0.9748, 492.8494: 0.9367, 499.9654: 0.9207, 506.4346: 0.9220,
                                     514.3270: 0.9411, 522.6722: 0.9781, 529.6589: 1.0255, 535.4811: 1.0656,
                                     541.0446: 1.0969, 544.2792: 1.1136, 548.1607: 1.1243, 552.0422: 1.1263,
                                     556.3765: 1.1176, 559.9345: 1.1043, 565.4333: 1.0709, 573.3904: 1.0018,
                                     577.1425: 0.9694, 583.5469: 0.9357, 590.5336: 0.9187, 599.0729: 0.9194,
                                     607.3534: 0.9397, 615.7633: 0.9868, 620.8093: 1.0352, 627.2137: 1.0963,
                                     634.5239: 1.1620, 646.8153: 1.2721, 651.8612: 1.2998, 659.4301: 1.3205,
                                     666.3521: 1.3212, 674.3739: 1.3002, 682.8485: 1.2388, 699.6683: 1.0609}
        red_sd = {400.6639: 0.3877, 408.7503: 0.3360, 422.9825: 0.2826, 437.2146: 0.2242,
                                454.0344: 0.1641, 470.2073: 0.1257, 492.5259: 0.1104, 517.7556: 0.1174,
                                540.0742: 0.1614, 559.1582: 0.2392, 573.0669: 0.3410, 588.2694: 0.4828,
                                609.9411: 0.7331, 626.9550: 0.9407, 640.6696: 1.1036, 646.4918: 1.1570,
                                653.2844: 1.1971, 661.6943: 1.2204, 669.4573: 1.2204, 679.1611: 1.1920,
                                686.6006: 1.1336, 699.2155: 1.0102}
        green_sd = {400.3404: 0.2092, 405.8392: 0.1891, 413.6022: 0.1841, 422.4650: 0.1962,
                                  431.3924: 0.1992, 440.4492: 0.1928, 446.5949: 0.1791, 454.6814: 0.1758,
                                  464.1910: 0.1968, 475.0592: 0.2499, 480.4286: 0.2899, 485.8627: 0.3276,
                                  499.2538: 0.4892, 503.7822: 0.5449, 509.3457: 0.5896, 514.3917: 0.6370,
                                  521.3137: 0.7331, 526.1655: 0.8032, 533.9285: 0.8700, 539.7508: 0.9067,
                                  545.5730: 0.9267, 550.6189: 0.9254, 555.9236: 0.9000, 563.2338: 0.8066,
                                  583.8057: 0.5246, 596.0971: 0.3537, 601.0784: 0.2919, 608.9707: 0.2285,
                                  622.1031: 0.1735, 639.0523: 0.1307, 660.4005: 0.0944, 684.0129: 0.0677,
                                  698.8920: 0.0573}
        blue_sd = {400.7933: 0.5042, 406.7449: 0.5322, 413.9904: 0.6077, 420.9124: 0.6998,
                                 430.6808: 0.7952, 437.6675: 0.8553, 443.6838: 0.8900, 451.7702: 0.9084,
                                 459.2098: 0.9040, 466.6493: 0.8700, 470.8542: 0.8299, 487.6094: 0.6197,
                                 502.1650: 0.4211, 510.6396: 0.3110, 522.3487: 0.2222, 535.2223: 0.1591,
                                 553.3360: 0.1024, 577.9188: 0.0630, 605.0892: 0.0590, 634.2004: 0.0690,
                                 668.1635: 0.0707, 698.8920: 0.0573}
        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]
        self.d_ref_sd = colour.SpectralDistribution(midscale_sd)

        self.calibrate()
