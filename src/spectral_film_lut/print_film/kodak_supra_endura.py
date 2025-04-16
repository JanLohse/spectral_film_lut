from spectral_film_lut.film_spectral import *


class KodakSupraEndura(FilmSpectral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lad = [0.8, 0.8, 0.8]
        self.density_measure = 'status_a'
        self.exposure_kelvin = None
        self.projection_kelvin = 6500

        # spectral sensitivity
        red_log_sensitivity = {494.2737: -1.4187, 500.1397: -1.3608, 504.7486: -1.2962, 511.4525: -1.2517,
                               516.8994: -1.2383, 522.7654: -1.2517, 530.3073: -1.2160, 534.0782: -1.1893,
                               539.1061: -1.1114, 542.8771: -1.0557, 546.6480: -1.0512, 550.2095: -1.1114,
                               552.9330: -1.2116, 557.1229: -1.3341, 560.0559: -1.3987, 564.6648: -1.4187,
                               569.6927: -1.4521, 573.8827: -1.4143, 581.0056: -1.4053, 591.8994: -1.3408,
                               600.9078: -1.2873, 611.5922: -1.1782, 621.2291: -1.0624, 629.6089: -0.9688,
                               636.3128: -0.9220, 641.7598: -0.9131, 648.4637: -0.9265, 656.0056: -0.9332,
                               666.2709: -0.8597, 674.4413: -0.7461, 683.0307: -0.5679, 689.7346: -0.3808,
                               696.6480: -0.2339, 702.9330: -0.1670, 708.7989: -0.2027, 710.8939: -0.2695,
                               716.3408: -0.4254, 719.4832: -0.5546, 722.6257: -0.7194, 728.9106: -1.0757,
                               732.6816: -1.2762, 734.9860: -1.3987, 737.0810: -1.5256, 738.5475: -1.6058}
        green_log_sensitivity = {389.3156: -0.0200, 391.8296: -0.2561, 393.9246: -0.4209, 395.3911: -0.4588,
                                 400.4190: -0.4499, 405.8659: -0.4744, 410.0559: -0.5278, 413.1983: -0.6281,
                                 418.0168: -0.7038, 424.9302: -0.6949, 433.1006: -0.6080, 447.3464: -0.3898,
                                 455.9358: -0.2361, 463.4777: -0.1158, 473.3240: -0.0045, 479.8184: 0.1381,
                                 486.7318: 0.2962, 493.4358: 0.3920, 498.0447: 0.4833, 501.6061: 0.5167,
                                 505.1676: 0.5323, 511.4525: 0.5100, 516.0615: 0.4833, 519.8324: 0.4699,
                                 524.6508: 0.5033, 530.9358: 0.6147, 534.7067: 0.6815, 538.6872: 0.8129,
                                 543.2961: 0.9555, 546.4385: 1.0601, 548.5335: 1.1136, 552.7235: 1.0891,
                                 556.4944: 0.7283, 560.0559: 0.2472, 562.7793: -0.2428, 566.9693: -0.7372,
                                 572.6257: -1.1114, 577.8631: -1.4321, 583.9385: -1.7416}
        blue_log_sensitivity = {379.2598: -0.0512, 386.8017: 0.2539, 393.7151: 0.4477, 397.2765: 0.5568,
                                402.0950: 0.6837, 406.0754: 0.7261, 409.6369: 0.7149, 413.1983: 0.6793,
                                419.9022: 0.6993, 425.1397: 0.7906, 432.6816: 0.9198, 437.0810: 1.0022,
                                441.8994: 1.0668, 447.9749: 1.1203, 453.6313: 1.2316, 459.0782: 1.3742,
                                463.0587: 1.5011, 466.8296: 1.6080, 469.3436: 1.6726, 472.9050: 1.7149,
                                476.4665: 1.6682, 479.6089: 1.5212, 481.2849: 1.3719, 485.4749: 0.9911,
                                487.7793: 0.6526, 492.5978: 0.1626, 498.6732: -0.4744, 502.4441: -0.8931,
                                506.0056: -1.2316, 509.1480: -1.4878}
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry
        red_curve = {-2.8178: 0.0827, -2.4666: 0.0827, -2.3277: 0.0827, -2.1782: 0.0891, -2.0367: 0.1005,
                     -1.9056: 0.1329, -1.8270: 0.1718, -1.7274: 0.2480, -1.6592: 0.3306, -1.5662: 0.4895,
                     -1.5020: 0.6337, -1.4482: 0.8039, -1.3722: 1.1183, -1.3250: 1.3339, -1.2805: 1.5867,
                     -1.2385: 1.8006, -1.1940: 2.0016, -1.1533: 2.1669, -1.1140: 2.3096, -1.0721: 2.4117,
                     -1.0315: 2.4749, -0.9934: 2.5235, -0.9436: 2.5673, -0.8781: 2.5997, -0.8204: 2.6256,
                     -0.7366: 2.6451, -0.6343: 2.6580, -0.5636: 2.6467, -0.5059: 2.6272}
        green_curve = {-2.8165: 0.0827, -2.4797: 0.0827, -2.0839: 0.0859, -1.9817: 0.0972, -1.8899: 0.1135,
                       -1.8296: 0.1394, -1.7851: 0.1702, -1.7379: 0.2075, -1.6828: 0.2690, -1.6160: 0.3695,
                       -1.5413: 0.5138, -1.4758: 0.6759, -1.3958: 0.9530, -1.3172: 1.2853, -1.2333: 1.6872,
                       -1.1389: 2.0405, -1.0983: 2.1605, -1.0524: 2.2577, -0.9987: 2.3485, -0.9463: 2.4165,
                       -0.8912: 2.4668, -0.8283: 2.5138, -0.7733: 2.5462, -0.7077: 2.5737, -0.6474: 2.5900,
                       -0.5924: 2.5932, -0.5452: 2.5883, -0.5007: 2.5786}
        blue_curve = {-2.8126: 0.0859, -2.3696: 0.0843, -2.0970: 0.0924, -1.9633: 0.1167, -1.8716: 0.1475,
                      -1.7877: 0.1977, -1.7169: 0.2577, -1.6461: 0.3485, -1.5858: 0.4489, -1.5190: 0.5932,
                      -1.4666: 0.7342, -1.4063: 0.9579, -1.3460: 1.2237, -1.2962: 1.5024, -1.2464: 1.7715,
                      -1.1953: 1.9984, -1.1507: 2.1507, -1.1035: 2.2836, -1.0668: 2.3485, -1.0223: 2.4068,
                      -0.9725: 2.4489, -0.9253: 2.4765, -0.8729: 2.4895, -0.8152: 2.5008, -0.7680: 2.5089,
                      -0.7025: 2.5154, -0.6396: 2.5203, -0.5819: 2.5105, -0.5242: 2.4878, -0.4954: 2.4749}
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
        red_sd = {400.2987: 0.1155, 408.3624: 0.1081, 419.7113: 0.0932, 431.3589: 0.0708, 444.2011: 0.0472,
                  455.5500: 0.0373, 472.5734: 0.0311, 494.9726: 0.0385, 514.9826: 0.0584, 535.5898: 0.1031,
                  549.0294: 0.1516, 559.3330: 0.2149, 572.9219: 0.3242, 586.2120: 0.4795, 599.9502: 0.6683,
                  607.4166: 0.7776, 613.8377: 0.8720, 621.1548: 0.9565, 625.7840: 0.9888, 630.1145: 1.0012,
                  635.9383: 0.9925, 640.7168: 0.9652, 645.6446: 0.9118, 651.4684: 0.8360, 657.7402: 0.7416,
                  665.5052: 0.6298, 674.0169: 0.5217, 682.3793: 0.4211, 687.9044: 0.3652, 695.0722: 0.3081,
                  699.8507: 0.2720}
        green_sd = {400.2987: 0.0422, 415.8288: 0.0484, 429.8656: 0.0484, 445.0971: 0.0596, 460.0299: 0.1006,
                    470.9308: 0.1565, 479.5918: 0.2186, 489.7461: 0.3180, 499.6018: 0.4447, 507.3668: 0.5615,
                    514.6839: 0.6696, 520.3584: 0.7466, 527.8248: 0.8410, 533.3499: 0.9068, 539.3230: 0.9615,
                    544.5495: 0.9913, 549.0294: 1.0000, 553.9572: 0.9764, 559.1837: 0.9193, 564.7088: 0.8186,
                    570.2339: 0.6944, 575.0124: 0.5801, 580.8362: 0.4509, 586.8094: 0.3416, 593.5291: 0.2373,
                    602.9368: 0.1466, 615.6297: 0.0770, 627.8746: 0.0484, 640.4181: 0.0385, 655.9482: 0.0360,
                    675.0622: 0.0298, 691.7870: 0.0236, 700.4480: 0.0211}
        blue_sd = {400.2987: 0.4087, 406.7198: 0.5391, 410.9009: 0.6261, 414.6341: 0.7093, 420.4579: 0.8199,
                   425.6844: 0.8957, 430.7616: 0.9416, 436.1374: 0.9739, 440.9159: 0.9901, 446.5903: 1.0000,
                   452.5635: 0.9913, 457.6406: 0.9727, 463.1658: 0.9267, 467.9443: 0.8720, 473.4694: 0.7863,
                   481.2344: 0.6571, 488.4022: 0.5342, 493.9273: 0.4373, 503.1857: 0.2957, 510.5027: 0.2099,
                   517.6705: 0.1453, 526.6302: 0.0907, 540.0697: 0.0447, 558.2877: 0.0186, 574.7138: 0.0124,
                   590.8412: 0.0075, 606.0727: 0.0037, 641.3141: 0.0037, 668.7904: 0.0099, 678.6461: 0.0112,
                   695.9681: 0.0087, 699.5520: 0.0112}
        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]

        self.calibrate()
