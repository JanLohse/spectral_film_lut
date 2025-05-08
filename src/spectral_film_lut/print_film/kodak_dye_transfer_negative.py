import colour

from spectral_film_lut.film_spectral import *


class KodakDyeTransferNegative(FilmSpectral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lad = [0.8, 0.8, 0.8]
        self.density_measure = 'absolute'
        self.exposure_kelvin = None
        self.projection_kelvin = 6500

        # spectral sensitivity
        red_transmittance_29 = {349.4411: 0.0015, 591.2818: 0.0025, 598.1130: 0.0037, 602.3942: 0.0106,
                                606.1832: 0.0378, 608.4143: 0.0708, 610.4099: 0.1097, 611.5302: 0.1460,
                                613.9978: 0.2160, 618.5144: 0.3907, 624.9122: 0.6060, 630.2121: 0.7327,
                                633.0638: 0.7817, 636.4944: 0.8198, 640.5102: 0.8445, 644.8675: 0.8649,
                                650.7784: 0.8812, 659.1247: 0.8935, 675.1925: 0.9036, 687.3076: 0.9076,
                                716.8493: 0.9075, 749.9180: 0.9075}
        green_transmittance_61 = {349.2207: 0.0015, 465.4050: 0.0005, 476.6480: 0.0007, 480.6050: 0.0051,
                                  484.3394: 0.0103, 487.1857: 0.0181, 489.2424: 0.0329, 493.0132: 0.0672,
                                  496.8675: 0.1121, 499.8230: 0.1637, 505.8655: 0.2584, 509.3699: 0.3108,
                                  513.0016: 0.3565, 514.1862: 0.3675, 515.8094: 0.3794, 517.9795: 0.3930,
                                  520.0479: 0.4033, 521.6925: 0.4067, 523.5619: 0.4085, 526.7592: 0.4082,
                                  530.0870: 0.4000, 533.4266: 0.3871, 537.3355: 0.3671, 541.0389: 0.3411,
                                  544.2972: 0.3169, 548.9187: 0.2767, 553.7671: 0.2339, 557.4984: 0.1971,
                                  562.6817: 0.1527, 567.0646: 0.1196, 571.4368: 0.0908, 574.4650: 0.0703,
                                  577.8110: 0.0549, 582.2635: 0.0378, 587.2574: 0.0246, 593.0153: 0.0144,
                                  599.2033: 0.0084, 604.9495: 0.0028, 619.2816: 0.0018, 749.6783: 0.0036}
        blue_transmittance_47b = {349.5513: 0.0016, 385.2651: 0.0017, 390.0086: 0.0003, 396.9029: 0.1069,
                                  402.4008: 0.1990, 408.9077: 0.2845, 416.0289: 0.3885, 419.3342: 0.4325,
                                  422.5615: 0.4639, 424.7187: 0.4825, 427.7683: 0.4970, 430.6157: 0.5043,
                                  433.2577: 0.5057, 436.8070: 0.4971, 440.3863: 0.4766, 445.3684: 0.4246,
                                  466.7372: 0.1704, 473.2003: 0.0995, 475.7898: 0.0781, 478.9443: 0.0513,
                                  482.0688: 0.0363, 483.9660: 0.0272, 487.4040: 0.0189, 491.8362: 0.0099,
                                  495.3694: 0.0075, 501.7790: 0.0011, 506.4102: 0.0005, 749.7907: 0.0028}
        # red_density_29 = {x: -math.log10(max(0, y)) for x, y in red_transmittance_29.items()}
        # green_density_61 = {x: -math.log10(max(0, y)) for x, y in green_transmittance_61.items()}
        # blue_density_47b = {x: -math.log10(max(0, y)) for x, y in blue_transmittance_47b.items()}
        red_density_29 = SpectralDistribution(red_transmittance_29).align(self.spectral_shape, interpolator=colour.LinearInterpolator).align(self.spectral_shape)
        green_density_61 = SpectralDistribution(green_transmittance_61).align(self.spectral_shape, interpolator=colour.LinearInterpolator).align(self.spectral_shape)
        blue_density_47b = SpectralDistribution(blue_transmittance_47b).align(self.spectral_shape, interpolator=colour.LinearInterpolator).align(self.spectral_shape)
        red_density_29.values = -np.log10(red_density_29.values)
        green_density_61.values = -np.log10(green_density_61.values)
        blue_density_47b.values = -np.log10(blue_density_47b.values)

        # taken from pan separation film 4133:
        log_sensitvity = {399.4509: 1.4940, 403.6734: 1.5348, 411.2937: 1.5808, 419.9970: 1.5985, 429.9913: 1.5946,
                          440.9678: 1.5548, 450.2537: 1.5010, 457.7344: 1.4447, 468.0152: 1.3372, 476.1611: 1.2425,
                          484.5433: 1.1146, 492.9173: 0.9982, 502.5494: 0.9061, 516.5975: 0.8816, 532.1182: 0.8789,
                          551.0384: 0.8786, 568.0426: 0.8823, 582.9146: 0.8949, 588.2265: 0.8986, 592.6725: 0.9241,
                          599.0045: 0.9880, 605.6526: 1.0556, 612.1046: 1.1003, 619.3170: 1.1219, 625.2685: 1.1231,
                          634.2355: 1.0693, 641.5274: 0.9798, 646.7463: 0.8161, 651.5675: 0.6141, 662.6953: -0.0839,
                          670.7483: -0.6438}
        log_sensitvity = colour.SpectralDistribution(log_sensitvity).align(self.spectral_shape, extrapolator_kwargs={
            'method': 'linear'}).align(self.spectral_shape)

        red_log_sensitivity = log_sensitvity - red_density_29
        green_log_sensitivity = log_sensitvity - green_density_61
        blue_log_sensitivity = log_sensitvity - blue_density_47b
        print(red_log_sensitivity.shape)
        colour.plotting.plot_multi_sds((red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity),)
        red_log_sensitivity = dict(zip(red_log_sensitivity.wavelengths, red_log_sensitivity.values))
        green_log_sensitivity = dict(zip(green_log_sensitivity.wavelengths, green_log_sensitivity.values))
        blue_log_sensitivity = dict(zip(blue_log_sensitivity.wavelengths, blue_log_sensitivity.values))

        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry curve from kodak matrix film 4150
        red_curve = {-0.8901: 0.0337, -0.8083: 0.0450, -0.7327: 0.0582, -0.6203: 0.0795, -0.5184: 0.1129,
                     -0.4309: 0.1500, -0.3650: 0.1855, -0.2970: 0.2272, -0.2282: 0.2761, -0.1474: 0.3376,
                     -0.0246: 0.4576, 0.0716: 0.5770, 0.1933: 0.7584, 0.3935: 1.1285, 0.5697: 1.4631, 0.6975: 1.7012,
                     0.8412: 1.9662, 0.9197: 2.0985, 1.0265: 2.2539, 1.1249: 2.3620, 1.2107: 2.4380, 1.2977: 2.4977,
                     1.3779: 2.5387, 1.4553: 2.5663, 1.5962: 2.6006, 1.7229: 2.6213, 1.8457: 2.6357, 1.9573: 2.6437}
        green_curve = red_curve.copy()
        blue_curve = red_curve.copy()
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
        red_sd = {399.9333: 0.1744, 416.2330: 0.1563, 437.2710: 0.1569, 456.4138: 0.1708, 477.4518: 0.1968,
                  501.3328: 0.2447, 523.2237: 0.3199, 552.1273: 0.4851, 567.8584: 0.5867, 582.1681: 0.6796,
                  593.6348: 0.7644, 598.3731: 0.7951, 613.9147: 0.8621, 629.7406: 0.9218, 639.9753: 0.9580,
                  646.8932: 0.9775, 652.3896: 0.9830, 661.4871: 0.9660, 672.2905: 0.9171, 685.8420: 0.8424,
                  699.9621: 0.7477}
        green_sd = {399.7438: 0.1771, 406.7564: 0.1627, 412.8214: 0.1549, 420.2132: 0.1617, 430.0688: 0.1893,
                    441.2512: 0.2356, 461.6259: 0.3725, 481.3372: 0.5275, 504.7444: 0.7084, 517.4430: 0.7860,
                    527.2986: 0.8156, 537.3438: 0.8292, 546.8204: 0.8308, 556.8656: 0.8072, 563.4992: 0.7768,
                    571.0805: 0.7025, 589.4651: 0.4383, 603.5852: 0.2532, 617.1367: 0.1425, 642.3444: 0.0635,
                    661.8662: 0.0441, 683.0938: 0.0354, 699.1092: 0.0273}
        blue_sd = {400.1228: 0.9301, 405.8088: 0.9509, 412.2529: 0.9664, 417.4650: 0.9665, 424.1934: 0.9508,
                   433.0066: 0.9144, 442.0093: 0.8622, 451.9598: 0.7793, 460.9625: 0.6925, 471.1972: 0.5850,
                   483.7063: 0.4364, 495.0782: 0.3210, 506.8292: 0.2136, 519.7174: 0.1395, 536.0171: 0.0908,
                   548.3367: 0.0752, 563.3097: 0.0757, 595.9091: 0.0700, 630.7830: 0.0545, 655.2326: 0.0466,
                   679.1136: 0.0340, 693.7075: 0.0318, 699.7726: 0.0306}
        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]

        self.calibrate()
