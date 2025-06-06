from spectral_film_lut.film_spectral import *


class KodakEnduraPremier(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.lad = [0.8, 0.8, 0.8]
        self.density_measure = 'status_a'

        # spectral sensitivity
        red_log_sensitivity = {529.6309: -1.6360, 551.2775: -1.5910, 565.1171: -1.5122, 569.9077: -1.5272,
                               575.7630: -1.4447, 590.3123: -1.3152, 600.9581: -1.2101, 613.5557: -0.9831,
                               625.2661: -0.7561, 632.5408: -0.6604, 638.2186: -0.6229, 642.8318: -0.6191,
                               653.1228: -0.6191, 657.7360: -0.5985, 665.5429: -0.4653, 674.7693: -0.3077,
                               680.6246: -0.1820, 687.1895: -0.0901, 692.6899: -0.0338, 696.7708: -0.0300,
                               701.7388: -0.0788, 707.2392: -0.1914, 712.0298: -0.3415, 716.1107: -0.5141,
                               722.3208: -0.8143, 732.2569: -1.3058, 740.4187: -1.6829}
        green_log_sensitivity = {420.5110: -0.8987, 431.6891: -0.8124, 442.3350: -0.6454, 444.8190: -0.5966,
                                 450.1419: -0.5403, 459.5458: -0.3734, 465.5784: -0.3039, 475.3371: -0.1801,
                                 483.4989: 0.0263, 491.1285: 0.2308, 498.2257: 0.3809, 502.4840: 0.4315,
                                 508.3392: 0.4428, 514.7268: 0.4278, 519.5174: 0.4109, 525.3726: 0.4522,
                                 531.2278: 0.5741, 536.1959: 0.6735, 541.5188: 0.8612, 548.2612: 1.1088,
                                 549.6806: 1.1388, 552.3421: 1.0901, 554.4713: 0.9700, 559.4393: 0.2664,
                                 564.7622: -0.4653, 568.3109: -0.8330, 571.3272: -1.0244, 576.2952: -1.2495,
                                 584.9894: -1.6961}
        blue_log_sensitivity = {380.4116: 0.0657, 385.3797: 0.2439, 389.8155: 0.3340, 397.2676: 0.5253,
                                400.4613: 0.6004, 405.0745: 0.6360, 410.3974: 0.6041, 414.3009: 0.5722,
                                420.1561: 0.6098, 425.4791: 0.7242, 434.8829: 0.8762, 438.9638: 0.9550,
                                442.6899: 0.9812, 446.7708: 1.0150, 450.8517: 1.0807, 458.3038: 1.2871,
                                465.7559: 1.5047, 469.4819: 1.6191, 473.3854: 1.6435, 475.8694: 1.6079,
                                479.9503: 1.3771, 485.8055: 0.7730, 490.7736: 0.2402, 496.8062: -0.3602,
                                502.4840: -0.8968, 510.1136: -1.5460}
        self.log_sensitivity = [red_log_sensitivity, green_log_sensitivity, blue_log_sensitivity]

        # sensiometry
        red_log_exposure = xp.array(
            [-3.0000, -1.9802, -1.8642, -1.7542, -1.6501, -1.5610, -1.4584, -1.3826, -1.2993, -1.2131, -1.1581, -1.0585,
             -1.0302, -0.9856, -0.9395, -0.8756, -0.7924, -0.6838, -0.5411, -0.3776, -0.2379, -0.1487], dtype=default_dtype)
        red_density_curve = xp.array(
            [0.1100, 0.1130, 0.1278, 0.1635, 0.2393, 0.3449, 0.5619, 0.7582, 1.0525, 1.4182, 1.6769, 2.1169, 2.2180,
             2.3459, 2.4529, 2.5659, 2.6596, 2.7161, 2.7428, 2.7636, 2.7547, 2.7413], dtype=default_dtype)
        green_log_exposure = xp.array(
            [-3.0015, -1.9683, -1.8345, -1.7364, -1.6487, -1.5461, -1.4420, -1.3543, -1.2815, -1.2384, -1.2101, -1.1700,
             -1.1001, -1.0362, -0.9604, -0.8697, -0.7849, -0.7255, -0.6630, -0.5114, -0.3687, -0.2468, -0.1487], dtype=default_dtype)
        green_density_curve = xp.array(
            [0.0833, 0.0892, 0.1056, 0.1442, 0.2126, 0.3612, 0.5976, 0.9068, 1.2190, 1.4034, 1.5223, 1.6799, 1.9415,
             2.1348, 2.2805, 2.3979, 2.4559, 2.4797, 2.4945, 2.5168, 2.5302, 2.5302, 2.5213], dtype=default_dtype)
        blue_log_exposure = xp.array(
            [-2.9985, -1.9817, -1.8479, -1.7631, -1.6650, -1.5847, -1.4836, -1.4435, -1.3974, -1.3216, -1.2443, -1.1982,
             -1.1402, -1.0852, -1.0406, -0.9916, -0.9187, -0.8518, -0.7820, -0.7002, -0.6184, -0.5055, -0.3033,
             -0.1516], dtype=default_dtype)
        blue_density_curve = xp.array(
            [0.1085, 0.1145, 0.1293, 0.1606, 0.2230, 0.3241, 0.5114, 0.6155, 0.7582, 1.0377, 1.3707, 1.5654, 1.7958,
             2.0040, 2.1199, 2.2255, 2.3162, 2.3741, 2.4039, 2.4277, 2.4277, 2.4395, 2.4485, 2.4381], dtype=default_dtype)
        self.log_exposure = [red_log_exposure, green_log_exposure, blue_log_exposure]
        self.density_curve = [red_density_curve, green_density_curve, blue_density_curve]

        # spectral dye density
        red_sd = {400.0000: 0.1257, 421.7812: 0.0629, 449.3708: 0.0314, 495.2565: 0.0411, 533.8819: 0.0943,
                  562.7783: 0.2454, 582.0910: 0.4352, 600.0968: 0.6987, 608.8093: 0.8184, 619.8451: 0.9055,
                  635.9632: 0.9707, 643.9497: 0.9961, 648.3059: 1.0058, 654.8403: 0.9901, 661.6651: 0.9454,
                  669.9419: 0.8583, 677.3475: 0.7423, 688.6738: 0.5670, 700.0000: 0.4231}
        green_sd = {400.2904: 0.0834, 418.0058: 0.0592, 437.7541: 0.0641, 459.5353: 0.1257, 478.8480: 0.2599,
                    501.6457: 0.5319, 511.2294: 0.6746, 520.8132: 0.7967, 534.1723: 0.9224, 543.6108: 0.9828,
                    548.4027: 1.0034, 554.7919: 0.9889, 560.4550: 0.9429, 567.7154: 0.8245, 577.4443: 0.6165,
                    590.9487: 0.3433, 600.2420: 0.2164, 611.4230: 0.1209, 629.1384: 0.0532, 658.1801: 0.0181,
                    699.7096: 0.0121}
        blue_sd = {400.2904: 0.5924, 414.6660: 0.7773, 424.8306: 0.8897, 431.9458: 0.9562, 436.5924: 0.9865,
                   441.2391: 1.0022, 446.7570: 0.9973, 454.3078: 0.9695, 462.7299: 0.9067, 474.4918: 0.7592,
                   496.1278: 0.4183, 509.4869: 0.2490, 520.0871: 0.1584, 532.4298: 0.0943, 553.3398: 0.0411,
                   584.9952: 0.0097, 699.7096: 0.0085}
        self.spectral_density = [colour.SpectralDistribution(x) for x in (red_sd, green_sd, blue_sd)]

        self.calibrate()
