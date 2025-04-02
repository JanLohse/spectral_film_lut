from spectral_film_lut.film_spectral import *


class Kodak2393(FilmSpectral):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lad = [1.09, 1.06, 1.03]
        self.density_measure = 'status_a'
        self.exposure_kelvin = 3200
        self.projection_kelvin = 6500

        # spectral sensitivity
        self.red_log_sensitivity = {579.3658: -2.9972, 580.1652: -2.6570, 583.2741: -2.4597, 585.9389: -2.4287,
                                    593.0449: -2.3930, 598.5521: -2.3329, 601.6610: -2.2680, 606.3688: -2.2342,
                                    609.7442: -2.1478, 615.5179: -2.0679, 618.6268: -2.0068, 621.8245: -1.9091,
                                    627.1540: -1.8311, 637.1025: -1.7747, 650.2487: -1.6789, 661.7961: -1.5483,
                                    668.1027: -1.4261, 671.3892: -1.3387, 676.1858: -1.2476, 679.4724: -1.0785,
                                    683.0254: -0.9093, 687.2002: -0.8482, 691.6415: -0.8661, 695.6387: -0.9760,
                                    699.2805: -1.1113, 702.4782: -1.2100, 707.8078: -1.3416, 711.9826: -1.5389,
                                    716.3351: -1.7785, 722.5529: -2.2953, 728.7707: -2.9831}
        self.green_log_sensitivity = {442.0412: -2.9859, 444.2619: -2.6054, 448.2590: -2.0886, 451.3679: -1.8114,
                                      454.5656: -1.6723, 463.8035: -1.5107, 472.5973: -1.3735, 478.3709: -1.2654,
                                      485.1217: -1.0973, 484.9440: -1.0973, 489.1188: -1.0023, 495.1590: -0.9168,
                                      501.1103: -0.8670, 504.2192: -0.8539, 511.3253: -0.8614, 516.6548: -0.8670,
                                      525.5374: -0.8445, 530.7781: -0.7731, 536.3741: -0.6368, 540.5489: -0.4677,
                                      543.4802: -0.3362, 545.9673: -0.2563, 549.0762: -0.2225, 551.6522: -0.2657,
                                      556.2711: -0.4912, 561.8671: -0.8482, 574.9245: -1.6563, 582.8300: -2.3000,
                                      588.3372: -2.9812}
        self.blue_log_sensitivity = {357.6568: 0.0820, 363.6969: 0.0406, 368.7600: -0.0073, 373.5566: -0.0778,
                                     375.7772: -0.1482, 377.4649: -0.2657, 378.3532: -0.3596, 379.4191: -0.4348,
                                     382.0839: -0.5100, 386.9693: -0.5194, 392.6541: -0.4602, 398.0725: -0.3737,
                                     403.9350: -0.3202, 408.1986: -0.2741, 414.1499: -0.1924, 428.1844: 0.0688,
                                     442.8406: 0.3319, 455.6316: 0.5424, 458.6516: 0.5875, 466.8236: 0.6298,
                                     462.7376: 0.6157, 469.3995: 0.5471, 478.0156: -0.0026, 484.6776: -0.5664,
                                     491.5171: -1.1818, 498.1791: -1.9664, 500.8438: -2.3329, 504.3080: -2.9906}

        # sensiometry
        self.red_log_exposure = np.array(sorted(
            [2.2808, 2.0761, 1.8568, 1.7151, 1.5926, 1.4172, 1.2536, 1.1129, 1.0242, 0.9146, 0.8140, 0.6495, 0.7373,
             0.5527, 0.3909, 0.1798, -0.0715, -0.4645, -0.6975]))
        self.red_density_curve = np.array(sorted(
            [5.2592, 5.1963, 5.0689, 4.9087, 4.6855, 4.1283, 3.3437, 2.6317, 2.1552, 1.6212, 1.2008, 0.6969, 0.9366,
             0.4984, 0.2807, 0.1424, 0.0876, 0.0657, 0.0548]))
        self.green_log_exposure = np.array(sorted(
            [2.2808, 2.0130, 1.8111, 1.6676, 1.5369, 1.4044, 1.2554, 1.0964, 0.9557, 0.7647, 0.5380, 0.6596, 0.3781,
             0.1953, -0.0761, -0.4736, -0.7076]))
        self.green_density_curve = np.array(sorted(
            [5.5235, 5.4290, 5.2538, 4.9950, 4.6061, 4.0721, 3.3670, 2.5728, 1.9429, 1.2309, 0.6353, 0.9119, 0.3656,
             0.1780, 0.0917, 0.0644, 0.0575]))
        self.blue_log_exposure = np.array(sorted(
            [2.2808, 2.0806, 1.8385, 1.6676, 1.5725, 1.4812, 1.3943, 1.3276, 1.2618, 1.1924, 1.1211, 1.0498, 0.8954,
             0.7948, 0.5910, 0.7016, 0.2474, 0.4403, -0.1748, 0.0464, -0.6957]))
        self.blue_density_curve = np.array(sorted(
            [5.3866, 5.3811, 5.2990, 5.1360, 4.9594, 4.6883, 4.3391, 4.0379, 3.7230, 3.3464, 2.9630, 2.5591, 1.7855,
             1.3473, 0.7421, 1.0310, 0.2738, 0.4669, 0.0917, 0.1506, 0.0630]))

        self.exposure_base = 10

        # spectral dye density
        red_spectral_density = {349.7274: 0.2693, 360.2681: 0.2718, 372.5352: 0.2851, 382.0763: 0.2982,
                                393.3439: 0.2975, 401.3403: 0.2842, 410.4271: 0.2509, 418.6052: 0.2048,
                                428.9641: 0.1508, 437.0513: 0.1175, 449.8637: 0.0749, 460.1318: 0.0457,
                                469.0368: 0.0337, 488.8460: 0.0308, 507.4739: 0.0314, 519.0141: 0.0476,
                                535.1886: 0.0937, 555.7247: 0.1839, 574.9886: 0.3258, 596.0700: 0.5462,
                                613.3348: 0.7621, 627.2376: 0.9107, 640.3226: 1.0018, 648.5915: 1.0450,
                                659.6774: 1.0742, 671.4902: 1.0752, 680.7587: 1.0488, 692.8442: 0.9764,
                                705.1113: 0.8637, 720.2862: 0.6890, 732.2808: 0.5446, 749.5457: 0.3455}
        green_spectral_density = {350.9087: 0.0826, 361.6311: 0.0597, 375.7156: 0.0432, 388.8914: 0.0413,
                                  408.8823: 0.0594, 421.1495: 0.0794, 430.2363: 0.0876, 439.7774: 0.0832,
                                  452.4989: 0.0876, 469.1277: 0.1515, 483.3030: 0.2699, 496.7515: 0.4414,
                                  509.0186: 0.6049, 516.2881: 0.6963, 524.7388: 0.7764, 534.0073: 0.8329,
                                  541.2767: 0.8586, 546.0927: 0.8589, 554.5434: 0.8208, 562.1763: 0.7510,
                                  568.4462: 0.6668, 581.4403: 0.4588, 590.3453: 0.3302, 601.6129: 0.2112,
                                  613.5166: 0.1302, 624.5116: 0.0851, 641.0495: 0.0533, 661.2222: 0.0305,
                                  681.0313: 0.0191, 702.8396: 0.0095, 726.9196: 0.0032, 744.6388: 0.0010}
        blue_spectral_density = {350.9995: 0.2890, 357.5420: 0.2588, 365.1749: 0.2321, 373.8982: 0.2229,
                                 381.3494: 0.2356, 391.4357: 0.3112, 402.6124: 0.4318, 413.7892: 0.5731,
                                 423.0577: 0.6700, 430.6906: 0.7287, 437.2331: 0.7576, 445.9564: 0.7694,
                                 454.3162: 0.7630, 461.4039: 0.7414, 467.9464: 0.7002, 474.2163: 0.6382,
                                 485.3021: 0.4954, 496.1154: 0.3461, 508.5643: 0.2064, 522.8305: 0.1064,
                                 541.8219: 0.0479, 559.2685: 0.0229, 575.8973: 0.0156, 594.2526: 0.0102,
                                 645.1386: 0.0108, 710.1090: 0.0108, 720.1045: 0.0089, 730.0091: 0.0019,
                                 748.7279: 0.0006}
        midscale_spectral_density = {350.6361: 1.2098, 361.1767: 1.0399, 370.1726: 0.8780, 356.0881: 1.1288,
                                     373.1713: 0.8240, 375.2612: 0.7980, 379.3503: 0.7773, 386.2562: 0.7735,
                                     392.2535: 0.7843, 397.0695: 0.8161, 402.8851: 0.8843, 413.5166: 0.9955,
                                     422.9668: 1.0644, 428.2372: 1.0856, 432.9623: 1.0910, 440.5952: 1.0764,
                                     453.1349: 1.0256, 470.0363: 0.9545, 481.9400: 0.9078, 488.6642: 0.8916,
                                     495.6611: 0.8913, 507.2013: 0.9256, 521.7401: 0.9932, 534.9159: 1.0637,
                                     541.1858: 1.0929, 547.8192: 1.1047, 553.9982: 1.0996, 559.9046: 1.0790,
                                     567.7192: 1.0329, 575.3521: 0.9780, 581.9855: 0.9320, 588.1645: 0.9059,
                                     593.8891: 0.8967, 601.7946: 0.9088, 608.3371: 0.9374, 619.5139: 1.0142,
                                     632.5080: 1.0987, 642.0491: 1.1482, 650.1363: 1.1755, 661.0404: 1.1939,
                                     671.9446: 1.1774, 681.8492: 1.1399, 691.2085: 1.0764, 699.6592: 1.0002,
                                     714.6524: 0.8288, 732.4625: 0.6160, 749.7274: 0.4064}

        self.red_sd = colour.SpectralDistribution(red_spectral_density)
        self.green_sd = colour.SpectralDistribution(green_spectral_density)
        self.blue_sd = colour.SpectralDistribution(blue_spectral_density)
        self.d_ref_sd = colour.SpectralDistribution(midscale_spectral_density)

        self.calibrate()
