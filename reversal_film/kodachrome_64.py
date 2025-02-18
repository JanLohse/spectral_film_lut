from film_spectral import *


class Kodachrome64(FilmSpectral):
    def __init__(self):
        super().__init__()

        self.iso = 64
        self.density_measure = "status_a"
        self.white_xy = [0.3127, 0.329]
        self.white_sd = colour.SDS_ILLUMINANTS['D65']

        # spectral sensitivity
        self.red_log_sensitivity = {481.3051: -1.8515, 495.4145: -1.7974, 503.3510: -1.7516, 513.0511: -1.5836,
                                    523.9859: -1.4192, 533.9506: -1.3240, 544.7972: -1.2474, 553.4392: -1.1447,
                                    561.0229: -0.9486, 574.1623: -0.5845, 584.8325: -0.1643, 589.7707: 0.0271,
                                    596.1199: 0.1811, 603.8801: 0.2838, 616.2257: 0.4099, 622.0459: 0.4986,
                                    630.8642: 0.7087, 641.0053: 0.8534, 646.2081: 0.9169, 650.4409: 0.9300,
                                    654.8501: 0.8814, 661.7284: 0.6573, 668.4303: 0.3585, 673.0159: 0.0784,
                                    680.7760: -0.5752, 687.4780: -1.0233, 700.7055: -1.6956}
        self.green_log_sensitivity = {400.0882: -1.7890, 440.0353: -1.2428, 460.4938: -0.9972, 467.3721: -0.9412,
                                      472.6631: -0.8553, 481.4815: -0.5817, 491.3580: -0.3511, 507.0547: 0.0084,
                                      516.0494: 0.2007, 524.2504: 0.3399, 534.3915: 0.5079, 542.5044: 0.6443,
                                      547.4427: 0.7077, 553.4392: 0.7255, 559.8765: 0.6956, 565.0794: 0.6246,
                                      570.8995: 0.4986, 574.7795: 0.3632, 580.7760: 0.0317, 590.6526: -0.7806,
                                      594.7090: -1.1261, 598.5891: -1.3968, 604.0564: -1.6863, 610.2293: -1.9524}
        self.blue_log_sensitivity = {400.1764: -1.5, 402.5: -1., 405: -.5, 410.3175: 0.5733, 412.3457: 0.7470,
                                     416.3139: 0.9150, 419.4885: 0.9832, 423.1041: 1.0000, 430.3351: 0.9813,
                                     439.0653: 0.9141, 447.0018: 0.8207, 457.3192: 0.7134, 465.4321: 0.6293,
                                     471.6931: 0.5500, 478.4832: 0.4099, 485.2734: 0.2232, 500.9700: -0.3511,
                                     511.4638: -0.7993, 520.5467: -1.2241, 530.4233: -1.6909, 536.1552: -1.9888}

        # sensiometry
        self.red_log_exposure = np.array(
            [-2.3944, -2.2583, -2.1231, -2.0260, -1.9119, -1.8058, -1.6997, -1.5636, -1.3924, -1.1732, -0.9409, -0.7377,
             -0.5526, -0.3684, -0.2392, -0.0661, 0.0631, 0.2262, 0.3834])
        self.red_density_curve = np.array(
            [3.6807, 3.6577, 3.6036, 3.5415, 3.4434, 3.3233, 3.1782, 2.9379, 2.5626, 2.0270, 1.5365, 1.1662, 0.8659,
             0.6156, 0.4905, 0.3594, 0.2773, 0.2072, 0.1832])
        self.green_log_exposure = np.array(
            [-2.3891, -2.1927, -2.0218, -1.8873, -1.7364, -1.6036, -1.3709, -1.1582, -0.9473, -0.7600, -0.6018, -0.4545,
             -0.2764, -0.0218, 0.2000, 0.3945])
        self.green_density_curve = np.array(
            [3.4486, 3.4067, 3.3194, 3.1956, 2.9918, 2.7552, 2.2420, 1.7616, 1.3576, 1.0227, 0.7953, 0.6151, 0.4531,
             0.2930, 0.2202, 0.1965])
        self.blue_log_exposure = np.array(
            [-2.3904, -2.1281, -1.9730, -1.8569, -1.7447, -1.5966, -1.3924, -1.2332, -0.9990, -0.8569, -0.6246, -0.3794,
             -0.0561, 0.1291, 0.2593, 0.3904])
        self.blue_density_curve = np.array(
            [3.3263, 3.2012, 3.0831, 2.9630, 2.8028, 2.5375, 2.1021, 1.7718, 1.3213, 1.0911, 0.7758, 0.5305, 0.3143,
             0.2442, 0.2142, 0.2012])

        self.exposure_base = 10

        # spectral dye density
        red_spectral_density = {419.8569: 0.1521, 430.7245: 0.1047, 443.2692: 0.0616, 457.8936: 0.0327,
                                475.0000: 0.0214, 500.4919: 0.0182, 520.2818: 0.0230, 538.3945: 0.0379,
                                554.8301: 0.0814, 569.7898: 0.1456, 581.7308: 0.2296, 590.9213: 0.3414,
                                603.7343: 0.5765, 617.1512: 0.8975, 625.8721: 1.0921, 631.2388: 1.1894,
                                635.1297: 1.2374, 639.0206: 1.2601, 642.1735: 1.2652, 646.2657: 1.2552,
                                652.1020: 1.2147, 656.5966: 1.1660, 665.7200: 1.0370, 676.7889: 0.8684,
                                689.6691: 0.6981, 699.6646: 0.5824}
        green_spectral_density = {419.9911: 0.3187, 432.0662: 0.2750, 443.2692: 0.2438, 453.5331: 0.2299,
                                  462.3882: 0.2432, 469.3649: 0.2811, 479.9642: 0.3917, 497.5403: 0.6219,
                                  508.8104: 0.7954, 517.1959: 0.9202, 525.1118: 1.0010, 531.2835: 1.0431,
                                  537.1869: 1.0613, 542.5537: 1.0603, 549.6646: 1.0331, 556.5742: 0.9770,
                                  565.3623: 0.8732, 577.7728: 0.6819, 588.9088: 0.5019, 597.6297: 0.3820,
                                  611.9857: 0.2361, 626.9454: 0.1420, 619.3649: 0.1822, 634.5930: 0.1145,
                                  645.8631: 0.0856, 664.3113: 0.0577, 699.6646: 0.0217}
        blue_spectral_density = {399.8658: 0.3593, 420.1252: 0.6433, 426.0957: 0.7040, 433.8104: 0.7529,
                                 441.4580: 0.7730, 449.3739: 0.7656, 456.6860: 0.7305, 464.4678: 0.6738,
                                 470.9750: 0.6073, 480.7692: 0.4711, 490.4964: 0.3398, 502.5045: 0.2117,
                                 520.2818: 0.0885, 551.1404: 0.0172, 598.0993: 0.0003}
        midscale_spectral_density = {420.7961: 1.1193, 430.9928: 1.1080, 439.9150: 1.0879, 446.9589: 1.0629,
                                     453.6673: 1.0250, 462.1199: 0.9673, 469.3649: 0.9261, 475.4696: 0.9005,
                                     481.9097: 0.8881, 487.8131: 0.8888, 495.2594: 0.9063, 501.4982: 0.9316,
                                     511.6950: 1.0055, 519.4097: 1.0655, 527.4597: 1.1132, 534.1011: 1.1359,
                                     539.1324: 1.1414, 544.6333: 1.1336, 550.5367: 1.1115, 556.2388: 1.0743,
                                     562.2093: 1.0256, 570.2594: 0.9462, 577.4374: 0.8765, 582.3345: 0.8408,
                                     586.7621: 0.8233, 590.7871: 0.8194, 595.6843: 0.8298, 600.1118: 0.8577,
                                     605.7469: 0.9283, 614.0653: 1.0678, 622.0483: 1.2023, 627.8175: 1.2834,
                                     632.3122: 1.3304, 636.2030: 1.3589, 640.3623: 1.3684, 644.5886: 1.3599,
                                     649.8882: 1.3217, 654.5841: 1.2688, 666.0555: 1.0921, 680.2102: 0.8732,
                                     699.9329: 0.6235}

        self.red_sd = colour.SpectralDistribution(red_spectral_density)
        self.green_sd = colour.SpectralDistribution(green_spectral_density)
        self.blue_sd = colour.SpectralDistribution(blue_spectral_density)
        self.d_ref_sd = colour.SpectralDistribution(midscale_spectral_density)

        self.calibrate()
