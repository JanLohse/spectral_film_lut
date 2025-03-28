import math
import time

import colour
import numpy as np
from colour import SpectralDistribution, MultiSpectralDistributions
from scipy.ndimage import gaussian_filter

default_dtype = np.float32
colour.utilities.set_default_float_dtype(default_dtype)


class FilmSpectral:
    def __init__(self, spectral_shape=None):
        if spectral_shape is None:
            self.spectral_shape = colour.SpectralShape(380, 780, 5)
        else:
            self.spectral_shape = spectral_shape
        self.xyz_cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].align(self.spectral_shape).values

        reference_sds = colour.characterisation.read_training_data_rawtoaces_v1().align(self.spectral_shape).values
        reference_xyz = reference_sds.T @ self.xyz_cmfs
        self.xyz_dual = np.linalg.lstsq(reference_xyz, reference_sds.T, rcond=None)[0].T

        status_a = [
            {599: 2.298, 600: 2.568, 610: 4.638, 620: 5.000, 630: 4.871, 640: 4.604, 650: 4.286, 660: 3.900, 670: 3.551,
             680: 3.165, 690: 2.776, 700: 2.383, 710: 1.970, 720: 1.551, 730: 1.141, 740: 0.741, 750: 0.341,
             751: 0.301},
            {499: 1.430, 500: 1.650, 510: 3.822, 520: 4.782, 530: 5.000, 540: 4.906, 550: 4.644, 560: 4.221, 570: 3.609,
             580: 2.766, 590: 1.579, 591: 1.409},
            {419: 3.222, 420: 3.602, 430: 4.819, 440: 5.000, 450: 4.912, 460: 4.620, 470: 4.040, 480: 2.989, 490: 1.566,
             500: 0.165, 501: 0.015}]

        status_m = [
            {619: 1.849, 620: 2.109, 630: 4.479, 640: 5.000, 650: 4.899, 660: 4.578, 670: 4.252, 680: 3.875, 690: 3.491,
             700: 3.099, 710: 2.687, 720: 2.269, 730: 1.859, 740: 1.449, 750: 1.054, 760: 0.654, 770: 0.254,
             771: 0.214},
            {469: 1.046, 470: 1.152, 480: 2.207, 490: 3.156, 500: 3.804, 510: 4.272, 520: 4.626, 530: 4.872, 540: 5.000,
             550: 4.995, 560: 4.818, 570: 4.458, 580: 3.915, 590: 3.172, 600: 2.239, 610: 1.070, 611: 0.950},
            {409: 1.853, 410: 2.103, 420: 4.111, 430: 4.632, 440: 4.871, 450: 5.000, 460: 4.955, 470: 4.743, 480: 4.343,
             490: 3.743, 500: 2.990, 510: 1.852, 511: 1.632}]

        apd = {360: (0.0000, 0.0000, 0.0000), 362: (0.0000, 0.0000, 0.0000), 364: (0.0000, 0.0000, 0.0000),
               366: (0.0000, 0.0000, 0.0000), 368: (0.0000, 0.0000, 0.0001), 370: (0.0000, 0.0000, 0.0001),
               372: (0.0000, 0.0001, 0.0003), 374: (0.0002, 0.0001, 0.0005), 376: (0.0005, 0.0002, 0.0007),
               378: (0.0009, 0.0002, 0.0010), 380: (0.0013, 0.0002, 0.0012), 382: (0.0013, 0.0002, 0.0013),
               384: (0.0010, 0.0001, 0.0013), 386: (0.0006, 0.0001, 0.0014), 388: (0.0002, 0.0000, 0.0016),
               390: (0.0000, 0.0000, 0.0020), 392: (0.0000, 0.0000, 0.0028), 394: (0.0000, 0.0000, 0.0037),
               396: (0.0000, 0.0000, 0.0050), 398: (0.0000, 0.0000, 0.0065), 400: (0.0000, 0.0000, 0.0083),
               402: (0.0000, 0.0000, 0.0107), 404: (0.0000, 0.0000, 0.0138), 406: (0.0000, 0.0000, 0.0175),
               408: (0.0000, 0.0000, 0.0219), 410: (0.0000, 0.0000, 0.0268), 412: (0.0000, 0.0000, 0.0325),
               414: (0.0000, 0.0000, 0.0392), 416: (0.0000, 0.0000, 0.0471), 418: (0.0000, 0.0000, 0.0562),
               420: (0.0000, 0.0000, 0.0667), 422: (0.0000, 0.0000, 0.0790), 424: (0.0000, 0.0000, 0.0935),
               426: (0.0000, 0.0000, 0.1095), 428: (0.0000, 0.0000, 0.1265), 430: (0.0000, 0.0000, 0.1434),
               432: (0.0000, 0.0000, 0.1591), 434: (0.0000, 0.0000, 0.1738), 436: (0.0000, 0.0000, 0.1891),
               438: (0.0000, 0.0000, 0.2068), 440: (0.0000, 0.0000, 0.2290), 442: (0.0000, 0.0003, 0.2557),
               444: (0.0000, 0.0010, 0.2866), 446: (0.0000, 0.0020, 0.3231), 448: (0.0000, 0.0030, 0.3670),
               450: (0.0000, 0.0039, 0.4204), 452: (0.0000, 0.0046, 0.4863), 454: (0.0000, 0.0052, 0.5635),
               456: (0.0000, 0.0058, 0.6478), 458: (0.0000, 0.0064, 0.7344), 460: (0.0000, 0.0071, 0.8174),
               462: (0.0000, 0.0079, 0.8952), 464: (0.0000, 0.0089, 0.9611), 466: (0.0000, 0.0099, 1.0000),
               468: (0.0000, 0.0108, 0.9959), 470: (0.0000, 0.0117, 0.9339), 472: (0.0000, 0.0126, 0.8125),
               474: (0.0000, 0.0134, 0.6538), 476: (0.0000, 0.0142, 0.4825), 478: (0.0000, 0.0150, 0.3240),
               480: (0.0000, 0.0158, 0.2028), 482: (0.0000, 0.0163, 0.1230), 484: (0.0000, 0.0165, 0.0710),
               486: (0.0000, 0.0166, 0.0407), 488: (0.0000, 0.0167, 0.0243), 490: (0.0000, 0.0174, 0.0150),
               492: (0.0000, 0.0184, 0.0087), 494: (0.0000, 0.0196, 0.0051), 496: (0.0000, 0.0211, 0.0033),
               498: (0.0000, 0.0230, 0.0025), 500: (0.0000, 0.0256, 0.0020), 502: (0.0000, 0.0296, 0.0014),
               504: (0.0000, 0.0356, 0.0009), 506: (0.0000, 0.0430, 0.0004), 508: (0.0000, 0.0514, 0.0001),
               510: (0.0000, 0.0600, 0.0000), 512: (0.0000, 0.0690, 0.0000), 514: (0.0000, 0.0787, 0.0000),
               516: (0.0000, 0.0888, 0.0000), 518: (0.0000, 0.0990, 0.0000), 520: (0.0000, 0.1095, 0.0000),
               522: (0.0000, 0.1201, 0.0000), 524: (0.0000, 0.1310, 0.0000), 526: (0.0000, 0.1427, 0.0000),
               528: (0.0000, 0.1556, 0.0000), 530: (0.0000, 0.1704, 0.0000), 532: (0.0000, 0.1829, 0.0000),
               534: (0.0000, 0.1944, 0.0000), 536: (0.0000, 0.2151, 0.0000), 538: (0.0000, 0.2556, 0.0000),
               540: (0.0000, 0.3269, 0.0000), 542: (0.0000, 0.4552, 0.0000), 544: (0.0000, 0.6303, 0.0000),
               546: (0.0000, 0.8082, 0.0000), 548: (0.0000, 0.9457, 0.0000), 550: (0.0000, 1.0000, 0.0000),
               552: (0.0000, 0.9408, 0.0000), 554: (0.0000, 0.7932, 0.0000), 556: (0.0000, 0.5983, 0.0000),
               558: (0.0000, 0.4023, 0.0000), 560: (0.0000, 0.2559, 0.0000), 562: (0.0000, 0.1744, 0.0000),
               564: (0.0000, 0.1285, 0.0000), 566: (0.0000, 0.1051, 0.0000), 568: (0.0000, 0.0907, 0.0000),
               570: (0.0000, 0.0718, 0.0000), 572: (0.0023, 0.0496, 0.0000), 574: (0.0075, 0.0330, 0.0000),
               576: (0.0133, 0.0211, 0.0000), 578: (0.0177, 0.0129, 0.0000), 580: (0.0197, 0.0071, 0.0000),
               582: (0.0194, 0.0033, 0.0000), 584: (0.0181, 0.0013, 0.0000), 586: (0.0163, 0.0003, 0.0000),
               588: (0.0146, 0.0000, 0.0000), 590: (0.0135, 0.0000, 0.0000), 592: (0.0128, 0.0000, 0.0000),
               594: (0.0125, 0.0000, 0.0000), 596: (0.0125, 0.0000, 0.0000), 598: (0.0128, 0.0000, 0.0000),
               600: (0.0134, 0.0000, 0.0000), 602: (0.0149, 0.0000, 0.0000), 604: (0.0176, 0.0000, 0.0000),
               606: (0.0214, 0.0000, 0.0000), 608: (0.0262, 0.0000, 0.0000), 610: (0.0318, 0.0000, 0.0000),
               612: (0.0383, 0.0000, 0.0000), 614: (0.0461, 0.0000, 0.0000), 616: (0.0553, 0.0000, 0.0000),
               618: (0.0655, 0.0000, 0.0000), 620: (0.0766, 0.0000, 0.0000), 622: (0.0890, 0.0000, 0.0000),
               624: (0.1028, 0.0000, 0.0000), 626: (0.1175, 0.0000, 0.0000), 628: (0.1323, 0.0000, 0.0000),
               630: (0.1464, 0.0000, 0.0000), 632: (0.1597, 0.0000, 0.0000), 634: (0.1724, 0.0000, 0.0000),
               636: (0.1843, 0.0000, 0.0000), 638: (0.1947, 0.0000, 0.0000), 640: (0.2033, 0.0000, 0.0000),
               642: (0.2094, 0.0000, 0.0000), 644: (0.2138, 0.0000, 0.0000), 646: (0.2174, 0.0000, 0.0000),
               648: (0.2215, 0.0000, 0.0000), 650: (0.2275, 0.0000, 0.0000), 652: (0.2348, 0.0000, 0.0000),
               654: (0.2432, 0.0000, 0.0000), 656: (0.2544, 0.0000, 0.0000), 658: (0.2702, 0.0000, 0.0000),
               660: (0.2923, 0.0000, 0.0000), 662: (0.3213, 0.0000, 0.0000), 664: (0.3560, 0.0000, 0.0000),
               666: (0.3954, 0.0000, 0.0000), 668: (0.4386, 0.0000, 0.0000), 670: (0.4845, 0.0000, 0.0000),
               672: (0.5337, 0.0000, 0.0000), 674: (0.5867, 0.0000, 0.0000), 676: (0.6418, 0.0000, 0.0000),
               678: (0.6977, 0.0000, 0.0000), 680: (0.7527, 0.0000, 0.0000), 682: (0.8112, 0.0000, 0.0000),
               684: (0.8727, 0.0000, 0.0000), 686: (0.9289, 0.0000, 0.0000), 688: (0.9721, 0.0000, 0.0000),
               690: (0.9950, 0.0000, 0.0000), 692: (1.0000, 0.0000, 0.0000), 694: (0.9928, 0.0000, 0.0000),
               696: (0.9714, 0.0000, 0.0000), 698: (0.9336, 0.0000, 0.0000), 700: (0.8776, 0.0000, 0.0000),
               702: (0.7997, 0.0000, 0.0000), 704: (0.7045, 0.0000, 0.0000), 706: (0.6028, 0.0000, 0.0000),
               708: (0.5049, 0.0000, 0.0000), 710: (0.4203, 0.0000, 0.0000), 712: (0.3498, 0.0000, 0.0000),
               714: (0.2876, 0.0000, 0.0000), 716: (0.2324, 0.0000, 0.0000), 718: (0.1834, 0.0000, 0.0000),
               720: (0.1396, 0.0000, 0.0000), 722: (0.0984, 0.0000, 0.0000), 724: (0.0603, 0.0000, 0.0000),
               726: (0.0289, 0.0000, 0.0000), 728: (0.0077, 0.0000, 0.0000), 730: (0.0000, 0.0000, 0.0000)}

        def interpolate_status_density(status):
            result = []
            for i, density in enumerate(status):
                density = SpectralDistribution(density).extrapolate(self.spectral_shape,
                                                                    extrapolator_kwargs={'method': 'linear'})
                density.values = 10 ** density.values
                density.interpolate(self.spectral_shape)
                density /= sum(density.values)
                result.append(density)
            result = MultiSpectralDistributions(result)
            return result.values

        self.status_a = interpolate_status_density(status_a)
        self.status_m = interpolate_status_density(status_m)
        self.apd = MultiSpectralDistributions(apd).align(self.spectral_shape).values
        self.apd /= np.sum(self.apd, axis=0)

        self.densiometry = {'status_a': self.status_a, 'status_m': self.status_m, 'apd': self.apd}

        printer_light = SpectralDistribution(
            {360: 0.0000, 362: 0.0000, 364: 0.0000, 366: 0.0000, 368: 0.0001, 370: 0.0001, 372: 0.0002, 374: 0.0005,
             376: 0.0007, 378: 0.0010, 380: 0.0013, 382: 0.0014, 384: 0.0016, 386: 0.0018, 388: 0.0022, 390: 0.0030,
             392: 0.0043, 394: 0.0061, 396: 0.0084, 398: 0.0113, 400: 0.0147, 402: 0.0189, 404: 0.0237, 406: 0.0290,
             408: 0.0345, 410: 0.0399, 412: 0.0451, 414: 0.0505, 416: 0.0560, 418: 0.0614, 420: 0.0669, 422: 0.0724,
             424: 0.0779, 426: 0.0834, 428: 0.0889, 430: 0.0944, 432: 0.0998, 434: 0.1052, 436: 0.1106, 438: 0.1159,
             440: 0.1211, 442: 0.1261, 444: 0.1309, 446: 0.1357, 448: 0.1406, 450: 0.1456, 452: 0.1508, 454: 0.1562,
             456: 0.1616, 458: 0.1671, 460: 0.1725, 462: 0.1787, 464: 0.1856, 466: 0.1921, 468: 0.1969, 470: 0.1989,
             472: 0.1988, 474: 0.1981, 476: 0.1964, 478: 0.1933, 480: 0.1885, 482: 0.1802, 484: 0.1684, 486: 0.1563,
             488: 0.1468, 490: 0.1428, 492: 0.1428, 494: 0.1439, 496: 0.1471, 498: 0.1534, 500: 0.1637, 502: 0.1825,
             504: 0.2114, 506: 0.2476, 508: 0.2883, 510: 0.3307, 512: 0.3762, 514: 0.4265, 516: 0.4794, 518: 0.5325,
             520: 0.5836, 522: 0.6338, 524: 0.6842, 526: 0.7325, 528: 0.7764, 530: 0.8137, 532: 0.8434, 534: 0.8669,
             536: 0.8852, 538: 0.8993, 540: 0.9104, 542: 0.9179, 544: 0.9220, 546: 0.9248, 548: 0.9285, 550: 0.9352,
             552: 0.9469, 554: 0.9621, 556: 0.9779, 558: 0.9915, 560: 1.0000, 562: 1.0036, 564: 1.0032, 566: 0.9969,
             568: 0.9832, 570: 0.9602, 572: 0.9235, 574: 0.8728, 576: 0.8134, 578: 0.7502, 580: 0.6882, 582: 0.6252,
             584: 0.5589, 586: 0.4947, 588: 0.4376, 590: 0.3930, 592: 0.3627, 594: 0.3436, 596: 0.3334, 598: 0.3297,
             600: 0.3304, 602: 0.3467, 604: 0.3866, 606: 0.4423, 608: 0.5062, 610: 0.5705, 612: 0.6370, 614: 0.7101,
             616: 0.7863, 618: 0.8623, 620: 0.9345, 622: 1.0029, 624: 1.0690, 626: 1.1317, 628: 1.1898, 630: 1.2424,
             632: 1.2890, 634: 1.3303, 636: 1.3667, 638: 1.3985, 640: 1.4261, 642: 1.4499, 644: 1.4700, 646: 1.4867,
             648: 1.5005, 650: 1.5118, 652: 1.5206, 654: 1.5273, 656: 1.5320, 658: 1.5348, 660: 1.5359, 662: 1.5355,
             664: 1.5336, 666: 1.5305, 668: 1.5263, 670: 1.5212, 672: 1.5151, 674: 1.5079, 676: 1.4994, 678: 1.4892,
             680: 1.4771, 682: 1.4631, 684: 1.4480, 686: 1.4332, 688: 1.4197, 690: 1.4088, 692: 1.4015, 694: 1.3965,
             696: 1.3926, 698: 1.3880, 700: 1.3813, 702: 1.3714, 704: 1.3590, 706: 1.3450, 708: 1.3305, 710: 1.3163,
             712: 1.3030, 714: 1.2904, 716: 1.2781, 718: 1.2656, 720: 1.2526, 722: 1.2387, 724: 1.2242, 726: 1.2091,
             728: 1.1937, 730: 1.1782})
        printer_light.align(self.spectral_shape, extrapolator_kwargs={'method': 'linear'})

        self.printer_lights = self.construct_spectral_density(printer_light, sigma=5)

    def calibrate(self):
        # target exposure of middle gray in log lux-seconds
        # normally use iso value, if not provided use target density of 1.0 on the green channel
        if hasattr(self, 'iso'):
            self.log_H_ref = np.ones(3) * math.log10(12.5 / self.iso)
        elif hasattr(self, 'lad'):
            self.log_H_ref = np.array([np.interp(self.lad[0], self.red_density_curve, self.red_log_exposure),
                                       np.interp(self.lad[1], self.green_density_curve, self.green_log_exposure),
                                       np.interp(self.lad[2], self.blue_density_curve, self.blue_log_exposure)])
        self.H_ref = 10 ** self.log_H_ref

        # extrapolate log_sensitivity to linear sensitivity
        self.red_log_sensitivity = colour.SpectralDistribution(self.red_log_sensitivity).align(self.spectral_shape,
            extrapolator_kwargs={'method': 'linear'}).align(self.spectral_shape)
        self.green_log_sensitivity = colour.SpectralDistribution(self.green_log_sensitivity).align(self.spectral_shape,
            extrapolator_kwargs={'method': 'linear'}).align(self.spectral_shape)
        self.blue_log_sensitivity = colour.SpectralDistribution(self.blue_log_sensitivity).align(self.spectral_shape,
            extrapolator_kwargs={'method': 'linear'}).align(self.spectral_shape)
        self.sensitivity = 10 ** np.stack(
            (self.red_log_sensitivity.values, self.green_log_sensitivity.values, self.blue_log_sensitivity.values)).T

        # convert relative camera exposure to absolute exposure in log lux-seconds for characteristic curve
        if self.exposure_base != 10:
            self.red_log_exposure = np.log10(self.exposure_base ** self.red_log_exposure * 10 ** self.log_H_ref[0])
            self.green_log_exposure = np.log10(self.exposure_base ** self.green_log_exposure * 10 ** self.log_H_ref[1])
            self.blue_log_exposure = np.log10(self.exposure_base ** self.blue_log_exposure * 10 ** self.log_H_ref[2])

        self.d_min = np.array(
            [np.min(x) for x in (self.red_density_curve, self.green_density_curve, self.blue_density_curve)])
        self.red_density_curve -= self.d_min[0]
        self.green_density_curve -= self.d_min[1]
        self.blue_density_curve -= self.d_min[2]
        self.d_ref = self.log_exposure_to_density(self.log_H_ref)
        self.d_max = np.array(
            [np.max(x) for x in (self.red_density_curve, self.green_density_curve, self.blue_density_curve)])

        # align spectral densities
        if hasattr(self, 'd_min_sd'):
            self.gaussian_extrapolation(self.d_min_sd)
            self.d_min_sd = self.d_min_sd.values
        else:
            self.d_min_sd = colour.sd_zeros(self.spectral_shape).values

        if hasattr(self, 'd_ref_sd'):
            self.gaussian_extrapolation(self.d_ref_sd)
        if (hasattr(self, 'red_sd') and hasattr(self, 'green_sd') and hasattr(self, 'blue_sd')):
            self.gaussian_extrapolation(self.red_sd)
            self.gaussian_extrapolation(self.green_sd)
            self.gaussian_extrapolation(self.blue_sd)
            self.spectral_density = np.stack((self.red_sd.values, self.green_sd.values, self.blue_sd.values)).T
        else:
            self.spectral_density = self.construct_spectral_density(self.d_ref_sd - self.d_min_sd)

        self.spectral_density @= np.linalg.inv(self.densiometry[self.density_measure].T @ self.spectral_density)
        self.d_min_sd = self.d_min_sd + self.spectral_density @ (
                self.d_min - self.densiometry[self.density_measure].T @ self.d_min_sd)
        self.d_ref_sd = self.spectral_density @ self.d_ref + self.d_min_sd

        if self.exposure_kelvin is None:
            self.exposure_kelvin = 6500

        gray = self.CCT_to_XYZ(self.exposure_kelvin, 0.18)
        self.XYZ_to_exp = self.sensitivity.T @ self.xyz_dual
        ref_exp = self.XYZ_to_exp @ gray
        correction_factors = self.H_ref / ref_exp
        self.XYZ_to_exp = (self.XYZ_to_exp.T * correction_factors).T

        if all([hasattr(self, x) for x in
                ['red_rms', 'green_rms', 'blue_rms', 'red_rms_density', 'green_rms_density', 'blue_rms_density']]):
            self.red_rms, self.red_rms_density = self.prepare_rms_data(self.red_rms, self.red_rms_density)
            self.green_rms, self.green_rms_density = self.prepare_rms_data(self.green_rms, self.green_rms_density)
            self.blue_rms, self.blue_rms_density = self.prepare_rms_data(self.blue_rms, self.blue_rms_density)
            if hasattr(self, 'rms'):
                ref_rms = np.interp(1, self.green_rms_density, self.green_rms)
                if self.rms > 1:
                    self.rms /= 1000
                factor = self.rms / ref_rms
                self.red_rms *= factor
                self.green_rms *= factor
                self.blue_rms *= factor

        for key, value in self.__dict__.items():
            if type(value) is np.ndarray and value.dtype is not default_dtype:
                self.__dict__[key] = value.astype(default_dtype)

    @staticmethod
    def prepare_rms_data(rms, density):
        xp = np.array(list(density.keys()), dtype=default_dtype)
        fp = np.array(list(density.values()), dtype=default_dtype)
        fp -= fp.min()
        density = np.interp(np.array(list(rms.keys()), dtype=default_dtype), xp, fp)
        rms = np.array(list(rms.values()), dtype=default_dtype)
        sorting = density.argsort()
        density = density[sorting]
        rms = rms[sorting]
        return rms, density

    def gaussian_extrapolation(self, sd: SpectralDistribution):
        def extrapolate(a_x, a_y, b_x, b_y, wavelengths, d_1=30, d_2=0.75):
            m = (a_y - b_y) / (a_x - b_x)
            d = d_1 * m / np.absolute(a_y) ** d_2
            a = a_y / np.exp(-d ** 2)
            c = a / m * -2 * d * np.exp(-d ** 2)
            b = a_x - c * d
            extrapolator = lambda x: a * np.exp(- (x - b) ** 2 / c ** 2)
            return extrapolator(wavelengths)

        sd.interpolate(self.spectral_shape)

        def_wv = self.spectral_shape.wavelengths
        wv_left = def_wv[def_wv < sd.wavelengths[0]]
        wv_right = def_wv[def_wv > sd.wavelengths[-1]]
        values_left = extrapolate(sd.wavelengths[0], sd.values[0], sd.wavelengths[1], sd.values[1], wv_left)
        values_right = extrapolate(sd.wavelengths[-1], sd.values[-1], sd.wavelengths[-2], sd.values[-2], wv_right)
        sd.values, sd.wavelengths = np.concatenate((values_left, sd.values, values_right)), np.concatenate(
            (wv_left, sd.wavelengths, wv_right))
        sd.interpolate(self.spectral_shape)

    @staticmethod
    def wavelength_argmax(distribution: SpectralDistribution, low=None, high=None):
        range = distribution.copy()
        if low is not None and high is not None:
            range.trim(colour.SpectralShape(low, high, 1))
        peak = range.wavelengths[range.values.argmax()]
        return peak

    @staticmethod
    def wavelength_argmin(distribution: SpectralDistribution, low=None, high=None):
        range = distribution.copy()
        if low is not None and high is not None:
            range.trim(colour.SpectralShape(low, high, 1))
        peak = range.wavelengths[range.values.argmin()]
        return peak

    def construct_spectral_density(self, ref_density, sigma=25):
        red_peak = FilmSpectral.wavelength_argmax(ref_density, 600, min(750, self.spectral_shape.end))
        green_peak = FilmSpectral.wavelength_argmax(ref_density, 500, 600)
        blue_peak = FilmSpectral.wavelength_argmax(ref_density, max(400, self.spectral_shape.start), 500)
        bg_cutoff = FilmSpectral.wavelength_argmin(ref_density, blue_peak, green_peak)
        gr_cutoff = FilmSpectral.wavelength_argmin(ref_density, green_peak, red_peak)

        wavelengths = ref_density.wavelengths
        factors = np.stack((np.where(gr_cutoff <= wavelengths, 1., 0.),
                            np.where((bg_cutoff < wavelengths) & (wavelengths < gr_cutoff), 1., 0.),
                            np.where(wavelengths <= bg_cutoff, 1., 0.)))
        factors = gaussian_filter(factors, sigma=sigma / self.spectral_shape.interval, axes=1)

        out = (factors * ref_density.values).T
        return out

    def log_exposure_to_density(self, log_exposure):
        red_density = np.interp(log_exposure[..., 0], self.red_log_exposure, self.red_density_curve)
        green_density = np.interp(log_exposure[..., 1], self.green_log_exposure, self.green_density_curve)
        blue_density = np.interp(log_exposure[..., 2], self.blue_log_exposure, self.blue_density_curve)

        return np.stack([red_density, green_density, blue_density], axis=-1, dtype=default_dtype)

    def compute_print_matrix(self, print_film, **kwargs):
        printer_light = self.compute_printer_light(print_film, **kwargs)
        # compute max exposure produced by unfiltered printer light
        peak_exposure = np.log10(print_film.sensitivity.T @ printer_light)
        # compute density matrix from print film sensitivity under adjusted printer light
        print_sensitivity = (print_film.sensitivity.T * printer_light).T
        print_sensitivity /= np.sum(print_sensitivity, axis=0)
        density_matrix = print_sensitivity.T @ self.spectral_density
        density_base = print_sensitivity.T @ self.d_min_sd
        return density_matrix, peak_exposure - density_base

    def compute_printer_light(self, print_film, compensation=None):
        if compensation is None:
            compensation = np.zeros(3, dtype=default_dtype)
        compensation = 2 ** compensation
        # transmitted printer lights by middle gray negative
        reduced_lights = (self.printer_lights.T * 10 ** -self.d_ref_sd).T
        # adjust printer lights to produce neutral exposure with middle gray negative
        light_factors = np.linalg.inv(print_film.sensitivity.T @ reduced_lights) @ np.multiply(print_film.H_ref,
                                                                                               compensation)
        printer_light = np.sum(self.printer_lights * light_factors, axis=1)
        return printer_light

    def compute_projection_light(self, projector_kelvin=5500, reference_kelvin=6504, white_point=1.):
        reference_light = colour.sd_blackbody(reference_kelvin).align(self.spectral_shape).normalise().values
        projector_light = colour.sd_blackbody(projector_kelvin).align(self.spectral_shape).normalise().values
        reference_white = colour.xyY_to_XYZ([*colour.CCT_to_xy(reference_kelvin), 1.])
        xyz_cmfs = self.xyz_cmfs * (reference_white / (self.xyz_cmfs.T @ reference_light))
        peak_xyz = colour.XYZ_to_RGB(xyz_cmfs.T @ (projector_light * 10 ** -self.d_min_sd), "sRGB")
        projector_light /= np.max(peak_xyz) / white_point
        return projector_light, xyz_cmfs

    @staticmethod
    def generate_conversion(negative_film, print_film=None, input_colourspace="ARRI Wide Gamut 4", measure_time=False,
                            output_colourspace="sRGB", projector_kelvin=6500, matrix_method=False, exp_comp=0,
                            printer_light_comp=None, white_point=1., mode='full', exposure_kelvin=6500, d_buffer=0.5,
                            **kwargs):
        pipeline = []
        if mode == 'negative' or mode == 'full':

            if input_colourspace is not None:
                pipeline.append((lambda x: colour.RGB_to_XYZ(x, input_colourspace, apply_cctf_decoding=True), "input"))

            if exposure_kelvin != negative_film.exposure_kelvin:
                pipeline.append((lambda x: colour.chromatic_adaptation(x, FilmSpectral.CCT_to_XYZ(exposure_kelvin), FilmSpectral.CCT_to_XYZ(negative_film.exposure_kelvin)), "chromatic adaptation"))

            exp_comp = 2 ** exp_comp
            pipeline.append((lambda x: np.log10(np.clip(np.dot(x * exp_comp, negative_film.XYZ_to_exp.T), 0.0001, None)),
                             "log exposure"))
            pipeline.append((negative_film.log_exposure_to_density, "characteristic curve"))

        density_scale = (negative_film.d_max.max() + d_buffer)
        if mode == 'negative':
            pipeline.append((lambda x: (x + d_buffer / 2) / density_scale, 'scale density'))
        elif mode == 'print':
            pipeline.append((lambda x: x * density_scale - d_buffer / 2, 'scale density'))

        if mode == 'print' or mode == 'full':
            if print_film is not None:
                if matrix_method:
                    density_matrix, peak_exposure = negative_film.compute_print_matrix(print_film,
                                                                                       compenation=printer_light_comp)
                    pipeline.append((lambda x: peak_exposure - np.dot(x, density_matrix.T), "printing matrix"))
                else:
                    printer_light = negative_film.compute_printer_light(print_film, compensation=printer_light_comp)
                    density_neg = negative_film.spectral_density.T
                    printing_mat = (print_film.sensitivity.T * printer_light * 10 ** -negative_film.d_min_sd).T
                    pipeline.append((lambda x: np.log10(
                        np.clip(np.dot(10 ** -np.dot(x, density_neg), printing_mat), 0.0001, None)), "printing"))
                pipeline.append((print_film.log_exposure_to_density, "characteristic curve print"))
                output_film = print_film
            else:
                output_film = negative_film

            projection_light, xyz_cmfs = output_film.compute_projection_light(projector_kelvin=projector_kelvin,
                                                                              white_point=white_point)
            d_min_sd = output_film.d_min_sd

            # TODO: matrix output here
            density_mat = output_film.spectral_density
            output_mat = (xyz_cmfs.T * projection_light * 10 ** -d_min_sd).T
            pipeline.append((lambda x: np.dot(10 ** -np.dot(x, density_mat.T), output_mat), "output matrix"))

            if output_colourspace is not None:
                pipeline.append((lambda x: colour.XYZ_to_RGB(x, output_colourspace, apply_cctf_encoding=True), "output"))

        def convert(x):
            start = time.time()
            for transform, title in pipeline:
                x = transform(x)
                if measure_time:
                    end = time.time()
                    print(f"{title:25} {end - start:.2f}s {x.dtype} {x.shape}")
                start = time.time()
            return np.clip(x, 0, 1)

        if mode == 'print' or mode == 'negative':
            return convert, density_scale
        else:
            return convert, 0

    @staticmethod
    def CCT_to_XYZ(CCT, Y=1.):
        xy = colour.CCT_to_xy(CCT)
        xyY = (xy[0], xy[1], Y)
        XYZ = colour.xyY_to_XYZ(xyY)
        return XYZ