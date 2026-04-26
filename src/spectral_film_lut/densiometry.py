"""
Here data and functions for handling and specifying density values is included.
"""

import colour
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from spectral_film_lut.config import DEFAULT_DTYPE, SPECTRAL_SHAPE

STATUS_A = [
    {
        599: 2.298,
        600: 2.568,
        610: 4.638,
        620: 5.000,
        630: 4.871,
        640: 4.604,
        650: 4.286,
        660: 3.900,
        670: 3.551,
        680: 3.165,
        690: 2.776,
        700: 2.383,
        710: 1.970,
        720: 1.551,
        730: 1.141,
        740: 0.741,
        750: 0.341,
        751: 0.301,
    },
    {
        499: 1.430,
        500: 1.650,
        510: 3.822,
        520: 4.782,
        530: 5.000,
        540: 4.906,
        550: 4.644,
        560: 4.221,
        570: 3.609,
        580: 2.766,
        590: 1.579,
        591: 1.409,
    },
    {
        419: 3.222,
        420: 3.602,
        430: 4.819,
        440: 5.000,
        450: 4.912,
        460: 4.620,
        470: 4.040,
        480: 2.989,
        490: 1.566,
        500: 0.165,
        501: 0.015,
    },
]
"""Status A density measure."""

STATUS_M = [
    {
        619: 1.849,
        620: 2.109,
        630: 4.479,
        640: 5.000,
        650: 4.899,
        660: 4.578,
        670: 4.252,
        680: 3.875,
        690: 3.491,
        700: 3.099,
        710: 2.687,
        720: 2.269,
        730: 1.859,
        740: 1.449,
        750: 1.054,
        760: 0.654,
        770: 0.254,
        771: 0.214,
    },
    {
        469: 1.046,
        470: 1.152,
        480: 2.207,
        490: 3.156,
        500: 3.804,
        510: 4.272,
        520: 4.626,
        530: 4.872,
        540: 5.000,
        550: 4.995,
        560: 4.818,
        570: 4.458,
        580: 3.915,
        590: 3.172,
        600: 2.239,
        610: 1.070,
        611: 0.950,
    },
    {
        409: 1.853,
        410: 2.103,
        420: 4.111,
        430: 4.632,
        440: 4.871,
        450: 5.000,
        460: 4.955,
        470: 4.743,
        480: 4.343,
        490: 3.743,
        500: 2.990,
        510: 1.852,
        511: 1.632,
    },
]
"""Status M density measure."""

APD = {
    360: (0.0000, 0.0000, 0.0000),
    362: (0.0000, 0.0000, 0.0000),
    364: (0.0000, 0.0000, 0.0000),
    366: (0.0000, 0.0000, 0.0000),
    368: (0.0000, 0.0000, 0.0001),
    370: (0.0000, 0.0000, 0.0001),
    372: (0.0000, 0.0001, 0.0003),
    374: (0.0002, 0.0001, 0.0005),
    376: (0.0005, 0.0002, 0.0007),
    378: (0.0009, 0.0002, 0.0010),
    380: (0.0013, 0.0002, 0.0012),
    382: (0.0013, 0.0002, 0.0013),
    384: (0.0010, 0.0001, 0.0013),
    386: (0.0006, 0.0001, 0.0014),
    388: (0.0002, 0.0000, 0.0016),
    390: (0.0000, 0.0000, 0.0020),
    392: (0.0000, 0.0000, 0.0028),
    394: (0.0000, 0.0000, 0.0037),
    396: (0.0000, 0.0000, 0.0050),
    398: (0.0000, 0.0000, 0.0065),
    400: (0.0000, 0.0000, 0.0083),
    402: (0.0000, 0.0000, 0.0107),
    404: (0.0000, 0.0000, 0.0138),
    406: (0.0000, 0.0000, 0.0175),
    408: (0.0000, 0.0000, 0.0219),
    410: (0.0000, 0.0000, 0.0268),
    412: (0.0000, 0.0000, 0.0325),
    414: (0.0000, 0.0000, 0.0392),
    416: (0.0000, 0.0000, 0.0471),
    418: (0.0000, 0.0000, 0.0562),
    420: (0.0000, 0.0000, 0.0667),
    422: (0.0000, 0.0000, 0.0790),
    424: (0.0000, 0.0000, 0.0935),
    426: (0.0000, 0.0000, 0.1095),
    428: (0.0000, 0.0000, 0.1265),
    430: (0.0000, 0.0000, 0.1434),
    432: (0.0000, 0.0000, 0.1591),
    434: (0.0000, 0.0000, 0.1738),
    436: (0.0000, 0.0000, 0.1891),
    438: (0.0000, 0.0000, 0.2068),
    440: (0.0000, 0.0000, 0.2290),
    442: (0.0000, 0.0003, 0.2557),
    444: (0.0000, 0.0010, 0.2866),
    446: (0.0000, 0.0020, 0.3231),
    448: (0.0000, 0.0030, 0.3670),
    450: (0.0000, 0.0039, 0.4204),
    452: (0.0000, 0.0046, 0.4863),
    454: (0.0000, 0.0052, 0.5635),
    456: (0.0000, 0.0058, 0.6478),
    458: (0.0000, 0.0064, 0.7344),
    460: (0.0000, 0.0071, 0.8174),
    462: (0.0000, 0.0079, 0.8952),
    464: (0.0000, 0.0089, 0.9611),
    466: (0.0000, 0.0099, 1.0000),
    468: (0.0000, 0.0108, 0.9959),
    470: (0.0000, 0.0117, 0.9339),
    472: (0.0000, 0.0126, 0.8125),
    474: (0.0000, 0.0134, 0.6538),
    476: (0.0000, 0.0142, 0.4825),
    478: (0.0000, 0.0150, 0.3240),
    480: (0.0000, 0.0158, 0.2028),
    482: (0.0000, 0.0163, 0.1230),
    484: (0.0000, 0.0165, 0.0710),
    486: (0.0000, 0.0166, 0.0407),
    488: (0.0000, 0.0167, 0.0243),
    490: (0.0000, 0.0174, 0.0150),
    492: (0.0000, 0.0184, 0.0087),
    494: (0.0000, 0.0196, 0.0051),
    496: (0.0000, 0.0211, 0.0033),
    498: (0.0000, 0.0230, 0.0025),
    500: (0.0000, 0.0256, 0.0020),
    502: (0.0000, 0.0296, 0.0014),
    504: (0.0000, 0.0356, 0.0009),
    506: (0.0000, 0.0430, 0.0004),
    508: (0.0000, 0.0514, 0.0001),
    510: (0.0000, 0.0600, 0.0000),
    512: (0.0000, 0.0690, 0.0000),
    514: (0.0000, 0.0787, 0.0000),
    516: (0.0000, 0.0888, 0.0000),
    518: (0.0000, 0.0990, 0.0000),
    520: (0.0000, 0.1095, 0.0000),
    522: (0.0000, 0.1201, 0.0000),
    524: (0.0000, 0.1310, 0.0000),
    526: (0.0000, 0.1427, 0.0000),
    528: (0.0000, 0.1556, 0.0000),
    530: (0.0000, 0.1704, 0.0000),
    532: (0.0000, 0.1829, 0.0000),
    534: (0.0000, 0.1944, 0.0000),
    536: (0.0000, 0.2151, 0.0000),
    538: (0.0000, 0.2556, 0.0000),
    540: (0.0000, 0.3269, 0.0000),
    542: (0.0000, 0.4552, 0.0000),
    544: (0.0000, 0.6303, 0.0000),
    546: (0.0000, 0.8082, 0.0000),
    548: (0.0000, 0.9457, 0.0000),
    550: (0.0000, 1.0000, 0.0000),
    552: (0.0000, 0.9408, 0.0000),
    554: (0.0000, 0.7932, 0.0000),
    556: (0.0000, 0.5983, 0.0000),
    558: (0.0000, 0.4023, 0.0000),
    560: (0.0000, 0.2559, 0.0000),
    562: (0.0000, 0.1744, 0.0000),
    564: (0.0000, 0.1285, 0.0000),
    566: (0.0000, 0.1051, 0.0000),
    568: (0.0000, 0.0907, 0.0000),
    570: (0.0000, 0.0718, 0.0000),
    572: (0.0023, 0.0496, 0.0000),
    574: (0.0075, 0.0330, 0.0000),
    576: (0.0133, 0.0211, 0.0000),
    578: (0.0177, 0.0129, 0.0000),
    580: (0.0197, 0.0071, 0.0000),
    582: (0.0194, 0.0033, 0.0000),
    584: (0.0181, 0.0013, 0.0000),
    586: (0.0163, 0.0003, 0.0000),
    588: (0.0146, 0.0000, 0.0000),
    590: (0.0135, 0.0000, 0.0000),
    592: (0.0128, 0.0000, 0.0000),
    594: (0.0125, 0.0000, 0.0000),
    596: (0.0125, 0.0000, 0.0000),
    598: (0.0128, 0.0000, 0.0000),
    600: (0.0134, 0.0000, 0.0000),
    602: (0.0149, 0.0000, 0.0000),
    604: (0.0176, 0.0000, 0.0000),
    606: (0.0214, 0.0000, 0.0000),
    608: (0.0262, 0.0000, 0.0000),
    610: (0.0318, 0.0000, 0.0000),
    612: (0.0383, 0.0000, 0.0000),
    614: (0.0461, 0.0000, 0.0000),
    616: (0.0553, 0.0000, 0.0000),
    618: (0.0655, 0.0000, 0.0000),
    620: (0.0766, 0.0000, 0.0000),
    622: (0.0890, 0.0000, 0.0000),
    624: (0.1028, 0.0000, 0.0000),
    626: (0.1175, 0.0000, 0.0000),
    628: (0.1323, 0.0000, 0.0000),
    630: (0.1464, 0.0000, 0.0000),
    632: (0.1597, 0.0000, 0.0000),
    634: (0.1724, 0.0000, 0.0000),
    636: (0.1843, 0.0000, 0.0000),
    638: (0.1947, 0.0000, 0.0000),
    640: (0.2033, 0.0000, 0.0000),
    642: (0.2094, 0.0000, 0.0000),
    644: (0.2138, 0.0000, 0.0000),
    646: (0.2174, 0.0000, 0.0000),
    648: (0.2215, 0.0000, 0.0000),
    650: (0.2275, 0.0000, 0.0000),
    652: (0.2348, 0.0000, 0.0000),
    654: (0.2432, 0.0000, 0.0000),
    656: (0.2544, 0.0000, 0.0000),
    658: (0.2702, 0.0000, 0.0000),
    660: (0.2923, 0.0000, 0.0000),
    662: (0.3213, 0.0000, 0.0000),
    664: (0.3560, 0.0000, 0.0000),
    666: (0.3954, 0.0000, 0.0000),
    668: (0.4386, 0.0000, 0.0000),
    670: (0.4845, 0.0000, 0.0000),
    672: (0.5337, 0.0000, 0.0000),
    674: (0.5867, 0.0000, 0.0000),
    676: (0.6418, 0.0000, 0.0000),
    678: (0.6977, 0.0000, 0.0000),
    680: (0.7527, 0.0000, 0.0000),
    682: (0.8112, 0.0000, 0.0000),
    684: (0.8727, 0.0000, 0.0000),
    686: (0.9289, 0.0000, 0.0000),
    688: (0.9721, 0.0000, 0.0000),
    690: (0.9950, 0.0000, 0.0000),
    692: (1.0000, 0.0000, 0.0000),
    694: (0.9928, 0.0000, 0.0000),
    696: (0.9714, 0.0000, 0.0000),
    698: (0.9336, 0.0000, 0.0000),
    700: (0.8776, 0.0000, 0.0000),
    702: (0.7997, 0.0000, 0.0000),
    704: (0.7045, 0.0000, 0.0000),
    706: (0.6028, 0.0000, 0.0000),
    708: (0.5049, 0.0000, 0.0000),
    710: (0.4203, 0.0000, 0.0000),
    712: (0.3498, 0.0000, 0.0000),
    714: (0.2876, 0.0000, 0.0000),
    716: (0.2324, 0.0000, 0.0000),
    718: (0.1834, 0.0000, 0.0000),
    720: (0.1396, 0.0000, 0.0000),
    722: (0.0984, 0.0000, 0.0000),
    724: (0.0603, 0.0000, 0.0000),
    726: (0.0289, 0.0000, 0.0000),
    728: (0.0077, 0.0000, 0.0000),
    730: (0.0000, 0.0000, 0.0000),
}
"""ACES printing density."""


def interpolate_status_density(
    status: list[dict[int, float]], spectral_shape=SPECTRAL_SHAPE
) -> np.ndarray:
    """
    Converts logarithmic status density values to a normalized linear status density
    matrix.
    """
    result = []
    for i, density in enumerate(status):
        density = colour.SpectralDistribution(density).extrapolate(
            spectral_shape, extrapolator_kwargs={"method": "linear"}
        )
        density.values = 10**density.values
        density.interpolate(spectral_shape)
        density /= sum(density.values)
        result.append(density)
    result = colour.MultiSpectralDistributions(result)
    return result.values


STATUS_A = np.asarray(interpolate_status_density(STATUS_A))
STATUS_M = np.asarray(interpolate_status_density(STATUS_M))
APD = np.asarray(
    colour.MultiSpectralDistributions(APD).align(SPECTRAL_SHAPE).values,
    dtype=DEFAULT_DTYPE,
)
APD /= np.sum(APD, axis=0)

DENSIOMETRY = {
    "status_a": STATUS_A,
    "status_m": STATUS_M,
    "apd": APD,
    "absolute": STATUS_A,
}
"""A dict with relevant density measures."""


def wavelength_argmax(
    distribution: colour.SpectralDistribution, low=None, high=None
) -> int:
    """Gets the argmax of a spectral distribution."""
    range = distribution.copy()
    if low is not None and high is not None:
        range.trim(colour.SpectralShape(low, high, 1))
    peak = range.wavelengths[range.values.argmax()]
    return peak


def wavelength_argmin(
    distribution: colour.SpectralDistribution, low=None, high=None
) -> int:
    """Gets the argmax of a spectral distribution."""
    range = distribution.copy()
    if low is not None and high is not None:
        range.trim(colour.SpectralShape(low, high, 1))
    peak = range.wavelengths[range.values.argmin()]
    return peak


def construct_spectral_density(
    ref_density: colour.SpectralDistribution, sigma=25
) -> np.ndarray:
    """Split single density curve into separate layers using local extrema."""
    red_peak = wavelength_argmax(ref_density, 600, min(750, SPECTRAL_SHAPE.end))
    green_peak = wavelength_argmax(ref_density, 500, 600)
    blue_peak = wavelength_argmax(ref_density, max(400, SPECTRAL_SHAPE.start), 500)
    bg_cutoff = wavelength_argmin(ref_density, blue_peak, green_peak)
    gr_cutoff = wavelength_argmin(ref_density, green_peak, red_peak)

    wavelengths = np.asarray(ref_density.wavelengths)
    factors = np.stack(
        (
            np.where(gr_cutoff <= wavelengths, 1.0, 0.0),
            np.where((bg_cutoff < wavelengths) & (wavelengths < gr_cutoff), 1.0, 0.0),
            np.where(wavelengths <= bg_cutoff, 1.0, 0.0),
        )
    )
    factors = ndimage.gaussian_filter(
        factors, sigma=(0, sigma / SPECTRAL_SHAPE.interval)
    ).astype(DEFAULT_DTYPE)

    out = (factors * np.asarray(ref_density.values)).T
    return out


def compute_printer_lights() -> np.ndarray:
    """
    Converts the APD printer light to a three light matrix by splitting into separate
    red, green, and blue lights.
    """
    printer_light = colour.SpectralDistribution(
        {
            360: 0.0000,
            362: 0.0000,
            364: 0.0000,
            366: 0.0000,
            368: 0.0001,
            370: 0.0001,
            372: 0.0002,
            374: 0.0005,
            376: 0.0007,
            378: 0.0010,
            380: 0.0013,
            382: 0.0014,
            384: 0.0016,
            386: 0.0018,
            388: 0.0022,
            390: 0.0030,
            392: 0.0043,
            394: 0.0061,
            396: 0.0084,
            398: 0.0113,
            400: 0.0147,
            402: 0.0189,
            404: 0.0237,
            406: 0.0290,
            408: 0.0345,
            410: 0.0399,
            412: 0.0451,
            414: 0.0505,
            416: 0.0560,
            418: 0.0614,
            420: 0.0669,
            422: 0.0724,
            424: 0.0779,
            426: 0.0834,
            428: 0.0889,
            430: 0.0944,
            432: 0.0998,
            434: 0.1052,
            436: 0.1106,
            438: 0.1159,
            440: 0.1211,
            442: 0.1261,
            444: 0.1309,
            446: 0.1357,
            448: 0.1406,
            450: 0.1456,
            452: 0.1508,
            454: 0.1562,
            456: 0.1616,
            458: 0.1671,
            460: 0.1725,
            462: 0.1787,
            464: 0.1856,
            466: 0.1921,
            468: 0.1969,
            470: 0.1989,
            472: 0.1988,
            474: 0.1981,
            476: 0.1964,
            478: 0.1933,
            480: 0.1885,
            482: 0.1802,
            484: 0.1684,
            486: 0.1563,
            488: 0.1468,
            490: 0.1428,
            492: 0.1428,
            494: 0.1439,
            496: 0.1471,
            498: 0.1534,
            500: 0.1637,
            502: 0.1825,
            504: 0.2114,
            506: 0.2476,
            508: 0.2883,
            510: 0.3307,
            512: 0.3762,
            514: 0.4265,
            516: 0.4794,
            518: 0.5325,
            520: 0.5836,
            522: 0.6338,
            524: 0.6842,
            526: 0.7325,
            528: 0.7764,
            530: 0.8137,
            532: 0.8434,
            534: 0.8669,
            536: 0.8852,
            538: 0.8993,
            540: 0.9104,
            542: 0.9179,
            544: 0.9220,
            546: 0.9248,
            548: 0.9285,
            550: 0.9352,
            552: 0.9469,
            554: 0.9621,
            556: 0.9779,
            558: 0.9915,
            560: 1.0000,
            562: 1.0036,
            564: 1.0032,
            566: 0.9969,
            568: 0.9832,
            570: 0.9602,
            572: 0.9235,
            574: 0.8728,
            576: 0.8134,
            578: 0.7502,
            580: 0.6882,
            582: 0.6252,
            584: 0.5589,
            586: 0.4947,
            588: 0.4376,
            590: 0.3930,
            592: 0.3627,
            594: 0.3436,
            596: 0.3334,
            598: 0.3297,
            600: 0.3304,
            602: 0.3467,
            604: 0.3866,
            606: 0.4423,
            608: 0.5062,
            610: 0.5705,
            612: 0.6370,
            614: 0.7101,
            616: 0.7863,
            618: 0.8623,
            620: 0.9345,
            622: 1.0029,
            624: 1.0690,
            626: 1.1317,
            628: 1.1898,
            630: 1.2424,
            632: 1.2890,
            634: 1.3303,
            636: 1.3667,
            638: 1.3985,
            640: 1.4261,
            642: 1.4499,
            644: 1.4700,
            646: 1.4867,
            648: 1.5005,
            650: 1.5118,
            652: 1.5206,
            654: 1.5273,
            656: 1.5320,
            658: 1.5348,
            660: 1.5359,
            662: 1.5355,
            664: 1.5336,
            666: 1.5305,
            668: 1.5263,
            670: 1.5212,
            672: 1.5151,
            674: 1.5079,
            676: 1.4994,
            678: 1.4892,
            680: 1.4771,
            682: 1.4631,
            684: 1.4480,
            686: 1.4332,
            688: 1.4197,
            690: 1.4088,
            692: 1.4015,
            694: 1.3965,
            696: 1.3926,
            698: 1.3880,
            700: 1.3813,
            702: 1.3714,
            704: 1.3590,
            706: 1.3450,
            708: 1.3305,
            710: 1.3163,
            712: 1.3030,
            714: 1.2904,
            716: 1.2781,
            718: 1.2656,
            720: 1.2526,
            722: 1.2387,
            724: 1.2242,
            726: 1.2091,
            728: 1.1937,
            730: 1.1782,
        },
        dtype=DEFAULT_DTYPE,
    )

    printer_light.align(SPECTRAL_SHAPE, extrapolator_kwargs={"method": "linear"})

    return construct_spectral_density(printer_light, sigma=5)


PRINTER_LIGHTS = compute_printer_lights()
"""The RGB printer lights build from the APD documentation."""


def adx16_encode(apd: np.ndarray, apd_min=None, scaling=1.0) -> np.ndarray:
    """
    Converts from absolute APD density values to ADX16 values.
    Instead of as 16 bit int, the output is encoded as a float in the range [0, 1].

    Args:
        apd: The density values.
        apd_min: The D-min values of the film to compensate for.
        scaling: Linear scaling applied to the values.
            For 1.0 it is ADX16-like and for 4.0 it is ADX10-like.

    Returns:
        ADX-like encoded values in the [0, 1] range.
    """
    if apd_min is None:
        apd_min = np.zeros(3, dtype=DEFAULT_DTYPE)
    # Scaling factors per channel: R, G, B
    scale = np.array([1.00, 0.92, 0.95], dtype=DEFAULT_DTYPE)

    if apd.shape[-1] != 3:
        scale = 1
        apd_min = apd_min[0]

    # Perform vectorized computation
    adx = scale * (8000.0 / 65535.0) * (apd - apd_min) + (1520.0 / 65535.0)

    # Round to nearest integer and clip to 0–65535 range
    adx = np.clip(adx * scaling, 0, 1)

    # Convert to uint16 to save space (fits ADX10 range)
    return adx


def adx16_decode(
    adx: np.ndarray, apd_min: np.ndarray | None = None, scaling: float = 1.0
):
    """
    Converts from ADX16 values to absolute APD density.
    Instead of as 16 bit int the output is expected as a float in the range [0, 1].

    Args:
        adx: ADX16 values in the range [0, 1].
        apd_min: The D-min values of the film that has been compensated for.
        scaling: Linear scaling removed from the values.
            For 1.0 it is ADX16-like and for 4.0 it is ADX10-like.

    Returns:
        Absolute APD density values.
    """
    if apd_min is None:
        apd_min = np.zeros(3, dtype=DEFAULT_DTYPE)
    # Scaling factors per channel: R, G, B
    scale = np.array([1.00, 0.92, 0.95], dtype=DEFAULT_DTYPE)

    if adx.shape[-1] != 3:
        scale = 1
        apd_min = apd_min[0]

    apd = (
        (adx.astype(DEFAULT_DTYPE) / scaling - 1520.0 / 65535.0)
        / (scale * 8000.0 / 65535.0)
    ) + apd_min

    return apd


def output_to_density(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray | None = None,
    method: str = "lm",
) -> np.ndarray:
    """
    Numerically invert the mapping:
        y = (10 ** -(x @ a.T)) @ b
    to recover x.

    Parameters
    ----------
    y : array-like, shape (p,)
        Output vector.
    a : array-like, shape (m, n)
        Density matrix (used in forward map).
    b : array-like, shape (m, p)
        Output matrix (used in forward map).
    x0 : array-like, shape (n,), optional
        Initial guess for x. Defaults to zeros.
    method : str, optional
        Least-squares solver method: 'lm', 'trf', or 'dogbox'.

    Returns
    -------
    x : ndarray, shape (n,)
        Estimated vector such that (10 ** -(x @ a.T)) @ b ≈ y
    """
    n = a.shape[1]
    if x0 is None:
        x0 = np.zeros(n)

    def residual(x):
        s = 10 ** -(x @ a.T)  # shape (m,)
        y_pred = s @ b  # shape (p,)
        return y_pred - y

    res = least_squares(residual, x0, method=method)
    return np.asarray(res.x)
