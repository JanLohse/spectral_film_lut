from spectral_film_lut.film_spectral import *


class Kodak5222(FilmSpectral):
    def __init__(self, dev_time=6):
        super().__init__()

        self.iso = 250
        self.density_measure = 'bw'

        # spectral sensitivity
        self.log_sensitivity = [
            {400.4723: 1.9996, 407.7061: 2.0319, 415.8733: 2.0641, 426.6073: 2.0827, 439.7915: 2.0579, 450.2922: 2.0009,
             464.4097: 1.8955, 480.5108: 1.7417, 490.6614: 1.6413, 499.7620: 1.5718, 507.2291: 1.5197, 516.6797: 1.5036,
             525.0803: 1.5185, 535.5809: 1.5582, 549.3485: 1.6127, 561.7159: 1.6524, 573.8501: 1.6760, 584.1174: 1.6549,
             591.2345: 1.6264, 598.7017: 1.5991, 610.0190: 1.5731, 619.1196: 1.5879, 622.3865: 1.5954, 629.8536: 1.5532,
             633.3539: 1.4838, 646.3047: 0.9208, 649.8049: 0.7124, 661.5890: 0.0304, 669.7562: -0.4904}]

        # sensiometry - characteristic curve
        curve = {
            4: {-2.9427: 0.2214, -2.8617: 0.2280, -2.7808: 0.2356, -2.6999: 0.2435, -2.6190: 0.2593, -2.5381: 0.2799,
                -2.4572: 0.3016, -2.3762: 0.3245, -2.2953: 0.3508, -2.2144: 0.3839, -2.1335: 0.4188, -2.0526: 0.4545,
                -1.9716: 0.4913, -1.8907: 0.5279, -1.8098: 0.5657, -1.7289: 0.6050, -1.6480: 0.6443, -1.5670: 0.6857,
                -1.4861: 0.7266, -1.4052: 0.7682, -1.3243: 0.8110, -1.2434: 0.8538, -1.1624: 0.8956, -1.0815: 0.9368,
                -1.0006: 0.9786, -0.9197: 1.0190, -0.8388: 1.0587, -0.7579: 1.0975, -0.6769: 1.1358, -0.5960: 1.1718,
                -0.5151: 1.2080, -0.4361: 1.2434},
            5: {-2.9388: 0.2275, -2.8579: 0.2368, -2.7770: 0.2477, -2.6961: 0.2692, -2.6151: 0.2924, -2.5342: 0.3089,
                -2.4533: 0.3344, -2.3724: 0.3622, -2.2915: 0.3935, -2.2105: 0.4300, -2.1296: 0.4691, -2.0487: 0.5115,
                -1.9678: 0.5556, -1.8869: 0.6008, -1.8059: 0.6453, -1.7250: 0.6900, -1.6441: 0.7343, -1.5632: 0.7806,
                -1.4823: 0.8270, -1.4014: 0.8745, -1.3204: 0.9224, -1.2395: 0.9693, -1.1586: 1.0165, -1.0777: 1.0632,
                -0.9968: 1.1085, -0.9158: 1.1534, -0.8349: 1.1976, -0.7540: 1.2401, -0.6731: 1.2820, -0.5922: 1.3213,
                -0.5112: 1.3594, -0.4342: 1.3940},
            6: {-2.9388: 0.2270, -2.8579: 0.2387, -2.7789: 0.2556, -2.6906: 0.2844, -2.6074: 0.3115, -2.5265: 0.3430,
                -2.4456: 0.3804, -2.3647: 0.4218, -2.2838: 0.4678, -2.2028: 0.5164, -2.1219: 0.5649, -2.0410: 0.6126,
                -1.9601: 0.6617, -1.8792: 0.7113, -1.7982: 0.7611, -1.7173: 0.8118, -1.6364: 0.8653, -1.5555: 0.9205,
                -1.4746: 0.9752, -1.3936: 1.0300, -1.3127: 1.0854, -1.2318: 1.1409, -1.1509: 1.1957, -1.0700: 1.2500,
                -0.9890: 1.3038, -0.9081: 1.3563, -0.8272: 1.4072, -0.7463: 1.4575, -0.6654: 1.5057, -0.5845: 1.5516,
                -0.5035: 1.5982, -0.4303: 1.6412},
            9: {-2.9388: 0.2710, -2.8579: 0.2863, -2.7770: 0.3082, -2.6961: 0.3388, -2.6151: 0.3770, -2.5342: 0.4222,
                -2.4533: 0.4747, -2.3724: 0.5342, -2.2915: 0.5988, -2.2105: 0.6665, -2.1296: 0.7356, -2.0487: 0.8041,
                -1.9678: 0.8706, -1.8869: 0.9366, -1.8059: 1.0021, -1.7250: 1.0666, -1.6441: 1.1301, -1.5632: 1.1924,
                -1.4823: 1.2539, -1.4014: 1.3158, -1.3204: 1.3777, -1.2395: 1.4393, -1.1586: 1.5003, -1.0777: 1.5605,
                -0.9968: 1.6187, -0.9158: 1.6759, -0.8349: 1.7316, -0.7540: 1.7861, -0.6731: 1.8387, -0.5922: 1.8885,
                -0.5112: 1.9365, -0.4342: 1.9800},
            12: {-2.9273: 0.2951, -2.8463: 0.3176, -2.7654: 0.3457, -2.6845: 0.3815, -2.6036: 0.4259, -2.5227: 0.4788,
                 -2.4417: 0.5403, -2.3608: 0.6099, -2.2818: 0.6867, -2.2048: 0.7665, -2.1296: 0.8487, -2.0564: 0.9295,
                 -1.9832: 1.0095, -1.9081: 1.0920, -1.8329: 1.1721, -1.7559: 1.2504, -1.6749: 1.3318, -1.5959: 1.4138,
                 -1.5169: 1.4937, -1.4380: 1.5758, -1.3590: 1.6565, -1.2800: 1.7382, -1.2010: 1.8182, -1.1201: 1.8994,
                 -1.0391: 1.9801, -0.9582: 2.0577, -0.8773: 2.1332, -0.7964: 2.2068, -0.7155: 2.2767, -0.6345: 2.3443,
                 -0.5536: 2.4082, -0.4727: 2.4680, -0.4149: 2.5084}}
        curve = curve[dev_time]
        self.log_exposure = [xp.array(list(curve.keys()), dtype=default_dtype)]
        self.density_curve = [xp.array(list(curve.values()), dtype=default_dtype)]

        self.mtf = [{2.4775: 1.1175, 3.0088: 1.1880, 3.9729: 1.2330, 5.2912: 1.2151, 7.7660: 1.1707, 12.6234: 1.0838,
                     17.6382: 0.9888, 23.0617: 0.8566, 31.0944: 0.6934, 40.6554: 0.5315, 51.6104: 0.3925,
                     63.4554: 0.2961, 77.6361: 0.2198, 100.1441: 0.1545}]

        # curve from kodak 2303
        self.rms_curve = [
            {-0.0046: 0.0624, 0.3563: 0.0684, 0.6135: 0.0805, 0.7653: 0.0865, 0.8861: 0.1147, 1.0194: 0.1851,
             1.1278: 0.2615, 1.2239: 0.3501, 1.3029: 0.4708, 1.4284: 0.7001, 1.6050: 1.1146, 1.9101: 2.0802,
             2.1425: 2.8689, 2.2603: 3.2270, 2.3176: 3.4000}]
        self.rms_density = [
            {-0.0017: 0.0018, 0.1229: 0.0020, 0.2782: 0.0019, 0.4386: 0.0023, 0.6195: 0.0024, 0.7833: 0.0027,
             0.8840: 0.0033, 1.0273: 0.0040, 1.2235: 0.0050, 1.4352: 0.0070, 1.6109: 0.0090, 1.8362: 0.0114,
             2.0802: 0.0140, 2.2747: 0.0158, 2.4215: 0.0175, 2.5580: 0.0211}]
        self.rms = 14

        self.calibrate()


class Kodak5222Dev4(Kodak5222):
    def __init__(self, *args, **kwargs):
        super().__init__(4, *args, **kwargs)


class Kodak5222Dev5(Kodak5222):
    def __init__(self, *args, **kwargs):
        super().__init__(5, *args, **kwargs)


class Kodak5222Dev6(Kodak5222):
    def __init__(self, *args, **kwargs):
        super().__init__(6, *args, **kwargs)


class Kodak5222Dev9(Kodak5222):
    def __init__(self, *args, **kwargs):
        super().__init__(9, *args, **kwargs)


class Kodak5222Dev12(Kodak5222):
    def __init__(self, *args, **kwargs):
        super().__init__(12, *args, **kwargs)
