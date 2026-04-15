import math
import time

import colour
import numpy as np
from numba import njit, prange
from scipy.optimize import nnls

from spectral_film_lut.utils import default_dtype, spectral_shape

XYZ_CMFS = np.asarray(
    colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    .align(spectral_shape)
    .values,
    dtype=default_dtype,
)


@njit(parallel=True)
def xys_to_XYZ(xys):
    h, w, c = xys.shape
    out = np.empty((h, w, c), dtype=np.float32)

    for j in prange(h):
        for i in prange(w):
            x = xys[j, i, 0]
            y = xys[j, i, 1]
            s = xys[j, i, 2]

            X = x * s
            Y = y * s
            Z = s - X - Y

            out[j, i, 0] = X
            out[j, i, 1] = Y
            out[j, i, 2] = Z

    return out


@njit(parallel=True)
def XYZ_to_xys(XYZ):
    h, w, c = XYZ.shape
    out = np.empty((h, w, c), dtype=np.float32)

    for j in prange(h):
        for i in prange(w):
            X = XYZ[j, i, 0]
            Y = XYZ[j, i, 1]
            Z = XYZ[j, i, 2]

            s = X + Y + Z
            x = X / s
            y = Y / s

            out[j, i, 0] = x
            out[j, i, 1] = y
            out[j, i, 2] = s

    return out


@njit(parallel=True)
def apply_2d_lut(image, lut):
    orig_shape = image.shape
    c = orig_shape[-1]

    # flatten spatial dims
    n = 1
    for i in range(len(orig_shape) - 1):
        n *= orig_shape[i]

    image_flat = image.reshape(n, c)

    lut_size = lut.shape[0]
    scaling = lut_size - 1

    out_flat = np.empty((n, c), dtype=np.float32)

    for i in prange(n):
        r = image_flat[i, 0]
        g = image_flat[i, 1]
        b = image_flat[i, 2]

        s = r + g + b

        if s == 0.0:
            out_flat[i, 0] = 0.0
            out_flat[i, 1] = 0.0
            out_flat[i, 2] = 0.0
            continue

        inv_sum = scaling / s

        r *= inv_sum
        g *= inv_sum

        r_ind = math.floor(r)
        g_ind = math.floor(g)

        r_factor = r % 1
        g_factor = g % 1

        factor_sum = r_factor + g_factor

        r_r, r_g, r_s = lut[r_ind + 1, g_ind]
        g_r, g_g, g_s = lut[r_ind, g_ind + 1]

        if factor_sum <= 1.0:
            s_r, s_g, s_s = lut[r_ind, g_ind]
            s_factor = 1.0 - factor_sum
        else:
            s_r, s_g, s_s = lut[r_ind + 1, g_ind + 1]
            r_factor, g_factor = 1 - g_factor, 1 - r_factor
            s_factor = factor_sum - 1.0

        out_flat[i, 0] = (r_r * r_factor + g_r * g_factor + s_r * s_factor) * s
        out_flat[i, 1] = (r_g * r_factor + g_g * g_factor + s_g * s_factor) * s
        out_flat[i, 2] = (r_s * r_factor + g_s * g_factor + s_s * s_factor) * s

    out_flat = out_flat.reshape(orig_shape)
    print(out_flat.shape)

    return out_flat


@njit(parallel=True)
def apply_2d_lut_mono(image, lut):
    orig_shape = image.shape
    c = orig_shape[-1]

    # flatten spatial dims
    n = 1
    for i in range(len(orig_shape) - 1):
        n *= orig_shape[i]

    image_flat = image.reshape(n, c)

    lut_size = lut.shape[0]
    scaling = lut_size - 1

    out_flat = np.empty((n, 1), dtype=np.float32)

    for i in prange(n):
        r = image_flat[i, 0]
        g = image_flat[i, 1]
        b = image_flat[i, 2]

        s = r + g + b

        if s == 0.0:
            out_flat[i, 0] = 0.0
            continue

        inv_sum = scaling / s

        r *= inv_sum
        g *= inv_sum

        r_ind = math.floor(r)
        g_ind = math.floor(g)

        r_factor = r % 1
        g_factor = g % 1

        factor_sum = r_factor + g_factor

        r_val = lut[r_ind + 1, g_ind, 0]
        g_val = lut[r_ind, g_ind + 1, 0]

        if factor_sum <= 1.0:
            s_val = lut[r_ind, g_ind, 0]
            s_factor = 1.0 - factor_sum
        else:
            s_val = lut[r_ind + 1, g_ind + 1, 0]
            r_factor, g_factor = 1 - g_factor, 1 - r_factor
            s_factor = factor_sum - 1.0

        out_flat[i, 0] = (r_val * r_factor + g_val * g_factor + s_val * s_factor) * s

    out_flat = out_flat.reshape(orig_shape[:-1] + (1,))

    return out_flat


def xy_to_spectrum_nnls(xy, loss_factor):
    x, y = xy
    if x + y > 1:
        return np.ones(XYZ_CMFS.shape[0])
    if not y:
        y = 1e-16

    X = x / y
    Z = (1 - x - y) / y
    XYZ = np.array([X, 1, Z])

    A = XYZ_CMFS.T  # (3×N)
    N = A.shape[1]

    # first derivative smoothness
    D1 = np.eye(N, k=0) - np.eye(N, k=1)

    # stack system instead of forming normal equations
    C = np.vstack([A, np.sqrt(loss_factor) * D1])
    d = np.concatenate([XYZ, np.zeros(N)])

    s, _ = nnls(C, d)
    return s


def xys_to_spectrum_nnls(xys, loss_factor) -> np.ndarray:
    x, y, s = xys
    spectrum = xy_to_spectrum_nnls((x, y), loss_factor)
    s_spectrum = (spectrum @ XYZ_CMFS).sum()
    spectrum *= s / s_spectrum
    return spectrum


def generate_spectral_sample_table(n, loss_factor):
    x = np.linspace(0, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    lut_id = np.stack([xx, yy, np.ones_like(xx)], axis=-1)

    flat = lut_id.reshape(-1, 3)

    out_flat = np.array(
        [xys_to_spectrum_nnls(point, loss_factor).astype(np.float32) for point in flat]
    )

    out = out_flat.reshape(n, n, -1)

    return out


start = time.time()
SPECTRUM_LUT = generate_spectral_sample_table(2**5, loss_factor=5)
print(time.time() - start)
