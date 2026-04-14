import math

import colour
import numpy as np
import scipy.interpolate
from matplotlib import pyplot as plt
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
def apply_2d_lut_XYZ(image, lut):
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

    return out_flat.reshape(orig_shape)


def distance_to_locus(xy) -> float:
    """
    Compute the distance to the locus. Points outside the locus return 0 distance.

    Args:
        xy: The input xy point.

    Returns:
        The distance to the locus.
    """
    p = np.array(xy)

    # Convert XYZ -> xy
    xy_cmfs = XYZ_CMFS[:, :2] / XYZ_CMFS.sum(axis=1, keepdims=True)

    # Segment vectors (previous point → current point)
    ab = xy_cmfs - np.roll(xy_cmfs, 1, axis=0)
    ap = p - xy_cmfs

    # Lengths of segments
    ab_len = np.linalg.norm(ab, axis=1)

    # Mask degenerate segments
    valid = ab_len >= 1e-4

    # 2D cross product (scalar z-component)
    cross = ap[:, 0] * ab[:, 1] - ap[:, 1] * ab[:, 0]

    # Signed distances
    d = np.ones_like(cross)
    d[valid] = cross[valid] / ab_len[valid]

    # Clamp to inside-only distances
    d = np.maximum(d, 0.0)

    return float(np.min(d))


def xy_to_loss_scale(xy, max_loss, max_distance=0.2) -> float:
    """
    Compute the correct loss scaling for the optimizer based on the distance of xy to
    the locus. We use a parabola whose vertex is in (max_distance, max_loss_scale), and
    which goes through (0, 0). This way the loss is 0 on the locus, and we smoothly
    reach the maximal locus towards the center.

    Args:
        xy: xy coordinates for which to compute the loss scaling.
        max_loss: The maximal loss scaling that is returned.
        max_distance: The distance at which the maximal loss scaling is returned.

    Returns:
        Loss scaling for xy to be used in xy_to_spectrum_nnls.
    """
    distance = distance_to_locus(xy)

    if distance > max_distance:
        return max_loss

    loss_scale = -max_loss / max_distance**2 * (distance - max_distance) ** 2 + max_loss

    return loss_scale


def xy_to_spectrum_nnls(xy, max_loss=0.1):
    x, y = xy
    if not y:
        y = 1e-16

    X = x / y
    Z = (1 - x - y) / y
    XYZ = np.array([X, 1, Z])

    A = XYZ_CMFS.T  # (3×N)
    N = A.shape[1]

    # first derivative smoothness
    D1 = np.eye(N, k=0) - np.eye(N, k=1)

    loss_scale = xy_to_loss_scale(xy, max_loss)

    # stack system instead of forming normal equations
    C = np.vstack([A, np.sqrt(loss_scale) * D1])
    d = np.concatenate([XYZ, np.zeros(N)])

    s, _ = nnls(C, d)
    return s


def spectrum_to_Y(spectrum):
    S = np.asarray(spectrum)

    if S.ndim == 1:
        return XYZ_CMFS[:, 1].T @ S

    return S @ XYZ_CMFS[:, 1]


def xys_to_spectrum_nnls(xys, max_loss=0.1):
    x, y, s = xys
    spectrum = xy_to_spectrum_nnls((x, y), max_loss)
    s_spectrum = (spectrum @ XYZ_CMFS).sum()
    spectrum *= s / s_spectrum
    return spectrum


def generate_spectral_sample_table(n=20, max_loss=0.05):
    x = np.linspace(0, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    lut_id = np.stack([xx, yy, np.ones_like(xx)], axis=-1)

    flat = lut_id.reshape(-1, 3)

    out_flat = np.array([xys_to_spectrum_nnls(point, max_loss) for point in flat])

    out = out_flat.reshape(n, n, -1)

    return out


def XYZ_to_xy(XYZ):
    if XYZ.ndim == 1:
        xy = XYZ[:2] / XYZ.sum()
        return xy

    xy = XYZ[:, :2] / XYZ.sum(axis=1, keepdims=True)
    return xy


def plot_cie(*dots, show_locus=True):

    xy_w = XYZ_CMFS[:, :-1] / XYZ_CMFS.sum(axis=1, keepdims=True)

    if show_locus:
        x = xy_w[:, 0]
        y = xy_w[:, 1]

        # parameter (assumes points are already ordered, e.g. by wavelength)
        t = np.arange(len(x))

        # cubic splines
        cs_x = scipy.interpolate.CubicSpline(t, x)
        cs_y = scipy.interpolate.CubicSpline(t, y)

        # smooth sampling
        t_fine = np.linspace(0, len(x) - 1, 500)

        x_smooth = cs_x(t_fine)
        y_smooth = cs_y(t_fine)

        plt.plot(x_smooth, y_smooth)

    plt.axis("square")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot((0, 1), (1, 0))

    for i, dot in enumerate(dots):
        print("lol", i)
        dot = np.array(dot)
        if len(dot.shape) == 1:
            label = f"#{i + 1} x={dot[0]:.3f} y={dot[1]:.3f}"
        else:
            label = f"#{i + 1}"
            if dot.shape[0] != 2:
                dot = dot.T
        plt.scatter(dot[0], dot[1], label=label)

    if dots:
        plt.legend()

    plt.show()


def spectrum_to_XYZ(spectrum):
    S = np.asarray(spectrum)

    if S.ndim == 1:
        return XYZ_CMFS.T @ S

    return S @ XYZ_CMFS


def spectrum_to_xy(spectrum):
    XYZ = spectrum_to_XYZ(spectrum)
    XYZ = np.asarray(XYZ)

    return XYZ_to_xy(XYZ)


def plot_spectra(*spectra, norm="Y"):
    wl = np.arange(380, 781, 5)
    any_legend = False

    for i, spectrum in enumerate(spectra):
        S = np.asarray(spectrum)

        # single spectrum
        if S.ndim == 1:
            if norm == "max":
                Y = S.max()
            elif norm in ["lum", "luminance", "Y", "y"]:
                Y = spectrum_to_Y(S)
            else:
                Y = 1
            if Y != 0:
                x, y = spectrum_to_xy(S)
                S = S / Y
            else:
                x, y = 0, 0
            plt.plot(wl, S, label=f"#{i + 1} {x=:.3f} {y=:.3f}")
            any_legend = True

        elif S.ndim == 2:
            if norm == "max":
                Y = S.max(axis=1)
            elif norm in ["lum", "luminance", "Y", "y"]:
                Y = spectrum_to_Y(S)
            else:
                Y = None
            if Y is not None:
                Y_safe = np.where(Y == 0, 1, Y)
                S_normalized = S / Y_safe[:, None]
                plt.plot(wl, S_normalized.T)
            else:
                plt.plot(wl, S.T)

        else:
            raise ValueError("Spectrum must be shape (81,) or (n, 81)")

    if any_legend:
        plt.legend()
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized spectral power")
    plt.show()


SPECTRUM_LUT = generate_spectral_sample_table(max_loss=0.01)
