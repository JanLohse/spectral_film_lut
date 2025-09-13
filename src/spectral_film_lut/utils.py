import math
import os
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import colour
import ffmpeg
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QRunnable, pyqtSlot
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QPushButton, QLabel, QWidget, QHBoxLayout, QFileDialog, QLineEdit, QSlider
from ffmpeg._run import compile
from ffmpeg.nodes import output_operator
from matplotlib import pyplot as plt
from numba import njit
import imageio.v3 as iio
import scipy

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    raise ImportError
    import cupy as xp
    from cupyx.scipy import ndimage as xdimage
    from cupyx.scipy import signal
    from cupyx.scipy.interpolate import PchipInterpolator

    cuda_available = True
except ImportError:
    import numpy as xp
    from scipy import ndimage as xdimage
    from scipy import signal
    from scipy.interpolate import PchipInterpolator

    cuda_available = False
import numpy as np


def to_numpy(x):
    if cuda_available:
        return xp.asnumpy(x)
    else:
        return x


spectral_shape = colour.SpectralShape(380, 780, 5)


def create_lut(negative_film, print_film=None, lut_size=33, name="test", cube=True, verbose=False, **kwargs):
    lut = colour.LUT3D(size=lut_size, name="test")
    transform, _ = negative_film.generate_conversion(negative_film, print_film, **kwargs)
    start = time.time()
    lut.table = to_numpy(transform(lut.table))
    if not cube:
        return lut.table
    if lut.table.shape[-1] == 1:
        lut.table = lut.table.repeat(3, -1)
    end = time.time()
    path = f"{name}.cube"
    if not os.path.exists("../../LUTs"):
        os.makedirs("../../LUTs")
    colour.io.write_LUT(lut, path)
    if verbose:
        print(f"created {path} in {end - start:.2f} seconds")
    return path


class WorkerSignals(QObject):
    """Signals from a running worker thread.

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc())

    result
        object data returned from processing, anything

    progress
        float indicating % progress
    """

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(float)


class Worker(QRunnable):
    """Worker thread.

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread.
                     Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class FileSelector(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

        file_browse = QPushButton('Browse')
        file_browse.setFixedWidth(55)
        file_browse.clicked.connect(self.open_file_dialog)
        self.filename_edit = QLineEdit()

        layout.addWidget(self.filename_edit)
        layout.addWidget(file_browse)

        self.textChanged = self.filename_edit.textChanged
        self.filetype = "Images (*.png *.jpg *)"

        self.show()

    def open_file_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(self, "Select a File", "", self.filetype)
        if filename:
            path = Path(filename)
            self.filename_edit.setText(str(path))

    def currentText(self):
        return self.filename_edit.text()


class Slider(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

        self.slider = QSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.setMinMaxTicks(0, 1, 1)

        self.text = QLabel()
        self.text.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.text.setFixedWidth(27)

        layout.addWidget(self.slider)
        layout.addWidget(self.text)

        self.slider.valueChanged.connect(self.sliderValueChanged)

        self.slider.valueChanged.connect(self.value_changed)

        self.slider.wheelEvent = self.customWheelEvent

        self.big_increment = 5
        self.small_increment = 2

    def setMinMaxTicks(self, min, max, enumerator=1, denominator=1):
        self.slider.setMinimum(0)
        number_of_steps = int(((max - min) * denominator) / enumerator)
        self.slider.setMaximum(number_of_steps)
        self.min = min
        self.enumerator = enumerator
        self.denominator = denominator
        self.big_increment = math.ceil(number_of_steps / 10)
        self.small_increment = math.ceil(number_of_steps / 30)

    def sliderValueChanged(self):
        value = self.getValue()
        if value.is_integer():
            self.text.setText(str(int(value)))
        else:
            self.text.setText(f"{value:.2f}")

    def getValue(self):
        return self.slider.value() * self.enumerator / self.denominator + self.min

    def setValue(self, value):
        self.slider.setValue(round((value - self.min) * self.denominator / self.enumerator))

    def setPosition(self, position):
        self.slider.setValue(position)

    def getPosition(self):
        return self.slider.value()

    def value_changed(self):
        self.valueChanged.emit(self.getValue())

    def increase(self, value=1):
        self.slider.setValue(self.slider.value() + value)

    def decrease(self, value=1):
        self.slider.setValue(self.slider.value() - value)

    def customWheelEvent(self, event: QWheelEvent):
        # Get the current value
        value = self.slider.value()

        # Determine the direction (positive = scroll up, negative = down)
        steps = event.angleDelta().y() // 120  # 120 per wheel step

        # Check if Shift or Ctrl is held
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            increment = self.big_increment
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            increment = 1
        else:
            increment = self.small_increment

        # Set the new value with bounds checking
        new_value = value + steps * increment
        new_value = max(self.slider.minimum(), min(self.slider.maximum(), new_value))
        self.slider.setValue(new_value)

        # Accept the event so the default handler doesn't interfere
        event.accept()


class Error(Exception):
    def __init__(self, cmd, stdout, stderr):
        super(Error, self).__init__(
            '{} error (see stderr output for detail)'.format(cmd)
        )
        self.stdout = stdout
        self.stderr = stderr

@output_operator()
def run_async(
    stream_spec,
    cmd='ffmpeg',
    pipe_stdin=False,
    pipe_stdout=False,
    pipe_stderr=False,
    quiet=False,
    overwrite_output=False,
):
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    args = compile(stream_spec, cmd, overwrite_output=overwrite_output)
    stdin_stream = subprocess.PIPE if pipe_stdin else None
    stdout_stream = subprocess.PIPE if pipe_stdout or quiet else None
    stderr_stream = subprocess.PIPE if pipe_stderr or quiet else None
    return subprocess.Popen(
        args, stdin=stdin_stream, stdout=stdout_stream, stderr=stderr_stream, startupinfo=startupinfo
    )

@output_operator()
def run(
    stream_spec,
    cmd='ffmpeg',
    capture_stdout=False,
    capture_stderr=False,
    input=None,
    quiet=False,
    overwrite_output=False,
):
    """Invoke ffmpeg for the supplied node graph.

    Args:
        capture_stdout: if True, capture stdout (to be used with
            ``pipe:`` ffmpeg outputs).
        capture_stderr: if True, capture stderr.
        quiet: shorthand for setting ``capture_stdout`` and ``capture_stderr``.
        input: text to be sent to stdin (to be used with ``pipe:``
            ffmpeg inputs)
        **kwargs: keyword-arguments passed to ``get_args()`` (e.g.
            ``overwrite_output=True``).

    Returns: (out, err) tuple containing captured stdout and stderr data.
    """
    process = run_async(
        stream_spec,
        cmd,
        pipe_stdin=input is not None,
        pipe_stdout=capture_stdout,
        pipe_stderr=capture_stderr,
        quiet=quiet,
        overwrite_output=overwrite_output,
    )
    out, err = process.communicate(input)
    retcode = process.poll()
    if retcode:
        raise Error('ffmpeg', out, err)
    return out, err


def multi_channel_interp(x, xps, fps, num_bins=1024, interpolate=False, left_extrapolate=False, right_extrapolate=False):
    """
    Resamples each (xp, fp) pair to a uniform grid for fast lookup.

    Returns:
    --------
    xp_common: np.ndarray, shape (num_bins,)
    fp_uniform: np.ndarray, shape (n_channels, num_bins)
    """
    if cuda_available:
        extrapolation_distance = 100
        if right_extrapolate:
            slopes = [(f_p[-1] - f_p[-2]) / (x_p[-1] - x_p[-2]) for x_p, f_p in zip(xps, fps)]
            xps = [xp.concatenate((x_p, xp.array([x_p[-1] + extrapolation_distance]))) for x_p in xps]
            fps = [xp.concatenate((f_p, xp.array([f_p[-1] + extrapolation_distance * slope]))) for f_p, slope in zip(fps, slopes)]
        if left_extrapolate:
            slopes = [(f_p[0] - f_p[1]) / (x_p[0] - x_p[1]) for x_p, f_p in zip(xps, fps)]
            xps = [xp.concatenate((xp.array([x_p[0] - extrapolation_distance]), x_p)) for x_p in xps]
            fps = [xp.concatenate((xp.array([f_p[0] - extrapolation_distance * slope])), f_p) for f_p, slope in
                   zip(fps, slopes)]
        return xp.stack(
            [xp.interp(xp.ascontiguousarray(x[..., i]), xp.ascontiguousarray(x_p), xp.ascontiguousarray(f_p)) for
             i, (x_p, f_p) in enumerate(zip(xps, fps))], dtype=np.float32, axis=-1)
    n_channels = len(xps)
    xp_min = min(x[0] for x in xps)
    xp_max = max(x[-1] for x in xps)
    xp_common = xp.linspace(xp_min, xp_max, num_bins).astype(xp.float32)

    fp_uniform = xp.empty((n_channels, num_bins), dtype=xp.float32)
    for ch in range(n_channels):
        fp_uniform[ch] = xp.interp(xp_common, xps[ch], fps[ch])
    return uniform_multi_channel_interp(x, xp_common, fp_uniform, interpolate, left_extrapolate, right_extrapolate)


@njit
def uniform_multi_channel_interp(x, xp_common, fp_uniform, interpolate=True, left_extrapolate=False, right_extrapolate=False):
    """
    Interpolates values in an N-D array `x` over the last dimension (channels)
    using a precomputed uniform grid (xp_common, fp_uniform), with optional
    linear extrapolation on both ends.

    Parameters:
    -----------
    x : np.ndarray, shape (..., channels)
    xp_common : np.ndarray, shape (num_bins,)
    fp_uniform : np.ndarray, shape (channels, num_bins)
    interpolate : bool
        Whether to linearly interpolate or just pick the nearest neighbor.
    left_extrapolate : bool
        Whether to linearly extrapolate for values < xp_common[0]
    right_extrapolate : bool
        Whether to linearly extrapolate for values > xp_common[-1]

    Returns:
    --------
    result : np.ndarray, same shape as `x`
    """
    shape = x.shape
    ndim = len(shape)
    n_channels = shape[ndim - 1]

    flat_size = 1
    for i in range(ndim - 1):
        flat_size *= shape[i]

    num_bins = xp_common.shape[0]
    xp_min = xp_common[0]
    xp_max = xp_common[-1]
    bin_width = (xp_max - xp_min) / (num_bins - 1)

    x_contig = np.ascontiguousarray(x)
    result = np.empty_like(x_contig, dtype=np.float32)

    x_flat = x_contig.reshape(flat_size, n_channels)
    r_flat = result.reshape(flat_size, n_channels)

    for idx in range(flat_size):
        for ch in range(n_channels):
            xi = x_flat[idx, ch]
            if xi <= xp_min:
                if left_extrapolate:
                    x0 = xp_common[0]
                    x1 = xp_common[1]
                    y0 = fp_uniform[ch, 0]
                    y1 = fp_uniform[ch, 1]
                    slope = (y1 - y0) / (x1 - x0)
                    r_flat[idx, ch] = y0 + slope * (xi - x0)
                else:
                    r_flat[idx, ch] = fp_uniform[ch, 0]
            elif xi >= xp_max:
                if right_extrapolate:
                    x0 = xp_common[-2]
                    x1 = xp_common[-1]
                    y0 = fp_uniform[ch, -2]
                    y1 = fp_uniform[ch, -1]
                    slope = (y1 - y0) / (x1 - x0)
                    r_flat[idx, ch] = y1 + slope * (xi - x1)
                else:
                    r_flat[idx, ch] = fp_uniform[ch, -1]
            else:
                pos = (xi - xp_min) / bin_width
                i = int(pos)
                if interpolate:
                    f = pos - i
                    y0 = fp_uniform[ch, i]
                    y1 = fp_uniform[ch, i + 1]
                    r_flat[idx, ch] = y0 + f * (y1 - y0)
                else:
                    r_flat[idx, ch] = fp_uniform[ch, i]

    return result


def construct_spectral_density(ref_density, sigma=25):
    red_peak = wavelength_argmax(ref_density, 600, min(750, spectral_shape.end))
    green_peak = wavelength_argmax(ref_density, 500, 600)
    blue_peak = wavelength_argmax(ref_density, max(400, spectral_shape.start), 500)
    bg_cutoff = wavelength_argmin(ref_density, blue_peak, green_peak)
    gr_cutoff = wavelength_argmin(ref_density, green_peak, red_peak)

    wavelengths = xp.asarray(ref_density.wavelengths)
    factors = xp.stack((xp.where(gr_cutoff <= wavelengths, 1., 0.),
                        xp.where((bg_cutoff < wavelengths) & (wavelengths < gr_cutoff), 1., 0.),
                        xp.where(wavelengths <= bg_cutoff, 1., 0.)))
    factors = xdimage.gaussian_filter(factors, sigma=(0, sigma / spectral_shape.interval)).astype(
        xp.float32)

    out = (factors * xp.asarray(ref_density.values)).T
    return out

def wavelength_argmax(distribution, low=None, high=None):
    range = distribution.copy()
    if low is not None and high is not None:
        range.trim(colour.SpectralShape(low, high, 1))
    peak = range.wavelengths[range.values.argmax()]
    return peak

def wavelength_argmin(distribution, low=None, high=None):
    range = distribution.copy()
    if low is not None and high is not None:
        range.trim(colour.SpectralShape(low, high, 1))
    peak = range.wavelengths[range.values.argmin()]
    return peak

def plot_gamuts(rgb_to_xyz, labels=None):
    rgb_to_xyz = [to_numpy(x) for x in rgb_to_xyz]

    # RGB cube corners (R, G, B) in [0, 1]
    rgb_primaries = np.identity(3)

    # Convert RGB primaries to XYZ
    xyz_primaries = [rgb_primaries @ x.T for x in rgb_to_xyz]

    # Convert XYZ to xy chromaticities
    xy_primaries = [colour.XYZ_to_xy(x) for x in xyz_primaries]

    # Compute the whitepoint (assumes white = [1, 1, 1] in RGB)
    xyz_white = [np.dot(x, np.ones(3)) for x in rgb_to_xyz]
    xy_white = [colour.XYZ_to_xy(x) for x in xyz_white]

    # Close the triangle by appending the first point at the end
    xy_gamut = [np.vstack((x, x[0])) for x in xy_primaries]

    # Plot the CIE 1931 Chromaticity Diagram
    colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)

    # Plot the RGB gamut triangle
    if labels is None or len(labels) != len(rgb_to_xyz):
        labels = list(range(len(rgb_to_xyz)))

    for i, x, y in zip(labels, xy_gamut, xy_white):
        line, = plt.plot(x[:, 0], x[:, 1], 'o-', label=i)
        # Use the same color to plot the white point
        plt.plot(y[0], y[1], 'x', color=line.get_color())

    plt.legend()
    plt.title("RGB Gamut on CIE 1931 Chromaticity Diagram")
    plt.show()

def plot_gamut(rgb_to_xyz, label=None):
    plot_gamuts([rgb_to_xyz], [label])


def plot_chromaticity(chromaticity, label=None):
    plot_chromaticties([chromaticity], [label])


def plot_chromaticties(chromaticies, labels=None):
    # Plot the CIE 1931 Chromaticity Diagram
    colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)

    if labels is None or len(labels) != len(chromaticies):
        labels = list(range(len(chromaticies)))

    for chromaticity, label in zip(chromaticies, labels):
        chromaticity = to_numpy(chromaticity)

        if chromaticity.ndim == 1:
            chromaticity = chromaticity.reshape(1, -1)

        # Convert XYZ to xy chromaticities
        xy = colour.XYZ_to_xy(chromaticity).T

        plt.scatter(xy[0], xy[1], label=label)

    plt.legend()
    plt.title("RGB Gamut on CIE 1931 Chromaticity Diagram")
    plt.show()


def generate_combinations(steps):
    values = np.linspace(0, 1, steps + 1)[:-1]
    result = []

    # First: zero in position 2 -> [a, b, 0]
    for a in values:
        b = 1 - a
        result.append([b, a, 0.0])

    # Second: zero in position 0 -> [0, a, b]
    for a in values:
        b = 1 - a
        result.append([0.0, b, a])

    # Third: zero in position 1 -> [a, 0, b]
    for a in values:
        b = 1 - a
        result.append([b, 0.0, a])

    return xp.array(result)


def generate_all_summing_to_one(steps):
    """
    Generate all (a, b, c) such that a + b + c = 1 and a, b, c ∈ [0, 1]
    with a specified number of steps (granularity).

    Parameters:
        steps (int): The number of divisions of the interval [0, 1].

    Returns:
        np.ndarray: Array of shape (n, 3) with all combinations summing to 1.
    """
    result = []
    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            result.append([i / steps, j / steps, k / steps])
    return xp.array(result)


def CCT_to_xy(CCT):
    CCT_3 = CCT ** 3
    CCT_2 = CCT ** 2

    if CCT <= 7000:
        x = -4.607 * 10 ** 9 / CCT_3 + 2.9678 * 10 ** 6 / CCT_2 + 0.09911 * 10 ** 3 / CCT + 0.244063
    else:
        x = -2.0064 * 10 ** 9 / CCT_3 + 1.9018 * 10 ** 6 / CCT_2 + 0.24748 * 10 ** 3 / CCT + 0.23704

    y = -3.000 * x ** 2 + 2.870 * x - 0.275
    return np.array([x, y])


def xy_to_illuminant_D(xy, spectral_shape=None):
    xy = np.concatenate((np.ones(1), xy))
    xy /= np.array([0.0241, 0.2562, -0.7341]) @ xy
    M = np.array([[-1.3515, -1.7703, 5.9114], [0.030, -31.4424, 30.0717]]) @ xy
    M = np.concatenate((np.ones(1), M))
    illuminant = np.array([
        [  0.04,   3.02,   6.00,  17.80,  29.60,  42.45,  55.30,  56.30,  57.30,  59.55,
          61.80,  61.65,  61.50,  65.15,  68.80,  66.10,  63.40,  64.60,  65.80,  80.30,
          94.80,  99.80, 104.80, 105.35, 105.90, 101.35,  96.80, 105.35, 113.90, 119.75,
         125.60, 125.55, 125.50, 123.40, 121.30, 121.30, 121.30, 117.40, 113.50, 113.30,
         113.10, 111.95, 110.80, 108.65, 106.50, 107.65, 108.80, 107.05, 105.30, 104.85,
         104.40, 102.20, 100.00,  98.00,  96.00,  95.55,  95.10,  92.10,  89.10,  89.80,
          90.50,  90.40,  90.30,  89.35,  88.40,  86.20,  84.00,  84.55,  85.10,  83.50,
          81.90,  82.25,  82.60,  83.75,  84.90,  83.10,  81.30,  76.60,  71.90,  73.10,
          74.30,  75.35,  76.40,  69.85,  63.30,  67.50,  71.70,  74.35,  77.00,  71.10,
          65.20,  56.45,  47.70,  58.15,  68.60,  66.80,  65.00,  65.50,  66.00,  63.50,
          61.00,  57.15,  53.30,  56.10,  58.90,  60.40,  61.90],

        [  0.02,   2.26,   4.50,  13.45,  22.40,  32.20,  42.00,  41.30,  40.60,  41.10,
          41.60,  39.80,  38.00,  40.20,  42.40,  40.45,  38.50,  36.75,  35.00,  39.20,
          43.40,  44.85,  46.30,  45.10,  43.90,  40.50,  37.10,  36.90,  36.70,  36.30,
          35.90,  34.25,  32.60,  30.25,  27.90,  26.10,  24.30,  22.20,  20.10,  18.15,
          16.20,  14.70,  13.20,  10.90,   8.60,   7.35,   6.10,   5.15,   4.20,   3.05,
           1.90,   0.95,   0.00,  -0.80,  -1.60,  -2.55,  -3.50,  -3.50,  -3.50,  -4.65,
          -5.80,  -6.50,  -7.20,  -7.90,  -8.60,  -9.05,  -9.50, -10.20, -10.90, -10.80,
         -10.70, -11.35, -12.00, -13.00, -14.00, -13.80, -13.60, -12.80, -12.00, -12.65,
         -13.30, -13.10, -12.90, -11.75, -10.60, -11.10, -11.60, -11.90, -12.20, -11.20,
         -10.20,  -9.00,  -7.80,  -9.50, -11.20, -10.80, -10.40, -10.50, -10.60, -10.15,
          -9.70,  -9.00,  -8.30,  -8.80,  -9.30,  -9.55,  -9.80],

        [  0.00,   1.00,   2.00,   3.00,   4.00,   6.25,   8.50,   8.15,   7.80,   7.25,
           6.70,   6.00,   5.30,   5.70,   6.10,   4.55,   3.00,   2.10,   1.20,   0.05,
          -1.10,  -0.80,  -0.50,  -0.60,  -0.70,  -0.95,  -1.20,  -1.90,  -2.60,  -2.75,
          -2.90,  -2.85,  -2.80,  -2.70,  -2.60,  -2.60,  -2.60,  -2.20,  -1.80,  -1.65,
          -1.50,  -1.40,  -1.30,  -1.25,  -1.20,  -1.10,  -1.00,  -0.75,  -0.50,  -0.40,
          -0.30,  -0.15,   0.00,   0.10,   0.20,   0.35,   0.50,   1.30,   2.10,   2.65,
           3.20,   3.65,   4.10,   4.40,   4.70,   4.90,   5.10,   5.90,   6.70,   7.00,
           7.30,   7.95,   8.60,   9.20,   9.80,  10.00,  10.20,   9.25,   8.30,   8.95,
           9.60,   9.05,   8.50,   7.75,   7.00,   7.30,   7.60,   7.80,   8.00,   7.35,
           6.70,   5.95,   5.20,   6.30,   7.40,   7.10,   6.80,   6.90,   7.00,   6.70,
           6.40,   5.95,   5.50,   5.80,   6.10,   6.30,   6.50]]).T @ M
    if spectral_shape is not None:
        illuminant = colour.SpectralDistribution(to_numpy(illuminant), colour.SpectralShape(300, 830, 5)).align(
            spectral_shape)
    return illuminant


def CCT_to_illuminant_D(CCT, spectral_shape=None):
    xy = CCT_to_xy(CCT)
    illuminant = xy_to_illuminant_D(xy, spectral_shape)
    return illuminant