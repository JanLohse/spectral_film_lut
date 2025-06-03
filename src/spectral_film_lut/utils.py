import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import colour
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QRunnable, pyqtSlot
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QPushButton, QLabel, QWidget, QHBoxLayout, QFileDialog, QLineEdit, QSlider
from ffmpeg._run import compile
from ffmpeg.nodes import output_operator
from numba import njit

spectral_shape = colour.SpectralShape(380, 780, 5)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import cupy as xp
    from cupyx.scipy import ndimage as xdimage
    from cupyx.scipy import signal

    cuda_available = True
except ImportError:
    import numpy as xp
    from scipy import ndimage as xdimage
    from scipy import signal
    import scipy

    cuda_available = False
import numpy as np


def to_numpy(x):
    if cuda_available:
        return xp.asnumpy(x)
    else:
        return x


def create_lut(negative_film, print_film=None, lut_size=33, name="test", verbose=False, **kwargs):
    lut = colour.LUT3D(size=lut_size, name="test")
    transform, _ = negative_film.generate_conversion(negative_film, print_film, **kwargs)
    start = time.time()
    lut.table = to_numpy(transform(lut.table))
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