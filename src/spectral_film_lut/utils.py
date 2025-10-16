import math
import sys
import time
import traceback
import warnings
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QRunnable, pyqtSlot, QRectF, QRect, QPoint, pyqtProperty, \
    QPropertyAnimation, QPointF
from PyQt6.QtGui import QWheelEvent, QFontMetrics, QPainter, QColor, QLinearGradient, QBrush
from PyQt6.QtWidgets import QPushButton, QLabel, QWidget, QHBoxLayout, QFileDialog, QLineEdit, QSlider, QComboBox, \
    QScrollArea, QFrame
from matplotlib import pyplot as plt
from numba import njit, prange, cuda
from scipy.linalg import fractional_matrix_power

from spectral_film_lut.css_theme import *

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
    table = transform(lut.table)
    if table.shape[-1] == 1:
        table = table.repeat(3, -1)
    if cube:
        lut.table = to_numpy(table)
    else:
        return table
    end = time.time()
    path = f"{name}.cube"
    if not os.path.exists("../../LUTs"):
        os.makedirs("../../LUTs")
    colour.io.write_LUT(lut, path)
    if verbose:
        print(f"created {path} in {end - start:.2f} seconds")
    return path


class RoundedScrollArea(QScrollArea):
    def __init__(self, radius=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if radius is None:
            radius = BORDER_RADIUS
        self.radius = radius
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)



class WideComboBox(QComboBox):
    def __init__(self, parent=None):
        super(WideComboBox, self).__init__(parent)
        self.setStyleSheet(f"QComboBox QAbstractItemView {{background-color: {MENU_COLOR};}}")

    def showPopup(self):
        # Measure the widest item text
        fm = QFontMetrics(self.view().font())
        max_width = max(fm.horizontalAdvance(self.itemText(i)) for i in range(self.count()))
        # Add some padding
        max_width += 30

        # Resize the popup view
        self.view().setMinimumWidth(max(self.width(), max_width))
        super().showPopup()


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


class GradientSlider(QSlider):
    def __init__(self, *args, reference_value=0, modern_design=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient = None
        self.set_color_gradient((0.3, 0., 0.), (0.7, 0., 0.))
        self.modern_design = modern_design
        self.setRange(-100, 100)
        self.setValue(0)
        self.reference_value = reference_value
        self._hover = False
        self._hoverProgress = 0.0  # property for animation
        self.anim = QPropertyAnimation(self, b"hoverProgress", self)
        self.anim.setDuration(150)
        self.setMouseTracking(True)
        self.setFixedHeight(30)
        self.setStyleSheet("QSlider { background: transparent; }")

    # --- animated property for smooth scaling ---
    def get_hover_progress(self):
        return self._hoverProgress

    def set_hover_progress(self, value):
        self._hoverProgress = value
        self.update()

    def set_color_gradient(self, start_color, end_color, steps=10, blend_in_lab=True):
        if blend_in_lab:
            start_color = colour.convert(start_color, "Oklch", "Oklab")
            end_color = colour.convert(end_color, "Oklch", "Oklab")
        source = "Oklab" if blend_in_lab else "Oklch"
        self.gradient = [(x, QColor(colour.convert(start_color * (1 - x) + end_color * x, source, "Hexadecimal"))) for
                         x in np.linspace(0, 1, steps)]

    hoverProgress = pyqtProperty(float, fget=get_hover_progress, fset=set_hover_progress)

    # --- hover detection ---
    def enterEvent(self, event):
        self._hover = True
        self.anim.stop()
        self.anim.setStartValue(self._hoverProgress)
        self.anim.setEndValue(1.0)
        self.anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover = False
        self.anim.stop()
        self.anim.setStartValue(self._hoverProgress)
        self.anim.setEndValue(0.0)
        self.anim.start()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        horizontal_padding = 7
        groove_thickness = 10 if self.modern_design else 4
        groove_rect = QRect(horizontal_padding, self.height() // 2 - groove_thickness // 2,
                            self.width() - horizontal_padding * 2, groove_thickness)

        min_val, max_val = self.minimum(), self.maximum()
        total_range = max_val - min_val

        def value_to_x(val):
            return groove_rect.left() + (val - min_val) / total_range * groove_rect.width()

        handle_x = int(value_to_x(self.value()))
        ref_x = int(value_to_x(self.reference_value))

        # background groove with gradient
        painter.setPen(Qt.PenStyle.NoPen)

        gradient = QLinearGradient(QPointF(groove_rect.topLeft()), QPointF(groove_rect.topRight()))
        for pos, color in self.gradient:
            gradient.setColorAt(pos, color)  # left color

        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(groove_rect, 3, 3)

        # active segment (ref -> handle)
        painter.setBrush(QColor(255, 255, 255, 85))
        if handle_x > ref_x:
            active_rect = QRect(ref_x, groove_rect.top(), handle_x - ref_x, groove_rect.height())
        else:
            active_rect = QRect(handle_x, groove_rect.top(), ref_x - handle_x, groove_rect.height())
        if self.reference_value in (self.minimum(), self.maximum()):
            painter.drawRoundedRect(active_rect, 3, 3)
        else:
            painter.drawRect(active_rect)

        # handle
        if self.modern_design:
            handle_bg_width = 6 + self._hoverProgress * 1
            handle_bg_rect = QRectF(handle_x - handle_bg_width, groove_rect.center().y() - groove_thickness, handle_bg_width * 2, groove_thickness * 2)
            painter.setBrush(QColor(BASE_COLOR))
            painter.drawRect(handle_bg_rect)

            handle_width = 1.25 + self._hoverProgress * 1
            handle_length = groove_thickness / 2 + 4
            handle_rect = QRectF(handle_x - handle_width, groove_rect.center().y() - handle_length, handle_width * 2, handle_length * 2)
            painter.setBrush(QColor(TEXT_PRIMARY))
            painter.drawRoundedRect(handle_rect, handle_width, handle_width)
        else:
            handle_center = QPoint(handle_x, groove_rect.center().y())
            hover_radius = 7
            inner_radius = round(2 + self._hoverProgress * 2)  # grow inner slightly too

            painter.setBrush(QColor(TEXT_PRIMARY))
            painter.drawEllipse(handle_center, hover_radius, hover_radius)

            painter.setBrush(QColor(BASE_COLOR))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(handle_center, inner_radius, inner_radius)

        painter.end()

    def set_reference_value(self, value: int):
        """Update the reference (neutral) value dynamically."""
        self.reference_value = value
        self.update()


class Slider(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.slider = GradientSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.setMinMaxTicks(0, 1, 1)

        self.text = QLabel()
        self.text.setAlignment((Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
        self.text.setFixedWidth(30)

        layout.addWidget(self.slider)
        layout.addWidget(self.text)

        self.slider.valueChanged.connect(self.sliderValueChanged)

        self.slider.valueChanged.connect(self.value_changed)

        self.slider.wheelEvent = self.customWheelEvent

        self.set_color_gradient = self.slider.set_color_gradient

        self.big_increment = 5
        self.small_increment = 2

    def setMinMaxTicks(self, min, max, enumerator=1, denominator=1, default=0):
        self.slider.setMinimum(0)
        number_of_steps = int(((max - min) * denominator) / enumerator)
        self.slider.setMaximum(number_of_steps)
        self.min = min
        self.enumerator = enumerator
        self.denominator = denominator
        self.big_increment = math.ceil(number_of_steps / 10)
        self.small_increment = math.ceil(number_of_steps / 30)
        self.setValue(default)
        self.slider.set_reference_value(self.getPosition())

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

@njit(parallel=True)
def apply_lut_tetrahedral_int(image, lut, exponent=16, out_exponent=8):
    """
    Apply a 3D LUT with tetrahedral interpolation.
    Input: uint16 image in [0, 65535]
    Output: uint8 image in [0, 255]

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W, 3), dtype=uint16.
    lut : np.ndarray
        LUT of shape (size, size, size, 3), dtype=uint8.

    Returns
    -------
    out : np.ndarray
        Output image of shape (H, W, 3), dtype=uint8.
    """
    h, w, c = image.shape
    size = lut.shape[0]
    max_value = 2 ** exponent - 1
    scale = max_value // (size - 1)
    scale_out = scale * 2 ** (exponent - out_exponent)

    out = np.empty((h, w, 3), dtype=np.uint8)

    for y in prange(h):
        for x in prange(w):
            r = image[y, x, 0]
            g = image[y, x, 1]
            b = image[y, x, 2]

            r0 = r // scale
            g0 = g // scale
            b0 = b // scale

            dr = r % scale
            dg = g % scale
            db = b % scale

            r1 = min(r0 + 1, size - 1)
            g1 = min(g0 + 1, size - 1)
            b1 = min(b0 + 1, size - 1)

            # Fetch cube corners
            c000 = lut[r0, g0, b0]
            c100 = lut[r1, g0, b0]
            c010 = lut[r0, g1, b0]
            c001 = lut[r0, g0, b1]
            c110 = lut[r1, g1, b0]
            c101 = lut[r1, g0, b1]
            c011 = lut[r0, g1, b1]
            c111 = lut[r1, g1, b1]

            # Tetrahedral interpolation
            if dr >= dg:
                if dg >= db:
                    c = (c000 * scale
                         + dr * (c100 - c000)
                         + dg * (c110 - c100)
                         + db * (c111 - c110))
                elif dr >= db:
                    c = (c000 * scale
                         + dr * (c100 - c000)
                         + db * (c101 - c100)
                         + dg * (c111 - c101))
                else:
                    c = (c000 * scale
                         + db * (c001 - c000)
                         + dr * (c101 - c001)
                         + dg * (c111 - c101))
            else:
                if db >= dg:
                    c = (c000 * scale
                         + db * (c001 - c000)
                         + dg * (c011 - c001)
                         + dr * (c111 - c011))
                elif db >= dr:
                    c = (c000 * scale
                         + dg * (c010 - c000)
                         + db * (c011 - c010)
                         + dr * (c111 - c011))
                else:
                    c = (c000 * scale
                         + dg * (c010 - c000)
                         + dr * (c110 - c010)
                         + db * (c111 - c110))

            # Convert back to uint8 safely
            out[y, x, 0] = np.uint8(c[0] // scale_out)
            out[y, x, 1] = np.uint8(c[1] // scale_out)
            out[y, x, 2] = np.uint8(c[2] // scale_out)

    return out

if cuda_available:
    @cuda.jit
    def apply_lut_tetrahedral_int_cuda(image, lut, out,
                                       size, scale, scale_out):
        """
        CUDA kernel: Apply a 3D LUT with tetrahedral interpolation.
        Input : uint16 image in [0, 65535]
        Output: uint8 image in [0, 255]
        """
        idx = cuda.grid(1)
        h, w, _ = image.shape
        total = h * w

        if idx >= total:
            return

        y = idx // w
        x = idx % w

        r = image[y, x, 0]
        g = image[y, x, 1]
        b = image[y, x, 2]

        r0 = r // scale
        g0 = g // scale
        b0 = b // scale

        dr = r % scale
        dg = g % scale
        db = b % scale

        r1 = min(r0 + 1, size - 1)
        g1 = min(g0 + 1, size - 1)
        b1 = min(b0 + 1, size - 1)

        # Fetch cube corners
        c000 = lut[r0, g0, b0]
        c100 = lut[r1, g0, b0]
        c010 = lut[r0, g1, b0]
        c001 = lut[r0, g0, b1]
        c110 = lut[r1, g1, b0]
        c101 = lut[r1, g0, b1]
        c011 = lut[r0, g1, b1]
        c111 = lut[r1, g1, b1]

        c = cuda.local.array(3, dtype=np.float32)

        # Tetrahedral interpolation in float
        for ch in range(3):
            if dr >= dg:
                if dg >= db:
                    c[ch] = c000[ch] * scale \
                            + dr * (c100[ch] - c000[ch]) \
                            + dg * (c110[ch] - c100[ch]) \
                            + db * (c111[ch] - c110[ch])
                elif dr >= db:
                    c[ch] = c000[ch] * scale \
                            + dr * (c100[ch] - c000[ch]) \
                            + db * (c101[ch] - c100[ch]) \
                            + dg * (c111[ch] - c101[ch])
                else:
                    c[ch] = c000[ch] * scale \
                            + db * (c001[ch] - c000[ch]) \
                            + dr * (c101[ch] - c001[ch]) \
                            + dg * (c111[ch] - c101[ch])
            else:
                if db >= dg:
                    c[ch] = c000[ch] * scale \
                            + db * (c001[ch] - c000[ch]) \
                            + dg * (c011[ch] - c001[ch]) \
                            + dr * (c111[ch] - c011[ch])
                elif db >= dr:
                    c[ch] = c000[ch] * scale \
                            + dg * (c010[ch] - c000[ch]) \
                            + db * (c011[ch] - c010[ch]) \
                            + dr * (c111[ch] - c011[ch])
                else:
                    c[ch] = c000[ch] * scale \
                            + dg * (c010[ch] - c000[ch]) \
                            + dr * (c110[ch] - c010[ch]) \
                            + db * (c111[ch] - c110[ch])

        # Normalize back to uint8
        out[y, x, 0] = min(255, max(0, int(c[0] / scale_out)))
        out[y, x, 1] = min(255, max(0, int(c[1] / scale_out)))
        out[y, x, 2] = min(255, max(0, int(c[2] / scale_out)))


    def run_lut_cuda(image: np.ndarray, lut: np.ndarray,
                     exponent=16, out_exponent=8) -> np.ndarray:
        """
        Wrapper: runs the CUDA kernel on an image + LUT.
        """
        h, w, _ = image.shape
        size = lut.shape[0]

        max_value = 2 ** exponent - 1
        scale = max_value // (size - 1)
        scale_out = scale * 2 ** (exponent - out_exponent)

        # Allocate GPU arrays
        d_image = cuda.to_device(image)
        d_lut = cuda.to_device(lut)
        d_out = cuda.device_array((h, w, 3), dtype=np.uint8)

        threads = 256
        blocks = (h * w + threads - 1) // threads

        apply_lut_tetrahedral_int_cuda[blocks, threads](
            d_image, d_lut, d_out, size, scale, scale_out
        )

        return d_out.copy_to_host()


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

@njit
def _rgb_to_hsv_pixel(r, g, b):
    """Convert one RGB pixel to HSV (all values in [0,1])."""
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    diff = c_max - c_min

    # Value
    v = c_max

    # Saturation
    if c_max == 0.0:
        s = 0.0
    else:
        s = diff / c_max

    # Hue
    if diff == 0.0:
        h = 0.0
    elif c_max == r:
        h = (g - b) / diff % 6.0
    elif c_max == g:
        h = (b - r) / diff + 2.0
    else:  # mx == b
        h = (r - g) / diff + 4.0

    h /= 6.0  # normalize to [0,1]

    return h, s, v

@njit
def _hsv_to_rgb_pixel(h, s, v):
    """Convert one HSV pixel to RGB (all values in [0,1])."""
    c = v * s
    x = c * (1 - abs(h * 6 % 2 - 1))
    m = v - c

    # rgb temp
    if 0 <= h < 1/6:
        r, g, b = c, x, 0
    elif 1/6 <= h < 2/6:
        r, g, b = x, c, 0
    elif 2/6 <= h < 3/6:
        r, g, b = 0, c, x
    elif 3/6 <= h < 4/6:
        r, g, b = 0, x, c
    elif 4/6 <= h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    # rgb
    r += m
    g += m
    b += m

    return r, g, b

@njit
def _rgb_to_hsl_pixel(r, g, b):
    """Convert one RGB pixel to HSL (all values in [0,1])."""
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    diff = c_max - c_min

    # Value
    l = (c_max + c_min) / 2

    # Saturation
    if c_max == 0.0 or l == 1.0:
        s = 0.0
    else:
        s = diff / (1 - abs(2 * l - 1))

    # Hue
    if diff == 0.0:
        h = 0.0
    elif c_max == r:
        h = (g - b) / diff % 6.0
    elif c_max == g:
        h = (b - r) / diff + 2.0
    else:  # mx == b
        h = (r - g) / diff + 4.0

    h /= 6.0  # normalize to [0,1]

    return h, s, l

@njit
def _hsl_to_rgb_pixel(h, s, l):
    """Convert one HSL pixel to RGB (all values in [0,1])."""
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs(h * 6 % 2 - 1))
    m = l - c / 2

    # rgb temp
    if 0 <= h < 1/6:
        r, g, b = c, x, 0
    elif 1/6 <= h < 2/6:
        r, g, b = x, c, 0
    elif 2/6 <= h < 3/6:
        r, g, b = 0, c, x
    elif 3/6 <= h < 4/6:
        r, g, b = 0, x, c
    elif 4/6 <= h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    # rgb
    r += m
    g += m
    b += m

    return r, g, b

# Reference white (D65)
REF_X = 0.95047
REF_Y = 1.00000
REF_Z = 1.08883

@njit
def _f(t):
    delta = 6/29
    if t > delta**3:
        return t ** (1/3)
    else:
        return (t / (3 * delta**2)) + (4/29)

@njit
def apply_per_pixel(rgb, function):
    """
    Convert a 3D (H, W, 3) or 4D (N, H, W, 3) RGB array to HSV.
    RGB values are expected in range [0, 1].
    Returns array of same shape with HSV values.

    H in [0, 1], S in [0, 1], V in [0, 1].
    """
    shape = rgb.shape
    ndim = rgb.ndim

    if ndim == 2:
        out = np.empty_like(rgb)
        for i in range(shape[0]):
            r, g, b = rgb[i]
            out[i] = np.array(function(r, g, b), dtype=np.float32)
        return out

    elif ndim == 3:  # (H, W, 3)
        out = np.empty_like(rgb)
        for i in range(shape[0]):
            for j in range(shape[1]):
                r, g, b = rgb[i, j]
                out[i, j] = np.array(function(r, g, b), dtype=np.float32)
        return out

    elif ndim == 4:  # (N, H, W, 3)
        out = np.empty_like(rgb)
        for n in range(shape[0]):
            for i in range(shape[1]):
                for j in range(shape[2]):
                    r, g, b = rgb[n, i, j]
                    out[n, i, j] = np.array(function(r, g, b), dtype=np.float32)
        return out

    else:
        raise ValueError("Input must be 3D or 4D with last dimension == 3")

@njit
def rgb_to_hsv(rgb):
    return apply_per_pixel(rgb, _rgb_to_hsv_pixel)

@njit
def hsv_to_rgb(hsv):
    return apply_per_pixel(hsv, _hsv_to_rgb_pixel)

@njit
def rgb_to_hsl(rgb):
    return apply_per_pixel(rgb, _rgb_to_hsl_pixel)

@njit
def hsl_to_rgb(hsl):
    return apply_per_pixel(hsl, _hsl_to_rgb_pixel)


@njit
def saturation_adjust_hsv_linear(rgb, sat_adjust, density=1):
    hsv = rgb_to_hsv(rgb)
    hsv[..., 1] *= sat_adjust
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)
    rgb = hsv_to_rgb(hsv)
    return rgb

@njit
def saturation_adjust_hsv_density(rgb, sat_adjust, density=1):
    hsv = rgb_to_hsv(rgb)
    if sat_adjust < 1:
        hsv[..., 1] *= sat_adjust
    else:
        hsv[..., 2] *= density * (1 - sat_adjust) *  hsv[..., 1] ** 2 + 1
        hsv[..., 1] = (1 - sat_adjust) * hsv[..., 1] ** 2 + sat_adjust * hsv[..., 1]

    rgb = hsv_to_rgb(hsv)
    return rgb


@njit
def saturation_adjust_hsl_linear(rgb, sat_adjust, density=1):
    hsl = rgb_to_hsl(rgb)
    hsl[..., 1] = np.clip(hsl[..., 1] * sat_adjust, 0, 1)
    rgb = hsl_to_rgb(hsl)
    return rgb

@njit
def saturation_adjust_hsl_density(rgb, sat_adjust, density=1):
    hsl = rgb_to_hsl(rgb)

    if sat_adjust < 1:
        hsl[..., 1] *= sat_adjust
    else:
        hsl[..., 2] *= density * (1 - sat_adjust) *  hsl[..., 1] ** 2 + 1
        hsl[..., 1] = (1 - sat_adjust) * hsl[..., 1] ** 2 + sat_adjust * hsl[..., 1]

    rgb = hsl_to_rgb(hsl)
    return rgb

def linear_gamut_compression(rgb, gamut_compression=0):
    A = xp.identity(3, rgb.dtype) * (1 - gamut_compression) + gamut_compression / 3
    A_inv = xp.linalg.inv(A)
    rgb = xp.clip(rgb @ A_inv, -0.1, None) @ A
    return rgb

def saturation_adjust_simple(rgb, sat_adjust, density=0.5, luminance_factors=None):
    if luminance_factors is None:
        luminance_factors = np.ones(3, dtype=np.float32) / 3
    else:
        luminance_factors = luminance_factors.astype(np.float32)
    Y = rgb @ luminance_factors.reshape(-1, 1)
    if sat_adjust <= 1:
        rgb = (1 - sat_adjust) * Y + sat_adjust * rgb
    else:
        rgb_saturated = (1 - sat_adjust) * Y + sat_adjust * rgb
        achromaticity = (np.divide(-rgb_saturated.min(axis=-1, keepdims=True), Y, where=(Y != 0)) + 1)
        achromaticity /= sat_adjust
        achromaticity = np.clip(achromaticity, 0, 1)
        rgb = (rgb * achromaticity + rgb_saturated * (1 - achromaticity)) * (1 - achromaticity * density * (sat_adjust - 1))
    return rgb

def gamut_compression_matrices(matrix, gamut_compression=0.):
    A = xp.identity(3, dtype=np.float32) * (1 - gamut_compression) + gamut_compression / 3
    A_inv = xp.linalg.inv(A)
    return matrix @ A_inv, A


def saturation_adjust_oklch(rgb, sat_adjust, white_point=None, luminance_factors=None):
    if luminance_factors is None:
        luminance_factors = np.ones(3, dtype=np.float32) / 3
    else:
        luminance_factors = luminance_factors.astype(np.float32)
    if white_point is None:
        white_point = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)

    l, c, h = 0.54, 0.054, 0.137

    samples_oklch = np.array([
        [l, c, h],
        [l, c, h + 1/3],
        [l, c, h + 2/3]
    ], dtype=np.float32)

    samples_xyz = colour.convert(samples_oklch, 'Oklch', 'CIE XYZ')

    samples_oklch_adj = samples_oklch.copy()
    samples_oklch_adj[:, 1] *= sat_adjust
    samples_xyz_adj = colour.convert(samples_oklch_adj, 'Oklch', 'sRGB', apply_cctf_encoding=False)

    S = samples_xyz.T
    T = samples_xyz_adj.T
    M = T @ np.linalg.inv(S)

    W_mapped = M @ white_point
    M = np.diag(white_point / W_mapped) @ M

    M = xp.asarray(M)

    if sat_adjust > 1:
        gamut_compression = (sat_adjust - 1) / 4
        # Compute original luminance
        Y = rgb[..., 1:2]
        rgb = rgb @ M.T

        # Apply gamut compression (your existing steps)
        a = rgb.max(axis=-1, keepdims=True)
        d = np.where(a != 0, (a - rgb) / np.abs(a), 0)
        d = gamut_compression * d / (d + 1) + (1 - gamut_compression) * d
        rgb_compressed = a - d * np.abs(a)

        # Compute new luminance
        Y_new = rgb_compressed @ luminance_factors.reshape(-1, 1)

        # Avoid division by zero
        scale = np.where(Y_new != 0, Y / Y_new, 1.0)

        # Rescale to preserve luminance
        rgb = rgb_compressed * scale
    else:
        rgb = rgb @ M.T

    return rgb



def convolution_filter(rgb, kernel, padding=False):
    if not cuda_available:
        return cv2.filter2D(rgb, -1, kernel)
    else:
        if len(kernel.shape) == 2:
            kernel = kernel[..., xp.newaxis]
        if padding:
            pad_h = kernel.shape[0] // 2
            pad_w = kernel.shape[1] // 2
            rgb = xp.pad(rgb, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'reflect')
            return signal.oaconvolve(rgb, kernel, mode='valid', axes=(0, 1))
        else:
            return signal.oaconvolve(rgb, kernel, mode='same', axes=(0, 1))
