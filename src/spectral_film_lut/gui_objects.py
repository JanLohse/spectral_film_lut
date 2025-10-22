import math
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import Qt, QObject, pyqtSignal, QRunnable, pyqtSlot, QRectF, QRect, QPoint, pyqtProperty, \
    QPropertyAnimation, QPointF, QEasingCurve
from PyQt6.QtGui import QWheelEvent, QFontMetrics, QPainter, QColor, QLinearGradient, QBrush
from PyQt6.QtWidgets import QLabel, QWidget, QHBoxLayout, QFileDialog, QLineEdit, QSlider, QComboBox, QScrollArea, \
    QFrame, QPushButton, QToolButton
from spectral_film_lut.utils import *


class RoundedScrollArea(QScrollArea):
    def __init__(self, radius=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if radius is None:
            radius = BORDER_RADIUS
        self.radius = radius
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)


class WideComboBox(QComboBox):
    def __init__(self, parent=None, base_color=None):
        super(WideComboBox, self).__init__(parent)
        self.setStyleSheet(f"QComboBox QAbstractItemView {{background-color: {MENU_COLOR};}}")

        # Colors (tweak these)
        if base_color is None:
            base_color = BASE_COLOR
        self._base_color = QColor(base_color)  # normal
        self._hover_color = QColor(HOVER_COLOR)  # hover
        self._pressed_color = QColor(PRESSED_COLOR)  # pressed

        # Current color property used by animation
        self._base_text_color = QColor(TEXT_PRIMARY)  # Normal text
        self._pressed_text_color = QColor(TEXT_SECONDARY)  # Pressed text color (slightly lighter or darker if you like)

        # Track current colors
        self._color = QColor(self._base_color)
        self._text_color = QColor(self._base_text_color)

        # Background animation
        self._anim = QPropertyAnimation(self, b"color", self)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))

    def showPopup(self):
        # Measure the widest item text
        fm = QFontMetrics(self.view().font())
        max_width = max(fm.horizontalAdvance(self.itemText(i)) for i in range(self.count()))
        # Add some padding
        max_width += 30

        # Resize the popup view
        self.view().setMinimumWidth(max(self.width(), max_width))
        super().showPopup()

    # ------------------- Events -------------------
    def enterEvent(self, event):
        self._start_color_animation(self._hover_color, HOVER_DURATION // 2)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._start_color_animation(self._base_color, HOVER_DURATION)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        # Animate background quickly and change text color instantly
        self._start_color_animation(self._pressed_color, PRESS_DURATION)
        self._set_text_color(self._pressed_text_color)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # Restore background and text color depending on mouse position
        target_bg = self._hover_color if self.underMouse() else self._base_color
        self._start_color_animation(target_bg, PRESS_DURATION)
        self._set_text_color(self._base_text_color)
        super().mouseReleaseEvent(event)

    # ------------------- Animation Logic -------------------
    def _start_color_animation(self, target: QColor, duration: int):
        self._anim.stop()
        self._anim.setDuration(duration)
        self._anim.setStartValue(QColor(self._color))
        self._anim.setEndValue(QColor(target))
        self._anim.start()

    def getColor(self):
        return self._color

    def setColor(self, color: QColor):
        self._color = QColor(color)
        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))

    color = pyqtProperty(QColor, fget=getColor, fset=setColor)

    # --- Stylesheet builder (applies background-color from self._color) ---
    @staticmethod
    def _build_stylesheet(bg_color: QColor, text_color: QColor) -> str:
        bg = qcolor_to_rgba_string(bg_color)
        fg = qcolor_to_rgba_string(text_color)
        # You can change font, radius, padding etc. here.
        return f"""
            WideComboBox {{
                background-color: {bg};
                color: {fg};
            }}
            
            QComboBox QAbstractItemView {{
                background-color: {MENU_COLOR};
            }}
        """

    def _set_text_color(self, color: QColor):
        """Instantly updates the text color (no animation)."""
        self._text_color = QColor(color)
        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))


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

        file_browse = AnimatedButton('Browse')
        file_browse.setFixedWidth(55)
        file_browse.clicked.connect(self.open_file_dialog)
        self.filename_edit = HoverLineEdit()

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
        self.gradient = [(x, QColor(colour.convert(start_color * (1 - x) + end_color * x, source, "Hexadecimal"))) for x
                         in np.linspace(0, 1, steps)]

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
            handle_bg_rect = QRectF(handle_x - handle_bg_width, groove_rect.center().y() - groove_thickness,
                                    handle_bg_width * 2, groove_thickness * 2)
            painter.setBrush(QColor(BASE_COLOR))
            painter.drawRect(handle_bg_rect)

            handle_width = 1.25 + self._hoverProgress * 1
            handle_length = groove_thickness / 2 + 4
            handle_rect = QRectF(handle_x - handle_width, groove_rect.center().y() - handle_length, handle_width * 2,
                                 handle_length * 2)
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


class SliderLog(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max = None
        self.steps = None
        self.min = None
        self.precision = None

        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.slider = GradientSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.setMinMaxSteps(1, 10, 100, 3)

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

    def setMinMaxSteps(self, min, max, steps=100, default=1, precision=None):
        self.min = min
        self.max = max
        self.steps = steps
        self.precision = precision

        self.slider.setMinimum(0)
        self.slider.setMaximum(steps)
        self.big_increment = math.ceil(steps / 10)
        self.small_increment = math.ceil(steps / 30)

        self.setValue(default)
        self.slider.set_reference_value(self.getPosition())

    def sliderValueChanged(self):
        value = self.getValue()
        if self.precision is not None:
            value = round(value, self.precision)
        if value.is_integer():
            self.text.setText(str(int(value)))
        else:
            self.text.setText(f"{value:.2f}")

    def getValue(self):
        fraction = self.slider.value() / self.steps
        return self.min * (self.max / self.min) ** fraction

    def setValue(self, value):
        fraction = math.log(value / self.min) / math.log(self.max / self.min)
        self.slider.setValue(round(fraction * self.steps))

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


class AnimatedButton(QPushButton):
    def __init__(self, text=None, base_color=None, *args, **kwargs):
        super().__init__(text=text, *args, **kwargs)

        # Colors (tweak these)
        if base_color is None:
            base_color = BASE_COLOR
        self._base_color = QColor(base_color)  # normal
        self._hover_color = QColor(HOVER_COLOR)  # hover
        self._pressed_color = QColor(PRESSED_COLOR)  # pressed

        # Current color property used by animation
        self._base_text_color = QColor(TEXT_PRIMARY)  # Normal text
        self._pressed_text_color = QColor(TEXT_SECONDARY)  # Pressed text color (slightly lighter or darker if you like)

        # Track current colors
        self._color = QColor(self._base_color)
        self._text_color = QColor(self._base_text_color)

        # Background animation
        self._anim = QPropertyAnimation(self, b"color", self)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))

    # ------------------- Events -------------------
    def enterEvent(self, event):
        self._start_color_animation(self._hover_color, HOVER_DURATION // 2)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._start_color_animation(self._base_color, HOVER_DURATION)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        # Animate background quickly and change text color instantly
        self._start_color_animation(self._pressed_color, PRESS_DURATION)
        self._set_text_color(self._pressed_text_color)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # Restore background and text color depending on mouse position
        target_bg = self._hover_color if self.underMouse() else self._base_color
        self._start_color_animation(target_bg, PRESS_DURATION)
        self._set_text_color(self._base_text_color)
        super().mouseReleaseEvent(event)

    # ------------------- Animation Logic -------------------
    def _start_color_animation(self, target: QColor, duration: int):
        self._anim.stop()
        self._anim.setDuration(duration)
        self._anim.setStartValue(QColor(self._color))
        self._anim.setEndValue(QColor(target))
        self._anim.start()

    def getColor(self):
        return self._color

    def setColor(self, color: QColor):
        self._color = QColor(color)
        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))

    color = pyqtProperty(QColor, fget=getColor, fset=setColor)

    # --- Stylesheet builder (applies background-color from self._color) ---
    @staticmethod
    def _build_stylesheet(bg_color: QColor, text_color: QColor) -> str:
        bg = qcolor_to_rgba_string(bg_color)
        fg = qcolor_to_rgba_string(text_color)
        # You can change font, radius, padding etc. here.
        return f"""
            AnimatedButton {{
                background-color: {bg};
                color: {fg};
            }}
        """

    def _set_text_color(self, color: QColor):
        """Instantly updates the text color (no animation)."""
        self._text_color = QColor(color)
        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))


class AnimatedToolButton(QToolButton):
    def __init__(self, text=None, base_color=None, *args, **kwargs):
        super().__init__(text=text, *args, **kwargs)

        if base_color is None:
            base_color = BASE_COLOR

        # Colors
        self._base_color = QColor(base_color)
        self._hover_color = QColor(HOVER_COLOR)
        self._pressed_color = QColor(PRESSED_COLOR)
        self._checked_color = QColor(CHECKED_COLOR)

        # Text colors
        self._base_text_color = QColor(TEXT_PRIMARY)
        self._pressed_text_color = QColor(TEXT_SECONDARY)

        # Current colors
        self._color = QColor(self._base_color)
        self._text_color = QColor(self._base_text_color)

        # Animation
        self._anim = QPropertyAnimation(self, b"color", self)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))

        # For toggled (checkable) buttons
        self.toggled.connect(self._update_state_color)

    # ------------------- Events -------------------
    def enterEvent(self, event):
        self._start_color_animation(self._hover_color, HOVER_DURATION // 2)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._start_color_animation(self._get_base_state_color(), HOVER_DURATION)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self._start_color_animation(self._pressed_color, PRESS_DURATION)
        self._set_text_color(self._pressed_text_color)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        # After releasing, go back depending on hover/toggle state
        target_bg = self._hover_color if self.underMouse() else self._get_base_state_color()
        self._start_color_animation(target_bg, PRESS_DURATION)
        self._set_text_color(self._base_text_color)
        super().mouseReleaseEvent(event)

    # ------------------- Helpers -------------------
    def _update_state_color(self, checked: bool):
        """Immediately updates background when toggled."""
        target = self._checked_color if checked else self._base_color
        self._start_color_animation(target, HOVER_DURATION)

    def _get_base_state_color(self):
        """Returns the correct base color depending on state."""
        if self.isCheckable() and self.isChecked():
            return self._checked_color
        return self._base_color

    # ------------------- Animation Logic -------------------
    def _start_color_animation(self, target: QColor, duration: int):
        self._anim.stop()
        self._anim.setDuration(duration)
        self._anim.setStartValue(QColor(self._color))
        self._anim.setEndValue(QColor(target))
        self._anim.start()

    def getColor(self):
        return self._color

    def setColor(self, color: QColor):
        self._color = QColor(color)
        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))

    color = pyqtProperty(QColor, fget=getColor, fset=setColor)

    @staticmethod
    def _build_stylesheet(bg_color: QColor, text_color: QColor) -> str:
        bg = qcolor_to_rgba_string(bg_color)
        fg = qcolor_to_rgba_string(text_color)
        return f"""
            QToolButton {{
                background-color: {bg};
                color: {fg};
            }}
        """

    def _set_text_color(self, color: QColor):
        self._text_color = QColor(color)
        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))


class HoverLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Colors
        self._base_color = QColor(LINEEDIT_COLOR)
        self._hover_color = QColor(HOVER_COLOR)
        self._color = QColor(self._base_color)

        # Animation setup
        self._anim = QPropertyAnimation(self, b"color", self)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        # Apply initial style
        self.setStyleSheet(self._build_stylesheet(self._color))

    # --- Hover events ---
    def enterEvent(self, event):
        self._start_color_animation(self._hover_color, HOVER_DURATION // 2)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._start_color_animation(self._base_color, HOVER_DURATION)
        super().leaveEvent(event)

    # --- Animation logic ---
    def _start_color_animation(self, target: QColor, duration: int):
        self._anim.stop()
        self._anim.setDuration(duration)
        self._anim.setStartValue(QColor(self._color))
        self._anim.setEndValue(QColor(target))
        self._anim.start()

    def getColor(self):
        return self._color

    def setColor(self, color: QColor):
        self._color = QColor(color)
        self.setStyleSheet(self._build_stylesheet(self._color))

    color = pyqtProperty(QColor, fget=getColor, fset=setColor)

    def _build_stylesheet(self, color: QColor) -> str:
        bg = qcolor_to_rgba_string(color)
        return f"""
            QLineEdit {{
                background-color: {bg};
            }}
        """

def qcolor_to_rgba_string(c: QColor) -> str:
    """Return a CSS rgba(...) string where alpha is 0..1 (Qt accepts that)."""
    r = c.red()
    g = c.green()
    b = c.blue()
    a = c.alpha() / 255.0
    return f"rgba({r}, {g}, {b}, {a:.3f})"
