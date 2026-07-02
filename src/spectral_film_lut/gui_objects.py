"""Additional GUI objects used by Spectral Film LUT."""

import math
import sys
import traceback
from pathlib import Path

import colour
import numpy as np
from PyQt6.QtCore import (
    QEasingCurve,
    QObject,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    QRunnable,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
    QWidget,
)

from spectral_film_lut.css_theme import (
    BACKGROUND_COLOR,
    BASE_COLOR,
    BORDER_RADIUS,
    CHECKED_COLOR,
    HOVER_COLOR,
    HOVER_DURATION,
    LINEEDIT_COLOR,
    OUTLINE_COLOR,
    PRESS_DURATION,
    PRESSED_COLOR,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


class RoundedScrollArea(QScrollArea):
    """A QScrollArea with rounded corners."""

    def __init__(self, radius=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if radius is None:
            radius = BORDER_RADIUS
        self.radius = radius
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)


class WideComboBox(QComboBox):
    """A combo box with animated hover and expanded and styled frame."""

    def __init__(self, parent=None, base_color=None):
        super().__init__(parent)

        # Colors (tweak these)
        if base_color is None:
            base_color = BASE_COLOR
        self._base_color = QColor(base_color)  # normal
        self._hover_color = QColor(HOVER_COLOR)  # hover
        self._pressed_color = QColor(PRESSED_COLOR)  # pressed

        # Current color property used by animation
        self._base_text_color = QColor(TEXT_PRIMARY)  # Normal text
        self._pressed_text_color = QColor(
            TEXT_SECONDARY
        )  # Pressed text color (slightly lighter or darker if you like)

        # Track current colors
        self._color = QColor(self._base_color)
        self._text_color = QColor(self._base_text_color)

        # Background animation
        self._anim = QPropertyAnimation(self, b"color", self)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.view().setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        popup = self.view().window()
        popup.setMaximumHeight(400)
        popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        popup.setStyleSheet(f"""
        QFrame {{
            background-color: {BACKGROUND_COLOR};
            border: 1px solid {OUTLINE_COLOR};
            border-radius: {BORDER_RADIUS}px;
        }}
        QAbstractItemView {{
            border: none;
        }}
        """)

        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))

    def showPopup(self):
        # Measure the widest item text
        fm = QFontMetrics(self.view().font())
        max_width = max(
            fm.horizontalAdvance(self.itemText(i)) for i in range(self.count())
        )
        # Add some padding
        max_width += 30

        # Resize the popup view
        self.view().setMinimumWidth(max(self.width(), max_width))
        super().showPopup()

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
        WideComboBox {{
            background-color: {bg};
            color: {fg};
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
    """A widget with a src line edit and a button to open a system file selector."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

        file_browse = AnimatedButton("Browse")
        file_browse.setFixedWidth(55)
        file_browse.clicked.connect(self.open_file_dialog)
        self.filename_edit = HoverLineEdit()

        layout.addWidget(self.filename_edit)
        layout.addWidget(file_browse)

        self.textChanged = self.filename_edit.textChanged
        self.filetype = "Images (*.png *.jpg *)"

        self.show()

    def open_file_dialog(self):
        filename, ok = QFileDialog.getOpenFileName(
            self, "Select a File", "", self.filetype
        )
        if filename:
            path = Path(filename)
            self.filename_edit.setText(str(path))

    def currentText(self):
        return self.filename_edit.text()


class GradientSlider(QSlider):
    """A styled slider with float value support and customizable color gradients."""

    def __init__(self, *args, reference_value=0, modern_design=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient = None
        self.set_color_gradient((0.3, 0.0, 0.0), (0.7, 0.0, 0.0))
        self.modern_design = modern_design
        self.setRange(-100, 100)
        self.setValue(0)
        self.reference_value = reference_value
        self._hover = False
        self._hoverProgress = 0.0  # property for animation
        self.anim = QPropertyAnimation(self, b"hoverProgress", self)
        self.anim.setDuration(150)
        self.setMouseTracking(True)
        self.setStyleSheet("QSlider { background: transparent; }")

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
        self.gradient = [
            (
                x,
                QColor(
                    colour.convert(
                        start_color * (1 - x) + end_color * x, source, "Hexadecimal"
                    )
                ),
            )
            for x in np.linspace(0, 1, steps)
        ]

    hoverProgress = pyqtProperty(
        float, fget=get_hover_progress, fset=set_hover_progress
    )

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
        groove_rect = QRect(
            horizontal_padding,
            self.height() // 2 - groove_thickness // 2,
            self.width() - horizontal_padding * 2,
            groove_thickness,
        )

        min_val, max_val = self.minimum(), self.maximum()
        total_range = max_val - min_val

        def value_to_x(val):
            return (
                groove_rect.left() + (val - min_val) / total_range * groove_rect.width()
            )

        handle_x = int(value_to_x(self.value()))
        ref_x = int(value_to_x(self.reference_value))

        # background groove with gradient
        painter.setPen(Qt.PenStyle.NoPen)

        gradient = QLinearGradient(
            QPointF(groove_rect.topLeft()), QPointF(groove_rect.topRight())
        )
        for pos, color in self.gradient:
            gradient.setColorAt(pos, color)  # left color

        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(groove_rect, 3, 3)

        # active segment (ref -> handle)
        painter.setBrush(QColor(255, 255, 255, 85))
        if handle_x > ref_x:
            active_rect = QRect(
                ref_x, groove_rect.top(), handle_x - ref_x, groove_rect.height()
            )
        else:
            active_rect = QRect(
                handle_x, groove_rect.top(), ref_x - handle_x, groove_rect.height()
            )
        if self.reference_value in (self.minimum(), self.maximum()):
            painter.drawRoundedRect(active_rect, 3, 3)
        else:
            painter.drawRect(active_rect)

        # handle
        if self.modern_design:
            handle_bg_width = 6 + self._hoverProgress * 1
            handle_bg_rect = QRectF(
                handle_x - handle_bg_width,
                groove_rect.center().y() - groove_thickness,
                handle_bg_width * 2,
                groove_thickness * 2,
            )
            painter.setBrush(QColor(BASE_COLOR))
            painter.drawRect(handle_bg_rect)

            handle_width = 1.25 + self._hoverProgress * 1
            handle_length = groove_thickness / 2 + 4
            handle_rect = QRectF(
                handle_x - handle_width,
                groove_rect.center().y() - handle_length,
                handle_width * 2,
                handle_length * 2,
            )
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
    """A slider with float value support."""

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
        self.text.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
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
        self.slider.setValue(
            round((value - self.min) * self.denominator / self.enumerator)
        )

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
    """A slider with float value support and a logarithmic value distribution."""

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
        self.text.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
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
        if value.is_integer():
            self.text.setText(str(int(value)))
        else:
            self.text.setText(f"{value:.3f}")

    def getValue(self):
        fraction = self.slider.value() / self.steps
        value = self.min * (self.max / self.min) ** fraction
        if self.precision is not None:
            value = round(value, self.precision)
        return value

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
    """A button with animated hover and press."""

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
        self._pressed_text_color = QColor(
            TEXT_SECONDARY
        )  # Pressed text color (slightly lighter or darker if you like)

        # Track current colors
        self._color = QColor(self._base_color)
        self._text_color = QColor(self._base_text_color)

        # Background animation
        self._anim = QPropertyAnimation(self, b"color", self)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))

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
        # You can change font, radius, padding etc. here.
        return f"""
            AnimatedButton {{
                background-color: {bg};
                color: {fg};
            }}

            AnimatedButton:disabled {{
                background-color: transparent;
                color: {TEXT_SECONDARY};
            }}
        """

    def _set_text_color(self, color: QColor):
        """Instantly updates the text color (no animation)."""
        self._text_color = QColor(color)
        self.setStyleSheet(self._build_stylesheet(self._color, self._text_color))


class AnimatedToolButton(QToolButton):
    """A tool button with animated hover and press."""

    def __init__(self, base_color=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        target_bg = (
            self._hover_color if self.underMouse() else self._get_base_state_color()
        )
        self._start_color_animation(target_bg, PRESS_DURATION)
        self._set_text_color(self._base_text_color)
        super().mouseReleaseEvent(event)

    def _update_state_color(self, checked: bool):
        """Immediately updates background when toggled."""
        target = self._checked_color if checked else self._base_color
        self._start_color_animation(target, HOVER_DURATION)

    def _get_base_state_color(self):
        """Returns the correct base color depending on state."""
        if self.isCheckable() and self.isChecked():
            return self._checked_color
        return self._base_color

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
    """A line edit with animated hover."""

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

    def enterEvent(self, event):
        self._start_color_animation(self._hover_color, HOVER_DURATION // 2)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._start_color_animation(self._base_color, HOVER_DURATION)
        super().leaveEvent(event)

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


class StyledContainer(QWidget):
    def __init__(self, parent=None, bg_color=BASE_COLOR, radius=16.0):
        super().__init__(parent)
        self.bg_color = QColor(bg_color)
        self.radius = radius

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect().toRectF(), self.radius, self.radius)
        painter.fillPath(path, self.bg_color)


class StyledPill(QWidget):
    def __init__(self, parent=None, color=TEXT_PRIMARY, pad_x=2.0, pad_y=5.0):
        super().__init__(parent)
        self.color = QColor(color)
        self._hover_progress = 0.0

        # Internal sizing baselines
        self.baseline_pad_x = pad_x
        self.baseline_pad_y = pad_y

    @pyqtProperty(float)
    def hoverProgress(self):
        return self._hover_progress

    @hoverProgress.setter
    def hoverProgress(self, val):
        self._hover_progress = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        base_rect = self.rect().toRectF()

        # Scaling margins based on exposed config settings
        current_pad_x = self.baseline_pad_x * (1.0 - self._hover_progress)
        current_pad_y = self.baseline_pad_y * (1.0 - self._hover_progress)

        pill_rect = QRectF(
            base_rect.x() + current_pad_x,
            base_rect.y() + current_pad_y,
            base_rect.width() - (current_pad_x * 2.0),
            base_rect.height() - (current_pad_y * 2.0),
        )

        path = QPainterPath()
        radius = pill_rect.height() / 2.0
        path.addRoundedRect(pill_rect, radius, radius)

        painter.fillPath(path, self.color)


class ModeSelector(QWidget):
    currentTextChanged = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        items=None,
        pill_thickness_pad: float = 2.0,
        pill_inset_pad_x: float = 2.0,
        hover_expand_duration: int = 150,
        slide_duration=150,
        bg_radius: float = 14.0,
        bg_padding: int = 2,
    ):
        super().__init__(parent)
        self._items = list(items or [])
        self._buttons = []
        self._current = ""

        self._start_rect = QRect()
        self._target_rect = QRect()
        self._anim_progress = 0.0

        # Save custom properties
        self.pill_thickness_pad = pill_thickness_pad
        self.pill_inset_pad_x = pill_inset_pad_x
        self.slide_duration = slide_duration

        self.setMouseTracking(True)

        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        outer_layout.addStretch()

        # Build Background Container using passed geometry params
        self._container = StyledContainer(self, bg_color=BASE_COLOR, radius=bg_radius)
        self._container.setMouseTracking(True)

        self._layout = QHBoxLayout(self._container)
        self._layout.setContentsMargins(
            bg_padding, bg_padding, bg_padding - 2, bg_padding
        )
        self._layout.setSpacing(2)

        outer_layout.addWidget(self._container)
        outer_layout.addStretch()

        # Build Pill using passed sizing parameters
        self._pill = StyledPill(
            self._container,
            color=TEXT_PRIMARY,
            pad_x=self.pill_inset_pad_x,
            pad_y=self.pill_thickness_pad,
        )
        self._pill.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._pill.lower()

        # Configured Hover Animation
        self._hover_anim = QPropertyAnimation(self._pill, b"hoverProgress")
        self._hover_anim.setDuration(hover_expand_duration)
        self._hover_anim.setEasingCurve(QEasingCurve.Type.OutQuad)

        # Configured Slide Animation
        self._anim = QPropertyAnimation(self, b"animProgress")
        self._anim.setDuration(self.slide_duration)
        self._anim.setEasingCurve(QEasingCurve.Type.Linear)
        offset = 3.0
        acceleration = 3.0
        anchor = 0.2
        x_1 = anchor ** (1 + offset)
        x_2 = (1 - anchor) ** (1 + offset)
        y_1 = anchor**acceleration
        y_2 = 1 - y_1
        self._leader_curve = QEasingCurve(QEasingCurve.Type.BezierSpline)
        self._leader_curve.addCubicBezierSegment(
            QPointF(x_1, y_1),
            QPointF(x_2, y_2),
            QPointF(1.0, 1.0),
        )
        self._follower_curve = QEasingCurve(QEasingCurve.Type.BezierSpline)
        self._follower_curve.addCubicBezierSegment(
            QPointF(1 - x_2, y_1),
            QPointF(1 - x_1, y_2),
            QPointF(1.0, 1.0),
        )

        if self._items:
            self.addItems(self._items)
            if self._current == "" and self._buttons:
                self._buttons[0].setChecked(True)
                self._current = self._buttons[0].text()

    @pyqtProperty(float)
    def animProgress(self):
        return self._anim_progress

    @animProgress.setter
    def animProgress(self, val):
        self._anim_progress = val
        if self._start_rect.isNull() or self._target_rect.isNull():
            return

        is_moving_right = self._target_rect.x() > self._start_rect.x()

        # Utilize configured curves to handle dynamic lag stretch
        curve_fast = QEasingCurve(self._leader_curve)
        curve_slow = QEasingCurve(self._follower_curve)

        if is_moving_right:
            p_right = curve_fast.valueForProgress(val)
            p_left = curve_slow.valueForProgress(val)
        else:
            p_left = curve_fast.valueForProgress(val)
            p_right = curve_slow.valueForProgress(val)

        left = int(
            self._start_rect.left()
            + (self._target_rect.left() - self._start_rect.left()) * p_left
        )
        right = int(
            self._start_rect.right()
            + (self._target_rect.right() - self._start_rect.right()) * p_right
        )

        new_geometry = QRect(
            left,
            self._target_rect.y(),
            max(1, right - left),
            self._target_rect.height(),
        )
        self._pill.setGeometry(new_geometry)

        self._handle_color_swap(val)

    def _handle_color_swap(self, progress):
        target_text = self._current
        if progress >= 0.5:
            for btn in self._buttons:
                if btn.text() == target_text:
                    if not btn.isChecked():
                        btn.setChecked(True)
                elif btn.isChecked():
                    btn.setChecked(False)
        else:
            old_text = getattr(self, "_old_text", "")
            if old_text:
                for btn in self._buttons:
                    if btn.text() == old_text:
                        if not btn.isChecked():
                            btn.setChecked(True)
                    elif btn.isChecked():
                        btn.setChecked(False)

    def enterEvent(self, event):
        self._hover_anim.stop()
        self._hover_anim.setStartValue(self._pill.hoverProgress)
        self._hover_anim.setEndValue(1.0)
        self._hover_anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hover_anim.stop()
        self._hover_anim.setStartValue(self._pill.hoverProgress)
        self._hover_anim.setEndValue(0.0)
        self._hover_anim.start()
        super().leaveEvent(event)

    def addItems(self, items):
        for text in items:
            btn = QPushButton(text, self._container)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setFlat(True)
            btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            btn.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

            checked_color = "rgba(20, 20, 20, 255)"
            btn.setStyleSheet(
                f"QPushButton {{ background: transparent; border: none; "
                f"color: {TEXT_SECONDARY}; padding: 4px 12px; margin: 0; }} "
                f"QPushButton:checked {{ color: {checked_color}; }}"
            )
            self._layout.addWidget(btn)
            self._buttons.append(btn)

        self._container.updateGeometry()
        QTimer.singleShot(50, self._position_pill_on_current)

    def setCurrentText(self, text):
        if self._current == text:
            return

        self._old_text = self._current
        self._current = text

        for btn in self._buttons:
            if btn.text() == text:
                self._animate_pill_to(btn)
                self.currentTextChanged.emit(text)
                return

    def currentText(self):
        return self._current

    def _select_button_at_pos(self, pos):
        container_pos = self._container.mapFrom(self, pos)
        for btn in self._buttons:
            if btn.geometry().contains(container_pos):
                self.setCurrentText(btn.text())
                break

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._select_button_at_pos(event.position().toPoint())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._select_button_at_pos(event.position().toPoint())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def wheelEvent(self, event):
        if not self._buttons:
            event.ignore()
            return

        current_idx = -1
        for i, btn in enumerate(self._buttons):
            if btn.text() == self._current:
                current_idx = i
                break

        if current_idx == -1:
            current_idx = 0

        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta < 0:
            next_idx = max(0, current_idx - 1)
        elif delta > 0:
            next_idx = min(len(self._buttons) - 1, current_idx + 1)
        else:
            next_idx = current_idx

        if next_idx != current_idx:
            self.setCurrentText(self._buttons[next_idx].text())
            event.accept()
        else:
            super().wheelEvent(event)

    def _get_target_rect(self, btn: QPushButton) -> QRect:
        target = btn.geometry()
        fm = QFontMetrics(btn.font())
        text_width = fm.horizontalAdvance(btn.text())
        horiz_pad = 12 * 2
        pill_w = text_width + horiz_pad
        x = target.x() + max(0, (target.width() - pill_w) // 2)

        return QRect(
            max(0, x),
            max(0, target.y()),
            max(1, pill_w),
            max(1, target.height()),
        )

    def _animate_pill_to(self, btn: QPushButton):
        if btn is None:
            return
        target_rect = self._get_target_rect(btn)

        if not self._pill.geometry().isValid() or self._pill.geometry().isEmpty():
            self._pill.setGeometry(target_rect)
            self._pill.lower()
            return

        self._anim.stop()

        self._start_rect = self._pill.geometry()
        self._target_rect = target_rect

        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.start()
        self._pill.lower()

    def _position_pill_on_current(self):
        cur = self._current
        if not cur and self._buttons:
            cur = self._buttons[0].text()
            self._current = cur

        for btn in self._buttons:
            if btn.text() == cur:
                btn.setChecked(True)
                target_rect = self._get_target_rect(btn)
                self._pill.setGeometry(target_rect)
                self._pill.lower()
                break

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._position_pill_on_current)


def qcolor_to_rgba_string(c: QColor) -> str:
    """Return a CSS rgba(...) string where alpha is 0..1 (Qt accepts that)."""
    r = c.red()
    g = c.green()
    b = c.blue()
    a = c.alpha() / 255.0
    return f"rgba({r}, {g}, {b}, {a:.3f})"


class AboutDialog(QMessageBox):
    def __init__(
        self,
        parent=None,
        app_name="Application",
        version="1.0.0",
        author="Author",
        year="2026",
        license_type="MIT License",
        links=None,
    ):
        """
        A highly customizable and styled About Dialog for PyQt6 applications.

        :param links: Dict of {'Link Text': 'URL'}
        """
        super().__init__(parent)
        self.setWindowTitle(f"About {app_name}")
        self.setIcon(QMessageBox.Icon.Information)

        # Default to an empty dict if no links are provided
        if links is None:
            links = {}

        # Build the HTML strings for links dynamically
        links_html = ""
        for label, url in links.items():
            links_html += f"• <a href='{url}'>{label}</a><br>\n"

        # Complete HTML with built-in styling
        about_text = f"""
        <style>
            div {{
                font-family: sans-serif;
            }}
            .title {{
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 8px;
            }}
            .meta {{
                font-size: 12px;
                margin-bottom: 12px;
                line-height: 1.4;
            }}
            .links a {{
                text-decoration: none;
            }}
            .links a:hover {{
                text-decoration: underline;
            }}
        </style>

        <div style="padding-right: 35px; padding-bottom: 5px;">
            <div class="title">{app_name} {version}</div>

            <div class="meta">
                Created by {author}<br>
                © {year} {author} | {license_type}
            </div>

            <div class="links">
                {links_html}
            </div>
        </div>
        """

        self.setText(about_text)
        self.setTextInteractionFlags(self.textInteractionFlags().LinksAccessibleByMouse)
