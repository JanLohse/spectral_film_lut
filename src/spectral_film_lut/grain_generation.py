from functools import cache

import imageio
import imageio.v3 as iio
from PyQt6.QtCore import QRegularExpression
from PyQt6.QtGui import QIntValidator, QRegularExpressionValidator
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QCheckBox, QProgressDialog, QApplication)

from spectral_film_lut.file_formats import FILE_FORMATS
from spectral_film_lut.gui_objects import *


@cache
def gaussian_noise_cache(shape):
    return xp.random.default_rng().standard_normal(shape, dtype=default_dtype)


def gaussian_noise(shape, cached=False):
    if not cached:
        return xp.random.default_rng().standard_normal(shape, dtype=default_dtype)
    noise_size = ((max(shape[:2]) + 100) // 1024 + 1) * 1024
    noise_map = gaussian_noise_cache((noise_size, noise_size))
    if cuda_available:
        offsets = xp.stack(
            [xp.random.randint(0, x, size=shape[2]) for x in [noise_size - shape[0] + 1, noise_size - shape[1] + 1]]).T
    else:
        offsets = xp.random.randint([0, 0], [noise_size - shape[0] + 1, noise_size - shape[1] + 1], size=(shape[2], 2))
    noise = xp.stack([noise_map[x:shape[0] + x, y:shape[1] + y] for x, y in offsets], axis=-1)
    return noise


def grain_kernel(pixel_size_mm, dye_size1_mm=0.0065, dye_size2_mm=0.015):
    # based on the paper:
    # Simulating Film Grain using the Noise-Power Spectrum by Ian Stephenson and Arthur Saunders
    kernel_size_mm = 4.24 * max(dye_size1_mm, dye_size2_mm)
    kernel_size = round(kernel_size_mm / pixel_size_mm)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        return None

    # Frequency grid (cycles per mm)
    fx = xp.fft.fftfreq(kernel_size, d=pixel_size_mm)
    fy = xp.fft.fftfreq(kernel_size, d=pixel_size_mm)
    FX, FY = xp.meshgrid(fx, fy)
    f = xp.sqrt(FX ** 2 + FY ** 2)  # radial frequency

    # Gaussian model for dye NPS: exp(-(pi*f*D)^2)
    nps1 = xp.exp(-(xp.pi * f * dye_size1_mm) ** 2)
    nps2 = xp.exp(-(xp.pi * f * dye_size2_mm) ** 2)

    # Total NPS (weighted sum)
    nps = (nps1 + nps2)
    # generate convolution kernel
    # Get spatial kernel by inverse FFT
    kernel = xp.fft.ifft2(xp.sqrt(nps))
    kernel = xp.fft.fftshift(kernel.real)  # center it

    # normalize kernel
    kernel /= xp.sqrt(xp.sum(kernel))

    return kernel


def generate_grain(shape, scale, grain_size=1., bw_grain=False, cached=False, **kwargs):
    # compute scaling factor of exposure rms in regard to measuring device size
    if bw_grain:
        shape = (shape[0], shape[1], 1)
    noise = gaussian_noise(shape, cached=cached)
    kernel = grain_kernel(1 / scale, 6.5 * grain_size / 1000, 15 * grain_size / 1000)
    if kernel is not None:
        noise = convolution_filter(noise, kernel)
    if len(noise.shape) == 2:
        noise = noise[..., xp.newaxis]
    if bw_grain:
        noise /= 1.5
    return noise


def generate_grain_frame(width, height, channels, scale, grain_size = 1., std_div=0.001):
    noise = generate_grain((height, width, channels), scale, grain_size=grain_size) * scale
    noise = (xp.clip(noise * std_div + 0.5, 0, 1) * 255).astype(xp.uint8)
    noise = to_numpy(noise)
    return noise


def output_file(noise, file_format, filename, ext):
    kwargs = FILE_FORMATS[file_format]["kwargs"]
    # --- Write output ---
    if file_format.endswith("Sequence"):
        os.makedirs(filename, exist_ok=True)
        for i, frame in enumerate(noise):
            frame_filename = os.path.join(filename, f"{os.path.basename(filename).split('.')[0]}_{i:04d}{ext}")
            iio.imwrite(frame_filename, frame, **kwargs)
    else:
        iio.imwrite(filename, noise, fps=24, **kwargs)


class ExportGrainDialog(QDialog):
    """Dialog for exporting random noise (grain) as images or video."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Noise")
        self.setStyleSheet(f"""
        QWidget {{
            border-radius: {BUTTON_RADIUS}px; 
            background-color: {BASE_COLOR};
        }}
        
        HoverLineEdit {{
            background-color: {SCROLLBAR_HOVER_COLOR};
        }}
        
        AnimatedButton, QComboBox, QToolButton {{
            padding: 3px;
        }}
        
        AnimatedButton::hover, QComboBox::hover, HoverLineEdit::hover {{
            background-color: {HOVER_COLOR};
        }}
        """)

        # --- UI Elements ---
        layout = QVBoxLayout()

        # Width
        layout.addWidget(QLabel("Width:"))
        self.width_field = HoverLineEdit("1920")
        self.width_field.setValidator(QIntValidator())
        layout.addWidget(self.width_field)

        # Height
        layout.addWidget(QLabel("Height:"))
        self.height_field = HoverLineEdit("1080")
        self.height_field.setValidator(QIntValidator())
        layout.addWidget(self.height_field)

        # Frames
        layout.addWidget(QLabel("Frame count:"))
        self.frame_field = HoverLineEdit("100")
        self.frame_field.setValidator(QIntValidator())
        layout.addWidget(self.frame_field)

        # Channels
        self.color_box = QCheckBox("Full color")
        self.color_box.setChecked(True)
        layout.addWidget(self.color_box)

        # Format selector
        layout.addWidget(QLabel("File format:"))
        self.format_selector = WideComboBox()
        self.format_selector.addItems(list(FILE_FORMATS.keys()))
        self.format_selector.setCurrentText("TIFF Sequence")
        layout.addWidget(self.format_selector)

        regex = QRegularExpression(r"[0-9]*|[0-9]+\.[0-9]*")
        double_validator = QRegularExpressionValidator(regex)
        layout.addWidget(QLabel("Frame rate:"))
        self.frame_rate = WideComboBox()
        self.frame_rate.setEditable(True)
        self.frame_rate.addItems(["23.976", "24", "25", "29.97", "30", "50", "59.94", "60"])
        self.frame_rate.setValidator(double_validator)
        self.frame_rate.setCurrentText("24")
        layout.addWidget(self.frame_rate)

        layout.addWidget(QLabel("Frame width (mm):"))
        self.frame_width_field = WideComboBox()
        self.frame_width_field.setEditable(True)
        self.frame_width_field.addItems(["5.79", "6.30", "12.42", "24.90", "36", "52.15", "70.41"])
        self.frame_width_field.setValidator(double_validator)
        self.frame_width_field.setCurrentText("36")
        layout.addWidget(self.frame_width_field)

        layout.addWidget(QLabel("Grain scale:"))
        self.grain_size_field = Slider()
        self.grain_size_field.setMinMaxTicks(0.5, 2, 1, 10)
        self.grain_size_field.setValue(1.)
        layout.addWidget(self.grain_size_field)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = AnimatedButton("Export")
        cancel_button = AnimatedButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.cancelled = False

        # --- Connections ---
        ok_button.clicked.connect(self.export_noise)
        cancel_button.clicked.connect(self.reject)

    def get_values(self):
        width = int(self.width_field.text() or 1920)
        height = int(self.height_field.text() or 1080)
        frames = int(self.frame_field.text() or 1)
        channels = 3 if self.color_box.isChecked() else 1
        file_format = self.format_selector.currentText()
        frame_rate = float(self.frame_rate.currentText())
        scale = width / float(self.frame_width_field.currentText())
        grain_size = float(self.grain_size_field.getValue())
        return width, height, frames, channels, file_format, frame_rate, scale, grain_size

    def export_noise(self):
        width, height, frames, channels, file_format, frame_rate, scale, grain_size = self.get_values()
        ext = FILE_FORMATS[file_format]["extension"]

        filename, ok = QFileDialog.getSaveFileName(self, "Choose output file", "", "*" + ext)
        if not ok or not filename:
            return

        if not filename.endswith(ext):
            filename += ext

        # --- Setup QProgressDialog ---
        self.progress_dialog = QProgressDialog("Preparing export...", "Cancel", 0, frames + 1, self)
        self.progress_dialog.setWindowTitle("Export grain overlay")
        self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        # self.progress_dialog.canceled.connect(self.cancel_export)
        QApplication.processEvents()

        self.cancelled = False
        self.progress_dialog.canceled.connect(self.cancel)

        kwargs = FILE_FORMATS[file_format]["kwargs"]
        if file_format.endswith("Sequence"):
            os.makedirs(filename, exist_ok=True)
            for i in range(frames):
                if self.cancelled:
                    break
                frame = generate_grain_frame(width, height, channels, scale, grain_size)
                frame_filename = os.path.join(filename, f"{os.path.basename(filename).split('.')[0]}_{i:04d}{ext}")
                iio.imwrite(frame_filename, frame, **kwargs)
                self.progress_dialog.setLabelText(f"Exported {i + 1}/{frames} frames")
                self.progress_dialog.setValue(i + 1)
        else:
            with imageio.get_writer(filename, fps=frame_rate, **kwargs) as writer:
                for i in range(frames):
                    if self.cancelled:
                        break
                    frame = generate_grain_frame(width, height, channels, scale, grain_size)
                    writer.append_data(frame)
                    self.progress_dialog.setLabelText(f"Exported {i + 1}/{frames} frames")
                    self.progress_dialog.setValue(i + 1)
                self.progress_dialog.setLabelText("Finishing up video file")
                QApplication.processEvents()

        self.progress_dialog.setValue(frames + 1)
        self.accept()
        QApplication.processEvents()

    def cancel(self):
        self.cancelled = True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = ExportGrainDialog()
    dlg.show()
    sys.exit(app.exec())
