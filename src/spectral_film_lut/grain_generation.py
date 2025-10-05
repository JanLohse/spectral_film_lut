import os
import time
import numpy as np
import imageio.v3 as iio
from PyQt6.QtCore import Qt
from spectral_film_lut.file_formats import FILE_FORMATS

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox,
    QComboBox, QPushButton, QFileDialog, QProgressDialog, QApplication
)
from PyQt6.QtGui import QIntValidator



def sample_noise(frames, height, width, channels):
    noise = np.random.default_rng().standard_normal(
        (frames, height, width, channels), dtype=np.float32
    )
    return noise


def process_noise(noise):
    noise = noise / 10 + 0.5
    noise *= 255
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    return noise


def output_file(noise, file_format, filename, ext):
    kwargs = FILE_FORMATS[file_format]["kwargs"]
    # --- Write output ---
    if file_format.endswith("Sequence"):
        os.makedirs(filename, exist_ok=True)
        for i, frame in enumerate(noise):
            frame_filename = os.path.join(
                filename, f"{os.path.basename(filename).split('.')[0]}_{i:04d}{ext}"
            )
            iio.imwrite(frame_filename, frame, **kwargs)
    else:
        iio.imwrite(filename, noise, fps=24, **kwargs)


class ExportGrainDialog(QDialog):
    """Dialog for exporting random noise (grain) as images or video."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Noise")

        # --- UI Elements ---
        layout = QVBoxLayout()

        # Width
        layout.addWidget(QLabel("Width:"))
        self.width_field = QLineEdit("1920")
        self.width_field.setValidator(QIntValidator())
        layout.addWidget(self.width_field)

        # Height
        layout.addWidget(QLabel("Height:"))
        self.height_field = QLineEdit("1080")
        self.height_field.setValidator(QIntValidator())
        layout.addWidget(self.height_field)

        # Frames
        layout.addWidget(QLabel("Frame count:"))
        self.frame_field = QLineEdit("1")
        self.frame_field.setValidator(QIntValidator())
        layout.addWidget(self.frame_field)

        # Channels
        self.color_box = QCheckBox("Full color")
        self.color_box.setChecked(True)
        layout.addWidget(self.color_box)

        # Format selector
        layout.addWidget(QLabel("File format:"))
        self.format_selector = QComboBox()
        self.format_selector.addItems(list(FILE_FORMATS.keys()))
        layout.addWidget(self.format_selector)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Export")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # --- Connections ---
        ok_button.clicked.connect(self.export_noise)
        cancel_button.clicked.connect(self.reject)

    # -----------------------------------------------------
    # Collect user input
    # -----------------------------------------------------
    def get_values(self):
        width = int(self.width_field.text() or 1920)
        height = int(self.height_field.text() or 1080)
        frames = int(self.frame_field.text() or 1)
        channels = 3 if self.color_box.isChecked() else 1
        file_format = self.format_selector.currentText()
        return width, height, frames, channels, file_format

    # -----------------------------------------------------
    # Export logic
    # -----------------------------------------------------
    def export_noise(self):
        width, height, frames, channels, file_format = self.get_values()
        ext = FILE_FORMATS[file_format]["extension"]

        filename, ok = QFileDialog.getSaveFileName(self, "Choose output file", "", "*" + ext)
        if not ok or not filename:
            return

        if not filename.endswith(ext):
            filename += ext

        self.progress_dialog = QProgressDialog("Sampling white noise...", "Cancel", 0, 3, self)
        self.progress_dialog.setWindowTitle("Export grain overlay")
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        QApplication.processEvents()

        noise = sample_noise(frames, height, width, channels)
        self.progress_dialog.setValue(1)
        self.progress_dialog.setLabelText("Processing noise to grain...")
        noise = process_noise(noise)
        self.progress_dialog.setValue(2)
        self.progress_dialog.setLabelText("Writing grain to file...")
        output_file(noise, file_format, filename, ext)
        self.progress_dialog.setLabelText("Done!")
        self.progress_dialog.setValue(3)
        QApplication.processEvents()
        time.sleep(0.5)
        self.progress_dialog.close()
        self.accept()
