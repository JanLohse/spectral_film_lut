import os
import sys
import time
import traceback
from pathlib import Path
import subprocess

import colour
from PyQt6.QtCore import Qt, QObject, pyqtSignal, QRunnable, pyqtSlot
from PyQt6.QtWidgets import QPushButton, QLabel, QWidget, QHBoxLayout, QFileDialog, QLineEdit, QSlider

from ffmpeg.nodes import output_operator
from ffmpeg._run import compile

from spectral_film_lut.film_spectral import FilmSpectral



def create_lut(negative_film, print_film=None, lut_size=33, name="test", verbose=False, **kwargs):
    lut = colour.LUT3D(size=lut_size, name="test")
    transform, _ = FilmSpectral.generate_conversion(negative_film, print_film, **kwargs)
    start = time.time()
    lut.table = transform(lut.table)
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
        filename, ok = QFileDialog.getOpenFileName(self, "Select a File", "D:\\icons\\avatar\\",
                                                   self.filetype)
        if filename:
            path = Path(filename)
            self.filename_edit.setText(str(path))

    def currentText(self):
        return self.filename_edit.text()


class Slider(QWidget):
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
        self.text.setFixedWidth(30)

        layout.addWidget(self.slider)
        layout.addWidget(self.text)

        self.slider.valueChanged.connect(self.sliderValueChanged)

        self.valueChanged = self.slider.valueChanged

    def setMinMaxTicks(self, min, max, enumerator=1, denominator=1):
        self.slider.setMinimum(0)
        self.slider.setMaximum(int(((max - min) * denominator) / enumerator))
        self.min = min
        self.enumerator = enumerator
        self.denominator = denominator

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

