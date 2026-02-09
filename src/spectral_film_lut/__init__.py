from pathlib import Path

try:
    from ._version import __version__
except ImportError:
    __version__ = ""

BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = str(BASE_DIR).replace("\\", "/")