"""Configure pytest for the headless application test environment."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
