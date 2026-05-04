"""Define the core parameters for color math."""

import colour
import numpy as np

DEFAULT_DTYPE = np.float32
"""The dtype used for the color pipeline."""
colour.utilities.set_default_float_dtype(DEFAULT_DTYPE)
SPECTRAL_SHAPE = colour.SpectralShape(380, 780, 5)
"""The wavelengths used for all spectral simulations and data."""
