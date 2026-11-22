"""Build film-stock metadata used by the GUI selectors."""

import numpy as np
from PyQt6.QtGui import QImage


def build_filmstock_info(filmstocks: dict) -> dict:
    """Build selector metadata and thumbnails for all available film stocks."""
    type_names = {
        "camerapositive": "Slide",
        "cameranegative": "Negative",
        "printnegative": "Print",
        "printpositive": "SlidePrint",
    }

    return {
        name: {
            "Year": stock.year,
            "Manufacturer": stock.manufacturer,
            "Type": type_names[stock.stage + stock.film_type],
            "Medium": stock.medium,
            "Sensitivity": f"ISO {stock.iso}" if stock.iso is not None else None,
            "sensitivity": stock.iso if stock.iso is not None else None,
            "resolution": f"{stock.resolution} lines/mm" if stock.resolution is not None else None,
            "Resolution": stock.resolution if stock.resolution is not None else None,
            "Granularity": f"{stock.rms} rms" if stock.rms is not None else None,
            "Decade": f"{stock.year // 10 * 10}s" if stock.year is not None else None,
            "stage": stock.stage,
            "Chromaticity": "BW" if stock.density_measure == "bw" else "Color",
            "image": QImage(
                np.require(stock.color_checker, np.uint8, "C"),
                6,
                4,
                18,
                QImage.Format.Format_RGB888,
            ),
            "Gamma": round(stock.gamma, 3),
            "Alias": stock.alias,
            "Comment": stock.comment,
            "D-max": f"{stock.d_max.max():.2f}",
        }
        for name, stock in filmstocks.items()
    }
