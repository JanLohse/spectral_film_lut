import os
import sys

import colour


def oklch_to_hex(l, c=0., h=0.):
    return colour.convert((l, c, h), "Oklch", "Hexadecimal")

BASE_COLOR = oklch_to_hex(0.2175)
BACKGROUND_COLOR = oklch_to_hex(0.276)
ACCENT_COLOR = oklch_to_hex(0.55)
BORDER_RADIUS = 12
BUTTON_RADIUS = 6
HOVER_COLOR = oklch_to_hex(0.35)
OUTLINE_COLOR = oklch_to_hex(0.4)
PRESSED_COLOR = oklch_to_hex(0.3)
CHECKED_COLOR = oklch_to_hex(0.24675)
MENU_COLOR = oklch_to_hex(0.25)
TEXT_PRIMARY = oklch_to_hex(0.9)
TEXT_SECONDARY = oklch_to_hex(0.7)
LINEEDIT_COLOR = oklch_to_hex(0.24675)
SCROLLBAR_HOVER_COLOR = oklch_to_hex(0.24)
SCROLLBAR_THICKNESS = 11
SCROLLBAR_MARGIN = 2
SCROLLBAR_HANDLE_COLOR = oklch_to_hex(0.3633)
SCROLLBAR_HANDLE_HOVER = oklch_to_hex(0.4535)
HIGHLIGHT_COLOR = oklch_to_hex(0.75)
HOVER_DURATION = 150
PRESS_DURATION = 75

def resource_path(relative_path):
    """Get absolute path to resource (works for dev and PyInstaller)."""
    if hasattr(sys, '_MEIPASS'):
        # Running from the PyInstaller bundle
        base_path = sys._MEIPASS
    else:
        # Running from source
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path).replace("\\", "/")


THEME = f"""
QMainWindow > QWidget, QDialog > QWidget {{
    background-color: {BACKGROUND_COLOR};
    color: {TEXT_PRIMARY};
    border-radius: {BUTTON_RADIUS}px;
}}

MainWindow {{
    background-color: {BACKGROUND_COLOR};
}}

MainWindow > QWidget {{
    border-radius: 0px;
}}

QScrollArea {{
    border-radius: {BORDER_RADIUS}px;
}}

QWidget {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
    border: none;
    border-radius: {BUTTON_RADIUS}px;
}}

/* Button */
QPushButton, QComboBox, QToolButton {{
    padding: 3px;
}}


QPushButton:hover, QComboBox:hover, QToolButton:hover {{
    background-color: {HOVER_COLOR};
}}

QPushButton:pressed, QComboBox:pressed, QToolButton:pressed {{
    background-color: {PRESSED_COLOR};
}}

QPushButton:disabled, QComboBox:disabled, QToolButton:disabled, AnimatedButton:disabled {{
    color: {TEXT_SECONDARY};
}}

/* ComboBox */
QComboBox QAbstractItemView {{
    padding: 5px 5px;
    outline: none;
    border: 1px solid {OUTLINE_COLOR};
}}

QComboBox QAbstractItemView::item {{
    background-color: transparent;
    padding: 3px 4px;
    border-radius: {BUTTON_RADIUS}px;
}}

QComboBox QAbstractItemView::item:selected {{
    background-color: {HOVER_COLOR};
}}

QComboBox::down-arrow {{
    image: url("{resource_path("resources/combo_arrow.svg")}");
    border: none;
    outline: none;
    background: transparent;
    width: 12px;
    height: 12px;
}}

QComboBox::drop-down {{
    border: none;
    background: transparent;
}}

/* Button icons */

#plus {{
    image: url("{resource_path("resources/plus.svg")}");
}}
#right {{
    image: url("{resource_path("resources/right.svg")}");
}}
#left {{
    image: url("{resource_path("resources/left.svg")}");
}}
#flip {{
    image: url("{resource_path("resources/flip.svg")}");
}}

/* LineEdit */
QLineEdit {{
    background-color: {SCROLLBAR_HOVER_COLOR};
    border-radius: {BUTTON_RADIUS}px;
    padding: 2px;
}}

QLineEdit:hover {{
    background-color: {HOVER_COLOR};
}}

/* CheckBox */
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 5px;
}}

QCheckBox::indicator:hover {{
    background-color: {HOVER_COLOR};
}}

QCheckBox::indicator:pressed {{
    background-color: {PRESSED_COLOR};
}}

QCheckBox::indicator:unchecked {{
    image: url("{resource_path("resources/checkbox_unchecked.svg")}");
}}

QCheckBox::indicator:checked {{
    image: url("{resource_path("resources/checkbox_checked.svg")}");
}}

QCheckBox::indicator:indeterminate {{
    image: url("{resource_path("resources/checkbox_halfway.svg")}");
}}

/* MenuBar */
QMenuBar {{
    background: transparent;
    spacing: 3px;
}}

QMenuBar::item {{
    padding: 5px 7px;
    border-radius: {BUTTON_RADIUS}px;
}}

QMenuBar::item:selected {{
    background-color: {HOVER_COLOR};
}}

QMenuBar::item:pressed {{
    background-color: {PRESSED_COLOR};
}}

QMenu {{
    background-color: {MENU_COLOR};
    padding: 10px;
    border: 1px solid {OUTLINE_COLOR};
    border-radius: {BORDER_RADIUS}px;
}}

/* Menu */
QMenu::item {{
    padding: 5px 7px;
}}

QMenu::item:selected {{
    background-color: {HOVER_COLOR};
    border: none;
    border-radius: {BUTTON_RADIUS}px;
}}

QMenu::item:pressed {{
    background-color: {PRESSED_COLOR};
    border: none;
    border-radius: {BUTTON_RADIUS}px;
}}

/* ScrollBar */
QScrollBar:vertical, QScrollBar:horizontal {{
    background: transparent;
    border: none;
    height: {SCROLLBAR_THICKNESS}px;
    width: {SCROLLBAR_THICKNESS}px;
}}

QScrollBar:vertical:hover, QScrollBar:horizontal:hover {{
    background: {SCROLLBAR_HOVER_COLOR};
}}

QScrollBar::handle:vertical {{
    background: {SCROLLBAR_HANDLE_COLOR};
    border-radius: {(SCROLLBAR_THICKNESS - 2 * SCROLLBAR_MARGIN) // 2}px;
    margin: {SCROLLBAR_MARGIN}px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background: {SCROLLBAR_HANDLE_HOVER};
}}

QScrollBar::handle:horizontal {{
    background: {SCROLLBAR_HANDLE_COLOR};
    border-radius: {(SCROLLBAR_THICKNESS - 2 * SCROLLBAR_MARGIN) // 2}px;
    margin: {SCROLLBAR_MARGIN}px;
    min-width: 20px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {SCROLLBAR_HANDLE_HOVER};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    background: none;
    border: none;
    width: 0;
    height: 0;
}}

/* === Remove page areas (optional, keeps track clean) === */
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: none;
}}
    
/* Progress bar */
QProgressDialog {{
    background-color: {BASE_COLOR}; color: {TEXT_PRIMARY};
}}

QProgressBar {{
    border-radius: {BUTTON_RADIUS}px;
    text-align: center;
    background-color: {PRESSED_COLOR};
    height: 16px;  /* thicker bar */
    color: {TEXT_PRIMARY};
}}

QProgressBar::chunk {{
    background-color: {OUTLINE_COLOR};
    border-radius: {BUTTON_RADIUS}px;
}}

#scroll {{
    background-color: {BASE_COLOR};
    border-radius: {BORDER_RADIUS}px;
}}
"""
