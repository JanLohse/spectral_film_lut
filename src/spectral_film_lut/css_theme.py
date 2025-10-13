import os

BASE_COLOR = "#191a1c"
BACKGROUND_COLOR = "#26282b"
ACCENT_COLOR = "#426dc3"
BORDER_RADIUS = 12
BUTTON_RADIUS = 6
HOVER_COLOR = "#46484b"
OUTLINE_COLOR = "#46484b"
PRESSED_COLOR = "#393b3d"
MENU_COLOR = "#202225"
TEXT_PRIMARY = "white"
SCROLLBAR_HOVER_COLOR = "#232426"
SCROLLBAR_THICKNESS = 11
SCROLLBAR_MARGIN = 2
SCROLLBAR_HANDLE_COLOR = "#3d3e3f"
SCROLLBAR_HANDLE_HOVER = "#565657"
HIGHLIGHT_COLOR = "#b0b5bd"

base_dir = os.path.dirname(__file__)
icon_path = os.path.join(base_dir, "resources").replace("\\", "/")

THEME = f"""
QMainWindow > QWidget, QDialog > QWidget {{
    background-color: {BACKGROUND_COLOR};
    color: {TEXT_PRIMARY};
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
    image: url("{icon_path}/down_arrow.svg");
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

/* LineEdit */
QLineEdit {{
    background-color: {PRESSED_COLOR};
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
    image: url("{icon_path}/checkbox_unchecked.svg");
}}

QCheckBox::indicator:checked {{
    image: url("{icon_path}/checkbox_checked.svg");
}}

QCheckBox::indicator:indeterminate {{
    image: url("{icon_path}/checkbox_halfway.svg");
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
"""
