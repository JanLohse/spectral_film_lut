import sys

from PyQt6.QtCore import Qt, QSize, QEvent, QTimer
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QComboBox, QStackedWidget, QListWidget, QListWidgetItem, QScrollArea,
                             QWidget, QGridLayout, QToolButton, QLabel, QHBoxLayout, QPushButton, QApplication,
                             QSizePolicy, QSplitter)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Film Stock Selector")

        self.filmstocks = [
            {'id': 'Kodak Vision3 500T', 'saturation': 4, 'granularity': 2, 'resolution': 5, 'year': 2007,
             'manufacturer': 'Kodak', },
            {'id': 'Fujifilm Eterna 250D', 'saturation': 3, 'granularity': 3, 'resolution': 4, 'year': 2005,
             'manufacturer': 'Fujifilm', },
            {'id': 'Agfa Vista Plus 200', 'saturation': 5, 'granularity': 4, 'resolution': 3, 'year': 2010,
             'manufacturer': 'Agfa', },
            {'id': 'Kodak Portra 400', 'saturation': 3, 'granularity': 2, 'resolution': 4, 'year': 1998,
             'manufacturer': 'Kodak', },
            {'id': 'Fujifilm Superia X-TRA 400', 'saturation': 4, 'granularity': 3, 'resolution': 3, 'year': 2003,
             'manufacturer': 'Fujifilm', },
            {'id': 'Ilford HP5 Plus 400', 'saturation': None, 'granularity': 2, 'resolution': 5, 'year': 1989,
             'manufacturer': 'Ilford', },
            {'id': 'Cinestill 800T', 'saturation': 4, 'granularity': 3, 'resolution': 4, 'year': 2015,
             'manufacturer': 'Cinestill', },
            {'id': 'Rollei Retro 80S', 'saturation': 2, 'granularity': 1, 'resolution': 5, 'year': 2011,
             'manufacturer': 'Rollei', },
            {'id': 'Lomography Color Negative 100', 'saturation': 5, 'granularity': 4, 'resolution': 2, 'year': 2017,
             'manufacturer': 'Lomography', },
            {'id': 'Adox Color Implosion', 'saturation': 5, 'granularity': 5, 'resolution': 2, 'year': 2012,
             'manufacturer': 'Adox', },
            {'id': 'Kodak Ektar 100', 'saturation': 5, 'granularity': 1, 'resolution': 5, 'year': 2008,
             'manufacturer': 'Kodak', },
            {'id': 'Fujifilm Pro 400H', 'saturation': 3, 'granularity': 2, 'resolution': 4, 'year': 2004,
             'manufacturer': 'Fujifilm', }
        ]

        self.film_combo = QComboBox()
        self.film_combo.addItem("Select film stock")

        self.select_button = QPushButton("Browse...")
        self.select_button.clicked.connect(self.open_selector)

        layout = QHBoxLayout()
        layout.addWidget(self.film_combo)
        layout.addWidget(self.select_button)
        self.setLayout(layout)

    def open_selector(self):
        dialog = FilmStockSelector(self, self.filmstocks)
        if dialog.exec():
            selected_stock = dialog.get_selected_film_stock()
            self.film_combo.clear()
            self.film_combo.addItem(selected_stock['name'])


class FilmStockSelector(QDialog):
    UNKNOWN_LABEL = "Unknown"

    def __init__(self, parent=None, film_stocks=None, id_key='id', sort_keys=None, group_keys=None, list_keys=None, sidebar_keys=None):
        super().__init__(parent)
        self.setWindowTitle("Select Film Stock")
        self.resize(800, 500)

        self.selected_film = None
        self.highlighted_widget = None
        self.highlighted_stock = None

        self.film_stocks = film_stocks

        all_keys = list({key for d in self.film_stocks for key in d})
        self.id_key = all_keys[0] if id_key is None else id_key
        self.sort_keys = sort_keys or all_keys
        self.group_keys = group_keys or all_keys
        self.list_keys = list_keys or all_keys
        self.sidebar_keys = sidebar_keys or all_keys

        self.current_max_cols = None

        self.sort_combo = QComboBox()
        self.group_combo = QComboBox()
        self.sort_combo.addItems(self.sort_keys)
        self.group_combo.addItems(['none'] + self.group_keys)

        self.sort_combo.currentTextChanged.connect(self.update_views)
        self.group_combo.currentTextChanged.connect(self.update_views)

        self.view_toggle = QToolButton()
        self.view_toggle.setText("Toggle View")
        self.view_toggle.setCheckable(True)
        self.view_toggle.toggled.connect(self.toggle_view)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.confirm_selection)

        self.stacked_view = QStackedWidget()

        self.list_scroll = QScrollArea()
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout()
        self.list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.list_container.setLayout(self.list_layout)
        self.list_scroll.setWidget(self.list_container)
        self.list_scroll.setWidgetResizable(True)

        self.grid_scroll = QScrollArea()
        self.grid_container = QWidget()
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(0)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_container.setLayout(self.grid_layout)
        self.grid_scroll.setWidget(self.grid_container)
        self.grid_scroll.setWidgetResizable(True)
        self.grid_scroll.viewport().installEventFilter(self)

        self.populate_list_view()
        self.populate_grid_view()

        self.stacked_view.addWidget(self.list_scroll)
        self.stacked_view.addWidget(self.grid_scroll)

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Sort by:"))
        control_layout.addWidget(self.sort_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Group by:"))
        control_layout.addWidget(self.group_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.view_toggle)
        control_layout.addStretch()

        self.detail_image = QLabel("[Image]")
        self.detail_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_image.setFixedSize(QSize(120, 120))
        self.detail_name = QLabel()
        self.detail_name.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.detail_area = QVBoxLayout()
        self.detail_area.addWidget(self.detail_image)
        self.detail_area.addWidget(self.detail_name)
        self.detail_labels = []

        for _ in range(len(self.sidebar_keys)):
            label = QLabel()
            self.detail_labels.append(label)
            self.detail_area.addWidget(label)

        self.detail_area.addStretch()
        self.detail_area.addWidget(self.ok_button)

        self.detail_widget = QWidget()
        self.detail_widget.setLayout(self.detail_area)
        self.detail_widget.setFixedWidth(200)

        main_split = QSplitter()
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addLayout(control_layout)
        left_layout.addWidget(self.stacked_view)
        left_widget.setLayout(left_layout)

        main_split.addWidget(left_widget)
        main_split.addWidget(self.detail_widget)
        main_split.setStretchFactor(0, 3)
        main_split.setStretchFactor(1, 1)

        layout = QVBoxLayout()
        layout.addWidget(main_split)
        self.setLayout(layout)

    def eventFilter(self, obj, event):
        if obj is self.grid_scroll.viewport() and event.type() == QEvent.Type.Resize:
            col_width = 150
            view_width = self.grid_scroll.viewport().width()
            new_max_cols = max(1, view_width // col_width)
            if new_max_cols != self.current_max_cols:
                self.current_max_cols = new_max_cols
                QTimer.singleShot(0, self.populate_grid_view)
        return super().eventFilter(obj, event)

    def sort_and_group_stocks(self):
        sort_key = self.sort_combo.currentText()
        grouped = self.group_combo.currentText()

        def safe_key(stock, key):
            val = stock.get(key)
            if isinstance(val, str):
                return (val is not None, val.lower())
            return (val is not None, val)

        sorted_stocks = sorted(self.film_stocks, key=lambda x: safe_key(x, sort_key))

        if grouped == 'none':
            return [(None, sorted_stocks)]

        groups = {}
        for stock in sorted_stocks:
            key = stock.get(grouped)
            display_key = key if key is not None else self.UNKNOWN_LABEL
            groups.setdefault(display_key, []).append(stock)

        return sorted(groups.items(), key=lambda x: (x[0] != self.UNKNOWN_LABEL, x[0]))

    def update_sidebar(self, stock):
        self.detail_image.setText("[Image]")
        self.detail_name.setText(stock.get(self.id_key, ''))
        for i, key in enumerate(self.sidebar_keys):
            self.detail_labels[i].setText(f"{key}: {stock.get(key, '')}")

    def highlight_widget(self, widget, stock):
        if self.highlighted_widget:
            self.highlighted_widget.setStyleSheet("")
        self.highlighted_widget = widget
        self.highlighted_stock = stock
        widget.setStyleSheet("background-color: lightblue;")
        self.update_sidebar(stock)

    def confirm_selection(self):
        if self.highlighted_stock:
            self.selected_film = self.highlighted_stock
            self.accept()

    def populate_list_view(self):
        for i in reversed(range(self.list_layout.count())):
            widget = self.list_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for group_key, group in self.sort_and_group_stocks():
            if group_key is not None:
                group_label = QLabel(str(group_key))
                group_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
                self.list_layout.addWidget(group_label)

            for stock in group:
                item_widget = QLabel()
                item_layout = QHBoxLayout()
                item_layout.setContentsMargins(5, 5, 5, 5)

                name_label = QLabel(stock.get(self.id_key, ""))
                name_label.setStyleSheet("font-weight: bold;")
                name_label.setMinimumWidth(150)
                item_layout.addWidget(name_label)

                for key in self.list_keys:
                    val = stock.get(key)
                    val_str = str(val) if val is not None else ""
                    attr_label = QLabel(val_str)
                    attr_label.setMinimumWidth(2)
                    attr_label.setMaximumWidth(70)
                    attr_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                    attr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    item_layout.addWidget(attr_label)

                item_widget.setLayout(item_layout)
                item_widget.setFrameStyle(QLabel.Shape.Box)
                item_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                item_widget.setMinimumHeight(40)
                item_widget.mousePressEvent = lambda e, w=item_widget, s=stock: self.highlight_widget(w, s)
                item_widget.mouseDoubleClickEvent = lambda e, s=stock: self.confirm_selection()

                self.list_layout.addWidget(item_widget)

    def populate_grid_view(self):
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        col_width = 150
        view_width = self.grid_scroll.viewport().width()
        max_cols = max(1, view_width // col_width)
        self.current_max_cols = max_cols

        row = 0

        for group_key, group in self.sort_and_group_stocks():
            if group_key is not None:
                title = QLabel(str(group_key))
                title.setStyleSheet("font-weight: bold; margin: 5px 0;")
                title.setAlignment(Qt.AlignmentFlag.AlignLeft)
                self.grid_layout.addWidget(title, row, 0, 1, max_cols)
                row += 1

            col = 0
            for stock in group:
                label = QLabel("[Image]\n" + stock[self.id_key])
                label.setFrameShape(QLabel.Shape.Box)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setFixedSize(QSize(col_width, 100))
                label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                label.mousePressEvent = lambda e, w=label, s=stock: self.highlight_widget(w, s)
                label.mouseDoubleClickEvent = lambda e, s=stock: self.confirm_selection()

                self.grid_layout.addWidget(label, row, col)
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

            if col != 0:
                row += 1

        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.grid_container.adjustSize()

    def update_views(self):
        self.populate_list_view()
        self.populate_grid_view()

    def toggle_view(self, checked):
        self.stacked_view.setCurrentIndex(1 if checked else 0)

    def get_selected_film_stock(self):
        return self.selected_film


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == '__main__':
    main()
