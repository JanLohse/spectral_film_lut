from PyQt6.QtCore import Qt, QSize, QEvent, QTimer
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QComboBox, QStackedWidget, QScrollArea, QWidget, QGridLayout,
                             QToolButton, QLabel, QHBoxLayout, QPushButton, QSizePolicy, QSplitter, QLineEdit)


class FilmStockSelector(QWidget):
    def __init__(self, film_stocks, **kwargs):
        super().__init__()
        self.film_stocks = film_stocks
        self.kwargs = kwargs

        self.film_combo = QComboBox()
        self.film_combo.addItems(self.film_stocks.keys())
        self.select_button = QPushButton("🔍")
        self.select_button.setFixedWidth(25)
        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.film_combo)
        layout.addWidget(self.select_button)
        layout.setContentsMargins(0, 0, 0, 0)
        self.select_button.clicked.connect(self.open_selector)

        self.setCurrentText = self.film_combo.setCurrentText
        self.currentTextChanged = self.film_combo.currentTextChanged
        self.currentText = self.film_combo.currentText

    def open_selector(self):
        current_stock = self.film_combo.currentText()
        dialog = FilmStockSelectorWindow(self, self.film_stocks, highlighted_stock=current_stock, **self.kwargs)
        if dialog.exec():
            selected_stock = dialog.get_selected_film_stock()
            self.film_combo.setCurrentText(selected_stock)


class FilmStockSelectorWindow(QDialog):
    UNKNOWN_LABEL = "Unknown"

    def __init__(self, parent=None, film_stocks=None, sort_keys=None, group_keys=None, list_keys=None,
                 sidebar_keys=None, default_sort=None, default_group=None, highlighted_stock=None, ):
        super().__init__(parent)
        self.setWindowTitle("Select Film Stock")
        self.resize(800, 500)

        self.selected_film = None
        self.highlighted_stock = None

        self.film_stocks = film_stocks
        self.film_tags = {x: " ".join((str(z) for z in y.values())) + " " + x for x, y in self.film_stocks.items()}

        if type(film_stocks) is dict:
            all_keys = list({key for d in self.film_stocks.values() for key in d})
        else:
            all_keys = []
        self.sort_keys = sort_keys or ['Name'] + all_keys
        self.group_keys = group_keys or all_keys
        self.list_keys = list_keys or all_keys
        self.sidebar_keys = sidebar_keys or all_keys

        self.grid_widgets = {}
        self.list_widgets = {}

        self.current_max_cols = None

        self.sort_combo = QComboBox()
        self.group_combo = QComboBox()
        self.sort_combo.addItems(self.sort_keys)
        self.group_combo.addItems(['none'] + self.group_keys)
        if default_group is not None and default_group in self.group_keys:
            self.group_combo.setCurrentText(default_group)
        if default_sort is not None and default_sort in self.sort_keys:
            self.sort_combo.setCurrentText(default_sort)
        self.search_bar = QLineEdit()

        self.sort_combo.currentTextChanged.connect(self.update_views)
        self.group_combo.currentTextChanged.connect(self.update_views)
        self.search_bar.textChanged.connect(self.update_views)

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
        control_layout.addWidget(QLabel("Sort:"))
        control_layout.addWidget(self.sort_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Group:"))
        control_layout.addWidget(self.group_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(QLabel("Filter:"))
        control_layout.addWidget(self.search_bar)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.view_toggle)
        # control_layout.addStretch()

        self.detail_image = QLabel("[Image]")
        self.detail_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail_image.setFixedSize(QSize(120, 120))
        self.detail_name = QLabel()
        self.detail_name.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.detail_area = QVBoxLayout()
        self.detail_area.addWidget(self.detail_image)
        self.detail_area.addWidget(self.detail_name)
        self.detail_label = QLabel()
        self.detail_area.addWidget(self.detail_label)

        self.detail_area.addStretch()
        self.detail_area.addWidget(self.ok_button)

        self.detail_widget = QWidget()
        self.detail_widget.setLayout(self.detail_area)
        self.detail_widget.setFixedWidth(200)

        main_split = QSplitter()
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.stacked_view)
        left_widget.setLayout(left_layout)

        main_split.addWidget(left_widget)
        main_split.addWidget(self.detail_widget)
        main_split.setStretchFactor(0, 3)
        main_split.setStretchFactor(1, 1)

        layout = QVBoxLayout()
        layout.addLayout(control_layout)
        layout.addWidget(main_split)
        self.setLayout(layout)

        if highlighted_stock is not None:
            self.highlight_widget(highlighted_stock)
            self.ensure_highlighted_visible()

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
        group_key = self.group_combo.currentText()
        filter = self.search_bar.text()

        def safe_key(stock, key):
            val = stock.get(key)
            if isinstance(val, str):
                return (val is not None, val.lower())
            return (val is not None, val)

        def filter_search(tags, filter):
            return all([x in tags for x in filter.split(' ')])

        filtered_stocks = {x: y for x, y in self.film_stocks.items() if filter_search(self.film_tags[x], filter)}

        if sort_key.lower() in ['', 'name', 'id', 'none'] or sort_key is None:
            sorted_stocks = sorted(filtered_stocks)
        else:
            sorted_stocks = sorted(filtered_stocks, key=lambda x: safe_key(filtered_stocks[x], sort_key))

        if group_key == 'none' or group_key is None:
            return [(None, sorted_stocks)]

        groups = {}
        for stock in sorted_stocks:
            key = self.film_stocks[stock].get(group_key)
            display_key = key if key is not None else self.UNKNOWN_LABEL
            groups.setdefault(display_key, []).append(stock)

        return sorted(groups.items(), key=lambda x: (x[0] != self.UNKNOWN_LABEL, x[0]))

    def update_sidebar(self, stock):
        self.detail_image.setText("[Image]")
        self.detail_name.setText(stock)
        detail_text = ""
        for key in self.sidebar_keys:
            if self.film_stocks[stock].get(key) is not None:
                detail_text += f"{key}: {self.film_stocks[stock].get(key, '')}\n"
        self.detail_label.setText(detail_text)

    def highlight_widget(self, stock=None):
        if stock is not None:
            if stock in self.grid_widgets and stock in self.list_widgets:
                grid_widget = self.grid_widgets[stock]
                list_widget = self.list_widgets[stock]
            else:
                return
            if self.highlighted_stock is not None:
                if self.highlighted_stock in self.grid_widgets:
                    self.grid_widgets[self.highlighted_stock].setStyleSheet("")
                if self.highlighted_stock in self.list_widgets:
                    self.list_widgets[self.highlighted_stock].setStyleSheet("")
            self.highlighted_stock = stock
        elif self.highlighted_stock is not None and self.highlighted_stock in self.grid_widgets and self.highlighted_stock in self.list_widgets:
            grid_widget = self.grid_widgets[self.highlighted_stock]
            list_widget = self.list_widgets[self.highlighted_stock]
        else:
            return
        grid_widget.setStyleSheet("background-color: lightblue;")
        list_widget.setStyleSheet("background-color: lightblue;")
        if stock is not None:
            self.update_sidebar(stock)

    def confirm_selection(self):
        if self.highlighted_stock:
            self.selected_film = self.highlighted_stock
            self.accept()

    def populate_list_view(self):
        self.list_widgets = {}

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

                name_label = QLabel(stock)
                name_label.setStyleSheet("font-weight: bold;")
                name_label.setMinimumWidth(150)
                item_layout.addWidget(name_label)

                for key in self.list_keys:
                    val = self.film_stocks[stock].get(key)
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
                item_widget.mousePressEvent = lambda e, s=stock: self.highlight_widget(s)
                item_widget.mouseDoubleClickEvent = lambda e, s=stock: self.confirm_selection()

                self.list_layout.addWidget(item_widget)

                self.list_widgets[stock] = item_widget

    def populate_grid_view(self):
        self.grid_widgets = {}

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
                label = QLabel("[Image]\n" + stock)
                label.setFrameShape(QLabel.Shape.Box)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setFixedSize(QSize(col_width, 100))
                label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                label.mousePressEvent = lambda e, s=stock: self.highlight_widget(s)
                label.mouseDoubleClickEvent = lambda e, s=stock: self.confirm_selection()

                self.grid_layout.addWidget(label, row, col)
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

                self.grid_widgets[stock] = label

            if col != 0:
                row += 1

        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.grid_container.adjustSize()

        self.highlight_widget()
        self.ensure_highlighted_visible()

    def update_views(self):
        self.populate_list_view()
        self.populate_grid_view()

    def toggle_view(self, checked):
        self.stacked_view.setCurrentIndex(1 if checked else 0)
        self.ensure_highlighted_visible()

    def get_selected_film_stock(self):
        return self.selected_film

    def ensure_highlighted_visible(self):
        if self.highlighted_stock is None:
            return

        if self.stacked_view.currentIndex() == 0:
            scroll_area = self.list_scroll
            widgets = self.list_widgets
        else:
            scroll_area = self.grid_scroll
            widgets = self.grid_widgets

        if self.highlighted_stock in widgets:
            widget = widgets[self.highlighted_stock]
        else:
            return

        QTimer.singleShot(0, lambda: scroll_area.ensureWidgetVisible(widget))
