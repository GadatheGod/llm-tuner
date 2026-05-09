from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QGroupBox, QTextEdit, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from llmtuner.core.model_db import search_models, get_use_case_categories, get_model_details
from llmtuner.core.recommender import recommend_models
from llmtuner.core.system_info import SystemInfo


class ModelBrowserTab(QWidget):
    model_selected = Signal(dict)

    def __init__(self):
        super().__init__()
        self.system_info = None
        self._build_ui()
        QTimer.singleShot(200, self._load_categories)

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        search_bar = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search models (e.g., llama, mistral, qwen, phi)...")
        self.search_input.returnPressed.connect(self._search)
        search_bar.addWidget(self.search_input, 1)

        self.category_combo = QComboBox()
        self.category_combo.setMinimumWidth(180)
        self.category_combo.currentIndexChanged.connect(self._on_category_change)
        search_bar.addWidget(self.category_combo)

        self.search_btn = QPushButton("Search HuggingFace")
        self.search_btn.clicked.connect(self._search)
        search_bar.addWidget(self.search_btn)

        self.recommend_btn = QPushButton("Auto-Recommend")
        self.recommend_btn.clicked.connect(self._auto_recommend)
        search_bar.addWidget(self.recommend_btn)

        layout.addLayout(search_bar)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setStyleSheet("QProgressBar::chunk { background: #e67e22; }")
        self.progress.hide()
        layout.addWidget(self.progress)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Model", "Family", "Params", "Context", "Downloads", "Categories", "Quantizations", "Action"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMaximumHeight(120)
        self.info_box.setPlaceholderText("Select a model to see details...")
        layout.addWidget(self.info_box, 0)

        self.setLayout(layout)
        self.models_cache = []

    def update_system_info(self, system_info: SystemInfo):
        self.system_info = system_info

    def _load_categories(self):
        categories = get_use_case_categories()
        self.category_combo.addItem("All Categories", "")
        for cat in categories:
            self.category_combo.addItem(cat["name"], cat["id"])

    def _on_category_change(self, index):
        self._search()

    def _search(self):
        query = self.search_input.text().strip()
        cat_id = self.category_combo.currentData()
        self.progress.show()
        self.progress.setValue(0)

        QTimer.singleShot(10, lambda: self._do_search(query, cat_id))

    def _do_search(self, query: str, cat_id: str):
        categories = [cat_id] if cat_id else None
        try:
            max_params = "70B"
            if self.system_info:
                gpu = self.system_info.gpu[0] if self.system_info.gpu else None
                if gpu and gpu.vram_total_mb > 0:
                    from llmtuner.core.recommender import _determine_max_params
                    max_params = _determine_max_params(self.system_info)

            self.models_cache = search_models(
                query=query,
                categories=categories,
                max_params=max_params,
                limit=30
            )
        except Exception as e:
            self.models_cache = []
            self.info_box.setText(f"Search error: {e}. Using local models.")
            self.models_cache = search_models(query=query, categories=categories, use_cache=False)

        self._populate_table(self.models_cache)
        self.progress.hide()

    def _populate_table(self, models: list):
        self.table.setRowCount(len(models))
        for row, model in enumerate(models):
            self.table.setItem(row, 0, QTableWidgetItem(model.get("name", "Unknown")))
            self.table.setItem(row, 1, QTableWidgetItem(model.get("family", "")))
            self.table.setItem(row, 2, QTableWidgetItem(model.get("params", "")))
            ctx = model.get("context_default", "")
            self.table.setItem(row, 3, QTableWidgetItem(f"{ctx}" if ctx else "4096"))
            dl = model.get("downloads", 0)
            self.table.setItem(row, 4, QTableWidgetItem(f"{dl:,}" if dl else "local"))

            cats = model.get("categories", model.get("tags", []))
            self.table.setItem(row, 5, QTableWidgetItem(", ".join(str(c) for c in cats[:3])))
            qants = model.get("quantizations", [])
            self.table.setItem(row, 6, QTableWidgetItem(", ".join(str(q) for q in qants[:3])))

            btn = QPushButton("Select")
            btn.setFixedWidth(70)
            btn.clicked.connect(lambda checked, m=model: self._select_model(m))
            self.table.setCellWidget(row, 7, btn)

    def _auto_recommend(self):
        if not self.system_info:
            self.info_box.setText("Please scan system first (System tab)")
            return
        self.progress.show()
        self.info_box.setText("Finding best models for your hardware...")
        QTimer.singleShot(10, self._do_recommend)

    def _do_recommend(self):
        cat_id = self.category_combo.currentData() or "chat"
        recommendations = recommend_models(self.system_info, use_case=cat_id, top=5)
        models = [r["model"] for r in recommendations]
        self.models_cache = models
        self._populate_table(models)
        self.progress.hide()
        if recommendations:
            top = recommendations[0]
            self.info_box.setText(
                f"Top pick: {top['model'].get('name', 'Unknown')} "
                f"(Score: {top['score']}) - {top['reason']}"
            )

    def _select_model(self, model: dict):
        details = get_model_details(model.get("id", ""))
        if details:
            model.update(details)

        self.selected_model = model
        self.model_selected.emit(model)

        lines = [f"Selected: {model.get('name', 'Unknown')}"]
        if model.get("family"):
            lines.append(f"Family: {model['family']}")
        if model.get("params"):
            lines.append(f"Parameters: {model['params']}")
        if model.get("context_default"):
            lines.append(f"Context: {model['context_default']} tokens")
        if model.get("quantizations"):
            lines.append(f"Quantizations: {', '.join(str(q) for q in model['quantizations'][:4])}")
        if model.get("hf_url"):
            lines.append(f"HF: {model['hf_url']}")
        self.info_box.setText("\n".join(lines) + "\nGo to Configure tab to see recommended settings.")

    def _on_double_click(self, row, col):
        if row < len(self.models_cache):
            self._select_model(self.models_cache[row])
