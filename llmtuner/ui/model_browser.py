from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QGroupBox, QTextEdit, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QTimer, Signal
from typing import Optional
from llmtuner.core.model_db import search_models, get_use_case_categories, get_model_details
from llmtuner.core.recommender import recommend_models, estimate_tok_per_sec
from llmtuner.core.system_info import SystemInfo


class ModelBrowserTab(QWidget):
    model_selected = Signal(dict)

    def __init__(self):
        super().__init__()
        self.system_info = None
        self.selected_engine = "llama.cpp"
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
        self.category_combo.setMinimumWidth(160)
        self.category_combo.currentIndexChanged.connect(self._on_category_change)
        search_bar.addWidget(self.category_combo)

        self.sort_combo = QComboBox()
        self.sort_combo.addItems(["Most Downloaded", "Trending", "Best Performance", "Smallest"])
        self.sort_combo.setMinimumWidth(130)
        self.sort_combo.currentIndexChanged.connect(self._search)
        search_bar.addWidget(self.sort_combo)

        self.search_btn = QPushButton("Search HF")
        self.search_btn.clicked.connect(self._search)
        search_bar.addWidget(self.search_btn)

        self.recommend_btn = QPushButton("Auto-Recommend")
        self.recommend_btn.clicked.connect(self._auto_recommend)
        search_bar.addWidget(self.recommend_btn)

        layout.addLayout(search_bar)

        engine_bar = QHBoxLayout()
        engine_bar.addWidget(QLabel("Engine:"))
        self.engine_group = QButtonGroup()
        self.engine_llama = QRadioButton("llama.cpp (GGUF)")
        self.engine_llama.setChecked(True)
        self.engine_llama.toggled.connect(self._on_engine_change)
        self.engine_ollama = QRadioButton("Ollama")
        self.engine_ollama.toggled.connect(self._on_engine_change)
        self.engine_group.addButton(self.engine_llama)
        self.engine_group.addButton(self.engine_ollama)
        engine_bar.addWidget(self.engine_llama)
        engine_bar.addWidget(self.engine_ollama)
        engine_bar.addStretch()
        layout.addLayout(engine_bar)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setStyleSheet("QProgressBar::chunk { background: #e67e22; }")
        self.progress.hide()
        layout.addWidget(self.progress)

        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            "Model", "Family", "Params", "Context", "Downloads",
            "Q4 Size", "Q8 Size", "Expected tok/s", "Engine", "Quantizations", "Action"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 11):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table, 1)

        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMaximumHeight(120)
        self.info_box.setPlaceholderText("Select a model to see details...")
        layout.addWidget(self.info_box, 0)

        self.setLayout(layout)
        self.models_cache = []

    def update_system_info(self, system_info: SystemInfo):
        self.system_info = system_info
        if not self.models_cache:
            QTimer.singleShot(600, self._auto_search_after_scan)

    def _auto_search_after_scan(self):
        if self.system_info and not self.models_cache:
            self.info_box.setText("Auto-searching models for your hardware...")
            self._search()

    def _load_categories(self):
        categories = get_use_case_categories()
        self.category_combo.addItem("All Categories", "")
        for cat in categories:
            self.category_combo.addItem(cat["name"], cat["id"])

    def _on_category_change(self, index):
        self._search()

    def _on_engine_change(self):
        self.selected_engine = "llama.cpp" if self.engine_llama.isChecked() else "ollama"

    def _search(self):
        query = self.search_input.text().strip()
        cat_id = self.category_combo.currentData() or None
        sort_mode = self.sort_combo.currentIndex()
        self.progress.show()
        self.progress.setValue(0)
        self._do_search(query, cat_id, sort_mode)

    def _do_search(self, query: str, cat_id: Optional[str], sort_mode: int):
        categories = [cat_id] if cat_id else None
        try:
            max_params = "70B"
            if self.system_info:
                from llmtuner.core.recommender import _determine_max_params
                max_params = _determine_max_params(self.system_info)

            self.models_cache = search_models(
                query=query,
                categories=categories,
                max_params=max_params,
                limit=30
            )

            if sort_mode == 1:
                self.models_cache.sort(key=lambda m: m.get("likes", 0), reverse=True)
            elif sort_mode == 2:
                self.models_cache.sort(key=lambda m: m.get("params_raw", 0), reverse=True)
            elif sort_mode == 3:
                self.models_cache.sort(key=lambda m: m.get("params_raw", 0))

            if self.models_cache:
                self.info_box.setText(f"Found {len(self.models_cache)} models. Double-click to select.")
            else:
                self.info_box.setText("No models found. Try a different search or category.")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.models_cache = []
            self.info_box.setText(f"Search error: {e}\n{tb}")
            self.progress.hide()
            return

        self._populate_table(self.models_cache)
        self.progress.hide()

    def _format_size(self, size_bytes: int) -> str:
        if not size_bytes:
            return "N/A"
        gb = size_bytes / 1e9
        if gb >= 1:
            return f"{gb:.1f} GB"
        return f"{size_bytes / 1e6:.0f} MB"

    def _populate_table(self, models: list):
        self.table.setRowCount(len(models))
        for row, model in enumerate(models):
            self.table.setItem(row, 0, QTableWidgetItem(model.get("name", "Unknown")))
            self.table.setItem(row, 1, QTableWidgetItem(model.get("family", "")))
            self.table.setItem(row, 2, QTableWidgetItem(model.get("params", "")))
            ctx = model.get("context_default", "")
            self.table.setItem(row, 3, QTableWidgetItem(str(ctx) if ctx else "4096"))

            dl = model.get("downloads", 0)
            self.table.setItem(row, 4, QTableWidgetItem(f"{dl:,}" if dl else "local"))

            size_bytes = model.get("size_bytes", {})
            q4_size = size_bytes.get("q4_k_m", 0)
            q8_size = size_bytes.get("q8_0", 0)
            self.table.setItem(row, 5, QTableWidgetItem(self._format_size(q4_size)))
            self.table.setItem(row, 6, QTableWidgetItem(self._format_size(q8_size)))

            if self.system_info:
                params = model.get("params", "8B")
                tok_s = estimate_tok_per_sec(self.system_info, params, "q4_k_m")
            else:
                tok_s = "Scan first"
            self.table.setItem(row, 7, QTableWidgetItem(tok_s))

            compat = "llama.cpp OK" if self.selected_engine == "llama.cpp" else "Ollama OK"
            compat_item = QTableWidgetItem(compat)
            compat_item.setForeground(Qt.darkGreen if "OK" in compat else Qt.darkGray)
            self.table.setItem(row, 8, compat_item)

            cats = model.get("categories", model.get("tags", []))
            self.table.setItem(row, 9, QTableWidgetItem(", ".join(str(c) for c in cats[:3])))

            btn = QPushButton("Select")
            btn.setFixedWidth(70)
            btn.clicked.connect(lambda checked, m=model: self._select_model(m))
            self.table.setCellWidget(row, 10, btn)

    def _auto_recommend(self):
        if not self.system_info:
            self.info_box.setText("Please scan system first (System tab)")
            return
        self.progress.show()
        self.info_box.setText("Finding best models for your hardware...")
        QTimer.singleShot(10, self._do_recommend)

    def _do_recommend(self):
        if not self.system_info:
            self.info_box.setText("Please scan system first (System tab)")
            self.progress.hide()
            return
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
        model["engine"] = self.selected_engine
        self.selected_model = model
        self.model_selected.emit(model)

        lines = [f"Selected: {model.get('name', 'Unknown')}"]
        if model.get("family"):
            lines.append(f"Family: {model['family']}")
        if model.get("params"):
            lines.append(f"Parameters: {model['params']}")
        size_bytes = model.get("size_bytes", {})
        if size_bytes:
            lines.append(f"Q4 Size: {self._format_size(size_bytes.get('q4_k_m', 0))}")
            lines.append(f"Q8 Size: {self._format_size(size_bytes.get('q8_0', 0))}")
        if model.get("context_default"):
            lines.append(f"Context: {model['context_default']} tokens")
        if model.get("quantizations"):
            lines.append(f"Quantizations: {', '.join(str(q) for q in model['quantizations'][:4])}")
        if self.system_info and model.get("params"):
            tok = estimate_tok_per_sec(self.system_info, model["params"], "q4_k_m")
            lines.append(f"Expected tok/s: {tok}")
        lines.append(f"Engine: {self.selected_engine}")
        if model.get("hf_url"):
            lines.append(f"HF: {model['hf_url']}")
        self.info_box.setText("\n".join(lines) + "\nGo to Configure tab to see recommended settings.")

    def _on_double_click(self, row, col):
        if row < len(self.models_cache):
            self._select_model(self.models_cache[row])
