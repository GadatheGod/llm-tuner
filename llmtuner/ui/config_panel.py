from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QComboBox, QLineEdit, QCheckBox,
    QTextEdit, QTabWidget, QFormLayout
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
from llmtuner.core.recommender import recommend_config
from llmtuner.core.system_info import SystemInfo


class ConfigPanelTab(QWidget):
    config_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.system_info = None
        self.selected_model = None
        self.current_config = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.addWidget(QLabel("Configuration Panel"))
        header.addStretch()

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["Optimum (Speed)", "Balanced", "Max Performance"])
        self.profile_combo.currentIndexChanged.connect(self._regenerate)
        header.addWidget(QLabel("Profile:"))
        header.addWidget(self.profile_combo)

        self.use_case_combo = QComboBox()
        use_cases = ["chat", "code", "creative_writing", "rag", "translation",
                     "math", "roleplay", "summarization", "agent", "vision", "fine_tune"]
        use_case_names = ["General Chat", "Code Generation", "Creative Writing", "RAG/Knowledge",
                          "Translation", "Math & Reasoning", "Roleplay", "Summarization",
                          "AI Agent", "Vision/Multimodal", "Fine-Tuning"]
        for uid, name in zip(use_cases, use_case_names):
            self.use_case_combo.addItem(name, uid)
        self.use_case_combo.currentIndexChanged.connect(self._regenerate)
        header.addWidget(QLabel("Use Case:"))
        header.addWidget(self.use_case_combo)

        layout.addLayout(header)

        self.tabs = QTabWidget()

        self.params_tab = self._build_params_tab()
        self.tabs.addTab(self.params_tab, "Parameters")

        self.summary_tab = QTextEdit()
        self.summary_tab.setReadOnly(True)
        self.tabs.addTab(self.summary_tab, "Summary")

        layout.addWidget(self.tabs)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        self.apply_btn = QPushButton("Apply Configuration")
        self.apply_btn.clicked.connect(self._apply_config)
        self.apply_btn.setEnabled(False)
        btn_bar.addWidget(self.apply_btn)
        layout.addLayout(btn_bar)

        self.setLayout(layout)

    def _build_params_tab(self):
        widget = QWidget()
        layout = QFormLayout()
        layout.setSpacing(8)

        self.n_gpu_layers_edit = QLineEdit("0")
        layout.addRow("GPU Layers (-ngl):", self.n_gpu_layers_edit)

        self.n_ctx_edit = QLineEdit("4096")
        layout.addRow("Context Size (-c):", self.n_ctx_edit)

        self.n_batch_edit = QLineEdit("2048")
        layout.addRow("Batch Size (-b):", self.n_batch_edit)

        self.n_threads_edit = QLineEdit("8")
        layout.addRow("Threads (-t):", self.n_threads_edit)

        self.n_threads_batch_edit = QLineEdit("8")
        layout.addRow("Batch Threads:", self.n_threads_batch_edit)

        self.n_predict_edit = QLineEdit("256")
        layout.addRow("Predict Tokens (-n):", self.n_predict_edit)

        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["q4_k_m", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "f16"])
        layout.addRow("Quantization:", self.quant_combo)

        self.flash_attention_check = QCheckBox("Flash Attention")
        self.flash_attention_check.setChecked(True)
        layout.addRow("", self.flash_attention_check)

        self.mmap_check = QCheckBox("Memory Map (mmap)")
        self.mmap_check.setChecked(True)
        layout.addRow("", self.mmap_check)

        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to .gguf model file...")
        layout.addRow("Model Path:", self.model_path_edit)

        for editor in [self.n_gpu_layers_edit, self.n_ctx_edit, self.n_batch_edit,
                        self.n_threads_edit, self.n_threads_batch_edit, self.n_predict_edit]:
            editor.editingFinished.connect(self._on_edit_changed)
        self.quant_combo.currentIndexChanged.connect(self._on_edit_changed)
        self.flash_attention_check.toggled.connect(self._on_edit_changed)
        self.mmap_check.toggled.connect(self._on_edit_changed)

        widget.setLayout(layout)
        return widget

    def update_recommendation(self, system_info: SystemInfo, model_data: dict):
        self.system_info = system_info
        self.selected_model = model_data
        self._regenerate()

    def set_model_path(self, path: str):
        self.model_path_edit.setText(path)

    def _regenerate(self):
        if not self.system_info:
            self._show_placeholder()
            return

        profile = self.profile_combo.currentIndex()
        profile_names = ["optimum", "balanced", "max_performance"]
        profile_name = profile_names[profile]

        use_case = self.use_case_combo.currentData() or "chat"
        model_params = self.selected_model.get("params", "8B") if self.selected_model else "8B"

        try:
            config = recommend_config(
                system=self.system_info,
                model_params=model_params,
                use_case=use_case,
                profile=profile_name
            )
            self.current_config = config
            self._populate_fields(config)
            self._update_summary(config)
            self.apply_btn.setEnabled(True)
            self.config_changed.emit(config)
        except Exception as e:
            self.summary_tab.setText(f"Error generating config: {e}")

    def _populate_fields(self, config: dict):
        self.n_gpu_layers_edit.setText(str(config.get("n_gpu_layers", 0)))
        self.n_ctx_edit.setText(str(config.get("n_ctx", 4096)))
        self.n_batch_edit.setText(str(config.get("n_batch", 2048)))
        self.n_threads_edit.setText(str(config.get("n_threads", 8)))
        self.n_threads_batch_edit.setText(str(config.get("n_threads_batch", 8)))
        self.n_predict_edit.setText(str(config.get("n_predict", 256)))

        quant_idx = self.quant_combo.findText(config.get("quantization", "q4_k_m"))
        if quant_idx >= 0:
            self.quant_combo.setCurrentIndex(quant_idx)

        self.flash_attention_check.setChecked(config.get("flash_attention", True))
        self.mmap_check.setChecked(config.get("mmap", True))

    def _update_summary(self, config: dict):
        lines = []
        lines.append("=== LLM-Tuner Configuration ===")
        lines.append("")
        lines.append(f"Profile: {config.get('profile', 'N/A')}")
        lines.append(f"Model Params: {config.get('model_params', 'N/A')}")
        lines.append(f"Use Case: {config.get('use_case', 'N/A')}")
        lines.append(f"Quantization: {config.get('quantization', 'N/A')}")
        lines.append("")
        lines.append("=== Parameters ===")
        lines.append(f"GPU Layers:      {config.get('n_gpu_layers', 0)}")
        lines.append(f"Context Size:    {config.get('n_ctx', 4096)}")
        lines.append(f"Batch Size:      {config.get('n_batch', 2048)}")
        lines.append(f"Threads:         {config.get('n_threads', 8)}")
        lines.append(f"Batch Threads:   {config.get('n_threads_batch', 8)}")
        lines.append(f"Predict Tokens:  {config.get('n_predict', 256)}")
        lines.append(f"Flash Attention: {config.get('flash_attention', True)}")
        lines.append(f"Memory Map:      {config.get('mmap', True)}")
        lines.append("")
        lines.append("=== Hardware ===")
        if self.system_info:
            lines.append(f"CPU: {self.system_info.cpu.model}")
            lines.append(f"Cores: {self.system_info.cpu.physical_cores}P / {self.system_info.cpu.logical_cores}L")
            if self.system_info.gpu:
                g = self.system_info.gpu[0]
                lines.append(f"GPU: {g.model} ({g.vram_total_mb} MB VRAM)")
            else:
                lines.append("GPU: None (CPU inference)")
            lines.append(f"RAM: {self.system_info.ram_total_gb} GB")
        self.summary_tab.setPlainText("\n".join(lines))

    def _show_placeholder(self):
        self.summary_tab.setText(
            "No system info available.\n"
            "Please scan your system first from the System tab, then select a model."
        )

    def _on_edit_changed(self):
        if not self.current_config:
            return
        self.current_config["n_gpu_layers"] = int(self.n_gpu_layers_edit.text() or "0")
        self.current_config["n_ctx"] = int(self.n_ctx_edit.text() or "4096")
        self.current_config["n_batch"] = int(self.n_batch_edit.text() or "2048")
        self.current_config["n_threads"] = int(self.n_threads_edit.text() or "8")
        self.current_config["n_threads_batch"] = int(self.n_threads_batch_edit.text() or "8")
        self.current_config["n_predict"] = int(self.n_predict_edit.text() or "256")
        self.current_config["quantization"] = self.quant_combo.currentText()
        self.current_config["flash_attention"] = self.flash_attention_check.isChecked()
        self.current_config["mmap"] = self.mmap_check.isChecked()
        if self.model_path_edit.text():
            self.current_config["model_path"] = self.model_path_edit.text()
        self._update_summary(self.current_config)
        self.config_changed.emit(self.current_config)

    def _apply_config(self):
        self.apply_btn.setText("Applied!")
        self.apply_btn.setEnabled(False)
        def re_enable():
            self.apply_btn.setText("Apply Configuration")
            self.apply_btn.setEnabled(True)
        QTimer.singleShot(1500, re_enable)
