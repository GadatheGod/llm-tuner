from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QComboBox, QTextEdit, QLineEdit,
    QFileDialog, QFrame
)
from PySide6.QtCore import Qt
from llmtuner.core.config_export import (
    export_ollama_modelfile, export_llama_cpp_config, export_json_config, launch_config
)


class ExportLaunchTab(QWidget):
    def __init__(self):
        super().__init__()
        self.current_config = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(12)

        self.config_label = QLabel("No configuration loaded.")
        self.config_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.config_label)

        model_section = QGroupBox("Model Path")
        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to .gguf model or Ollama model name...")
        model_layout.addWidget(self.model_path_edit)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_model)
        model_layout.addWidget(self.browse_btn)
        model_section.setLayout(model_layout)
        layout.addWidget(model_section)

        export_section = QGroupBox("Export Configuration")
        export_layout = QVBoxLayout()
        export_layout.setSpacing(8)

        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Output directory (default: current directory)")
        export_layout.addWidget(self.output_path_edit)

        format_bar = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems([
            "Ollama Modelfile",
            "llama.cpp .bat/.sh",
            "JSON Config",
        ])
        format_bar.addWidget(QLabel("Format:"))
        format_bar.addWidget(self.format_combo)
        format_bar.addStretch()
        export_layout.addLayout(format_bar)

        self.export_btn = QPushButton("Export Config File")
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)

        export_section.setLayout(export_layout)
        layout.addWidget(export_section)

        launch_section = QGroupBox("Auto-Launch")
        launch_layout = QVBoxLayout()
        launch_layout.setSpacing(8)

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["llama.cpp", "Ollama"])
        launch_layout.addWidget(QLabel("Engine:"))
        launch_layout.addWidget(self.engine_combo)

        launch_btn_bar = QHBoxLayout()
        self.launch_btn = QPushButton("Launch with Config")
        self.launch_btn.clicked.connect(self._launch)
        self.launch_btn.setEnabled(False)
        self.launch_btn.setStyleSheet("""
            QPushButton {
                background: #27ae60;
                padding: 10px 24px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background: #2ecc71; }
        """)
        launch_btn_bar.addWidget(self.launch_btn)
        launch_btn_bar.addStretch()
        launch_layout.addLayout(launch_btn_bar)

        launch_section.setLayout(launch_layout)
        layout.addWidget(launch_section)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        self.status_text.setPlaceholderText("Export/launch status will appear here...")
        layout.addWidget(self.status_text)

        self.setLayout(layout)

    def update_config(self, config: dict):
        self.current_config = config
        self.export_btn.setEnabled(True)
        self.launch_btn.setEnabled(True)

        profile = config.get("profile", "balanced")
        params = config.get("model_params", "N/A")
        ngl = config.get("n_gpu_layers", 0)
        ctx = config.get("n_ctx", 4096)
        self.config_label.setText(
            f"Active config: {profile} profile | {params} params | "
            f"GPU layers: {ngl} | Context: {ctx}"
        )

    def set_model_path(self, path: str):
        self.model_path_edit.setText(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "GGUF Files (*.gguf);;All Files (*)"
        )
        if path:
            self.model_path_edit.setText(path)

    def _export(self):
        if not self.current_config:
            self.status_text.append("No configuration to export.")
            return

        model_path = self.model_path_edit.text().strip()
        fmt = self.format_combo.currentIndex()

        base_name = "llm_config"
        if model_path:
            import os
            base_name = os.path.splitext(os.path.basename(model_path))[0]

        output_dir = self.output_path_edit.text().strip()
        if not output_dir:
            output_dir = "."

        try:
            if fmt == 0:
                out_path = f"{output_dir}/{base_name}.Modelfile"
                export_ollama_modelfile(out_path, self.current_config, model_path)
                self.status_text.append(f"Exported: {out_path}")
            elif fmt == 1:
                out_path = f"{output_dir}/{base_name}_run.bat"
                export_llama_cpp_config(out_path, self.current_config, model_path)
                self.status_text.append(f"Exported: {out_path}")
            else:
                out_path = f"{output_dir}/{base_name}.json"
                export_json_config(out_path, self.current_config, model_path)
                self.status_text.append(f"Exported: {out_path}")
        except Exception as e:
            self.status_text.append(f"Export error: {e}")

    def _launch(self):
        if not self.current_config:
            self.status_text.append("No configuration to launch.")
            return

        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self.status_text.append("Please set a model path first.")
            return

        engine = self.engine_combo.currentText().lower()
        self.status_text.append(f"Launching {engine} with model: {model_path}")
        self.launch_btn.setEnabled(False)

        try:
            result = launch_config(self.current_config, model_path, engine)
            if result:
                self.status_text.append(f"Launch result: {result}")
            else:
                self.status_text.append("Launch complete.")
        except Exception as e:
            self.status_text.append(f"Launch error: {e}")
        finally:
            self.launch_btn.setEnabled(True)
