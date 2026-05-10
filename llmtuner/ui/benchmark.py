from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGroupBox, QTextEdit,
    QLineEdit, QFileDialog, QRadioButton, QButtonGroup, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from llmtuner.core.benchmark_runner import BenchmarkRunner, BenchmarkResult
from llmtuner.core.benchmark_thread import BenchmarkThread
from llmtuner.core.config_export import export_llama_cpp_config, export_ollama_modelfile
from llmtuner.utils.persistence import set_pref, get_pref


class BenchmarkTab(QWidget):
    result_ready = Signal(object)

    def __init__(self):
        super().__init__()
        self.model_path = ""
        self.llama_path = ""
        self.ollama_path = ""
        self.current_config = None
        self.selected_engine = "llama.cpp"
        self.runner = BenchmarkRunner(on_progress=self._on_progress)
        self._thread = None
        self._build_ui()

    def _init_thread(self):
        self._thread = BenchmarkThread(self.runner, "llama")
        self._thread.progress.connect(self._on_progress)
        self._thread.result.connect(self._show_result)
        self._thread.error.connect(lambda e: self.output_box.append(f"Error: {e}"))

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Engine selector
        engine_group = QGroupBox("Engine")
        engine_layout = QHBoxLayout()
        self.engine_grp = QButtonGroup()
        self.llama_radio = QRadioButton("llama.cpp")
        self.llama_radio.setChecked(True)
        self.llama_radio.toggled.connect(self._on_engine_change)
        self.ollama_radio = QRadioButton("Ollama")
        self.ollama_radio.toggled.connect(self._on_engine_change)
        self.engine_grp.addButton(self.llama_radio)
        self.engine_grp.addButton(self.ollama_radio)
        engine_layout.addWidget(self.llama_radio)
        engine_layout.addWidget(self.ollama_radio)
        engine_layout.addStretch()
        engine_group.setLayout(engine_layout)
        layout.addWidget(engine_group)

        # Paths
        paths_group = QGroupBox("Paths")
        paths_layout = QVBoxLayout()
        paths_layout.setSpacing(6)

        # Model path
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to .gguf model file...")
        model_row.addWidget(self.model_path_edit, 1)
        self.browse_model_btn = QPushButton("Browse")
        self.browse_model_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.browse_model_btn)
        paths_layout.addLayout(model_row)

        # Engine path
        engine_row = QHBoxLayout()
        engine_row.addWidget(QLabel("Engine:"))
        self.engine_path_edit = QLineEdit()
        self.engine_path_edit.setPlaceholderText("Path to llama-cli.exe or ollama...")
        engine_row.addWidget(self.engine_path_edit, 1)
        self.browse_engine_btn = QPushButton("Browse")
        self.browse_engine_btn.clicked.connect(self._browse_engine)
        engine_row.addWidget(self.browse_engine_btn)
        auto_detect_btn = QPushButton("Auto-Detect")
        auto_detect_btn.clicked.connect(self._auto_detect_paths)
        engine_row.addWidget(auto_detect_btn)
        paths_layout.addLayout(engine_row)

        # Ollama model name
        ollama_row = QHBoxLayout()
        ollama_row.addWidget(QLabel("Ollama Name:"))
        self.ollama_name_edit = QLineEdit()
        self.ollama_name_edit.setPlaceholderText("Ollama model name (e.g., llama3)")
        ollama_row.addWidget(self.ollama_name_edit, 1)
        self.ollama_name_edit.hide()
        paths_layout.addLayout(ollama_row)

        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)

        # Run buttons
        run_group = QGroupBox("Run Benchmark")
        run_layout = QHBoxLayout()
        self.run_btn = QPushButton("Run Benchmark")
        self.run_btn.clicked.connect(self._run_benchmark)
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("""
            QPushButton { background: #2d5f8a; padding: 10px 24px;
                font-weight: bold; font-size: 13px; border-radius: 4px; }
            QPushButton:hover { background: #3a7cb8; }
        """)
        run_layout.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        run_layout.addWidget(self.cancel_btn)

        self.accuracy_btn = QPushButton("Run Accuracy Test (20 Q&A)")
        self.accuracy_btn.clicked.connect(self._run_accuracy)
        self.accuracy_btn.setEnabled(False)
        run_layout.addWidget(self.accuracy_btn)
        run_layout.addStretch()
        run_group.setLayout(run_layout)
        layout.addWidget(run_group)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("Running benchmark...")
        self.progress.hide()
        layout.addWidget(self.progress)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #666; font-style: italic;")
        self.progress_label.hide()
        layout.addWidget(self.progress_label)

        # Results
        results_frame = QGroupBox("Results")
        results_layout = QVBoxLayout()

        self.metrics_bar = QHBoxLayout()
        self.tps_label = QLabel("Tokens/s: --")
        self.tps_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #2d5f8a;")
        self.metrics_bar.addWidget(self.tps_label)

        self.ptps_label = QLabel("Prompt tok/s: --")
        self.ptps_label.setStyleSheet("font-size: 14px; color: #666;")
        self.metrics_bar.addWidget(self.ptps_label)

        self.accuracy_label = QLabel("Accuracy: --")
        self.accuracy_label.setStyleSheet("font-size: 14px; color: #666;")
        self.metrics_bar.addWidget(self.accuracy_label)

        self.load_time_label = QLabel("Load time: --")
        self.load_time_label.setStyleSheet("font-size: 12px; color: #999;")
        self.metrics_bar.addWidget(self.load_time_label)
        self.metrics_bar.addStretch()
        results_layout.addLayout(self.metrics_bar)

        # Export button
        export_row = QHBoxLayout()
        self.export_btn = QPushButton("Export Config & Launch")
        self.export_btn.clicked.connect(self._export_and_launch)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("""
            QPushButton { background: #27ae60; padding: 8px 16px;
                font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background: #2ecc71; }
        """)
        export_row.addWidget(self.export_btn)
        export_row.addStretch()
        results_layout.addLayout(export_row)
        results_layout.addStretch()
        results_frame.setLayout(results_layout)
        layout.addWidget(results_frame)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText("Benchmark output will appear here...")
        self.output_box.setFont(QFont("Consolas", 9))
        self.output_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.output_box, 1)

        self.setLayout(layout)

    def update_config(self, config: dict):
        self.current_config = config
        if config:
            if config.get("model_path"):
                self.model_path = config["model_path"]
                self.model_path_edit.setText(config["model_path"])
            if config.get("llama_cpp_path"):
                self.llama_path = config["llama_cpp_path"]
                self.engine_path_edit.setText(self.llama_path)
            self.run_btn.setEnabled(True)

    def set_model_path(self, path: str):
        self.model_path = path
        self.model_path_edit.setText(path)
        self.run_btn.setEnabled(True)

    def set_llama_cpp_path(self, path: str):
        self.llama_path = path
        self.engine_path_edit.setText(path)
        self.runner.set_llama_cpp_path(path)

    def _on_engine_change(self):
        self.selected_engine = "llama.cpp" if self.llama_radio.isChecked() else "ollama"
        if self.selected_engine == "ollama":
            self.ollama_name_edit.show()
        else:
            self.ollama_name_edit.hide()

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "GGUF Files (*.gguf);;All Files (*)"
        )
        if path:
            self.set_model_path(path)

    def _browse_engine(self):
        if self.selected_engine == "llama.cpp":
            path, _ = QFileDialog.getOpenFileName(
                self, "Select llama-cli", "",
                "Executables (*.exe);;All Files (*)"
            )
            if path:
                self.llama_path = path
                self.engine_path_edit.setText(path)
                self.runner.set_llama_cpp_path(path)
                set_pref("llama_cpp_path", path)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Select ollama", "",
                "Executables (*.exe);;All Files (*)"
            )
            if path:
                self.ollama_path = path
                self.engine_path_edit.setText(path)
                set_pref("ollama_path", path)

    def _auto_detect_paths(self):
        import shutil
        detected = []
        # Auto-detect llama.cpp
        llama_path = get_pref("llama_cpp_path", "")
        if not llama_path:
            for name in ["llama-cli", "llama-cli.exe", "main"]:
                found = shutil.which(name)
                if found:
                    llama_path = found
                    break
            if not llama_path:
                import os
                common = [
                    os.path.expanduser("~/.llm-tuner/bin/llama-cli"),
                    os.path.join(os.environ.get("LOCALAPPDATA", ""), "llama.cpp/llama-cli.exe"),
                ]
                for p in common:
                    if os.path.isfile(p):
                        llama_path = p
                        break
        if llama_path:
            self.llama_path = llama_path
            if self.selected_engine == "llama.cpp":
                self.engine_path_edit.setText(llama_path)
                self.runner.set_llama_cpp_path(llama_path)
            detected.append(f"llama.cpp: {llama_path}")

        # Auto-detect ollama
        ollama_path = get_pref("ollama_path", "")
        if not ollama_path:
            found = shutil.which("ollama")
            if found:
                ollama_path = found
            else:
                import os
                ollama_common = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs/ollama/ollama.exe")
                if os.path.isfile(ollama_common):
                    ollama_path = ollama_common
        if ollama_path:
            self.ollama_path = ollama_path
            if self.selected_engine == "ollama":
                self.engine_path_edit.setText(ollama_path)
            detected.append(f"ollama: {ollama_path}")

        if detected:
            self.output_box.append("\n=== Auto-Detect Results ===\n" + "\n".join(detected))
        else:
            self.output_box.append("\nNo engines found. Download llama.cpp or Ollama first.")

    def _run_benchmark(self):
        if self._thread and self._thread.isRunning():
            return
        if self.selected_engine == "llama.cpp":
            model_path = self.model_path_edit.text().strip()
            if not model_path:
                self.output_box.append("Please select a model file first.")
                return
            llama_path = self.engine_path_edit.text().strip()
            if llama_path:
                self.runner.set_llama_cpp_path(llama_path)
            else:
                saved = get_pref("llama_cpp_path", "")
                if saved:
                    self.runner.set_llama_cpp_path(saved)
            config = self.current_config or {
                "n_threads": 8, "n_ctx": 4096, "n_batch": 2048, "n_predict": 256,
                "n_gpu_layers": 0, "flash_attention": False,
            }
            self._thread = BenchmarkThread(self.runner, "llama", model_path, config)
        else:
            model_name = self.ollama_name_edit.text().strip()
            if not model_name:
                self.output_box.append("Please enter an Ollama model name (e.g., llama3).")
                return
            self._thread = BenchmarkThread(self.runner, "ollama", model_name)
        self._thread.progress.connect(self._on_progress)
        self._thread.result.connect(self._show_result)
        self._thread.error.connect(lambda e: self.output_box.append(f"Error: {e}"))
        self._thread.start()
        self._start_bench()

    def _run_accuracy(self):
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self.output_box.append("No model loaded for accuracy test.")
            return
        config = self.current_config or {}
        result = BenchmarkResult(model=model_path, engine="llama.cpp")
        result.config = config
        self._start_bench()
        self.progress.setFormat("Running accuracy test...")
        self._thread = BenchmarkThread(self.runner, "accuracy", result, config=config)
        self._thread.progress.connect(self._on_progress)
        self._thread.result.connect(self._on_accuracy_done)
        self._thread.error.connect(lambda e: self.output_box.append(f"Accuracy test error: {e}"))
        self._thread.start()

    def _on_accuracy_done(self, result):
        self.accuracy_label.setText(
            f"Accuracy: {result.accuracy_score:.0f}% ({result.accuracy_correct}/{result.accuracy_total})"
        )
        self.output_box.append(
            f"\n=== Accuracy Test ===\n"
            f"Score: {result.accuracy_score:.1f}% "
            f"({result.accuracy_correct}/{result.accuracy_total} correct)"
        )
        self._end_bench()

    def _show_result(self, result: BenchmarkResult):
        if result.error:
            self.output_box.append(f"\nError: {result.error}")
            self.tps_label.setText("Tokens/s: ERROR")
        else:
            self.tps_label.setText(f"Tokens/s: {result.tokens_per_second:.1f}")
            self.ptps_label.setText(f"Prompt tok/s: {result.prompt_tokens_per_second:.1f}")
            self.load_time_label.setText(f"Load: {result.load_time_ms:.0f}ms")
            color = "#27ae60" if result.tokens_per_second > 30 else "#e67e22" if result.tokens_per_second > 10 else "#e74c3c"
            self.tps_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
            self.output_box.append(f"\n=== Benchmark Result ===")
            self.output_box.append(f"Model: {result.model}")
            self.output_box.append(f"Engine: {result.engine}")
            self.output_box.append(f"Tokens/s: {result.tokens_per_second:.2f}")
            self.output_box.append(f"Prompt tokens/s: {result.prompt_tokens_per_second:.2f}")
            self.output_box.append(f"Total tokens: {result.total_tokens}")
            self.output_box.append(f"Load time: {result.load_time_ms:.1f}ms")
            if result.raw_output:
                self.output_box.append(f"\n--- Raw Output ---\n{result.raw_output[-2000:]}")
        self.accuracy_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.result_ready.emit(result)

    def _export_and_launch(self):
        if not self.current_config:
            self.output_box.append("No configuration to export.")
            return
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self.output_box.append("Please set a model path first.")
            return
        try:
            import os
            base_name = os.path.splitext(os.path.basename(model_path))[0]
            if self.selected_engine == "llama.cpp":
                out_path = f"{base_name}_run.bat"
                export_llama_cpp_config(out_path, self.current_config, model_path)
                self.output_box.append(f"Exported: {out_path}")
            else:
                out_path = f"{base_name}.Modelfile"
                export_ollama_modelfile(out_path, self.current_config, model_path)
                self.output_box.append(f"Exported: {out_path}")
            self.output_box.append("Config exported successfully!")
        except Exception as e:
            self.output_box.append(f"Export error: {e}")

    def _start_bench(self):
        self.progress.show()
        self.progress_label.show()
        self.progress_label.setText("Running benchmark...")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def _end_bench(self):
        self.progress.hide()
        self.progress_label.hide()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_progress(self, msg: str):
        self.progress_label.setText(msg)

    def _cancel(self):
        if self._thread:
            self._thread.cancel()
        self.output_box.append("Benchmark cancelled.")
        self._end_bench()
