from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGroupBox, QFormLayout, QTextEdit,
    QLineEdit, QFileDialog, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QColor
from llmtuner.core.benchmark_runner import BenchmarkRunner, BenchmarkResult


class BenchmarkTab(QWidget):
    result_ready = Signal(object)

    def __init__(self):
        super().__init__()
        self.model_path = ""
        self.llama_path = ""
        self.runner = BenchmarkRunner(on_progress=self._on_progress)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        model_section = QGroupBox("Model")
        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to .gguf model file...")
        self.model_path_edit.setMinimumWidth(400)
        model_layout.addWidget(self.model_path_edit)

        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_model)
        model_layout.addWidget(self.browse_btn)
        model_section.setLayout(model_layout)
        layout.addWidget(model_section)

        engine_section = QGroupBox("Engine")
        engine_layout = QHBoxLayout()

        self.llama_btn = QPushButton("Run llama.cpp Benchmark")
        self.llama_btn.clicked.connect(self._run_llama_bench)
        self.llama_btn.setEnabled(False)
        engine_layout.addWidget(self.llama_btn)

        self.ollama_name_edit = QLineEdit()
        self.ollama_name_edit.setPlaceholderText("Ollama model name (e.g., llama3)")
        engine_layout.addWidget(self.ollama_name_edit)

        self.ollama_btn = QPushButton("Run Ollama Benchmark")
        self.ollama_btn.clicked.connect(self._run_ollama_bench)
        engine_layout.addWidget(self.ollama_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        engine_layout.addWidget(self.cancel_btn)

        engine_section.setLayout(engine_layout)
        layout.addWidget(engine_section)

        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("Running benchmark...")
        self.progress.hide()
        layout.addWidget(self.progress)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #666; font-style: italic;")
        self.progress_label.hide()
        layout.addWidget(self.progress_label)

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

        self.accuracy_btn = QPushButton("Run Accuracy Test (20 Q&A)")
        self.accuracy_btn.clicked.connect(self._run_accuracy)
        self.accuracy_btn.setEnabled(False)
        results_layout.addWidget(self.accuracy_btn)
        results_layout.addStretch()
        results_frame.setLayout(results_layout)
        layout.addWidget(results_frame)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setPlaceholderText("Benchmark output will appear here...")
        self.output_box.setFont(QFont("Consolas", 9))
        self.output_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.output_box)

        self.setLayout(layout)

    def set_model_path(self, path: str):
        self.model_path = path
        self.model_path_edit.setText(path)
        self.llama_btn.setEnabled(True)

    def set_llama_cpp_path(self, path: str):
        self.llama_path = path
        self.runner.set_llama_cpp_path(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "GGUF Files (*.gguf);;All Files (*)"
        )
        if path:
            self.set_model_path(path)

    def _run_llama_bench(self):
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self.output_box.append("Please select a model file first.")
            return

        from llmtuner.utils.persistence import get_pref
        llama_path = get_pref("llama_cpp_path", "")
        if llama_path:
            self.runner.set_llama_cpp_path(llama_path)

        self._start_bench()
        try:
            config = {
                "n_threads": 8,
                "n_ctx": 4096,
                "n_batch": 2048,
                "n_predict": 256,
                "n_gpu_layers": 0,
                "flash_attention": False,
            }
            result = self.runner.run_llama_cpp_benchmark(model_path, config)
            self._show_result(result)
        except Exception as e:
            self.output_box.append(f"Error: {e}")
        finally:
            self._end_bench()

    def _run_ollama_bench(self):
        model_name = self.ollama_name_edit.text().strip()
        if not model_name:
            self.output_box.append("Please enter an Ollama model name (e.g., llama3).")
            return

        self._start_bench()
        try:
            result = self.runner.run_ollama_benchmark(model_name)
            self._show_result(result)
        except Exception as e:
            self.output_box.append(f"Error: {e}")
        finally:
            self._end_bench()

    def _run_accuracy(self):
        model_path = self.model_path_edit.text().strip()
        if not model_path:
            self.output_box.append("No model loaded for accuracy test.")
            return

        from llmtuner.core.benchmark_runner import BenchmarkResult
        result = BenchmarkResult(model=model_path, engine="llama.cpp")
        result.config = {
            "n_threads": int(self.tps_label.text().split()[-1] if "Threads" in self.tps_label.text() else "8"),
        }

        self._start_bench()
        self.progress.setFormat("Running accuracy test...")
        try:
            result = self.runner.run_accuracy_test(result)
            self.accuracy_label.setText(f"Accuracy: {result.accuracy_score:.0f}% ({result.accuracy_correct}/{result.accuracy_total})")
            self.output_box.append(f"\n=== Accuracy Test ===")
            self.output_box.append(f"Score: {result.accuracy_score:.1f}% ({result.accuracy_correct}/{result.accuracy_total} correct)")
        except Exception as e:
            self.output_box.append(f"Accuracy test error: {e}")
        finally:
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
        self.result_ready.emit(result)

    def _start_bench(self):
        self.progress.show()
        self.progress_label.show()
        self.progress_label.setText("Running benchmark...")
        self.llama_btn.setEnabled(False)
        self.ollama_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def _end_bench(self):
        self.progress.hide()
        self.progress_label.hide()
        self.llama_btn.setEnabled(True)
        self.ollama_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_progress(self, msg: str):
        self.progress_label.setText(msg)

    def _cancel(self):
        self.runner.cancel()
        self.output_box.append("Benchmark cancelled.")
        self._end_bench()
