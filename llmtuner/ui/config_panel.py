from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QCheckBox,
    QTextEdit, QFormLayout, QTabWidget
)
from PySide6.QtCore import Qt, Signal, QTimer
from llmtuner.core.recommender import recommend_config, estimate_tok_per_sec, _kv_cache_mb_per_token
from llmtuner.core.constants import QUANT_SIZES, KV_SCALE
from llmtuner.core.model_db import get_model_details
from llmtuner.core.system_info import SystemInfo
import re


def _infer_params_from_name(model_data: dict) -> str:
    for f in ["name", "id"]:
        text = str(model_data.get(f, ""))
        match = re.search(r'(\d+\.?\d*)[Bb]', text)
        if match:
            val = match.group(1)
            return f"{val}B" if "." in val else f"{int(float(val))}B"
    return ""


class ConfigPanelTab(QWidget):
    config_changed = Signal(dict)

    def __init__(self):
        super().__init__()
        self.system_info = None
        self.selected_model = None
        self.current_config = None
        self.selected_engine = "llama.cpp"
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
        self.tabs.addTab(self.summary_tab, "Summary & Memory")

        layout.addWidget(self.tabs, 1)

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
        self.n_gpu_layers_edit.setToolTip("Number of layers to offload to GPU (0 = CPU only, 99 = all). Higher = faster but needs more VRAM.")
        layout.addRow("GPU Layers (-ngl):", self.n_gpu_layers_edit)

        self.n_ctx_edit = QLineEdit("4096")
        self.n_ctx_edit.setToolTip("Maximum context length in tokens. More context = more VRAM used for KV cache. 4096 for short chats, 16384+ for RAG.")
        layout.addRow("Context Size (-c):", self.n_ctx_edit)

        self.n_batch_edit = QLineEdit("2048")
        self.n_batch_edit.setToolTip("Processing batch size. Larger = faster prompt processing but uses more VRAM. Should be <= context size.")
        layout.addRow("Batch Size (-b):", self.n_batch_edit)

        self.n_threads_edit = QLineEdit("8")
        self.n_threads_edit.setToolTip("Number of CPU threads for generation. Set to your physical core count for best performance.")
        layout.addRow("Threads (-t):", self.n_threads_edit)

        self.n_threads_batch_edit = QLineEdit("8")
        self.n_threads_batch_edit.setToolTip("Number of CPU threads for prompt processing. Can be higher than threads if prompt processing is the bottleneck.")
        layout.addRow("Batch Threads:", self.n_threads_batch_edit)

        self.n_predict_edit = QLineEdit("256")
        self.n_predict_edit.setToolTip("Maximum number of tokens to generate in one response. 256 for short replies, 1024+ for long outputs.")
        layout.addRow("Predict Tokens (-n):", self.n_predict_edit)

        self.quant_combo = QComboBox()
        self.quant_combo.addItems(["q2_k", "q3_k_m", "q4_0", "q4_k_m", "q4_k_s", "q5_0", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "f16"])
        layout.addRow("Quantization:", self.quant_combo)

        self.temperature_edit = QLineEdit("0.8")
        layout.addRow("Temperature:", self.temperature_edit)

        self.repeat_penalty_edit = QLineEdit("1.1")
        layout.addRow("Repeat Penalty:", self.repeat_penalty_edit)

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
                        self.n_threads_edit, self.n_threads_batch_edit, self.n_predict_edit,
                        self.temperature_edit, self.repeat_penalty_edit]:
            editor.editingFinished.connect(self._on_edit_changed)
        self.quant_combo.currentIndexChanged.connect(self._on_edit_changed)
        self.flash_attention_check.toggled.connect(self._on_edit_changed)
        self.mmap_check.toggled.connect(self._on_edit_changed)

        widget.setLayout(layout)
        return widget

    def update_recommendation(self, system_info: SystemInfo, model_data: dict):
        self.system_info = system_info
        self.selected_model = model_data
        self.selected_engine = model_data.get("engine", "llama.cpp")
        details = get_model_details(model_data.get("id", ""))
        if details:
            model_data.update(details)
            self.selected_model = model_data
        self._regenerate()

    def set_model_path(self, path: str):
        self.model_path_edit.setText(path)

    def _get_total_vram_mb(self) -> int:
        if not self.system_info or not self.system_info.gpu:
            return 0
        return sum(g.vram_total_mb for g in self.system_info.gpu if g.vram_total_mb)

    def _get_params_raw(self) -> float:
        if self.selected_model and self.selected_model.get("params_raw"):
            return float(self.selected_model["params_raw"])
        p = self.selected_model.get("params", "8B") if self.selected_model else "8B"
        match = re.search(r'(\d+\.?\d*)', p or "")
        if match:
            val = float(match.group(1))
            return val * 1e9 if "B" in p else val * 1e6
        return 8e9

    def _regenerate(self):
        if not self.system_info:
            self._show_placeholder()
            return

        profile = self.profile_combo.currentIndex()
        profile_names = ["optimum", "balanced", "max_performance"]
        profile_name = profile_names[profile]
        use_case = self.use_case_combo.currentData() or "chat"

        model_params = "8B"
        if self.selected_model:
            model_params = self.selected_model.get("params", "")
            if not model_params:
                model_params = _infer_params_from_name(self.selected_model)
            if not model_params:
                model_params = "8B"

        try:
            config = recommend_config(
                system=self.system_info, model_params=model_params,
                use_case=use_case, profile=profile_name
            )
            config["engine"] = self.selected_engine
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
        lines = ["=== LLM-Tuner Configuration ===", ""]
        lines.append(f"Profile: {config.get('profile', 'N/A')}")
        lines.append(f"Model Params: {config.get('model_params', 'N/A')}")
        lines.append(f"Use Case: {config.get('use_case', 'N/A')}")
        lines.append(f"Quantization: {config.get('quantization', 'N/A')}")
        lines.append(f"Engine: {config.get('engine', 'llama.cpp')}")
        lines.append(f"Temperature: {self.temperature_edit.text() or '0.8'}")
        lines.append(f"Repeat Penalty: {self.repeat_penalty_edit.text() or '1.1'}")
        lines.append("")

        # Parameters
        lines.append("=== Parameters ===")
        lines.append(f"GPU Layers:      {config.get('n_gpu_layers', 0)}")
        lines.append(f"Context Size:    {config.get('n_ctx', 4096)}")
        lines.append(f"Batch Size:      {config.get('n_batch', 2048)}")
        lines.append(f"Threads:         {config.get('n_threads', 8)}")
        lines.append(f"Predict Tokens:  {config.get('n_predict', 256)}")
        lines.append(f"Flash Attention: {config.get('flash_attention', True)}")
        lines.append(f"Memory Map:      {config.get('mmap', True)}")
        lines.append("")

        # Hardware
        lines.append("=== Hardware ===")
        if self.system_info:
            lines.append(f"CPU: {self.system_info.cpu.model}")
            lines.append(f"Cores: {self.system_info.cpu.physical_cores}P / {self.system_info.cpu.logical_cores}L")
            if self.system_info.cpu.l3_cache_kb:
                lines.append(f"L3 Cache: {self.system_info.cpu.l3_cache_kb / 1024:.1f} MB")
            lines.append(f"ISA: {self.system_info.cpu.instruction_sets}")

            gpu_count = len(self.system_info.gpu) if self.system_info.gpu else 0
            if self.system_info.gpu:
                gpu_names = " + ".join(g.model for g in self.system_info.gpu)
                lines.append(f"GPU: {gpu_names}")
                gpu_vram_gb = round(self._get_total_vram_mb() / 1024, 1)
                lines.append(f"Total VRAM: {gpu_vram_gb} GB ({gpu_count}x GPU)")
                best_gpu = max(self.system_info.gpu, key=lambda g: g.vram_total_mb or 0)
                if best_gpu.compute_capability != "N/A":
                    lines.append(f"Compute Cap: {best_gpu.compute_capability}")
            else:
                lines.append("GPU: None (CPU inference)")
            lines.append(f"RAM: {self.system_info.ram_total_gb} GB ({self.system_info.ram_type})")
            lines.append(f"PCIe: {self.system_info.pcie_version}")
            lines.append(f"Disk: {self.system_info.disk_type} ({self.system_info.disk_free_gb} GB free)")
        lines.append("")

# Memory Breakdown
        lines.append("=== Memory Breakdown ===")
        params_raw = self._get_params_raw()
        params_b = params_raw / 1e9
        params_name = f"{params_b:.1f}B" if params_b >= 1 else f"{params_raw / 1e6:.0f}M"
        lines.append(f"Model: {config.get('model_params', params_name)}")

        ctx_size = config.get("n_ctx", 4096)

        # KV cache multiplier from empirical llama.cpp data (q4_0 cache type):
        # iq1_ns=1bit(0.25x), q2_0=2bit(0.5x), q3_1=3bit(0.75x),
        # q4_0=4bit(1x), q5_0=5bit(1.25x), q8_0=8bit(2x), f16=16bit(4x)
        kv_scale = {
            "q2_k": 0.25, "q3_k_m": 0.75, "q4_k_m": 1.0, "q4_0": 1.0,
            "q5_k_m": 1.25, "q5_k_s": 1.25, "q5_0": 1.25,
            "q6_k": 2.0, "q8_0": 2.0, "f16": 4.0,
        }

        # Get KV cache per token at q4_0 using accurate architecture-based calculation
        arch_params = self.selected_model.get("arch_params", {}) if self.selected_model else {}
        if not isinstance(arch_params, dict):
            arch_params = {}
        kv_mb_per_tok = _kv_cache_mb_per_token(
            params_raw,
            model_family=self.selected_model.get("family", "") if self.selected_model else "",
            n_layers=arch_params.get("n_layers", 0),
            hidden_dim=arch_params.get("hidden_dim", 0),
            n_kv_heads=arch_params.get("n_kv_heads", 0),
            n_heads=arch_params.get("n_heads", 0),
        )
        base_kv_gb = round(kv_mb_per_tok * ctx_size / 1024, 2)

        # Show all quantization levels
        lines.append("")
        lines.append(f"{'Quant':>8s}  {'Model GB':>10s}  {'KV Cache GB':>12s}  {'Total GB':>10s}  {'Fits VRAM':>10s}  {'tok/s':>10s}")
        lines.append("-" * 72)

        total_vram_gb = self._get_total_vram_mb() / 1024
        selected_quant = config.get("quantization", "q4_k_m")
        show_quants = ["q2_k", "q3_k_m", "q4_0", "q4_k_m", "q4_k_s", "q5_0", "q5_k_m", "q5_k_s", "q6_k", "q8_0", "f16"]

        for q in show_quants:
            mult = QUANT_SIZES.get(q, 0.4)
            model_gb = round(params_raw * mult / 1e9, 2)
            scale = kv_scale.get(q, 1.0)
            kv_gb = round(base_kv_gb * scale, 2)
            total_gb = round(model_gb + kv_gb, 2)
            fits = ""
            if total_vram_gb > 0:
                fits = "YES" if total_gb <= total_vram_gb * 0.9 else "PARTIAL" if total_gb <= total_vram_gb * 1.2 else "NO"
            else:
                ram_gb = self.system_info.ram_total_gb if self.system_info else 0
                fits = "YES" if total_gb <= ram_gb * 0.7 else "NO" if ram_gb > 0 else "N/A"

            tok_s = ""
            if self.system_info:
                tok_s_data = estimate_tok_per_sec(self.system_info, config.get("model_params", "8B"), q)
                tok_s = tok_s_data.get("gen", str(tok_s_data))

            marker = " <--" if q == selected_quant else ""
            lines.append(f"{q:>8s}  {model_gb:>9.2f} GB  {kv_gb:>11.2f} GB  {total_gb:>9.2f} GB  {fits:>10s}  {tok_s:>10s}{marker}")

        lines.append("")
        cur_q = config.get("quantization", "q4_k_m")
        cur_scale = kv_scale.get(cur_q, 1.0)
        cur_kv = round(base_kv_gb * cur_scale, 2)
        lines.append(f"KV Cache ({cur_q}): {cur_kv} GB at {ctx_size} context")
        lines.append(f"  At 4096 ctx: {round(kv_mb_per_tok * 4096 / 1024 * cur_scale, 2)} GB  |  At 16384 ctx: {round(kv_mb_per_tok * 16384 / 1024 * cur_scale, 2)} GB  |  At 32768 ctx: {round(kv_mb_per_tok * 32768 / 1024 * cur_scale, 2)} GB")
        lines.append("")
        if total_vram_gb > 0:
            lines.append(f"Total GPU VRAM Available: {total_vram_gb:.1f} GB")
            cur_mult = QUANT_SIZES.get(cur_q, 0.4)
            cur_model_gb = round(params_raw * cur_mult / 1e9, 2)
            cur_total = round(cur_model_gb + cur_kv, 2)
            if cur_total <= total_vram_gb * 0.9:
                lines.append(f"Status: {cur_q} fits entirely in VRAM")
            elif cur_total <= total_vram_gb:
                lines.append(f"Status: {cur_q} fits in VRAM (tight)")
            else:
                lines.append(f"Status: {cur_q} exceeds VRAM - partial CPU offload needed")
                for q in ["q6_k", "q5_k_m", "q4_k_m", "q3_k_m"]:
                    m = QUANT_SIZES.get(q, 0.4)
                    qs = kv_scale.get(q, 1.0)
                    qkv = round(kv_mb_per_tok * ctx_size / 1024 * qs, 2)
                    qtotal = round(params_raw * m / 1e9 + qkv, 2)
                    if qtotal <= total_vram_gb * 0.9:
                        lines.append(f"  -> Suggest: {q} would fit in VRAM ({qtotal:.1f} GB total)")
                        break

        lines.append("")
        if self.system_info:
            tok_s_data = estimate_tok_per_sec(self.system_info, config.get("model_params", "8B"), config.get("quantization", "q4_k_m"))
            lines.append(f"Expected Performance: {tok_s_data.get('gen', str(tok_s_data))}")

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
        self.current_config["temperature"] = self.temperature_edit.text() or "0.8"
        self.current_config["repeat_penalty"] = self.repeat_penalty_edit.text() or "1.1"
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
