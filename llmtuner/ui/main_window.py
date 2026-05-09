from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QMenuBar, QMenu,
    QStatusBar, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon
from llmtuner.ui.system_scan import SystemScanTab
from llmtuner.ui.model_browser import ModelBrowserTab
from llmtuner.ui.config_panel import ConfigPanelTab
from llmtuner.ui.benchmark import BenchmarkTab
from llmtuner.ui.export_launch import ExportLaunchTab
from llmtuner.core.system_info import scan_system, SystemInfo
from llmtuner.utils.logger import logger


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM-Tuner v1.0")
        self.setMinimumSize(1100, 700)
        self.resize(1200, 800)

        self.system_info = None
        self.selected_model = None
        self.current_config = None
        self.benchmark_result = None

        self._build_menu()
        self._build_tabs()
        self._build_statusbar()

        QTimer.singleShot(100, self._initial_scan)

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        scan_action = QAction("&Scan System", self)
        scan_action.setShortcut("Ctrl+S")
        scan_action.triggered.connect(self._rescan_system)
        file_menu.addAction(scan_action)

        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        tools_menu = menubar.addMenu("&Tools")
        open_model_action = QAction("Open &Model File...", self)
        open_model_action.setShortcut("Ctrl+O")
        open_model_action.triggered.connect(self._open_model_file)
        tools_menu.addAction(open_model_action)

        set_llama_path_action = QAction("Set llama.cpp &Path...", self)
        set_llama_path_action.triggered.connect(self._set_llama_path)
        tools_menu.addAction(set_llama_path_action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About LLM-Tuner", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_tabs(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(False)
        self.tab_widget.setMovable(True)

        self.system_tab = SystemScanTab()
        self.system_tab.scan_complete.connect(self._on_scan_complete)
        self.tab_widget.addTab(self.system_tab, "System")

        self.model_tab = ModelBrowserTab()
        self.model_tab.model_selected.connect(self._on_model_selected)
        self.tab_widget.addTab(self.model_tab, "Models")

        self.config_tab = ConfigPanelTab()
        self.config_tab.config_changed.connect(self._on_config_changed)
        self.tab_widget.addTab(self.config_tab, "Configure")

        self.bench_tab = BenchmarkTab()
        self.bench_tab.result_ready.connect(self._on_benchmark_ready)
        self.tab_widget.addTab(self.bench_tab, "Benchmark")

        self.export_tab = ExportLaunchTab()
        self.tab_widget.addTab(self.export_tab, "Export & Launch")

        self.setCentralWidget(self.tab_widget)

    def _build_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")

    def _initial_scan(self):
        self.system_tab.start_scan()

    def _rescan_system(self):
        self.system_tab.start_scan()

    def _on_scan_complete(self, system_info: SystemInfo):
        self.system_info = system_info
        logger.info(f"System scan complete: {len(system_info.gpu)} GPU(s), {system_info.cpu.logical_cores} CPU cores, {system_info.ram_total_gb}GB RAM")
        self.statusbar.showMessage(f"System scanned: {system_info.cpu.model} | {system_info.gpu[0].model if system_info.gpu else 'No GPU'}")

        self.model_tab.update_system_info(system_info)

    def _on_model_selected(self, model_data: dict):
        self.selected_model = model_data
        logger.info(f"Model selected: {model_data.get('name', 'Unknown')}")
        self.statusbar.showMessage(f"Selected: {model_data.get('name', 'Unknown')}")

        if self.system_info:
            self.config_tab.update_recommendation(self.system_info, model_data)

    def _on_config_changed(self, config: dict):
        self.current_config = config
        self.export_tab.update_config(config)

    def _on_benchmark_ready(self, result):
        self.benchmark_result = result
        if result.error:
            self.statusbar.showMessage(f"Benchmark error: {result.error}")
        else:
            self.statusbar.showMessage(f"Benchmark: {result.tokens_per_second:.1f} tok/s")

    def _open_model_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "GGUF Files (*.gguf);;All Files (*)"
        )
        if path:
            self.config_tab.set_model_path(path)
            self.bench_tab.set_model_path(path)
            self.statusbar.showMessage(f"Model loaded: {path}")

    def _set_llama_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select llama-cli", "",
            "Executables (*.exe);;All Files (*)"
        )
        if path:
            from llmtuner.utils.persistence import set_pref
            set_pref("llama_cpp_path", path)
            self.bench_tab.set_llama_cpp_path(path)
            self.statusbar.showMessage(f"llama.cpp path set: {path}")

    def _show_about(self):
        QMessageBox.about(
            self, "About LLM-Tuner",
            "LLM-Tuner v1.0\n\n"
            "AI-powered LLM configuration optimizer for\n"
            "llama.cpp, Ollama, and vLLM.\n\n"
            "Scans your system, recommends optimal parameters,\n"
            "runs benchmarks, and generates config files."
        )
