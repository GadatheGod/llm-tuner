from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGroupBox, QGridLayout, QFrame, QTextEdit
)
from PySide6.QtCore import Qt, QTimer, Signal
from llmtuner.core.system_info import scan_system, SystemInfo


class SystemScanTab(QWidget):
    scan_complete = Signal(object)

    def __init__(self):
        super().__init__()
        self.system_info = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        top_bar = QHBoxLayout()
        self.scan_btn = QPushButton("Scan System")
        self.scan_btn.setStyleSheet("""
            QPushButton {
                background: #27ae60;
                border: none;
                padding: 8px 24px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover { background: #2ecc71; }
        """)
        self.scan_btn.clicked.connect(self.start_scan)
        top_bar.addWidget(self.scan_btn)

        self.scan_label = QLabel("Click to scan your system")
        self.scan_label.setStyleSheet("color: #666; font-style: italic;")
        top_bar.addWidget(self.scan_label)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.setStyleSheet("QProgressBar::chunk { background: #2d5f8a; }")
        self.progress.hide()
        layout.addWidget(self.progress)

        self.content_area = QGridLayout()
        self.content_area.setSpacing(12)
        layout.addLayout(self.content_area)
        layout.addStretch()

        self.setLayout(layout)

    def start_scan(self):
        self.scan_btn.setEnabled(False)
        self.scan_label.setText("Scanning...")
        self.progress.show()
        self.progress.setValue(0)

        QTimer.singleShot(50, self._do_scan)

    def _do_scan(self):
        self.progress.setValue(10)
        from llmtuner.core.system_info import get_cpu_info
        self.system_info = SystemInfo()
        self.system_info.cpu = get_cpu_info()
        self._update_cpu_section()

        self.progress.setValue(30)
        from llmtuner.core.system_info import get_gpu_info
        self.system_info.gpu = get_gpu_info()
        self._update_gpu_section()

        self.progress.setValue(50)
        from llmtuner.core.system_info import get_ram_info, get_disk_info, get_ram_speed, get_ram_type
        total, avail, used = get_ram_info()
        self.system_info.ram_total_gb = round(total, 1)
        self.system_info.ram_available_gb = round(avail, 1)
        self.system_info.ram_used_gb = round(used, 1)
        self.system_info.ram_speed_mhz = get_ram_speed()
        self.system_info.ram_type = get_ram_type()
        self._update_ram_section()

        self.progress.setValue(70)
        import platform
        self.system_info.os_name = platform.system()
        self.system_info.os_version = platform.version()
        self.system_info.os_arch = platform.machine()
        total_disk, free_disk, disk_type, disk_model = get_disk_info()
        self.system_info.disk_total_gb = round(total_disk, 1)
        self.system_info.disk_free_gb = round(free_disk, 1)
        self.system_info.disk_type = disk_type
        self.system_info.disk_model = disk_model
        self._update_disk_section()

        self.progress.setValue(90)
        from llmtuner.core.system_info import get_pcie_version
        self.system_info.pcie_version = get_pcie_version()

        self.progress.setValue(100)
        self.scan_label.setText("Scan complete!")
        self.scan_btn.setEnabled(True)

        QTimer.singleShot(500, self.progress.hide)
        self.scan_complete.emit(self.system_info)

    def _make_pair(self, label: str, value: str):
        l = QLabel(f"{label}:")
        l.setStyleSheet("color: #555;")
        v = QLabel(str(value) if value else "N/A")
        v.setStyleSheet("font-weight: bold; color: #2d5f8a;")
        return l, v

    def _update_cpu_section(self):
        cpu = self.system_info.cpu
        group = QGroupBox("CPU")
        grid = QGridLayout()
        grid.setSpacing(4)

        pairs = [
            ("Model", cpu.model),
            ("Vendor", cpu.vendor),
            ("Physical Cores", str(cpu.physical_cores)),
            ("Logical Cores", str(cpu.logical_cores)),
            ("Max Clock", f"{cpu.max_mhz:.0f} MHz" if cpu.max_mhz else "N/A"),
            ("Current Clock", f"{cpu.current_mhz:.0f} MHz" if cpu.current_mhz else "N/A"),
            ("L2 Cache", f"{cpu.l2_cache_kb / 1024:.1f} MB" if cpu.l2_cache_kb else "N/A"),
            ("L3 Cache", f"{cpu.l3_cache_kb / 1024:.1f} MB" if cpu.l3_cache_kb else "N/A"),
            ("Instruction Sets", cpu.instruction_sets),
        ]
        for i, (label, val) in enumerate(pairs):
            lbl, v = self._make_pair(label, val)
            grid.addWidget(lbl, i, 0, Qt.AlignLeft | Qt.AlignTop)
            grid.addWidget(v, i, 1, Qt.AlignLeft)

        group.setLayout(grid)
        self._add_widget(group, 0, 0, 1, 2)

    def _update_gpu_section(self):
        gpus = self.system_info.gpu
        for idx, gpu in enumerate(gpus):
            group = QGroupBox(f"GPU {idx} - {gpu.vendor}")
            grid = QGridLayout()
            grid.setSpacing(4)

            pairs = [
                ("Model", gpu.model),
                ("Architecture", gpu.architecture),
                ("VRAM Total", f"{gpu.vram_total_mb} MB" if gpu.vram_total_mb else "N/A"),
                ("VRAM Free", f"{gpu.vram_free_mb} MB" if gpu.vram_free_mb else "N/A"),
                ("VRAM Used", f"{gpu.vram_used_mb} MB" if gpu.vram_used_mb else "N/A"),
                ("Compute Capability", gpu.compute_capability),
                ("CUDA Cores", str(gpu.core_count) if gpu.core_count else "N/A"),
                ("Clock Speed", f"{gpu.clock_mhz} MHz" if gpu.clock_mhz else "N/A"),
                ("Memory BW", f"{gpu.memory_bandwidth_gbps} GB/s" if gpu.memory_bandwidth_gbps else "N/A"),
                ("Driver", gpu.driver_version),
            ]
            for i, (label, val) in enumerate(pairs):
                lbl, v = self._make_pair(label, val)
                grid.addWidget(lbl, i, 0, Qt.AlignLeft | Qt.AlignTop)
                grid.addWidget(v, i, 1, Qt.AlignLeft)

            group.setLayout(grid)
            self._add_widget(group, idx + 1, 0, 1, 2)

        if not gpus:
            group = QGroupBox("GPU")
            vl = QVBoxLayout()
            lbl = QLabel("No GPU detected - will use CPU inference")
            lbl.setStyleSheet("color: #e67e22; font-weight: bold;")
            lbl.setAlignment(Qt.AlignCenter)
            vl.addWidget(lbl)
            group.setLayout(vl)
            self._add_widget(group, 1, 0, 1, 2)

    def _update_ram_section(self):
        group = QGroupBox("Memory (RAM)")
        grid = QGridLayout()
        grid.setSpacing(4)

        usage_pct = 0
        if self.system_info.ram_total_gb > 0:
            usage_pct = int((self.system_info.ram_used_gb / self.system_info.ram_total_gb) * 100)

        pairs = [
            ("Total", f"{self.system_info.ram_total_gb} GB"),
            ("Available", f"{self.system_info.ram_available_gb} GB"),
            ("Used", f"{self.system_info.ram_used_gb} GB ({usage_pct}%)"),
            ("Type", self.system_info.ram_type),
            ("Speed", f"{self.system_info.ram_speed_mhz} MHz" if self.system_info.ram_speed_mhz else "N/A"),
        ]
        for i, (label, val) in enumerate(pairs):
            lbl, v = self._make_pair(label, val)
            grid.addWidget(lbl, i, 0, Qt.AlignLeft | Qt.AlignTop)
            grid.addWidget(v, i, 1, Qt.AlignLeft)

        group.setLayout(grid)
        self._add_widget(group, len(self.system_info.gpu) + 1, 2, 1, 2)

    def _update_disk_section(self):
        group = QGroupBox("Disk & System")
        grid = QGridLayout()
        grid.setSpacing(4)

        pairs = [
            ("Disk Model", self.system_info.disk_model),
            ("Disk Type", self.system_info.disk_type),
            ("Total", f"{self.system_info.disk_total_gb} GB"),
            ("Free", f"{self.system_info.disk_free_gb} GB"),
            ("PCIe Version", self.system_info.pcie_version),
            ("OS", f"{self.system_info.os_name} {self.system_info.os_arch}"),
        ]
        for i, (label, val) in enumerate(pairs):
            lbl, v = self._make_pair(label, val)
            grid.addWidget(lbl, i, 0, Qt.AlignLeft | Qt.AlignTop)
            grid.addWidget(v, i, 1, Qt.AlignLeft)

        group.setLayout(grid)
        self._add_widget(group, len(self.system_info.gpu) + 2, 0, 1, 2)

    def _add_widget(self, widget, row, col, row_span, col_span):
        while self.content_area.count() > 0:
            child = self.content_area.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.content_area.addWidget(widget, row, col, row_span, col_span)
