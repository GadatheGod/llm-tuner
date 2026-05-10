from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QGroupBox, QGridLayout, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal


class SystemScanTab(QWidget):
    scan_complete = Signal(object)

    def __init__(self):
        super().__init__()
        self.system_info = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setSpacing(10)

        # Top bar
        top_bar = QHBoxLayout()
        self.scan_btn = QPushButton("Scan System")
        self.scan_btn.setStyleSheet("""
            QPushButton {
                background: #27ae60; border: none; padding: 8px 24px;
                font-weight: bold; font-size: 13px; border-radius: 4px;
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

        # Scrollable content area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(10)
        self.scroll_layout.addStretch()
        scroll.setWidget(self.scroll_content)
        layout.addWidget(scroll, 1)

        self.setLayout(layout)

    def start_scan(self):
        self.scan_btn.setEnabled(False)
        self.scan_label.setText("Scanning...")
        self.progress.show()
        self.progress.setValue(0)
        QTimer.singleShot(50, self._do_scan)

    def _do_scan(self):
        # Clear all previous groups
        while self.scroll_layout.count() > 0:
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.progress.setValue(10)
        from llmtuner.core.system_info import get_cpu_info
        self.system_info = None  # Reset
        from llmtuner.core.system_info import SystemInfo
        self.system_info = SystemInfo()
        self.system_info.cpu = get_cpu_info()
        self._add_group(self._make_cpu_group())

        self.progress.setValue(30)
        from llmtuner.core.system_info import get_gpu_info
        self.system_info.gpu = get_gpu_info()
        for g in self._make_gpu_groups():
            self._add_group(g)

        self.progress.setValue(50)
        from llmtuner.core.system_info import get_ram_info, get_disk_info, get_ram_speed, get_ram_type
        total, avail, used = get_ram_info()
        self.system_info.ram_total_gb = round(total, 1)
        self.system_info.ram_available_gb = round(avail, 1)
        self.system_info.ram_used_gb = round(used, 1)
        self.system_info.ram_speed_mhz = get_ram_speed()
        self.system_info.ram_type = get_ram_type()
        self._add_group(self._make_ram_group())

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
        self._add_group(self._make_disk_group())

        self.progress.setValue(75)
        warnings = self._detect_health_warnings()
        if warnings:
            self._add_group(self._make_warnings_group(warnings))

        self.progress.setValue(85)
        from llmtuner.core.system_info import get_pcie_version
        self.system_info.pcie_version = get_pcie_version()

        self.progress.setValue(95)
        from llmtuner.core.system_info import get_bios_info, get_display_info, get_sound_info, get_network_info
        bios = get_bios_info()
        self.system_info.bios_vendor = bios.get("vendor", "Unknown")
        self.system_info.bios_version = bios.get("version", "Unknown")
        self.system_info.bios_date = bios.get("date", "Unknown")
        self._add_group(self._make_bios_group())

        display = get_display_info()
        self.system_info.display_resolution = display.get("resolution", "N/A")
        self.system_info.display_bits = display.get("bits", "N/A")
        self.system_info.display_hz = display.get("hz", "N/A")
        self._add_group(self._make_display_group())

        sound = get_sound_info()
        self.system_info.sound_device = sound
        self._add_group(self._make_sound_group())

        network = get_network_info()
        self.system_info.network_adapter = network.get("name", "Unknown")
        self.system_info.network_speed = network.get("speed", "Unknown")
        self.system_info.network_ipv4 = network.get("ipv4", "Unknown")
        self._add_group(self._make_network_group())

        self.progress.setValue(100)
        self.scan_label.setText("Scan complete!")
        self.scan_btn.setEnabled(True)
        QTimer.singleShot(500, self.progress.hide)
        self.scan_complete.emit(self.system_info)

    def _add_group(self, group):
        self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, group)

    def _make_pair(self, label, value):
        l = QLabel(str(label) + ":")
        l.setStyleSheet("color: #555;")
        v = QLabel(str(value) if value else "N/A")
        v.setStyleSheet("font-weight: bold; color: #2d5f8a;")
        return l, v

    def _make_cpu_group(self):
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
        return group

    def _make_gpu_groups(self):
        groups = []
        for idx, gpu in enumerate(self.system_info.gpu):
            group = QGroupBox(f"GPU {idx} — {gpu.vendor}")
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
                ("Driver Version", gpu.driver_version),
                ("PCIe", self.system_info.pcie_version),
            ]
            for i, (label, val) in enumerate(pairs):
                lbl, v = self._make_pair(label, val)
                grid.addWidget(lbl, i, 0, Qt.AlignLeft | Qt.AlignTop)
                grid.addWidget(v, i, 1, Qt.AlignLeft)
            group.setLayout(grid)
            groups.append(group)
        if not groups:
            group = QGroupBox("GPU")
            vl = QVBoxLayout()
            lbl = QLabel("No GPU detected — will use CPU inference")
            lbl.setStyleSheet("color: #e67e22; font-weight: bold;")
            lbl.setAlignment(Qt.AlignCenter)
            vl.addWidget(lbl)
            group.setLayout(vl)
            groups.append(group)
        return groups

    def _make_ram_group(self):
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
        return group

    def _make_disk_group(self):
        group = QGroupBox("Disk & Storage")
        grid = QGridLayout()
        grid.setSpacing(4)
        pairs = [
            ("Disk Model", self.system_info.disk_model),
            ("Disk Type", self.system_info.disk_type),
            ("Total", f"{self.system_info.disk_total_gb} GB"),
            ("Free", f"{self.system_info.disk_free_gb} GB"),
        ]
        for i, (label, val) in enumerate(pairs):
            lbl, v = self._make_pair(label, val)
            grid.addWidget(lbl, i, 0, Qt.AlignLeft | Qt.AlignTop)
            grid.addWidget(v, i, 1, Qt.AlignLeft)
        group.setLayout(grid)
        return group

    def _make_bios_group(self):
        group = QGroupBox("BIOS / Firmware")
        grid = QGridLayout()
        grid.setSpacing(4)
        pairs = [
            ("BIOS Vendor", self.system_info.bios_vendor),
            ("BIOS Version", self.system_info.bios_version),
            ("BIOS Date", self.system_info.bios_date),
        ]
        for i, (label, val) in enumerate(pairs):
            lbl, v = self._make_pair(label, val)
            grid.addWidget(lbl, i, 0, Qt.AlignLeft | Qt.AlignTop)
            grid.addWidget(v, i, 1, Qt.AlignLeft)
        group.setLayout(grid)
        return group

    def _make_display_group(self):
        group = QGroupBox("Display")
        grid = QGridLayout()
        grid.setSpacing(4)
        pairs = [
            ("Resolution", self.system_info.display_resolution),
            ("Color Depth", self.system_info.display_bits),
            ("Refresh Rate", self.system_info.display_hz),
        ]
        for i, (label, val) in enumerate(pairs):
            lbl, v = self._make_pair(label, val)
            grid.addWidget(lbl, i, 0, Qt.AlignLeft | Qt.AlignTop)
            grid.addWidget(v, i, 1, Qt.AlignLeft)
        group.setLayout(grid)
        return group

    def _make_sound_group(self):
        group = QGroupBox("Sound")
        grid = QGridLayout()
        grid.setSpacing(4)
        lbl, v = self._make_pair("Device", self.system_info.sound_device or "None detected")
        grid.addWidget(lbl, 0, 0, Qt.AlignLeft)
        grid.addWidget(v, 0, 1, Qt.AlignLeft)
        group.setLayout(grid)
        return group

    def _make_network_group(self):
        group = QGroupBox("Network & System")
        grid = QGridLayout()
        grid.setSpacing(4)
        pairs = [
            ("Adapter", self.system_info.network_adapter),
            ("Speed", self.system_info.network_speed),
            ("IPv4 Address", self.system_info.network_ipv4),
            ("OS", f"{self.system_info.os_name} {self.system_info.os_arch}"),
       ("OS Version", self.system_info.os_version),
        ]
        for i, (label, val) in enumerate(pairs):
            lbl, v = self._make_pair(label, val)
            grid.addWidget(lbl, i, 0, Qt.AlignLeft | Qt.AlignTop)
            grid.addWidget(v, i, 1, Qt.AlignLeft)
        group.setLayout(grid)
        return group

    def _detect_health_warnings(self):
        """Detect system health issues that could affect LLM performance."""
        warnings = []
        if not self.system_info:
            return warnings
        gpu = self.system_info.gpu
        ram = self.system_info.ram_total_gb
        disk_free = self.system_info.disk_free_gb
        disk_type = self.system_info.disk_type

        total_vram = sum(g.vram_total_mb for g in gpu) if gpu else 0
        if total_vram > 0 and total_vram < 6144:
            warnings.append(("Low VRAM", f"Only {total_vram // 1024}GB VRAM. Models larger than 7B will be very slow. Consider q2_k/q3_k quantization.", "#e74c3c"))
        elif not gpu:
            warnings.append(("No GPU", "No GPU detected. All inference will run on CPU, which is significantly slower.", "#e67e22"))

        if ram > 0 and ram < 16:
            warnings.append(("Low RAM", f"Only {ram}GB RAM. Large models (13B+) may not fit or will be very slow.", "#e74c3c"))

        if disk_free > 0 and disk_free < 10:
            warnings.append(("Low Disk Space", f"Only {disk_free:.0f}GB free. Need space for model files (typically 2-30GB each).", "#e67e22"))

        if disk_type and ("HDD" in disk_type or "Mechanical" in disk_type):
            warnings.append(("Slow Disk", "Using mechanical HDD. Model loading will be slow — consider SSD.", "#e67e22"))

        return warnings

    def _make_warnings_group(self, warnings):
        """Create a warning group showing system health issues."""
        group = QGroupBox("System Health")
        layout = QVBoxLayout()
        layout.setSpacing(6)
        for title, message, color in warnings:
            label = QLabel(f"{title}: {message}")
            label.setStyleSheet(f"color: {color}; font-weight: bold; padding: 4px; border-left: 3px solid {color}; border-radius: 2px; padding-left: 8px;")
            layout.addWidget(label)
        if not warnings:
            ok_label = QLabel("All checks passed — system is suitable for LLM inference.")
            ok_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            layout.addWidget(ok_label)
        group.setLayout(layout)
        return group
