import pytest
from llmtuner.core.system_info import (
    CPUInfo, GPUInfo, SystemInfo, scan_system,
    get_cpu_info, get_gpu_info, get_ram_info
)


def test_cpu_info_creation():
    cpu = CPUInfo()
    assert cpu.model == "Unknown"
    assert cpu.vendor == "Unknown"
    assert cpu.logical_cores == 0


def test_gpu_info_creation():
    gpu = GPUInfo()
    assert gpu.vendor == "None"
    assert gpu.vram_total_mb == 0
    assert gpu.compute_capability == "N/A"


def test_system_info_creation():
    info = SystemInfo()
    assert isinstance(info.cpu, CPUInfo)
    assert isinstance(info.gpu, list)
    assert info.gpu == []


def test_system_info_to_dict():
    gpu = GPUInfo()
    gpu.model = "Test GPU"
    gpu.vram_total_mb = 8192

    info = SystemInfo()
    info.gpu = [gpu]
    info.ram_total_gb = 16.0

    d = info.to_dict()
    assert "cpu" in d
    assert "gpu" in d
    assert len(d["gpu"]) == 1
    assert d["gpu"][0]["model"] == "Test GPU"
    assert d["ram_total_gb"] == 16.0


def test_get_cpu_info():
    cpu = get_cpu_info()
    assert cpu.logical_cores > 0
    assert cpu.physical_cores > 0
    assert cpu.model != "Unknown"


def test_get_ram_info():
    total, avail, used = get_ram_info()
    assert total > 0
    assert avail > 0
    assert used >= 0


def test_scan_system():
    info = scan_system()
    assert info.cpu.logical_cores > 0
    assert info.ram_total_gb > 0
    assert info.os_name in ["Windows", "Linux", "Darwin"]


def test_get_gpu_info_returns_list():
    gpus = get_gpu_info()
    assert isinstance(gpus, list)
