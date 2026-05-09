import os
import platform
import subprocess
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, List


@dataclass
class GPUInfo:
    vendor: str = "None"
    model: str = "Unknown"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    vram_used_mb: int = 0
    compute_capability: str = "N/A"
    cuda_version: str = "N/A"
    driver_version: str = "N/A"
    core_count: int = 0
    clock_mhz: int = 0
    memory_bandwidth_gbps: int = 0


@dataclass
class CPUInfo:
    model: str = "Unknown"
    vendor: str = "Unknown"
    logical_cores: int = 0
    physical_cores: int = 0
    max_mhz: float = 0.0
    current_mhz: float = 0.0
    l2_cache_kb: int = 0
    l3_cache_kb: int = 0


@dataclass
class SystemInfo:
    cpu: CPUInfo = field(default_factory=CPUInfo)
    gpu: List[GPUInfo] = field(default_factory=list)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_speed_mhz: int = 0
    os_name: str = ""
    os_version: str = ""
    os_arch: str = ""
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_type: str = "Unknown"
    pcie_version: str = "N/A"

    def __post_init__(self):
        if self.gpu is None:
            self.gpu = []

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gpu"] = [asdict(g) for g in self.gpu]
        return d


def get_cpu_info() -> CPUInfo:
    import psutil
    import cpuinfo

    cpu = CPUInfo()
    info = cpuinfo.get_cpu_info()
    cpu.model = info.get("brand_raw", "Unknown")
    cpu.vendor = info.get("vendor_id", "Unknown")
    cpu.logical_cores = psutil.cpu_count(logical=True) or 0
    cpu.physical_cores = psutil.cpu_count(logical=False) or 0

    freq = psutil.cpu_freq()
    if freq:
        cpu.current_mhz = freq.current
        cpu.max_mhz = freq.max

    return cpu


def get_gpu_info() -> List[GPUInfo]:
    gpus = []

    try:
        import GPUtil
        for gpu in GPUtil.getGPUs():
            g = GPUInfo()
            g.vendor = "NVIDIA"
            g.model = gpu.name
            g.vram_total_mb = gpu.memoryTotal * 1024
            g.vram_free_mb = gpu.memoryFree * 1024
            g.vram_used_mb = gpu.memoryUsed * 1024
            g.clock_mhz = int(gpu.load * 100)
            gpus.append(g)
    except Exception:
        pass

    if not gpus:
        try:
            nvml_result = _try_pynvml()
            if nvml_result:
                return nvml_result
        except Exception:
            pass

    if not gpus:
        try:
            gpus = _try_wmic_gpu()
        except Exception:
            pass

    if not gpus:
        gpus = _try_dxdiag_gpu()

    return gpus


def _try_pynvml() -> List[GPUInfo]:
    try:
        import pynvml
        pynvml.nvmlInit()
        gpus = []
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            g = GPUInfo()
            g.vendor = "NVIDIA"
            g.model = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            g.vram_total_mb = mem_info.total // (1024 * 1024)
            g.vram_free_mb = mem_info.free // (1024 * 1024)
            g.vram_used_mb = mem_info.used // (1024 * 1024)
            cc = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            g.compute_capability = f"{cc.major}.{cc.minor}"
            driver = pynvml.nvmlSystemGetDriverVersion()
            g.driver_version = str(driver)
            try:
                g.core_count = pynvml.nvmlDeviceGetAttribute(
                    handle, pynvml.NVML_DEVICE_ATTRIBUTE_MULTI_INSTANCE_GPU
                )
            except Exception:
                g.core_count = 0
            try:
                clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                g.clock_mhz = clock
            except Exception:
                pass
            gpus.append(g)
        return gpus
    except Exception:
        return []


def _try_wmic_gpu() -> List[GPUInfo]:
    try:
        result = subprocess.run(
            ["wmic", "PATH", "Win32_VideoController",
             "get", "Name,AdapterRAM,DriverVersion", "/format:csv"],
            capture_output=True, text=True, timeout=10
        )
        gpus = []
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if line.startswith("Name") or not line.strip():
                continue
            parts = [p.strip('"') for p in line.split(",")]
            if len(parts) >= 2:
                g = GPUInfo()
                g.model = parts[0]
                g.vendor = "NVIDIA" if "NVIDIA" in parts[0] else "AMD" if "AMD" in parts[0] or "Radeon" in parts[0] else "Intel" if "Intel" in parts[0] else "Unknown"
                if len(parts) >= 3 and parts[2]:
                    try:
                        g.vram_total_mb = int(parts[2]) // (1024 * 1024)
                    except ValueError:
                        pass
                gpus.append(g)
        return gpus
    except Exception:
        return []


def _try_dxdiag_gpu() -> List[GPUInfo]:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion | ConvertTo-Json -Depth 2"],
            capture_output=True, text=True, timeout=15
        )
        data = json.loads(result.stdout)
        gpus = []
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not item.get("Name"):
                continue
            g = GPUInfo()
            g.model = item.get("Name", "Unknown")
            g.vendor = "NVIDIA" if "NVIDIA" in g.model else "AMD" if "AMD" in g.model or "Radeon" in g.model else "Intel" if "Intel" in g.model else "Unknown"
            ram = item.get("AdapterRAM")
            if ram:
                g.vram_total_mb = int(ram) // (1024 * 1024)
            g.driver_version = item.get("DriverVersion", "N/A")
            gpus.append(g)
        return gpus
    except Exception:
        return []


def get_ram_info():
    import psutil
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024 ** 3)
    avail_gb = vm.available / (1024 ** 3)
    used_gb = vm.used / (1024 ** 3)
    return total_gb, avail_gb, used_gb


def get_disk_info(path: Optional[str] = None) -> tuple:
    import psutil
    if path is None:
        path = os.path.expanduser("~")
    usage = psutil.disk_usage(path)
    total_gb = usage.total / (1024 ** 3)
    free_gb = usage.free / (1024 ** 3)
    disk_type = "Unknown"
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             f"$disk = Get-Volume | Where-Object {{ $_.DriveLetter -eq '{os.path.splitdrive(path)[0].replace(chr(58), '')}' }}; $disk.FileSystemLabel"],
            capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            disk_type = result.stdout.strip()
    except Exception:
        pass
    return total_gb, free_gb, disk_type


def get_ram_speed() -> int:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_PhysicalMemory | Select-Object -First 1 -ExpandProperty ConfiguredClockSpeed"],
            capture_output=True, text=True, timeout=5
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def get_pcie_version() -> str:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_PNPEntity | Where-Object { $_.Name -match 'NVIDIA|AMD|Radeon|GeForce' } | Select-Object -First 1 -ExpandProperty PNPClass"],
            capture_output=True, text=True, timeout=5
        )
        return "PCIe" if result.stdout.strip() else "N/A"
    except Exception:
        return "N/A"


def scan_system() -> SystemInfo:
    import psutil
    sysinfo = SystemInfo()
    sysinfo.cpu = get_cpu_info()
    sysinfo.gpu = get_gpu_info()

    total, avail, used = get_ram_info()
    sysinfo.ram_total_gb = round(total, 1)
    sysinfo.ram_available_gb = round(avail, 1)
    sysinfo.ram_used_gb = round(used, 1)
    sysinfo.ram_speed_mhz = get_ram_speed()

    sysinfo.os_name = platform.system()
    sysinfo.os_version = platform.version()
    sysinfo.os_arch = platform.machine()

    total_disk, free_disk, disk_type = get_disk_info()
    sysinfo.disk_total_gb = round(total_disk, 1)
    sysinfo.disk_free_gb = round(free_disk, 1)
    sysinfo.disk_type = disk_type
    sysinfo.pcie_version = get_pcie_version()

    return sysinfo
