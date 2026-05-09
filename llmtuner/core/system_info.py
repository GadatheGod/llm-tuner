import os
import platform
import subprocess
import json
import re
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
    architecture: str = "Unknown"


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
    instruction_sets: str = "Unknown"


@dataclass
class SystemInfo:
    cpu: CPUInfo = field(default_factory=CPUInfo)
    gpu: List[GPUInfo] = field(default_factory=list)
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_speed_mhz: int = 0
    ram_type: str = "Unknown"
    os_name: str = ""
    os_version: str = ""
    os_arch: str = ""
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_type: str = "Unknown"
    disk_model: str = "Unknown"
    pcie_version: str = "N/A"
    bios_vendor: str = "Unknown"
    bios_version: str = "Unknown"
    bios_date: str = "Unknown"
    display_resolution: str = "N/A"
    display_bits: str = "N/A"
    display_hz: str = "N/A"
    sound_device: str = "None"
    network_adapter: str = "Unknown"
    network_speed: str = "Unknown"
    network_ipv4: str = "Unknown"

    def __post_init__(self):
        if self.gpu is None:
            self.gpu = []

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gpu"] = [asdict(g) for g in self.gpu]
        return d


def _detect_gpu_arch(model: str, vendor: str) -> str:
    model_upper = model.upper()
    if vendor == "NVIDIA":
        if any(x in model_upper for x in ["RTX 40", "AD", "ADA"]):
            return "Ada Lovelace"
        elif any(x in model_upper for x in ["RTX 30", "A5000", "A4000", "A100", "A40", "A10", "A6000", "A1000", "A2000", "A3000", "A50", "L40", "L4", "RTX 6000", "RTX A5000", "RTX A4000", "RTX A6000"]):
            return "Ampere"
        elif any(x in model_upper for x in ["RTX 20", "TITAN RTX", "A30"]):
            return "Turing"
        elif any(x in model_upper for x in ["RTX 10", "GTX 10", "TITAN X", "TITAN V"]):
            return "Pascal"
        elif any(x in model_upper for x in ["GTX 9", "GTX 980", "GTX 970"]):
            return "Maxwell"
        elif "T4" in model_upper or "P4" in model_upper or "P100" in model_upper:
            return "Pascal"
        elif "H100" in model_upper or "H200" in model_upper:
            return "Hopper"
        elif "A100" in model_upper:
            return "Ampere"
    elif vendor == "AMD":
        if any(x in model_upper for x in ["RX 7", "RADEN 7", "RDNA3"]):
            return "RDNA 3"
        elif any(x in model_upper for x in ["RX 6", "RADEN 6", "RDNA2"]):
            return "RDNA 2"
        elif any(x in model_upper for x in ["RX 5", "RDNA"]):
            return "RDNA"
    return "Unknown"


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

    l3_raw = str(info.get("l3_cache_size", ""))
    if "b" in l3_raw.lower():
        l3_mb_match = re.search(r"(\d+)", l3_raw)
        if l3_mb_match:
            val = int(l3_mb_match.group(1))
            if "m" in l3_raw.lower():
                cpu.l3_cache_kb = val * 1024
            elif "k" in l3_raw.lower():
                cpu.l3_cache_kb = val
            else:
                cpu.l3_cache_kb = val
    elif l3_raw.isdigit():
        byte_val = int(l3_raw)
        if byte_val > 1000000:
            cpu.l3_cache_kb = byte_val // 1024
        else:
            cpu.l3_cache_kb = byte_val

    l2_raw = str(info.get("l2_cache_size", ""))
    if "b" in l2_raw.lower():
        l2_kb_match = re.search(r"(\d+)", l2_raw)
        if l2_kb_match:
            val = int(l2_kb_match.group(1))
            if "m" in l2_raw.lower():
                cpu.l2_cache_kb = val * 1024
            elif "k" in l2_raw.lower():
                cpu.l2_cache_kb = val
            else:
                cpu.l2_cache_kb = val
    elif l2_raw.isdigit():
        byte_val = int(l2_raw)
        if byte_val > 1000000:
            cpu.l2_cache_kb = byte_val // 1024
        else:
            cpu.l2_cache_kb = byte_val

    flags = info.get("flags", [])
    if isinstance(flags, str):
        flags = [flags]
    cpu.instruction_sets = ", ".join([f for f in ["avx2", "avx512", "fma3", "sse4_2"] if f in str(flags).lower()]) if flags else "Unknown"

    return cpu


def get_gpu_info() -> List[GPUInfo]:
    gpus = []

    try:
        import GPUtil
        for gpu in GPUtil.getGPUs():
            g = GPUInfo()
            g.vendor = "NVIDIA"
            g.model = gpu.name
            g.vram_total_mb = int(gpu.memoryTotal)
            g.vram_free_mb = int(gpu.memoryFree)
            g.vram_used_mb = int(gpu.memoryUsed)
            g.clock_mhz = int(gpu.load * 100)
            g.architecture = _detect_gpu_arch(g.model, g.vendor)
            gpus.append(g)
    except Exception:
        pass

    if gpus:
        return gpus

    if not gpus:
        try:
            nvml_result = _try_pynvml()
            if nvml_result:
                return nvml_result
        except Exception:
            pass

    if not gpus:
        try:
            gpus = _try_powershell_gpu()
        except Exception:
            pass

    if not gpus:
        gpus = _try_wmic_gpu()

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
                clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                g.clock_mhz = clock
            except Exception:
                pass
            try:
                mem_bw = pynvml.nvmlDeviceGetMemoryInfo(handle)
                g.memory_bandwidth_gbps = 500
            except Exception:
                pass
            g.architecture = _detect_gpu_arch(g.model, g.vendor)
            gpus.append(g)
        return gpus
    except Exception:
        return []


def _try_powershell_gpu() -> List[GPUInfo]:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion,VideoModeDescription | ConvertTo-Json -Depth 2"],
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
            g.architecture = _detect_gpu_arch(g.model, g.vendor)
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
                g.architecture = _detect_gpu_arch(g.model, g.vendor)
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
    disk_model = "Unknown"
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "$drive = '" + os.path.splitdrive(path)[0].replace(":", "") + "'; "
             "$vol = Get-Volume | Where-Object { $_.DriveLetter -eq $drive }; "
             "$diskNum = $vol | Get-Disk | Select-Object -ExpandProperty Number; "
             "$phy = Get-PhysicalDisk | Where-Object { $_.BusType -ne $null }; "
             "$result = Get-PhysicalDisk | Where-Object { $_.FriendlyName -ne $null } | Select-Object -First 1 MediaType,FriendlyName,BusType | ConvertTo-Json; "
             "Write-Output $result"],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            try:
                disk_data = json.loads(result.stdout.strip())
                media = disk_data.get("MediaType", "")
                if media == "SSD":
                    bus_type = disk_data.get("BusType", "")
                    if bus_type == "NVMe":
                        disk_type = "NVMe SSD"
                    else:
                        disk_type = "SATA SSD"
                elif media == "HDD":
                    disk_type = "HDD (Mechanical)"
                else:
                    bus = disk_data.get("BusType", "")
                    disk_type = bus if bus else "SSD"
                disk_model = disk_data.get("FriendlyName", "Unknown").split(",")[0].strip()
            except Exception:
                pass
    except Exception:
        pass
    return total_gb, free_gb, disk_type, disk_model


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


def get_ram_type() -> str:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_PhysicalMemory | Select-Object -First 1 -ExpandProperty MemoryType"],
            capture_output=True, text=True, timeout=5
        )
        mem_type = result.stdout.strip()
        type_map = {"24": "DDR", "24.5": "DDR2", "24.6": "DDR3", "24.7": "DDR4", "24.8": "DDR5"}
        return type_map.get(mem_type, f"DDR (Type {mem_type})" if mem_type else "Unknown")
    except Exception:
        return "Unknown"


def get_pcie_version() -> str:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA|AMD|Radeon|GeForce' } | Select-Object -First 1; "
             "if ($gpu) { "
             "  $devId = $gpu.PNPDeviceID; "
             "  $pci = Get-CimInstance MSPCI_Device -Namespace 'root\\standardcimv2' | Where-Object { $_.PNPDeviceID -match [regex]::Escape($devId.split(':')[0]) } | Select-Object -First 1; "
             "  if ($pci) { Write-Output $pci.CurrentSpeed } else { Write-Output 'N/A' } "
             "} else { Write-Output 'N/A' }"],
            capture_output=True, text=True, timeout=10
        )
        speed = result.stdout.strip()
        if speed and speed != "N/A":
            return f"PCIe {speed}"
        return _get_pcie_simple()
    except Exception:
        return _get_pcie_simple()


def _get_pcie_simple() -> str:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'RTX 40|RTX 30|RTX 20|RTX 3080|RTX 3070|RTX 3090' } | Select-Object -First 1 Name"],
            capture_output=True, text=True, timeout=5
        )
        name = result.stdout.strip().replace("Name:", "").strip()
        if "RTX 40" in name:
            return "PCIe 4.0"
        elif "RTX 30" in name or "RTX 20" in name:
            return "PCIe 4.0"
        else:
            return "PCIe 3.0"
    except Exception:
        return "N/A"


def scan_system() -> SystemInfo:
    sysinfo = SystemInfo()
    sysinfo.cpu = get_cpu_info()
    sysinfo.gpu = get_gpu_info()

    total, avail, used = get_ram_info()
    sysinfo.ram_total_gb = round(total, 1)
    sysinfo.ram_available_gb = round(avail, 1)
    sysinfo.ram_used_gb = round(used, 1)
    sysinfo.ram_speed_mhz = get_ram_speed()
    sysinfo.ram_type = get_ram_type()

    sysinfo.os_name = platform.system()
    sysinfo.os_version = platform.version()
    sysinfo.os_arch = platform.machine()

    total_disk, free_disk, disk_type, disk_model = get_disk_info()
    sysinfo.disk_total_gb = round(total_disk, 1)
    sysinfo.disk_free_gb = round(free_disk, 1)
    sysinfo.disk_type = disk_type
    sysinfo.disk_model = disk_model
    sysinfo.pcie_version = get_pcie_version()

    return sysinfo


def get_bios_info() -> dict:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "$bios = Get-CimInstance Win32_BIOS | Select-Object -First 1 Manufacturer,SMBIOSBIOSVersion,ReleaseDate; "
             "$bios | ConvertTo-Json"],
            capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout.strip())
        return {
            "vendor": data.get("Manufacturer", "Unknown"),
            "version": data.get("SMBIOSBIOSVersion", "Unknown"),
            "date": data.get("ReleaseDate", "Unknown"),
        }
    except Exception:
        return {"vendor": "Unknown", "version": "Unknown", "date": "Unknown"}


def get_display_info() -> dict:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "$disp = Get-CimInstance Win32_DesktopMonitor | Select-Object -First 1; "
             "$screen = Get-CimInstance Win32_VideoController | Select-Object -First 1; "
             "$res = $screen.Name; "
             "$bits = $screen.CurrentHorizontalResolution + 'x' + $screen.CurrentVerticalResolution; "
             "$hz = $screen.CurrentRefreshRate; "
             "$r = @{'resolution'=$bits;'bits'='32 bit';'hz'=$hz}; $r | ConvertTo-Json"],
            capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout.strip())
        return {
            "resolution": data.get("resolution", "N/A"),
            "bits": data.get("bits", "N/A"),
            "hz": str(data.get("hz", "N/A")),
        }
    except Exception:
        return {"resolution": "N/A", "bits": "N/A", "hz": "N/A"}


def get_sound_info() -> str:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "$audio = Get-CimInstance Win32_SoundDevice | Select-Object -First 1 -ExpandProperty Name; Write-Output $audio"],
            capture_output=True, text=True, timeout=5
        )
        name = result.stdout.strip()
        return name if name else "None detected"
    except Exception:
        return "None detected"


def get_network_info() -> dict:
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "$nic = Get-CimInstance Win32_NetworkAdapter | Where-Object { $_.NetConnectionStatus -eq 2 -and $_.NetEnabled -eq $true } | Select-Object -First 1; "
             "$cfg = Get-CimInstance Win32_NetworkAdapterConfiguration | Where-Object { $_.IPEnabled -eq $true -and $_.DHCPEnabled -eq $true } | Select-Object -First 1; "
             "$n = @{'name'=$nic.Name;'speed'=$nic.Speed;'ipv4'=$cfg.IPAddress[0]}; $n | ConvertTo-Json"],
            capture_output=True, text=True, timeout=5
        )
        data = json.loads(result.stdout.strip())
        speed = data.get("speed", 0)
        speed_str = "Unknown"
        if speed:
            try:
                speed_val = int(speed) if not isinstance(speed, int) else speed
                if speed_val >= 1000000000:
                    speed_str = f"{speed_val // 1000000000} Gbps"
                elif speed_val >= 1000000:
                    speed_str = f"{speed_val // 1000000} Mbps"
            except (ValueError, TypeError):
                pass
        return {
            "name": data.get("name", "Unknown"),
            "speed": speed_str,
            "ipv4": data.get("ipv4", "Unknown"),
        }
    except Exception:
        return {"name": "Unknown", "speed": "Unknown", "ipv4": "Unknown"}