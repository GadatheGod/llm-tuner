import pytest
from llmtuner.core.system_info import SystemInfo, CPUInfo, GPUInfo
from llmtuner.core.recommender import recommend_config, recommend_models


def test_recommender_cpu_only():
    info = SystemInfo()
    info.cpu = CPUInfo()
    info.cpu.logical_cores = 8
    info.cpu.physical_cores = 4
    info.ram_total_gb = 16

    config = recommend_config(info, "8B", "chat", "balanced")
    assert config["n_gpu_layers"] == 0
    assert config["n_threads"] > 0
    assert config["n_ctx"] > 0
    assert "quantization" in config


def test_recommender_gpu_nvidia():
    gpu = GPUInfo()
    gpu.vendor = "NVIDIA"
    gpu.model = "GeForce RTX 4060"
    gpu.vram_total_mb = 8192
    gpu.vram_free_mb = 7168
    gpu.compute_capability = "8.9"

    info = SystemInfo()
    info.cpu = CPUInfo()
    info.cpu.logical_cores = 16
    info.cpu.physical_cores = 8
    info.gpu = [gpu]
    info.ram_total_gb = 32

    config = recommend_config(info, "8B", "chat", "balanced")
    assert config["n_gpu_layers"] == -1
    assert config["n_ctx"] >= 2048
    assert config["n_batch"] > 0


def test_recommender_optimum_profile():
    gpu = GPUInfo()
    gpu.vendor = "NVIDIA"
    gpu.vram_total_mb = 12288
    gpu.vram_free_mb = 11000

    info = SystemInfo()
    info.cpu = CPUInfo()
    info.cpu.logical_cores = 12
    info.cpu.physical_cores = 6
    info.gpu = [gpu]
    info.ram_total_gb = 32

    config = recommend_config(info, "8B", "chat", "optimum")
    assert config["profile"] == "optimum"
    assert config["n_ctx"] <= 8192


def test_recommender_max_performance():
    gpu = GPUInfo()
    gpu.vendor = "NVIDIA"
    gpu.vram_total_mb = 24576
    gpu.vram_free_mb = 23000

    info = SystemInfo()
    info.cpu = CPUInfo()
    info.cpu.logical_cores = 24
    info.cpu.physical_cores = 12
    info.gpu = [gpu]
    info.ram_total_gb = 64

    config = recommend_config(info, "13B", "code", "max_performance")
    assert config["profile"] == "max_performance"
    assert config["n_ctx"] >= 8192


def test_recommender_code_use_case():
    gpu = GPUInfo()
    gpu.vendor = "NVIDIA"
    gpu.vram_total_mb = 8192
    gpu.vram_free_mb = 7000

    info = SystemInfo()
    info.cpu = CPUInfo()
    info.cpu.logical_cores = 16
    info.cpu.physical_cores = 8
    info.gpu = [gpu]
    info.ram_total_gb = 32

    config = recommend_config(info, "8B", "code", "balanced")
    assert config["n_ctx"] >= 8192


def test_recommend_models():
    info = SystemInfo()
    info.cpu = CPUInfo()
    info.cpu.logical_cores = 8
    info.cpu.physical_cores = 4
    info.ram_total_gb = 16

    models = recommend_models(info, "chat", top=5)
    assert isinstance(models, list)
    assert len(models) > 0
    for m in models:
        assert "model" in m
        assert "score" in m
        assert "reason" in m


def test_determine_max_params_with_gpu():
    from llmtuner.core.recommender import _determine_max_params
    info = SystemInfo()
    gpu = GPUInfo()
    gpu.vram_total_mb = 24000
    info.gpu = [gpu]
    assert _determine_max_params(info) == "70B"


def test_determine_max_params_low_vram():
    from llmtuner.core.recommender import _determine_max_params
    info = SystemInfo()
    gpu = GPUInfo()
    gpu.vram_total_mb = 4096
    info.gpu = [gpu]
    assert _determine_max_params(info) == "4B"


def test_determine_max_params_cpu_only():
    from llmtuner.core.recommender import _determine_max_params
    info = SystemInfo()
    info.ram_total_gb = 32
    assert _determine_max_params(info) == "16B"
