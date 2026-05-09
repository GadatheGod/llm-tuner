from typing import Dict, Optional, List
from llmtuner.core.system_info import SystemInfo, GPUInfo


QUANT_SIZES = {
    "q2_k": 0.22,
    "q3_k_m": 0.28,
    "q4_0": 0.35,
    "q4_k_m": 0.4,
    "q4_k_s": 0.38,
    "q5_0": 0.48,
    "q5_k_m": 0.52,
    "q5_k_s": 0.50,
    "q6_k": 0.6,
    "q8_0": 0.8,
    "f16": 1.6,
}


CONTEXT_MAP = {
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "extreme": 32768,
}


def recommend_config(
    system: SystemInfo,
    model_params: str = "8B",
    use_case: str = "chat",
    profile: str = "balanced"
) -> Dict:
    params_raw = _parse_params(model_params)
    gpu = system.gpu[0] if system.gpu else None

    config = {
        "profile": profile,
        "model_params": model_params,
        "use_case": use_case,
        "n_gpu_layers": 0,
        "n_ctx": 4096,
        "n_batch": 2048,
        "n_threads": system.cpu.logical_cores,
        "n_threads_batch": system.cpu.logical_cores,
        "n_predict": 256,
        "quantization": "q4_k_m",
        "flash_attention": True,
        "mmap": True,
        "multiproc": False,
        "cache_type_k": "q4_0",
        "cache_type_v": "q4_0",
    }

    if gpu:
        config = _configure_gpu(gpu, params_raw, config, profile)
    else:
        config = _configure_cpu(system.cpu, params_raw, config)

    config = _apply_profile(config, profile, params_raw, use_case)
    return config


def _configure_gpu(gpu: GPUInfo, params_raw: float, config: Dict, profile: str) -> Dict:
    vram_mb = gpu.vram_free_mb or gpu.vram_total_mb
    if not vram_mb:
        return _configure_cpu(config.get("cpu"), params_raw, config)

    model_size_gb = params_raw * 0.4 / 1e9
    model_size_mb = model_size_gb * 1024

    if profile == "optimum":
        vram_budget = vram_mb * 0.85
    elif profile == "max_performance":
        vram_budget = vram_mb * 0.95
    else:
        vram_budget = vram_mb * 0.90

    config["n_gpu_layers"] = -1

    if gpu.vendor == "NVIDIA":
        if gpu.compute_capability != "N/A":
            cc = float(gpu.compute_capability.replace(".", ""))
            config["flash_attention"] = cc >= 80

    if model_size_mb * 0.5 > vram_budget:
        config["quantization"] = "q4_k_m"
    elif model_size_mb * 0.4 > vram_budget:
        config["quantization"] = "q3_k_m"
    else:
        config["quantization"] = "q5_k_m" if profile != "optimum" else "q4_k_m"

    ctx_mb_per_token = params_raw * 0.000001
    max_ctx_tokens = (vram_budget - model_size_mb * 0.5) / ctx_mb_per_token
    max_ctx_tokens = max(2048, min(int(max_ctx_tokens), 32768))

    if profile == "max_performance":
        config["n_ctx"] = max(4096, _next_power_of_2(max_ctx_tokens))
    elif profile == "optimum":
        config["n_ctx"] = min(8192, _next_power_of_2(max_ctx_tokens))
    else:
        config["n_ctx"] = _next_power_of_2(max_ctx_tokens)

    config["n_batch"] = min(config["n_ctx"], 4096 if profile == "max_performance" else 2048)
    return config


def _configure_cpu(cpu, params_raw: float, config: Dict) -> Dict:
    config["n_gpu_layers"] = 0
    config["n_threads"] = max(2, cpu.physical_cores)
    config["n_threads_batch"] = max(2, cpu.logical_cores)
    config["n_ctx"] = 4096
    config["n_batch"] = 1024
    config["quantization"] = "q4_k_m"

    ram_gb = cpu.physical_cores * 2 if hasattr(cpu, 'physical_cores') else 4
    if params_raw > 8e9:
        config["quantization"] = "q3_k_m"
    return config


def _apply_profile(config: Dict, profile: str, params_raw: float, use_case: str) -> Dict:
    if profile == "optimum":
        config["n_batch"] = max(2048, config.get("n_ctx", 4096))
        config["n_predict"] = 128
        if config["n_ctx"] > 8192:
            config["n_ctx"] = 8192
    elif profile == "max_performance":
        config["n_batch"] = min(8192, config.get("n_ctx", 8192))
        config["n_predict"] = 512
        config["quantization"] = _upgrade_quant(config["quantization"])
    else:
        pass

    if use_case == "code":
        config["n_ctx"] = max(config["n_ctx"], 8192)
    elif use_case == "rag":
        config["n_ctx"] = max(config["n_ctx"], 16384)
    return config


def _upgrade_quant(q: str) -> str:
    ladder = ["q3_k_m", "q4_0", "q4_k_m", "q5_0", "q5_k_m", "q6_k", "q8_0"]
    for i, step in enumerate(ladder):
        if q == step and i < len(ladder) - 1:
            return ladder[i + 1]
    return q


def _parse_params(params: str) -> float:
    if "b" in params.lower():
        return float(params.replace("B", "").replace("b", "")) * 1e9
    elif "m" in params.lower():
        return float(params.replace("M", "").replace("m", "")) * 1e6
    return int(params) * 1e9


def _next_power_of_2(n: int) -> int:
    n = max(256, n)
    p = 256
    while p < n:
        p *= 2
    return p


def recommend_models(
    system: SystemInfo,
    use_case: str = "chat",
    top: int = 5
) -> List[Dict]:
    from llmtuner.core.model_db import _get_local_models, get_use_case_categories

    categories = get_use_case_categories()
    cat_map = {c["id"]: c for c in categories}
    cat = cat_map.get(use_case, {})

    max_params = _determine_max_params(system)
    models = _get_local_models(categories=[use_case], max_params=max_params, limit=top * 2)
    scored = []
    for m in models:
        score = _score_model(m, system, cat)
        scored.append({"model": m, "score": score, "reason": _explain_score(score, m, system)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top]


def _determine_max_params(system: SystemInfo) -> str:
    gpu = system.gpu[0] if system.gpu else None
    if gpu:
        vram = gpu.vram_total_mb
        if vram >= 24000:
            return "70B"
        elif vram >= 12000:
            return "16B"
        elif vram >= 6000:
            return "9B"
        elif vram >= 6000:
            return "8B"
        else:
            return "4B"
    else:
        ram = system.ram_total_gb
        if ram >= 64:
            return "34B"
        elif ram >= 32:
            return "16B"
        elif ram >= 16:
            return "9B"
        else:
            return "4B"


def _score_model(model: Dict, system: SystemInfo, cat: Dict) -> float:
    score = 100
    gpu = system.gpu[0] if system.gpu else None

    if gpu:
        vram = gpu.vram_total_mb
        size = model.get("size_bytes", {}).get("q4_k_m", 0)
        if size and size > vram * 1024 * 0.7:
            score -= 30
    else:
        ram = system.ram_total_gb
        size = model.get("size_bytes", {}).get("q4_k_m", 0)
        if size and size > ram * 1024 * 1024 * 0.6:
            score -= 25

    if model.get("downloads", 0) > 100000:
        score += 10
    elif model.get("downloads", 0) > 50000:
        score += 5

    tags = model.get("tags", [])
    cat_tags = cat.get("tags", [])
    overlap = len(set(tags) & set(cat_tags))
    score += overlap * 5

    return max(0, score)


def _explain_score(score: float, model: Dict, system: SystemInfo) -> str:
    reasons = []
    gpu = system.gpu[0] if system.gpu else None

    if gpu:
        vram = gpu.vram_total_mb
        size = model.get("size_bytes", {}).get("q4_k_m", 0)
        if size <= vram * 1024 * 0.7:
            reasons.append("Fits in VRAM (Q4)")
        else:
            reasons.append("May need smaller quant or CPU offload")
    else:
        reasons.append("CPU inference - ensure enough RAM")

    if model.get("downloads", 0) > 100000:
        reasons.append("Popular model")

    return " | ".join(reasons)
