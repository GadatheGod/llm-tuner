from typing import Dict, Optional, List
from llmtuner.core.system_info import SystemInfo, GPUInfo, CPUInfo


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
    gpus = system.gpu if system.gpu else []

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
        "gpu_count": len(gpus),
        "gpu_names": [g.model for g in gpus] if gpus else [],
    }

    if gpus:
        config = _configure_gpu(system, gpus, params_raw, config, profile)
    else:
        config = _configure_cpu(system.cpu, params_raw, config)

    config = _apply_profile(config, profile, params_raw, use_case)
    return config


def _configure_gpu(system: SystemInfo, gpus: List[GPUInfo], params_raw: float, config: Dict, profile: str) -> Dict:
    total_vram_mb = sum(g.vram_total_mb for g in gpus if g.vram_total_mb)
    if not total_vram_mb:
        return config

    best_gpu = max(gpus, key=lambda g: g.vram_total_mb or 0)

    if profile == "optimum":
        vram_budget = total_vram_mb * 0.85
    elif profile == "max_performance":
        vram_budget = total_vram_mb * 0.95
    else:
        vram_budget = total_vram_mb * 0.90

    config["n_gpu_layers"] = 99

    quant_multiplier = 0.4
    model_size_mb = (params_raw * quant_multiplier / 1e9) * 1024

    if model_size_mb > vram_budget:
        for q, mult in sorted(QUANT_SIZES.items(), key=lambda x: x[1]):
            size = (params_raw * mult / 1e9) * 1024
            if size <= vram_budget:
                config["quantization"] = q
                model_size_mb = size
                break
        else:
            config["quantization"] = "q3_k_m"
            model_size_mb = (params_raw * 0.28 / 1e9) * 1024
    elif profile == "max_performance":
        config["quantization"] = "q6_k"
        model_size_mb = (params_raw * 0.6 / 1e9) * 1024
    else:
        config["quantization"] = "q5_k_m" if model_size_mb * 1.3 <= vram_budget else "q4_k_m"

    if best_gpu.vendor == "NVIDIA" and best_gpu.compute_capability != "N/A":
        try:
            cc = float(best_gpu.compute_capability)
            config["flash_attention"] = cc >= 8.0
        except ValueError:
            pass

    kv_cache_mb_per_token = (params_raw / 1e9) * 2.0 * 8 / 100
    remaining_vram = vram_budget - model_size_mb
    if kv_cache_mb_per_token > 0 and remaining_vram > 0:
        max_ctx_tokens = int(remaining_vram / kv_cache_mb_per_token)
    else:
        max_ctx_tokens = 8192

    max_ctx_tokens = max(2048, min(max_ctx_tokens, 32768))

    if profile == "max_performance":
        config["n_ctx"] = max(4096, _next_power_of_2(max_ctx_tokens))
    elif profile == "optimum":
        config["n_ctx"] = min(8192, max(4096, _next_power_of_2(max_ctx_tokens)))
    else:
        config["n_ctx"] = max(4096, _next_power_of_2(max_ctx_tokens))

    config["n_batch"] = min(config["n_ctx"], 4096 if profile == "max_performance" else 2048)
    return config


def _configure_cpu(cpu: CPUInfo, params_raw: float, config: Dict) -> Dict:
    config["n_gpu_layers"] = 0
    config["n_threads"] = max(2, cpu.physical_cores)
    config["n_threads_batch"] = max(2, cpu.logical_cores)
    config["n_ctx"] = 4096
    config["n_batch"] = 1024

    if params_raw > 8e9:
        config["quantization"] = "q3_k_m"
    else:
        config["quantization"] = "q4_k_m"
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
    elif use_case in ("roleplay", "creative_writing"):
        config["n_ctx"] = max(config["n_ctx"], 8192)
    return config


def _upgrade_quant(q: str) -> str:
    ladder = ["q3_k_m", "q4_0", "q4_k_m", "q5_0", "q5_k_m", "q6_k", "q8_0"]
    for i, step in enumerate(ladder):
        if q == step and i < len(ladder) - 1:
            return ladder[i + 1]
    return q


def _parse_params(params: str) -> float:
    if not params:
        return 8e9
    import re
    match = re.search(r'(\d+\.?\d*)', params)
    if not match:
        return 8e9
    val = float(match.group(1))
    if "b" in params.lower():
        return val * 1e9
    elif "m" in params.lower():
        return val * 1e6
    return val * 1e9


def _next_power_of_2(n: int) -> int:
    n = max(256, n)
    p = 256
    while p < n:
        p *= 2
    return p


def estimate_tok_per_sec(system: SystemInfo, model_params: str = "8B", quant: str = "q4_k_m") -> str:
    """Estimate tokens/sec based on GPU arch + model size + quantization."""
    gpus = system.gpu if system.gpu else []
    params_raw = _parse_params(model_params)
    quant_mult = QUANT_SIZES.get(quant, 0.4)
    effective_params = params_raw * quant_mult / 1e9

    # Multi-GPU scaling: approximate linear speedup up to 2 GPUs
    gpu_count = len(gpus) if gpus else 0

    if gpus and any(g.vram_total_mb > 0 for g in gpus):
        best_gpu = max(gpus, key=lambda g: g.vram_total_mb or 0)
        arch = best_gpu.architecture.lower()
        if "ada" in arch or "rtx 40" in best_gpu.model.lower():
            base = {"8": 120, "13": 80, "30": 40, "70": 18, "8x7": 35}
        elif "ampere" in arch or "rtx 30" in best_gpu.model.lower():
            base = {"8": 70, "13": 45, "30": 22, "70": 10, "8x7": 20}
        elif "turing" in arch or "rtx 20" in best_gpu.model.lower():
            base = {"8": 50, "13": 30, "30": 15, "70": 7, "8x7": 14}
        elif "pascal" in arch:
            base = {"8": 30, "13": 18, "30": 8, "70": 4, "8x7": 9}
        elif "hopper" in arch or "h100" in best_gpu.model.lower():
            base = {"8": 200, "13": 140, "30": 70, "70": 35, "8x7": 55}
        elif "rdna" in arch or "amd" in best_gpu.vendor.lower():
            base = {"8": 50, "13": 30, "30": 15, "70": 7, "8x7": 12}
        else:
            base = {"8": 40, "13": 25, "30": 12, "70": 6, "8x7": 10}

        size_key = str(int(effective_params)) if effective_params < 10 else (
            "30" if effective_params < 40 else "70"
        )
        if "8x" in best_gpu.model.lower() or "moe" in model_params.lower():
            size_key = "8x7"

        tok_s = base.get(size_key, base.get("8", 30))

        # Quantization speed factor (capped at 1.5x for q2_k)
        quant_speed = QUANT_SIZES.get(quant, 0.4)
        base_mult = 0.4
        if quant_speed > 0:
            adj = base_mult / quant_speed
            adj = min(adj, 1.5)  # cap: lower quant can't be infinitely faster
            tok_s = int(tok_s * adj)

        # Multi-GPU scaling factor
        if gpu_count > 1:
            tok_s = int(tok_s * (1 + (gpu_count - 1) * 0.5))

        high = tok_s + int(tok_s * 0.3)
        low = tok_s - int(tok_s * 0.3)
        return f"{max(1,low)}-{high} tok/s"
    else:
        cpu = system.cpu
        cores = cpu.physical_cores or 4
        quant_speed = QUANT_SIZES.get(quant, 0.4)
        base_mult = 0.4
        if quant_speed > 0:
            speed_adj = base_mult / quant_speed
        else:
            speed_adj = 1
        if effective_params < 10:
            base = max(2, cores * 0.8 * speed_adj)
        elif effective_params < 40:
            base = max(1, cores * 0.3 * speed_adj)
        else:
            base = max(1, cores * 0.15 * speed_adj)
        high = int(base * 1.5)
        low = int(base * 0.6)
        return f"{max(1,low)}-{high} tok/s (CPU)"


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
    models = _get_local_models(categories=[use_case], max_params=max_params, limit=top * 3)
    scored = []
    for m in models:
        score = _score_model(m, system, cat)
        scored.append({"model": m, "score": score, "reason": _explain_score(score, m, system)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top]


def _determine_max_params(system: SystemInfo) -> str:
    gpus = system.gpu if system.gpu else []
    if gpus:
        total_vram = sum(g.vram_total_mb for g in gpus if g.vram_total_mb)
        if total_vram >= 24000:
            return "70B"
        elif total_vram >= 16000:
            return "34B"
        elif total_vram >= 12000:
            return "16B"
        elif total_vram >= 8000:
            return "9B"
        elif total_vram >= 4000:
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
    gpus = system.gpu if system.gpu else []

    if gpus:
        total_vram_mb = sum(g.vram_total_mb for g in gpus if g.vram_total_mb)
        size = model.get("size_bytes", {}).get("q4_k_m", 0)
        if size and size > total_vram_mb * 1024 * 0.7:
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
    gpus = system.gpu if system.gpu else []

    if gpus:
        total_vram_mb = sum(g.vram_total_mb for g in gpus if g.vram_total_mb)
        size = model.get("size_bytes", {}).get("q4_k_m", 0)
        if size and size <= total_vram_mb * 1024 * 0.7:
            reasons.append("Fits in VRAM (Q4)")
        else:
            reasons.append("May need smaller quant or CPU offload")
    else:
        reasons.append("CPU inference - ensure enough RAM")

    if model.get("downloads", 0) > 100000:
        reasons.append("Popular model")

    return " | ".join(reasons)