from typing import Dict, Optional, List
from llmtuner.core.system_info import SystemInfo, GPUInfo, CPUInfo
from llmtuner.core.constants import QUANT_SIZES, KV_SCALE, CONTEXT_MAP


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
        config = _configure_cpu(system, system.cpu, params_raw, config)

    remaining = config.pop("_remaining_vram_mb", 0)
    config = _apply_profile(config, profile, params_raw, use_case, remaining)
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
    model_size_mb = params_raw * quant_multiplier / (1024 * 1024)

    if model_size_mb > vram_budget:
        for q, mult in sorted(QUANT_SIZES.items(), key=lambda x: x[1]):
            size = params_raw * mult / (1024 * 1024)
            if size <= vram_budget:
                config["quantization"] = q
                model_size_mb = size
                break
        else:
            config["quantization"] = "q3_k_m"
            model_size_mb = params_raw * 0.28 / (1024 * 1024)
    elif profile == "max_performance":
        config["quantization"] = "q6_k"
        model_size_mb = params_raw * 0.6 / (1024 * 1024)
    else:
        config["quantization"] = "q5_k_m" if model_size_mb * 1.3 <= vram_budget else "q4_k_m"

    if best_gpu.vendor == "NVIDIA" and best_gpu.compute_capability != "N/A":
        try:
            cc = float(best_gpu.compute_capability)
            config["flash_attention"] = cc >= 8.0
        except ValueError:
            pass

    kv_cache_mb_per_token = _kv_cache_mb_per_token(params_raw)
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
    config["_remaining_vram_mb"] = remaining_vram
    return config


def _configure_cpu(system: SystemInfo, cpu: CPUInfo, params_raw: float, config: Dict) -> Dict:
    ram_gb = system.ram_total_gb
    config["n_gpu_layers"] = 0
    config["n_threads"] = max(2, cpu.physical_cores)
    config["n_threads_batch"] = max(2, cpu.logical_cores)
    config["n_ctx"] = 4096
    config["n_batch"] = 1024

    # RAM-based quantization selection for CPU inference
    ram_budget_mb = ram_gb * 1024 * 0.7  # reserve 30% for OS
    if params_raw > 8e9:
        for q, mult in sorted(QUANT_SIZES.items(), key=lambda x: x[1]):
            size_mb = params_raw * mult / (1024 * 1024)
            if size_mb <= ram_budget_mb:
                config["quantization"] = q
                break
        else:
            config["quantization"] = "q2_k"
    else:
        for q, mult in sorted(QUANT_SIZES.items(), key=lambda x: x[1]):
            size_mb = params_raw * mult / (1024 * 1024)
            if size_mb <= ram_budget_mb:
                config["quantization"] = q
                break
        else:
            config["quantization"] = "q2_k"
    return config


def _apply_profile(config: Dict, profile: str, params_raw: float, use_case: str,
                   remaining_vram_mb: float = 0) -> Dict:
    params_b = params_raw / 1e9

    if profile == "optimum":
        config["n_ctx"] = min(config.get("n_ctx", 8192), 16384)
        config["n_ctx"] = max(4096, config["n_ctx"])
        config["n_batch"] = max(2048, config["n_ctx"])
        config["n_predict"] = 128
    elif profile == "max_performance":
        config["n_ctx"] = min(config.get("n_ctx", 32768), 65536)
        config["n_ctx"] = max(8192, config["n_ctx"])
        config["n_batch"] = min(8192, config["n_ctx"])
        config["n_predict"] = 512
        config["quantization"] = _upgrade_quant(config["quantization"])
    else:
        config["n_ctx"] = min(config.get("n_ctx", 8192), 32768)
        config["n_ctx"] = max(4096, config["n_ctx"])
        config["n_batch"] = min(config["n_ctx"], 4096)

    if use_case == "code":
        config["n_ctx"] = max(config["n_ctx"], 8192)
    elif use_case == "rag":
        config["n_ctx"] = max(config["n_ctx"], 16384)
    elif use_case in ("roleplay", "creative_writing"):
        config["n_ctx"] = max(config["n_ctx"], 8192)

    # Allow up to 262144 (256k) — model may support it
    config["n_ctx"] = min(config["n_ctx"], 262144)

    # Clamp context to VRAM capacity if available
    if remaining_vram_mb > 0:
        kv_per_token = _kv_cache_mb_per_token(params_raw)
        if kv_per_token > 0:
            max_ctx_from_vram = int(remaining_vram_mb / kv_per_token)
            config["n_ctx"] = min(config["n_ctx"], max_ctx_from_vram, 262144)
            config["n_ctx"] = max(config["n_ctx"], 2048)

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


def _model_size_key(k: str) -> int:
    """Sort key for model size strings like '8', '70', '8x7'."""
    if "x" in k:
        return 999
    return int(k)


def _select_size_key(params_b: float, base_gen: dict) -> str:
    """Choose the appropriate size bucket from base_gen based on parameter count."""
    bk = int(params_b)
    for sk in sorted(base_gen.keys(), key=_model_size_key):
        if bk <= _model_size_key(sk):
            return sk
    return max(base_gen.keys(), key=_model_size_key)


ARCH_KV_DEFAULTS = {
    "llama3": {"n_layers": 32, "hidden_dim": 4096, "n_heads": 32, "n_kv_heads": 8, "mq_ratio": 0.25},
    "llama3.2": {"n_layers": 26, "hidden_dim": 3072, "n_heads": 16, "n_kv_heads": 8, "mq_ratio": 0.5},
    "llama3_2": {"n_layers": 26, "hidden_dim": 3072, "n_heads": 16, "n_kv_heads": 8, "mq_ratio": 0.5},
    "llama": {"n_layers": 32, "hidden_dim": 4096, "n_heads": 32, "n_kv_heads": 8, "mq_ratio": 0.25},
    "mistral": {"n_layers": 32, "hidden_dim": 4096, "n_heads": 32, "n_kv_heads": 8, "mq_ratio": 0.25},
    "gemma2": {"n_layers": 28, "hidden_dim": 3584, "n_heads": 16, "n_kv_heads": 8, "mq_ratio": 0.5},
    "gemma_2": {"n_layers": 28, "hidden_dim": 3584, "n_heads": 16, "n_kv_heads": 8, "mq_ratio": 0.5},
    "gemma": {"n_layers": 28, "hidden_dim": 3584, "n_heads": 16, "n_kv_heads": 8, "mq_ratio": 0.5},
    "phi3": {"n_layers": 24, "hidden_dim": 3072, "n_heads": 32, "n_kv_heads": 32, "mq_ratio": 1.0},
    "phi_3": {"n_layers": 24, "hidden_dim": 3072, "n_heads": 32, "n_kv_heads": 32, "mq_ratio": 1.0},
    "phi": {"n_layers": 24, "hidden_dim": 3072, "n_heads": 32, "n_kv_heads": 32, "mq_ratio": 1.0},
    "qwen2.5": {"n_layers": 28, "hidden_dim": 3584, "n_heads": 28, "n_kv_heads": 4, "mq_ratio": 0.143},
    "qwen2_5": {"n_layers": 28, "hidden_dim": 3584, "n_heads": 28, "n_kv_heads": 4, "mq_ratio": 0.143},
    "qwen2": {"n_layers": 28, "hidden_dim": 3584, "n_heads": 28, "n_kv_heads": 4, "mq_ratio": 0.143},
    "qwen": {"n_layers": 28, "hidden_dim": 3584, "n_heads": 28, "n_kv_heads": 4, "mq_ratio": 0.143},
    "deepseek2": {"n_layers": 27, "hidden_dim": 4096, "n_heads": 32, "n_kv_heads": 8, "mq_ratio": 0.25},
    "deepseek_2": {"n_layers": 27, "hidden_dim": 4096, "n_heads": 32, "n_kv_heads": 8, "mq_ratio": 0.25},
    "deepseek": {"n_layers": 27, "hidden_dim": 4096, "n_heads": 32, "n_kv_heads": 8, "mq_ratio": 0.25},
}


def _kv_cache_mb_per_token(params_raw: float, model_family: str = "",
                           n_layers: int = 0, hidden_dim: int = 0,
                           n_kv_heads: int = 0, n_heads: int = 0) -> float:
    """Calculate KV cache memory per token at q4_0 quantization.
    
    Formula: ctx_tokens * 2 * n_layers * (n_kv_heads * head_dim) * 0.5 bytes
    Simplified: params_B * (n_layers / params_B) * hidden_dim * (n_kv_heads / n_heads) * 0.5
    
    At 1 token context, this gives MB/token for q4_0 cache.
    """
    params_b = params_raw / 1e9

    if n_layers and hidden_dim and n_kv_heads and n_heads:
        head_dim = hidden_dim / n_heads
        kv_per_token_bytes = 2 * n_layers * n_kv_heads * head_dim * 0.5
        return kv_per_token_bytes / (1024 * 1024)

    if model_family:
        fam = model_family.lower().replace(" ", "").replace("-", "")
        best_match = None
        best_len = 0
        for key, arch in ARCH_KV_DEFAULTS.items():
            if key in fam or fam in key:
                if len(key) > best_len:
                    best_len = len(key)
                    best_match = arch
        # Also try without dots (e.g. "llama-3.2" -> "llama32" should match "llama3.2")
        fam_dots = fam.replace(".", "")
        for key, arch in ARCH_KV_DEFAULTS.items():
            if (key in fam_dots or fam_dots in key) and len(key) > best_len:
                best_len = len(key)
                best_match = arch
        if best_match:
            n_layers = best_match["n_layers"]
            hidden_dim = best_match["hidden_dim"]
            n_heads = best_match["n_heads"]
            n_kv_heads = best_match["n_kv_heads"]

    if not n_layers:
        if params_b <= 0.5:
            n_layers, hidden_dim = 16, 2048
            n_heads, n_kv_heads = 16, 8
        elif params_b <= 1.5:
            n_layers, hidden_dim = 24, 2560
            n_heads, n_kv_heads = 16, 8
        elif params_b <= 3.5:
            n_layers, hidden_dim = 28, 3200
            n_heads, n_kv_heads = 32, 8
        elif params_b <= 7.5:
            n_layers, hidden_dim = 32, 4096
            n_heads, n_kv_heads = 32, 8
        elif params_b <= 10.5:
            n_layers, hidden_dim = 36, 4096
            n_heads, n_kv_heads = 32, 8
        elif params_b <= 14:
            n_layers, hidden_dim = 40, 5120
            n_heads, n_kv_heads = 40, 8
        elif params_b <= 22:
            n_layers, hidden_dim = 48, 6144
            n_heads, n_kv_heads = 48, 8
        elif params_b <= 40:
            n_layers, hidden_dim = 48, 8192
            n_heads, n_kv_heads = 64, 8
        elif params_b <= 100:
            n_layers, hidden_dim = 80, 8192
            n_heads, n_kv_heads = 64, 8
        else:
            n_layers, hidden_dim = 96, 12288
            n_heads, n_kv_heads = 96, 8

    if not n_heads:
        n_heads = n_layers

    if not n_kv_heads:
        n_kv_heads = max(1, n_heads // 8)

    head_dim = hidden_dim / n_heads
    kv_per_token_bytes = 2 * n_layers * n_kv_heads * head_dim * 0.5
    return kv_per_token_bytes / (1024 * 1024)


def estimate_tok_per_sec(system: SystemInfo, model_params: str = "8B", weight_quant: str = "q4_k_m", kv_cache_type: str = "q4_0") -> dict:
    """Estimate prompt eval + generation tok/s based on GPU arch + model size + quantization.
    Returns dict with 'prompt', 'gen', and 'combined' strings.
    Speed depends on WEIGHT quant, not cache type. Cache type only affects memory."""
    gpus = system.gpu if system.gpu else []
    params_raw = _parse_params(model_params)
    params_b = params_raw / 1e9

    # Weight quant affects speed; cache quant does not
    weight_mult = QUANT_SIZES.get(weight_quant, 0.4)
    effective_params = params_b * weight_mult

    gpu_count = len(gpus) if gpus else 0

    if gpus and any(g.vram_total_mb > 0 for g in gpus):
        best_gpu = max(gpus, key=lambda g: g.vram_total_mb or 0)
        arch = best_gpu.architecture.lower()
        model_name = best_gpu.model.lower()

        # Base generation tok/s per GPU for q4 weights (calibrated from real benchmarks)
        # Your data: 27B q4 on dual 3090 = 27.56 tok/s → single = ~18 tok/s
        if "ada" in arch or "rtx 40" in model_name:
            base_gen = {"8": 110, "13": 72, "27": 55, "30": 45, "70": 20, "8x7": 38}
            base_prompt = {"8": 800, "13": 500, "27": 350, "30": 280, "70": 130, "8x7": 250}
        elif "ampere" in arch or "rtx 30" in model_name:
            base_gen = {"8": 65, "13": 40, "27": 18, "30": 18, "70": 10, "8x7": 20}
            base_prompt = {"8": 500, "13": 300, "27": 180, "30": 160, "70": 80, "8x7": 120}
        elif "turing" in arch or "rtx 20" in model_name:
            base_gen = {"8": 45, "13": 28, "27": 13, "30": 13, "70": 7, "8x7": 12}
            base_prompt = {"8": 350, "13": 200, "27": 100, "30": 100, "70": 50, "8x7": 80}
        elif "pascal" in arch:
            base_gen = {"8": 25, "13": 16, "27": 8, "30": 8, "70": 4, "8x7": 8}
            base_prompt = {"8": 200, "13": 120, "27": 55, "30": 55, "70": 25, "8x7": 45}
        elif "hopper" in arch or "h100" in model_name:
            base_gen = {"8": 190, "13": 130, "27": 95, "30": 80, "70": 40, "8x7": 60}
            base_prompt = {"8": 1500, "13": 1000, "27": 700, "30": 600, "70": 300, "8x7": 450}
        elif "rdna" in arch or best_gpu.vendor == "AMD":
            base_gen = {"8": 45, "13": 28, "27": 12, "30": 12, "70": 6, "8x7": 10}
            base_prompt = {"8": 300, "13": 180, "27": 90, "30": 90, "70": 40, "8x7": 60}
        else:
            base_gen = {"8": 35, "13": 22, "27": 10, "30": 10, "70": 5, "8x7": 9}
            base_prompt = {"8": 250, "13": 150, "27": 70, "30": 70, "70": 30, "8x7": 50}

        # Size key
        if "moe" in model_params.lower() or "8x" in model_params.lower():
            size_key = "8x7"
        else:
            size_key = _select_size_key(params_b, base_gen)

        gen_tok_s = base_gen.get(size_key, 30)
        prompt_tok_s = base_prompt.get(size_key, 200)

        # Weight quant speed factor (q2 faster, q8 slower than q4 baseline)
        q_adj = 0.4 / max(weight_mult, 0.22)
        q_adj = min(q_adj, 1.5)  # cap low-quant speedup
        gen_tok_s = int(gen_tok_s * q_adj)
        prompt_tok_s = int(prompt_tok_s * q_adj)

        # Multi-GPU scaling (not linear — ~50% per extra GPU)
        if gpu_count > 1:
            scale = 1 + (gpu_count - 1) * 0.5
            gen_tok_s = int(gen_tok_s * scale)
            prompt_tok_s = int(prompt_tok_s * scale)

        # Long context penalty on generation (~5-10% slowdown at 128k+)
        # (not exposed here — user sets context in config panel)

        high_gen = gen_tok_s + int(gen_tok_s * 0.2)
        low_gen = gen_tok_s - int(gen_tok_s * 0.2)
        high_pr = prompt_tok_s + int(prompt_tok_s * 0.2)
        low_pr = prompt_tok_s - int(prompt_tok_s * 0.2)

        return {
            "gen": f"{max(1,low_gen)}-{high_gen} tok/s",
            "prompt": f"{max(1,low_pr)}-{high_pr} tok/s",
            "combined": f"gen {max(1,low_gen)}-{high_gen} | prompt {max(1,low_pr)}-{high_pr} tok/s"
        }
    else:
        cpu = system.cpu
        cores = cpu.physical_cores or 4
        q_adj = 0.4 / max(weight_mult, 0.22)
        if effective_params < 10:
            base_g = max(2, cores * 0.8 * q_adj)
        elif effective_params < 40:
            base_g = max(1, cores * 0.3 * q_adj)
        else:
            base_g = max(1, cores * 0.15 * q_adj)
        base_p = base_g * 1.2
        high_g = int(base_g * 1.4)
        low_g = int(base_g * 0.6)
        high_p = int(base_p * 1.4)
        low_p = int(base_p * 0.6)
        return {
            "gen": f"{max(1,low_g)}-{high_g} tok/s (CPU)",
            "prompt": f"{max(1,low_p)}-{high_p} tok/s (CPU)",
            "combined": f"gen {max(1,low_g)}-{high_g} | prompt {max(1,low_p)}-{high_p} tok/s (CPU)"
        }


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
        if size and size > total_vram_mb * 1024 * 1024 * 0.7:
            score -= 30
    else:
        ram = system.ram_total_gb
        size = model.get("size_bytes", {}).get("q4_k_m", 0)
        if size and size > ram * 1024 * 1024 * 1024 * 0.6:
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
        if size and size <= total_vram_mb * 1024 * 1024 * 0.7:
            reasons.append("Fits in VRAM (Q4)")
        else:
            reasons.append("May need smaller quant or CPU offload")
    else:
        reasons.append("CPU inference - ensure enough RAM")

    if model.get("downloads", 0) > 100000:
        reasons.append("Popular model")

    return " | ".join(reasons)