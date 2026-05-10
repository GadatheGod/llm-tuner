# Quantization size multipliers relative to parameter count.
# Values are empirically derived from actual GGUF file sizes.
QUANT_SIZES = {
    "q2_k": 0.22,
    "q3_k_m": 0.28,
    "q4_0": 0.35,
    "q4_k_m": 0.40,
    "q4_k_s": 0.38,
    "q5_0": 0.48,
    "q5_k_m": 0.52,
    "q5_k_s": 0.50,
    "q6_k": 0.60,
    "q8_0": 0.80,
    "f16": 1.60,
}

# KV cache scale factors relative to q4_0 (4-bit) cache.
# These represent (cache_bits / 4.0).
KV_SCALE = {
    "q2_k": 0.25, "q3_k_m": 0.75, "q4_k_m": 1.0, "q4_0": 1.0,
    "q5_k_m": 1.25, "q5_k_s": 1.25, "q5_0": 1.25,
    "q6_k": 2.0, "q8_0": 2.0, "f16": 4.0,
}

# Context size tiers for use case recommendations.
CONTEXT_MAP = {
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "extreme": 32768,
}
