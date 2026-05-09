import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

import requests

CACHE_DIR = Path(os.path.expanduser("~/.llm-tuner/cache"))
CACHE_FILE = CACHE_DIR / "hf_models.json"
CACHE_TTL = 3600


def _get_cached() -> Optional[List[Dict]]:
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text())
            if time.time() - data.get("timestamp", 0) < CACHE_TTL:
                return data.get("models", [])
        except Exception:
            pass
    return None


def _cache_results(models: List[Dict]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {"timestamp": time.time(), "models": models}
    CACHE_FILE.write_text(json.dumps(data))


def search_models(
    query: str = "",
    categories: Optional[List[str]] = None,
    max_params: str = "70B",
    limit: int = 20,
    use_cache: bool = True
) -> List[Dict]:
    # Only use cache for broad searches (no specific query typed)
    if use_cache and not query.strip():
        cached = _get_cached()
        if cached:
            return _filter_models(cached, query, categories, max_params, limit)

    try:
        search_query = query.strip() or "llama OR mistral OR qwen OR gemma OR phi"
        url = "https://huggingface.co/api/models"
        params = {
            "sort": "downloads",
            "direction": "-1",
            "limit": limit * 3,
            "search": search_query,
            "pipeline_tag": "text-generation",
            "tags": "gguf",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        models = resp.json()
        processed = [_process_hf_model(m) for m in models]
        _cache_results(processed)
        return _filter_models(processed, query, categories, max_params, limit)
    except Exception:
        return _get_local_models(query, categories, max_params, limit)


def _infer_params_from_id(model_id: str) -> str:
    match = re.search(r'(\d+\.?\d*)[Bb]', model_id)
    if match:
        val = match.group(1)
        if "." in val:
            return f"{val}B"
        return f"{int(float(val))}B"
    match = re.search(r'(\d+\.?\d*)[Mm]', model_id)
    if match:
        return f"{match.group(1)}M"
    return ""


def _infer_family(model_id: str) -> str:
    id_lower = model_id.lower()
    for family in ["llama-3", "llama-2", "qwen2.5-coder", "qwen2.5", "qwen2", "qwen", "mistral", "phi-3", "phi-2", "phi-1", "gemma-2", "gemma", "deepseek", "yi", "nomic", "codellama", "codestral"]:
        if family in id_lower:
            return family.replace("-", " ").title().replace(".", "")
    return id_lower.split("/")[0].split("-")[0].title() if "/" in model_id else id_lower.split("-")[0].title()


def _process_hf_model(model: Dict) -> Dict:
    model_id = model.get("id", "")
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    params = _infer_params_from_id(model_id) or _infer_params_from_id(name)

    tags = model.get("tags", [])
    quantizations = []
    if "q4_k_m" in tags or "q4" in tags or "gguf" in tags:
        quantizations = ["q4_k_m", "q5_k_m", "q8_0"]
    if "q5" in tags:
        quantizations.append("q5_k_m")

    card_data = model.get("cardData", {})
    context_length = card_data.get("max_model_length", 4096)
    if isinstance(context_length, str):
        try:
            context_length = int(context_length.replace(",", "").replace(" ", ""))
        except ValueError:
            context_length = 4096

    family = _infer_family(model_id)

    params_raw = _parse_param_value(params)

    # Calculate size_bytes per quantization level
    quant_multipliers = {"q2_k": 0.22, "q3_k_m": 0.28, "q4_k_m": 0.4, "q5_k_m": 0.52, "q6_k": 0.6, "q8_0": 0.8, "f16": 1.6}
    size_bytes = {}
    for q, mult in quant_multipliers.items():
        size_bytes[q] = int(params_raw * mult)

    return {
        "id": model_id,
        "name": name,
        "family": family,
        "params": params,
        "params_raw": params_raw,
        "size_bytes": size_bytes,
        "tags": tags,
        "categories": _tags_to_categories(tags),
        "downloads": model.get("downloads", 0),
        "likes": model.get("likes", 0),
        "pipeline_tag": model.get("pipeline_tag", ""),
        "quantizations": quantizations if quantizations else ["q4_k_m", "q5_k_m", "q8_0"],
        "context_default": context_length if isinstance(context_length, int) and context_length > 0 else 4096,
        "hf_url": f"https://huggingface.co/{model_id}",
        "hf_repo": model_id,
    }


def _tags_to_categories(tags: List[str]) -> List[str]:
    category_map = {
        "code": ["code", "programming"],
        "translation": ["translation", "multilingual"],
        "text-generation": ["chat", "creative_writing"],
    }
    cats = set()
    for tag in tags:
        tag_lower = tag.lower()
        if "code" in tag_lower:
            cats.add("code")
        if "translate" in tag_lower or "multilingual" in tag_lower:
            cats.add("translation")
        if "text-generation" in tag_lower:
            cats.add("chat")
        if "summarization" in tag_lower:
            cats.add("summarization")
        if "rag" in tag_lower or "retrieval" in tag_lower:
            cats.add("rag")
        if "vision" in tag_lower or "image" in tag_lower:
            cats.add("vision")
        if "math" in tag_lower:
            cats.add("math")
    return list(cats) if cats else ["chat"]


def _parse_param_value(params: str) -> int:
    if not params:
        return 0
    match = re.search(r'(\d+\.?\d*)', params)
    if not match:
        return 0
    val = float(match.group(1))
    if "B" in params.upper():
        return int(val * 1e9)
    elif "M" in params.upper():
        return int(val * 1e6)
    return int(val)


def _filter_models(
    models: List[Dict],
    query: str,
    categories: Optional[List[str]] = None,
    max_params: str = "70B",
    limit: int = 20
) -> List[Dict]:
    max_params_raw = _parse_param_value(max_params)
    results = []
    q = query.lower()
    for m in models:
        if q and q not in m.get("name", "").lower() and q not in m.get("id", "").lower():
            continue
        if categories:
            model_cats = m.get("categories", [])
            if not any(c in model_cats for c in categories):
                continue
        model_params_raw = m.get("params_raw", 0)
        if max_params_raw > 0 and model_params_raw > max_params_raw:
            continue
        results.append(m)
        if len(results) >= limit:
            break
    return results


def _get_local_models(
    query: str = "",
    categories: Optional[List[str]] = None,
    max_params: str = "70B",
    limit: int = 20
) -> List[Dict]:
    data_path = Path(__file__).parent.parent / "data" / "models.json"
    if not data_path.exists():
        return []
    data = json.loads(data_path.read_text())
    models = data.get("models", [])

    max_params_raw = _parse_param_value(max_params)

    results = []
    for m in models:
        if categories and not any(c in m.get("categories", []) for c in categories):
            continue
        if query and query.lower() not in m.get("name", "").lower():
            continue
        if max_params_raw > 0 and m.get("params_raw", 0) > max_params_raw:
            continue
        results.append({
            "id": m["id"],
            "name": m["name"],
            "family": m.get("family", ""),
            "params": m.get("params", ""),
            "params_raw": m.get("params_raw", 0),
            "quantizations": m.get("quantizations", ["q4_k_m"]),
            "size_bytes": m.get("size_bytes", {}),
            "context_default": m.get("context_default", 4096),
            "categories": m.get("categories", []),
            "hf_repo": m.get("hf_repo", m["id"]),
            "downloads": 0,
            "likes": 0,
            "pipeline_tag": "text-generation",
            "tags": m.get("categories", []),
            "hf_url": f"https://huggingface.co/{m.get('hf_repo', m['id'])}",
        })
        if len(results) >= limit:
            break
    return results


def get_use_case_categories() -> List[Dict]:
    data_path = Path(__file__).parent.parent / "data" / "use_cases.json"
    if not data_path.exists():
        return []
    data = json.loads(data_path.read_text())
    return data.get("categories", [])


def get_model_details(model_id: str) -> Optional[Dict]:
    data_path = Path(__file__).parent.parent / "data" / "models.json"
    if not data_path.exists():
        return None
    data = json.loads(data_path.read_text())
    for m in data.get("models", []):
        if m["id"] == model_id or m.get("hf_repo") == model_id:
            return m
        if model_id.lower() in m["id"].lower() or m["id"].lower() in model_id.lower():
            return m
    return None