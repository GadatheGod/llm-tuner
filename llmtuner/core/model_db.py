import json
import os
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
    if use_cache:
        cached = _get_cached()
        if cached:
            return _filter_models(cached, query, categories, max_params, limit)

    try:
        search_query = query.strip() or "llama OR mistral OR qwen OR gemma OR phi"
        url = "https://huggingface.co/api/models"
        params = {
            "sort": "downloads",
            "direction": "-1",
            "limit": limit * 2,
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


def _process_hf_model(model: Dict) -> Dict:
    return {
        "id": model.get("id", ""),
        "name": model.get("id", "").split("/")[-1],
        "tags": model.get("tags", []),
        "downloads": model.get("downloads", 0),
        "likes": model.get("likes", 0),
        "pipeline_tag": model.get("pipeline_tag", ""),
        "hf_url": f"https://huggingface.co/{model.get('id', '')}",
    }


def _filter_models(
    models: List[Dict],
    query: str,
    categories: Optional[List[str]] = None,
    max_params: str = "70B",
    limit: int = 20
) -> List[Dict]:
    results = []
    q = query.lower()
    for m in models:
        if q and q not in m.get("name", "").lower() and q not in m.get("id", "").lower():
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

    max_params_raw = int(max_params.replace("B", "").replace(".5", "/2*10").replace(".", "")) * 100000000
    if "B" in max_params:
        max_params_raw = float(max_params.replace("B", "")) * 1000000000

    results = []
    for m in models:
        if categories and not any(c in m.get("categories", []) for c in categories):
            continue
        if query and query.lower() not in m.get("name", "").lower():
            continue
        if m.get("params_raw", 0) > max_params_raw:
            continue
        results.append({
            "id": m["id"],
            "name": m["name"],
            "family": m.get("family", ""),
            "params": m.get("params", ""),
            "quantizations": m.get("quantizations", []),
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
    return None
