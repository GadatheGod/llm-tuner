import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


DATA_DIR = Path(os.path.expanduser("~/.llm-tuner"))
PREFS_FILE = DATA_DIR / "prefs.json"
HISTORY_FILE = DATA_DIR / "history.json"


def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load(filepath: Path, default: Any = None) -> Any:
    _ensure_dir()
    if not filepath.exists():
        return default if default is not None else {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default if default is not None else {}


def _save(filepath: Path, data: Any):
    _ensure_dir()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def get_prefs() -> Dict[str, Any]:
    return _load(PREFS_FILE, {})


def save_prefs(prefs: Dict[str, Any]):
    current = get_prefs()
    current.update(prefs)
    _save(PREFS_FILE, current)


def get_pref(key: str, default: Any = None) -> Any:
    prefs = get_prefs()
    return prefs.get(key, default)


def set_pref(key: str, value: Any):
    save_prefs({key: value})


def get_history() -> list:
    return _load(HISTORY_FILE, [])


def add_history(entry: Dict[str, Any]):
    history = get_history()
    entry["timestamp"] = _timestamp()
    history.insert(0, entry)
    if len(history) > 100:
        history = history[:100]
    _save(HISTORY_FILE, history)


def clear_history():
    _save(HISTORY_FILE, [])


def _timestamp() -> str:
    from datetime import datetime
    return datetime.now().isoformat()


def get_data_dir() -> Path:
    _ensure_dir()
    return DATA_DIR
