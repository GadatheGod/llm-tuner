import pytest
import tempfile
import os
from llmtuner.utils.persistence import (
    get_prefs, save_prefs, get_pref, set_pref,
    get_history, add_history, clear_history, _ensure_dir
)


def test_save_and_load_prefs():
    save_prefs({"test_key": "test_value"})
    prefs = get_prefs()
    assert prefs.get("test_key") == "test_value"


def test_get_set_pref():
    set_pref("color", "blue")
    val = get_pref("color")
    assert val == "blue"


def test_get_pref_default():
    val = get_pref("nonexistent_key", "default_val")
    assert val == "default_val"


def test_prefs_merge():
    save_prefs({"key1": "value1"})
    save_prefs({"key2": "value2"})
    prefs = get_prefs()
    assert prefs.get("key1") == "value1"
    assert prefs.get("key2") == "value2"


def test_history_add():
    clear_history()
    add_history({"action": "test", "model": "test-model"})
    history = get_history()
    assert len(history) == 1
    assert history[0]["action"] == "test"
    assert "timestamp" in history[0]


def test_history_limit():
    clear_history()
    for i in range(150):
        add_history({"action": f"test_{i}"})
    history = get_history()
    assert len(history) <= 100


def test_clear_history():
    add_history({"action": "to_be_cleared"})
    clear_history()
    assert get_history() == []
