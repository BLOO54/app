"""Remembers your choices (last model, size, safe mode...) between runs."""

from __future__ import annotations

import json
import threading

from . import paths

DEFAULTS = {
    "model": "juggernaut",
    "aspect": "square",
    "quality": "balanced",
    "style": "none",
    "count": 1,
    "safe_mode": True,
    "negative": "",
    "device": "",          # "cuda" or "cpu" - decided when the AI engine is installed
}

_lock = threading.Lock()


def _file():
    return paths.data_dir() / "settings.json"


def load() -> dict:
    data = dict(DEFAULTS)
    try:
        data.update(json.loads(_file().read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return data


def update(**changes) -> dict:
    with _lock:
        data = load()
        data.update({k: v for k, v in changes.items() if k in DEFAULTS})
        tmp = _file().with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(_file())
        return data
