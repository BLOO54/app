"""Keeps a list of the pictures you made (newest first) next to the pictures."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from . import paths

_lock = threading.Lock()


def _index() -> Path:
    return paths.pictures_dir() / "gallery.jsonl"


def add(entry: dict) -> None:
    with _lock, open(_index(), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def items(limit: int = 300) -> list[dict]:
    out = []
    try:
        lines = _index().read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    pics = paths.pictures_dir()
    for line in reversed(lines):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if (pics / e.get("file", "")).is_file():
            out.append(e)
        if len(out) >= limit:
            break
    return out


def safe_name(name: str) -> str | None:
    """Only plain file names inside the pictures folder (no ../ tricks)."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        return None
    if not name.lower().endswith(".png"):
        return None
    return name


def delete(name: str) -> bool:
    name = safe_name(name)
    if not name:
        return False
    with _lock:
        p = paths.pictures_dir() / name
        try:
            p.unlink()
        except OSError:
            return False
        try:
            lines = _index().read_text(encoding="utf-8").splitlines()
            keep = [l for l in lines if f'"file": "{name}"' not in l]
            _index().write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        except OSError:
            pass
    return True
