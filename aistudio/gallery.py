"""Keeps a list of the pictures you made (newest first) next to the pictures."""

from __future__ import annotations

import json
import re
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


# names the app gives its pictures, e.g. 20261113-181452-353030877.png
_OUR_NAME = re.compile(r"^\d{8}-\d{6}-\d+\.png$")


def clear_all() -> int:
    """Delete every picture the app made plus the list of prompts. Other files in the folder are left alone."""
    removed = 0
    with _lock:
        pics = paths.pictures_dir()
        names = set()
        try:
            for line in _index().read_text(encoding="utf-8").splitlines():
                try:
                    names.add(json.loads(line).get("file", ""))
                except ValueError:
                    pass
        except OSError:
            pass
        names |= {f.name for f in pics.iterdir() if f.is_file() and _OUR_NAME.match(f.name)}
        for name in names:
            if safe_name(name) and (pics / name).is_file():
                try:
                    (pics / name).unlink()
                    removed += 1
                except OSError:
                    pass
        _index().unlink(missing_ok=True)
    return removed
