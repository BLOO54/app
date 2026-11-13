"""Your own styles ("style packs" / LoRAs): ones you trained in the app or imported from a file.

Each style lives in %LOCALAPPDATA%\\AIImageStudio\\styles\\<id>\\
    meta.json                         name, trigger words, how it was made
    pytorch_lora_weights.safetensors  the style itself
    thumb.jpg                         small picture for the button
    train\\                            the pictures it was trained on (trained styles only)
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path

from . import paths

WEIGHTS = "pytorch_lora_weights.safetensors"


def root() -> Path:
    p = paths.data_dir() / "styles"
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_id(style_id: str) -> str | None:
    return style_id if isinstance(style_id, str) and re.fullmatch(r"[a-z0-9]{6,32}", style_id) else None


def folder(style_id: str) -> Path:
    sid = safe_id(style_id)
    if not sid:
        raise ValueError("bad style id")
    return root() / sid


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def make_trigger(name: str) -> str:
    """A made-up word the AI links to your style, e.g. 'My Roblox Art' -> 'myrobloxart_style'."""
    base = re.sub(r"[^a-z0-9]+", "", (name or "").lower())[:20] or "mystyle"
    return f"{base}_style"


def save_meta(style_id: str, meta: dict) -> None:
    f = folder(style_id)
    f.mkdir(parents=True, exist_ok=True)
    (f / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def get(style_id: str) -> dict | None:
    sid = safe_id(style_id)
    if not sid:
        return None
    try:
        meta = json.loads((root() / sid / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    meta["ready"] = (root() / sid / WEIGHTS).exists()
    meta["has_thumb"] = (root() / sid / "thumb.jpg").exists()
    return meta


def weights_file(style_id: str) -> Path:
    return folder(style_id) / WEIGHTS


def all_styles() -> list[dict]:
    out = []
    for d in sorted(root().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if d.is_dir():
            m = get(d.name)
            if m:
                out.append(m)
    return out


def delete(style_id: str) -> bool:
    sid = safe_id(style_id)
    if not sid:
        return False
    shutil.rmtree(root() / sid, ignore_errors=True)
    return True


def rename(style_id: str, name: str) -> bool:
    m = get(style_id)
    if not m:
        return False
    m["name"] = (name or "").strip()[:40] or m["name"]
    m.pop("ready", None)
    m.pop("has_thumb", None)
    save_meta(style_id, m)
    return True


def new_meta(style_id: str, name: str, kind: str, trigger: str = "", **extra) -> dict:
    return {"id": style_id, "name": (name or "My style").strip()[:40], "kind": kind,
            "trigger": (trigger or "").strip()[:80], "created": time.time(), **extra}
