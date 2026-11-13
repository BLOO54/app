"""Where things live on your PC.

- App data (settings, log, AI models):  %LOCALAPPDATA%\\AIImageStudio
- Your pictures:                          Pictures\\AI Image Studio
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _base() -> Path:
    override = os.environ.get("AISTUDIO_HOME")
    if override:
        return Path(override)
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(root) / "AIImageStudio"
    return Path.home() / ".aistudio"


def data_dir() -> Path:
    p = _base()
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def pictures_dir() -> Path:
    override = os.environ.get("AISTUDIO_PICTURES")
    if override:
        p = Path(override)
    else:
        p = Path.home() / "Pictures" / "AI Image Studio"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_file() -> Path:
    return data_dir() / "app.log"


def web_dir() -> Path:
    return Path(__file__).resolve().parent / "web"


def python_exe() -> str:
    """The console python.exe that runs this app (not pythonw), used for pip."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":
        cand = exe.with_name("python.exe")
        if cand.exists():
            return str(cand)
    return str(exe)
