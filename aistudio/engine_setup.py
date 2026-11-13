"""Installs the "AI engine" (PyTorch + diffusers) the first time you open the app.

Why not include it in the installer? It's ~3 GB, and keeping it separate means app
updates stay tiny (a few MB) - the engine is only downloaded once.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from collections import deque

from . import paths, settings

log = logging.getLogger(__name__)

TORCH_INDEX = {"cuda": "https://download.pytorch.org/whl/cu128",
               "cpu": "https://download.pytorch.org/whl/cpu"}
EXTRA_PACKAGES = ["diffusers>=0.32", "transformers>=4.45", "accelerate>=1.0", "safetensors",
                  "huggingface_hub>=0.26", "pillow"]
NEEDED_MODULES = ["torch", "diffusers", "transformers", "accelerate", "safetensors", "huggingface_hub", "PIL"]

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def has_nvidia() -> bool:
    if sys.platform != "win32":
        return shutil.which("nvidia-smi") is not None
    for cand in [shutil.which("nvidia-smi"), r"C:\Windows\System32\nvidia-smi.exe"]:
        if cand and os.path.exists(cand):
            try:
                out = subprocess.run([cand, "-L"], capture_output=True, text=True, timeout=15,
                                     creationflags=_NO_WINDOW)
                if "GPU" in (out.stdout or ""):
                    return True
            except (OSError, subprocess.SubprocessError):
                pass
    return False


def gpu_names() -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15, creationflags=_NO_WINDOW)
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def is_installed() -> bool:
    importlib.invalidate_caches()
    return all(importlib.util.find_spec(m) is not None for m in NEEDED_MODULES)


_RAW = re.compile(r"Progress (\d+) of (\d+)")


class EngineInstaller:
    def __init__(self):
        self._lock = threading.Lock()
        self.active = False
        self.error: str | None = None
        self.progress = 0.0
        self.message = ""
        self.lines: deque[str] = deque(maxlen=12)
        self.device = ""

    def status(self) -> dict:
        with self._lock:
            return {"active": self.active, "error": self.error, "progress": round(self.progress, 4),
                    "message": self.message, "log": list(self.lines), "device": self.device}

    def start(self, force_cpu: bool = False) -> bool:
        with self._lock:
            if self.active:
                return False
            self.active, self.error, self.progress = True, None, 0.0
            self.message = "Starting..."
        threading.Thread(target=self.run, args=(force_cpu,), daemon=True).start()
        return True

    def _say(self, msg=None, progress=None, line=None):
        with self._lock:
            if msg is not None:
                self.message = msg
            if progress is not None:
                self.progress = progress
            if line:
                self.lines.append(line)

    def _pip(self, args: list[str], lo: float, hi: float, label: str):
        cmd = [paths.python_exe(), "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check",
               "--no-warn-script-location", "--progress-bar", "raw", *args]
        log.info("running: %s", " ".join(cmd))
        self._say(label, lo)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
        tail = deque(maxlen=30)
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            m = _RAW.search(line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                if total > 50_000_000:   # only the big files move the bar
                    self._say(progress=lo + (hi - lo) * done / total)
                continue
            tail.append(line)
            log.info("pip: %s", line)
            self._say(line=line[:200])
        code = proc.wait()
        if code != 0:
            raise RuntimeError("Install failed:\n" + "\n".join(list(tail)[-8:]))
        self._say(progress=hi)

    def run(self, force_cpu: bool = False):
        try:
            device = "cpu" if force_cpu or not has_nvidia() else "cuda"
            with self._lock:
                self.device = device
            word = "NVIDIA graphics card" if device == "cuda" else "processor (no NVIDIA card found - slow)"
            self._pip(["torch", "--index-url", TORCH_INDEX[device]], 0.0, 0.85,
                      f"Downloading PyTorch for your {word} (biggest part, ~3 GB)...")
            self._pip(EXTRA_PACKAGES, 0.85, 0.99, "Downloading the image tools (diffusers)...")
            if not is_installed():
                raise RuntimeError("The engine installed but can't be found. Try again or restart the app.")
            settings.update(device=device)
            self._say("AI engine ready!", 1.0)
            log.info("engine installed (%s)", device)
        except Exception as e:
            log.exception("engine install failed")
            with self._lock:
                self.error = str(e)
                self.message = "Install failed"
        finally:
            with self._lock:
                self.active = False
