"""The AI models (the "brains" that draw the pictures) and how they get downloaded.

Models come from Hugging Face, a big free library of AI models. Each one is ~7 GB and is
downloaded once into %LOCALAPPDATA%\\AIImageStudio\\models. After that, no internet needed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

# classic downloads write a growing ".incomplete" file we can measure for the progress bar
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")   # we draw our own progress bar


@dataclass
class ModelInfo:
    id: str
    name: str
    repo: str
    about: str
    guidance: float
    steps: dict = field(default_factory=lambda: {"fast": 20, "balanced": 30, "best": 40})
    scheduler: str = "dpmpp_2m_karras"
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    negative: str = ""
    approx_gb: float = 7.0


CATALOG: list[ModelInfo] = [
    ModelInfo(
        id="juggernaut",
        name="Juggernaut XL",
        repo="RunDiffusion/Juggernaut-XL-v9",
        about="Best all-rounder: realistic photos, people, places, game art, anything.",
        guidance=5.0,
        steps={"fast": 20, "balanced": 30, "best": 40},
        scheduler="dpmpp_2m_karras",
        negative="blurry, low quality, worst quality, deformed, bad anatomy, extra fingers, watermark, text",
        approx_gb=6.5,
    ),
    ModelInfo(
        id="animagine",
        name="Animagine XL 4.0",
        repo="cagliostrolab/animagine-xl-4.0",
        about="Anime and cartoon style. Works best with short tags, like: 1girl, red hair, sword, night city",
        guidance=5.0,
        steps={"fast": 20, "balanced": 28, "best": 35},
        scheduler="euler_a",
        prompt_suffix=", masterpiece, high score, great score, absurdres",
        negative="lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, cropped, "
                 "worst quality, low quality, low score, bad score, average score, signature, watermark, blurry",
        approx_gb=6.5,
    ),
]


def get(model_id: str) -> ModelInfo:
    for m in CATALOG:
        if m.id == model_id:
            return m
    return CATALOG[0]


# ---------------------------------------------------------------- which files to download

_SKIP_EXT = (".bin", ".ckpt", ".onnx", ".onnx_data", ".msgpack", ".h5", ".pb", ".pt", ".pth")


def select_files(siblings: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], bool]:
    """Pick only the files diffusers needs, preferring the smaller fp16 ("half size") weights.

    `siblings` is [(path_in_repo, size_bytes)]. Returns (chosen_files, uses_fp16).
    Big single-file checkpoints in the top folder are skipped (diffusers uses the sub-folders).
    """
    config, weights = [], {}
    for name, size in siblings:
        low = name.lower()
        if low.endswith(_SKIP_EXT):
            continue
        if "/" not in name:
            if low == "model_index.json":
                config.append((name, size))
            continue
        folder = name.split("/", 1)[0]
        if low.endswith((".json", ".txt", ".model")):
            config.append((name, size))
        elif low.endswith(".safetensors"):
            weights.setdefault(folder, []).append((name, size))
    fp16 = bool(weights) and all(any(".fp16." in n for n, _ in files) for files in weights.values())
    chosen = list(config)
    for files in weights.values():
        if fp16:
            chosen += [f for f in files if ".fp16." in f[0]]
        else:
            chosen += [f for f in files if ".fp16." not in f[0]]
    return chosen, fp16


def check_layout(chosen: list[tuple[str, int]]) -> str | None:
    names = {n for n, _ in chosen}
    if "model_index.json" not in names:
        return "this model isn't in the format the app needs (no model_index.json)"
    if not any(n.startswith("unet/") and n.endswith(".safetensors") for n in names):
        return "this model has no drawing weights (unet) in a supported format"
    return None


# ---------------------------------------------------------------- on-disk state

def model_dir(model_id: str) -> Path:
    return paths.models_dir() / model_id


def is_ready(model_id: str) -> bool:
    return (model_dir(model_id) / "ready.json").exists()


def ready_info(model_id: str) -> dict:
    try:
        return json.loads((model_dir(model_id) / "ready.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def delete(model_id: str) -> None:
    shutil.rmtree(model_dir(model_id), ignore_errors=True)


def _dir_size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


# ---------------------------------------------------------------- downloading

class Downloads:
    """Runs model downloads in the background and reports progress."""

    def __init__(self):
        self._lock = threading.Lock()
        self.state: dict[str, dict] = {}   # model_id -> {"done":bytes,"total":bytes,"error":str|None,"active":bool}

    def status(self, model_id: str) -> dict:
        with self._lock:
            return dict(self.state.get(model_id, {}))

    def start(self, model_id: str) -> bool:
        with self._lock:
            st = self.state.get(model_id)
            if st and st.get("active"):
                return False
            self.state[model_id] = {"done": 0, "total": 0, "error": None, "active": True,
                                    "message": "Getting the file list..."}
        threading.Thread(target=self._run, args=(model_id,), daemon=True).start()
        return True

    def _set(self, model_id, **kw):
        with self._lock:
            self.state.setdefault(model_id, {}).update(kw)

    def _run(self, model_id: str):
        m = get(model_id)
        target = model_dir(model_id)
        try:
            from huggingface_hub import HfApi, hf_hub_download  # installed with the AI engine
            info = HfApi().model_info(m.repo, files_metadata=True)
            sib = [(s.rfilename, int(s.size or 0)) for s in info.siblings]
            chosen, fp16 = select_files(sib)
            problem = check_layout(chosen)
            if problem:
                raise RuntimeError(problem)
            total = sum(s for _, s in chosen)
            self._set(model_id, total=total, message="Downloading...")
            target.mkdir(parents=True, exist_ok=True)

            stop = threading.Event()

            def watch():
                while not stop.wait(0.7):
                    self._set(model_id, done=min(total, _dir_size(target)))
            threading.Thread(target=watch, daemon=True).start()
            try:
                for name, _size in sorted(chosen, key=lambda f: f[1]):
                    for attempt in range(4):
                        try:
                            hf_hub_download(m.repo, name, local_dir=str(target))
                            break
                        except Exception as e:  # network hiccup - try again (it resumes)
                            if attempt == 3:
                                raise
                            log.warning("download of %s failed (%s), retrying", name, e)
                            self._set(model_id, message="Connection hiccup, retrying...")
                            time.sleep(3 + attempt * 5)
                            self._set(model_id, message="Downloading...")
            finally:
                stop.set()
            shutil.rmtree(target / ".cache", ignore_errors=True)
            (target / "ready.json").write_text(json.dumps({"repo": m.repo, "fp16": fp16,
                                                           "files": len(chosen), "bytes": total}),
                                               encoding="utf-8")
            self._set(model_id, done=total, active=False, message="Ready")
            log.info("model %s downloaded (%d bytes)", model_id, total)
        except Exception as e:
            log.exception("model download failed")
            self._set(model_id, active=False, error=str(e) or e.__class__.__name__,
                      message="Download failed")
