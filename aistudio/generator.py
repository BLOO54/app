"""The part that actually draws pictures (uses PyTorch + diffusers, loaded only when needed)."""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
import uuid
from dataclasses import dataclass, field

from . import gallery, models, paths, safety, settings, styles

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    params: dict
    state: str = "queued"          # queued, loading, running, done, error, cancelled
    message: str = "Waiting..."
    image_index: int = 0           # which picture of the batch is being drawn (0-based)
    count: int = 1
    step: int = 0
    steps: int = 1
    images: list = field(default_factory=list)
    error: str | None = None
    cancel: bool = False
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {"id": self.id, "state": self.state, "message": self.message,
                "image_index": self.image_index, "count": self.count, "step": self.step,
                "steps": self.steps, "images": self.images, "error": self.error}


class NotAllowed(ValueError):
    pass


def make_job_params(raw: dict) -> dict:
    """Check and fill in what the page sent. Raises NotAllowed / ValueError on bad input."""
    prompt = str(raw.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Type what you want to see first.")
    if len(prompt) > 2000:
        raise ValueError("That prompt is too long (2000 characters max).")
    reason = safety.check_prompt(prompt)
    if reason:
        raise NotAllowed(reason)
    m = models.get(str(raw.get("model") or "juggernaut"))
    aspect = raw.get("aspect") if raw.get("aspect") in styles.ASPECTS else "square"
    quality = raw.get("quality") if raw.get("quality") in styles.QUALITY else "balanced"
    style = raw.get("style") if raw.get("style") in styles.STYLES else "none"
    try:
        count = max(1, min(8, int(raw.get("count") or 1)))
    except (TypeError, ValueError):
        count = 1
    seed = raw.get("seed")
    try:
        seed = int(seed) if seed not in (None, "", -1, "-1") else None
    except (TypeError, ValueError):
        seed = None
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)
    _label, w, h = styles.ASPECTS[aspect]
    user_neg = str(raw.get("negative") or "").strip()[:1000]
    safe_mode = bool(raw.get("safe_mode", True))
    ref_path = raw.get("ref_path") or None          # set by the server after it saved the upload
    ref_mode = raw.get("ref_mode") if raw.get("ref_mode") in ("style", "remix") else None
    if not ref_path:
        ref_mode = None
    if not ref_mode:
        ref_path = None
    try:
        ref_strength = float(raw.get("ref_strength", 0.6))
    except (TypeError, ValueError):
        ref_strength = 0.6
    lo, hi = (0.1, 1.0) if ref_mode == "style" else (0.2, 0.95)
    ref_strength = round(max(lo, min(hi, ref_strength)), 2)
    full_prompt = m.prompt_prefix + styles.apply_style(prompt, style) + m.prompt_suffix
    negative = ", ".join(x for x in [user_neg, m.negative] if x)
    negative = safety.build_negative(negative, safe_mode)
    return {"prompt": prompt, "full_prompt": full_prompt, "negative": negative, "user_negative": user_neg,
            "model": m.id, "aspect": aspect, "width": w, "height": h, "quality": quality,
            "steps": int(m.steps[quality]), "guidance": float(m.guidance), "style": style,
            "count": count, "seed": seed, "safe_mode": safe_mode,
            "ref_path": ref_path, "ref_mode": ref_mode, "ref_strength": ref_strength}


class Generator:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self.pipe = None
        self.i2i = None             # "remix" version of the pipeline (shares the same model parts)
        self.ip_loaded = False      # style copier attached?
        self.placed = None          # which pipeline is currently set up on the graphics card
        self.loaded_model: str | None = None
        self.offload = False
        self.device = "cpu"
        self.gpu = ""
        self.last_used = time.time()
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------ public
    def submit(self, params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], params=params, count=params["count"], steps=params["steps"])
        with self._lock:
            self.jobs[job.id] = job
            # forget old finished jobs
            for jid in [j.id for j in self.jobs.values()
                        if j.state in ("done", "error", "cancelled") and time.time() - j.created > 3600]:
                self.jobs.pop(jid, None)
        self._q.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job:
            job.cancel = True

    def busy(self) -> bool:
        return any(j.state in ("queued", "loading", "running") for j in list(self.jobs.values()))

    def unload(self) -> None:
        with self._lock:
            self._unload()

    # ------------------------------------------------------------ internals
    def _unload(self):
        if self.pipe is not None:
            self.pipe = None
            self.i2i = None
            self.ip_loaded = False
            self.placed = None
            self.loaded_model = None
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _set_scheduler(self, pipe, kind: str):
        from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
        cfg = pipe.scheduler.config
        if kind == "euler_a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(cfg)
        else:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                cfg, use_karras_sigmas=True, algorithm_type="dpmsolver++")

    def load_pipeline(self, source: str, fp16_files: bool, scheduler: str = "dpmpp_2m_karras",
                      force_offload: bool = False):
        """Load a model folder (or a Hugging Face name, used by the build tests)."""
        import os

        import torch
        from diffusers import StableDiffusionXLPipeline

        cuda = torch.cuda.is_available()
        self.device = "cuda" if cuda else "cpu"
        if cuda:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16 if os.environ.get("AISTUDIO_CPU_DTYPE") == "bf16" else torch.float32
        kw = {"torch_dtype": dtype}
        if os.path.isdir(source):
            kw["use_safetensors"] = True
        if fp16_files:
            kw["variant"] = "fp16"
        pipe = StableDiffusionXLPipeline.from_pretrained(source, **kw)
        self._set_scheduler(pipe, scheduler)
        pipe.set_progress_bar_config(disable=True)
        self.offload = False
        if cuda:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / 1024**3
            self.gpu = f"{props.name} ({vram_gb:.0f} GB)"
            if force_offload or vram_gb < 11.5:
                # 8 GB cards: keep only the part that's working on the graphics card
                pipe.enable_model_cpu_offload()
                self.offload = True
            else:
                pipe.to("cuda")
        else:
            self.gpu = "No NVIDIA graphics card - using the processor (slow)"
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        return pipe

    def _ensure_model(self, job: Job, force_offload=False):
        model_id = job.params["model"]
        if self.pipe is not None and self.loaded_model == model_id and not force_offload:
            return
        m = models.get(model_id)
        if not models.is_ready(model_id):
            raise RuntimeError(f"The {m.name} model isn't downloaded yet. Download it in the Models tab.")
        job.state, job.message = "loading", f"Loading {m.name} into your graphics card (takes ~30 s the first time)..."
        self._unload()
        info = models.ready_info(model_id)
        self.pipe = self.load_pipeline(str(models.model_dir(model_id)), bool(info.get("fp16")),
                                       m.scheduler, force_offload=force_offload)
        self.placed = self.pipe
        self.loaded_model = model_id

    # ------------------------------------------------------------ reference pictures
    def _place(self, pipe):
        """Put a pipeline on the graphics card (or set up low-memory mode for it)."""
        if self.placed is pipe:
            return
        if self.device == "cuda":
            if self.offload:
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        self.placed = pipe

    def _set_style_copier(self, on: bool):
        if on == self.ip_loaded:
            return
        if on:
            if not models.is_ready("style"):
                raise RuntimeError("Download the Style copier add-on first (Models tab).")
            self.pipe.load_ip_adapter(str(models.model_dir("style")), subfolder="sdxl_models",
                                      weight_name="ip-adapter_sdxl.safetensors")
        else:
            self.pipe.unload_ip_adapter()
        self.ip_loaded = on
        self.placed = None      # parts changed - set up the graphics card again
        self.i2i = None

    def _prepare(self, job: Job):
        """Pick the right pipeline for this job. Returns (pipe, extra arguments)."""
        p = job.params
        mode = p.get("ref_mode")
        if mode:
            from PIL import Image, ImageOps
            ref = Image.open(p["ref_path"]).convert("RGB")
        if mode == "style":
            job.message = "Getting the style copier ready..."
            self._set_style_copier(True)
            self._place(self.pipe)
            # "InstantStyle": only the style layers look at the reference, so it copies the look, not the content
            self.pipe.set_ip_adapter_scale({"up": {"block_0": [0.0, p["ref_strength"], 0.0]}})
            return self.pipe, {"width": p["width"], "height": p["height"], "ip_adapter_image": ref}
        self._set_style_copier(False)
        if mode == "remix":
            from diffusers import StableDiffusionXLImg2ImgPipeline
            if self.i2i is None:
                self.i2i = StableDiffusionXLImg2ImgPipeline.from_pipe(self.pipe)
                self.i2i.set_progress_bar_config(disable=True)
            self._place(self.i2i)
            start = ImageOps.fit(ref, (p["width"], p["height"]), method=Image.LANCZOS)
            job.steps = max(1, int(p["steps"] * p["ref_strength"]))
            return self.i2i, {"image": start, "strength": p["ref_strength"]}
        self._place(self.pipe)
        return self.pipe, {"width": p["width"], "height": p["height"]}

    def _worker(self):
        while True:
            job = self._q.get()
            if job.cancel:
                job.state, job.message = "cancelled", "Cancelled"
                continue
            try:
                with self._lock:
                    self._run(job)
            except Exception as e:
                log.exception("generation failed")
                job.state, job.error, job.message = "error", _friendly(e), "Something went wrong"
            self.last_used = time.time()

    def _run(self, job: Job):
        import torch
        try:
            self._ensure_model(job)
            self._draw_all(job)
        except torch.cuda.OutOfMemoryError:
            if self.offload:
                raise
            log.warning("out of graphics memory - retrying in low-memory mode")
            job.images.clear()
            self._unload()
            torch.cuda.empty_cache()
            self._ensure_model(job, force_offload=True)
            self._draw_all(job)

    def _draw_all(self, job: Job):
        import torch
        from PIL.PngImagePlugin import PngInfo
        p = job.params
        pipe, extra = self._prepare(job)
        job.state = "running"
        for i in range(p["count"]):
            if job.cancel:
                break
            job.image_index, job.step = i, 0
            seed = (p["seed"] + i) % (2**31)
            job.message = f"Drawing picture {i + 1} of {p['count']}..."

            def on_step(pipe, step, _t, kw):
                job.step = step + 1
                if job.cancel:
                    pipe._interrupt = True
                return kw

            gen = torch.Generator(device="cpu").manual_seed(seed)
            t0 = time.time()
            out = pipe(prompt=p["full_prompt"], negative_prompt=p["negative"] or None,
                       num_inference_steps=p["steps"], guidance_scale=p["guidance"], generator=gen,
                       callback_on_step_end=on_step, **extra)
            if job.cancel:
                break
            img = out.images[0]
            name = time.strftime("%Y%m%d-%H%M%S") + f"-{seed}.png"
            meta = PngInfo()
            meta.add_text("parameters",
                          f"{p['full_prompt']}\nNegative prompt: {p['negative']}\n"
                          f"Steps: {p['steps']}, CFG scale: {p['guidance']}, Seed: {seed}, "
                          f"Size: {p['width']}x{p['height']}, Model: {models.get(p['model']).name}"
                          + (f", Reference: {p['ref_mode']} {p['ref_strength']}" if p.get("ref_mode") else ""))
            img.save(paths.pictures_dir() / name, pnginfo=meta)
            entry = {"file": name, "prompt": p["prompt"], "negative": p["user_negative"], "model": p["model"],
                     "style": p["style"], "aspect": p["aspect"], "quality": p["quality"], "seed": seed,
                     "width": p["width"], "height": p["height"], "steps": p["steps"],
                     "seconds": round(time.time() - t0, 1), "time": time.time(),
                     "ref_mode": p.get("ref_mode"), "ref_strength": p.get("ref_strength")}
            gallery.add(entry)
            job.images.append(entry)
            log.info("made %s in %.1fs", name, time.time() - t0)
        if job.cancel:
            job.state, job.message = "cancelled", "Stopped"
        else:
            job.state, job.message = "done", "Done!"


def _friendly(e: Exception) -> str:
    text = str(e) or e.__class__.__name__
    low = text.lower()
    if "out of memory" in low:
        return ("Your graphics card ran out of memory. Close games or other heavy apps, "
                "or pick a smaller size / fewer pictures, then try again.")
    if "no module named" in low:
        return "The AI engine is missing a part. Open the Setup tab and press Repair."
    return text
