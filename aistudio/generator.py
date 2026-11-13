"""The part that actually draws pictures (uses PyTorch + diffusers, loaded only when needed)."""

from __future__ import annotations

import logging
import os
import queue
import random
import threading
import time
import uuid
from dataclasses import dataclass, field

from . import gallery, models, mystyles, paths, safety, settings, styles

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
    preview: bytes | None = None   # small JPEG of the picture as it forms
    preview_n: int = 0
    created: float = field(default_factory=time.time)

    def public(self) -> dict:
        return {"id": self.id, "state": self.state, "message": self.message,
                "image_index": self.image_index, "count": self.count, "step": self.step,
                "steps": self.steps, "images": self.images, "error": self.error, "preview": self.preview_n}


# rough "what does it look like so far" colors for SDXL's 4 hidden channels (no full decode needed)
_PREVIEW_RGB = [[0.3651, 0.4232, 0.4341], [-0.2533, -0.0042, 0.1068],
                [0.1076, 0.1111, -0.0362], [-0.3165, -0.2492, -0.2188]]
_PREVIEW_BIAS = [0.1084, -0.0175, -0.0011]


def latent_preview(latents) -> bytes:
    """Turn the half-finished picture (latents) into a small JPEG, in a few milliseconds."""
    import io

    import torch
    from PIL import Image
    lat = latents[0].float()
    m = torch.tensor(_PREVIEW_RGB, device=lat.device)
    b = torch.tensor(_PREVIEW_BIAS, device=lat.device)
    rgb = torch.einsum("chw,cr->hwr", lat, m) + b
    arr = ((rgb.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=80)
    return buf.getvalue()


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
    user_style = None
    us = raw.get("user_style")
    if us:
        meta = mystyles.get(str(us))
        if not meta or not meta.get("ready"):
            raise ValueError("That style isn't available anymore.")
        user_style = meta["id"]
        try:
            us_strength = float(raw.get("user_style_strength", 0.8))
        except (TypeError, ValueError):
            us_strength = 0.8
        us_strength = round(max(0.1, min(1.5, us_strength)), 2)
    full_prompt = m.prompt_prefix + styles.apply_style(prompt, style) + m.prompt_suffix
    if user_style:
        trig = mystyles.get(user_style).get("trigger") or ""
        if trig:
            full_prompt = f"{full_prompt}, {trig}"
    negative = ", ".join(x for x in [user_neg, m.negative] if x)
    negative = safety.build_negative(negative, safe_mode)
    return {"prompt": prompt, "full_prompt": full_prompt, "negative": negative, "user_negative": user_neg,
            "model": m.id, "aspect": aspect, "width": w, "height": h, "quality": quality,
            "steps": int(m.steps[quality]), "guidance": float(m.guidance), "style": style,
            "count": count, "seed": seed, "safe_mode": safe_mode,
            "ref_path": ref_path, "ref_mode": ref_mode, "ref_strength": ref_strength,
            "user_style": user_style, "user_style_strength": us_strength if user_style else None}


class Generator:
    """Draws pictures. Memory plan for the graphics card ("mode"):

    - full:    everything on the graphics card (cards with 12 GB or more)
    - split:   the big drawing part (UNet) + picture-finisher (VAE) stay on the card; the text readers
               and style copier wait in normal memory and hop onto the card only for the moment they're
               used (8-11 GB cards like the RTX 4060). Fast, and nothing big moves per picture.
    - offload: everything waits in normal memory and moves on when needed (small cards / emergency)
    - cpu:     no NVIDIA card
    """

    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._q: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        self.pipe = None
        self.i2i = None             # "remix" version of the pipeline (shares the same model parts)
        self.ip_loaded = False      # style copier attached?
        self.loras: dict = {}       # name -> loaded
        self.base_scheduler = None  # the model's own scheduler settings
        self.loaded_model: str | None = None
        self.mode = "cpu"
        self.offload = False
        self.device = "cpu"
        self.gpu = ""
        self.vram_gb = 0.0
        self.light_vae = False
        self.last_used = time.time()
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------ public
    def submit(self, params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], params=params, count=params["count"], steps=params["steps"])
        with self._lock:
            self.jobs[job.id] = job
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

    # ------------------------------------------------------------ loading
    def _unload(self):
        if self.pipe is not None:
            self.pipe = None
            self.i2i = None
            self.ip_loaded = False
            self.loras = {}
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
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
        cfg = self.base_scheduler or pipe.scheduler.config
        if kind == "euler_a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(cfg)
        elif kind == "ddim_trailing":
            pipe.scheduler = DDIMScheduler.from_config(cfg, timestep_spacing="trailing")
        else:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                cfg, use_karras_sigmas=True, algorithm_type="dpmsolver++")

    def load_pipeline(self, source: str, fp16_files: bool, scheduler: str = "dpmpp_2m_karras",
                      force_offload: bool = False, light_vae: bool = True):
        """Load a model folder (or a Hugging Face name, used by the build tests)."""
        import os

        import torch
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline

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
        # lighter picture-finisher from the Speed pack: works in half precision -> half the memory
        self.light_vae = False
        vae_dir = models.model_dir("speed") / "vae"
        if light_vae and models.is_ready("speed") and (vae_dir / "config.json").exists():
            try:
                kw["vae"] = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=dtype)
                self.light_vae = True
            except Exception:
                log.exception("couldn't load the light VAE - using the model's own")
        pipe = StableDiffusionXLPipeline.from_pretrained(source, **kw)
        self.base_scheduler = dict(pipe.scheduler.config)
        self._set_scheduler(pipe, scheduler)
        pipe.set_progress_bar_config(disable=True)

        if cuda:
            props = torch.cuda.get_device_properties(0)
            self.vram_gb = props.total_memory / 1024**3
            free, _total = torch.cuda.mem_get_info()
            log.info("graphics card: %s, %.1f GB total, %.1f GB free before loading",
                     props.name, self.vram_gb, free / 1024**3)
            if force_offload or self.vram_gb < 7.5:
                self.mode = "offload"
            elif self.vram_gb < 11.5:
                self.mode = "split"
            else:
                self.mode = "full"
            if self.mode == "offload":
                pipe.enable_model_cpu_offload()
            elif self.mode == "split":
                pipe.unet.to("cuda")
                pipe.vae.to("cuda")
            else:
                pipe.to("cuda")
            self.gpu = f"{props.name} ({self.vram_gb:.0f} GB)"
        else:
            self.mode = "cpu"
            self.gpu = "No NVIDIA graphics card - using the processor (slow)"
        self.offload = self.mode == "offload"
        if self.mode in ("split", "offload"):
            # finish the picture in tiles so it never runs out of graphics memory
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass
        log.info("memory plan: %s, light finisher: %s", self.mode, self.light_vae)
        return pipe

    def _ensure_model(self, job: Job, force_offload=False):
        model_id = job.params["model"]
        if self.pipe is not None and self.loaded_model == model_id and not force_offload:
            return
        m = models.get(model_id)
        test_source = os.environ.get("AISTUDIO_TEST_SOURCE")      # build tests: a tiny stand-in model
        if not test_source and not models.is_ready(model_id):
            raise RuntimeError(f"The {m.name} model isn't downloaded yet. Download it in the Models tab.")
        job.state, job.message = "loading", f"Loading {m.name} into your graphics card (takes ~30 s the first time)..."
        self._unload()
        info = models.ready_info(model_id)
        t0 = time.time()
        if test_source:
            self.pipe = self.load_pipeline(test_source, False, m.scheduler, light_vae=False)
        else:
            self.pipe = self.load_pipeline(str(models.model_dir(model_id)), bool(info.get("fp16")),
                                           m.scheduler, force_offload=force_offload)
        self.loaded_model = model_id
        log.info("loaded %s in %.1fs", model_id, time.time() - t0)

    # ------------------------------------------------------------ moving small parts on/off the card
    def _borrow(self, *modules):
        """In 'split' mode, put small parts on the card for a moment. Returns a function that puts them back."""
        mods = [m for m in modules if m is not None]
        if self.mode != "split" or not mods:
            return lambda: None
        for m in mods:
            m.to("cuda")

        def give_back():
            for m in mods:
                m.to("cpu")
        return give_back

    def _exec_device(self, pipe):
        try:
            return pipe._execution_device
        except Exception:
            return self.device

    def _encode_text(self, pipe, p) -> dict:
        give_back = self._borrow(pipe.text_encoder, pipe.text_encoder_2)
        try:
            pe, npe, ppe, nppe = pipe.encode_prompt(
                prompt=p["full_prompt"], device=self._exec_device(pipe), num_images_per_prompt=1,
                do_classifier_free_guidance=p["guidance"] > 1, negative_prompt=p["negative"] or None)
        finally:
            give_back()
        out = {"prompt_embeds": pe, "pooled_prompt_embeds": ppe}
        if p["guidance"] > 1:
            out.update(negative_prompt_embeds=npe, negative_pooled_prompt_embeds=nppe)
        return out

    def _encode_style(self, pipe, ref, p) -> list:
        give_back = self._borrow(getattr(pipe, "image_encoder", None))
        try:
            return pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=ref, ip_adapter_image_embeds=None, device=self._exec_device(pipe),
                num_images_per_prompt=1, do_classifier_free_guidance=p["guidance"] > 1)
        finally:
            give_back()

    # ------------------------------------------------------------ add-ons
    def _set_style_copier(self, on: bool):
        if on == self.ip_loaded:
            return
        if on:
            if not models.is_ready("style"):
                raise RuntimeError("Download the Style copier add-on first (Models tab).")
            if self.mode == "split":
                # load while the big parts sit in normal memory, so the 1.8 GB style reader lands there too
                self.pipe.unet.to("cpu")
                self.pipe.vae.to("cpu")
            try:
                self.pipe.load_ip_adapter(str(models.model_dir("style")), subfolder="sdxl_models",
                                          weight_name="ip-adapter_sdxl.safetensors")
            finally:
                if self.mode == "split":
                    if getattr(self.pipe, "image_encoder", None) is not None:
                        self.pipe.image_encoder.to("cpu")      # only hops on when used
                    self.pipe.unet.to("cuda")
                    self.pipe.vae.to("cuda")
            if self.mode == "offload":
                self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.unload_ip_adapter()
            if self.mode == "offload":
                self.pipe.enable_model_cpu_offload()
        self.ip_loaded = on
        self.i2i = None

    def _set_loras(self, wanted: dict):
        """wanted: {name: (file, strength)} - extra skills like Turbo. Empty = none."""
        pipe = self.pipe
        for name, (path, _w) in wanted.items():
            if name not in self.loras:
                from . import engine_setup
                engine_setup.ensure_packages(["peft"])
                pipe.load_lora_weights(str(path), adapter_name=name)
                self.loras[name] = True
                log.info("loaded extra skill %s", name)
        if not self.loras:
            return
        if wanted:
            pipe.enable_lora()
            pipe.set_adapters(list(wanted), [w for _p, w in wanted.values()])
        else:
            pipe.disable_lora()

    # ------------------------------------------------------------ jobs
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

    def _prepare(self, job: Job):
        """Pick the right pipeline + settings for this job. Returns (pipe, extra arguments)."""
        p = job.params
        mode = p.get("ref_mode")
        ref = None
        if mode:
            from PIL import Image
            ref = Image.open(p["ref_path"]).convert("RGB")

        # extra skills: Turbo (Speed pack) and/or one of your own styles
        wanted = {}
        if p["quality"] == "turbo":
            if not models.is_ready("speed"):
                raise RuntimeError("Turbo needs the Speed pack add-on (Models tab).")
            wanted["turbo"] = (models.model_dir("speed") / "Hyper-SDXL-8steps-CFG-lora.safetensors", 1.0)
        if p.get("user_style"):
            f = mystyles.weights_file(p["user_style"])
            if not f.exists():
                raise RuntimeError("That style's file is missing - it may have been deleted.")
            wanted["u" + p["user_style"]] = (f, p["user_style_strength"])
        self._set_loras(wanted)

        if mode == "style":
            job.message = "Getting the style copier ready..."
            self._set_style_copier(True)
            # "InstantStyle": only the style layers look at the reference, so it copies the look, not the content
            self.pipe.set_ip_adapter_scale({"up": {"block_0": [0.0, p["ref_strength"], 0.0]}})
            pipe = self.pipe
            extra = {"width": p["width"], "height": p["height"]}
        else:
            self._set_style_copier(False)
            if mode == "remix":
                from diffusers import StableDiffusionXLImg2ImgPipeline
                from PIL import Image, ImageOps
                if self.i2i is None:
                    self.i2i = StableDiffusionXLImg2ImgPipeline.from_pipe(self.pipe)
                    self.i2i.set_progress_bar_config(disable=True)
                pipe = self.i2i
                start = ImageOps.fit(ref, (p["width"], p["height"]), method=Image.LANCZOS)
                job.steps = max(1, int(p["steps"] * p["ref_strength"]))
                extra = {"image": start, "strength": p["ref_strength"]}
            else:
                pipe = self.pipe
                extra = {"width": p["width"], "height": p["height"]}

        sched = "ddim_trailing" if p["quality"] == "turbo" else models.get(p["model"]).scheduler
        self._set_scheduler(pipe, sched)
        if self.mode == "offload":
            pipe.enable_model_cpu_offload()      # (re)attach the memory-saving hooks to this pipeline

        job.message = "Reading your prompt..."
        extra.update(self._encode_text(pipe, p))
        if mode == "style":
            extra["ip_adapter_image_embeds"] = self._encode_style(pipe, ref, p)
        return pipe, extra

    def _draw_all(self, job: Job):
        import torch
        from PIL.PngImagePlugin import PngInfo
        p = job.params
        pipe, extra = self._prepare(job)
        cuda = self.device == "cuda"
        job.state = "running"
        for i in range(p["count"]):
            if job.cancel:
                break
            job.image_index, job.step = i, 0
            job.preview, job.preview_n = None, job.preview_n + 1
            seed = (p["seed"] + i) % (2**31)
            job.message = f"Drawing picture {i + 1} of {p['count']}..."
            marks = {}

            def on_step(pipe_, step, _t, kw):
                job.step = step + 1
                marks["last_step"] = time.time()
                lat = kw.get("latents")
                if lat is not None:
                    try:
                        job.preview = latent_preview(lat)
                        job.preview_n += 1
                    except Exception:
                        pass
                if job.step >= job.steps:
                    job.message = f"Finishing picture {i + 1} of {p['count']}..."
                if job.cancel:
                    pipe_._interrupt = True
                return kw

            gen = torch.Generator(device="cpu").manual_seed(seed)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            out = pipe(prompt=None, num_inference_steps=p["steps"], guidance_scale=p["guidance"],
                       generator=gen, callback_on_step_end=on_step, **extra)
            if job.cancel:
                break
            took = time.time() - t0
            drawing = marks.get("last_step", t0 + took) - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3 if cuda else 0
            img = out.images[0]
            name = time.strftime("%Y%m%d-%H%M%S") + f"-{seed}.png"
            meta = PngInfo()
            meta.add_text("parameters",
                          f"{p['full_prompt']}\nNegative prompt: {p['negative']}\n"
                          f"Steps: {p['steps']}, CFG scale: {p['guidance']}, Seed: {seed}, "
                          f"Size: {p['width']}x{p['height']}, Model: {models.get(p['model']).name}"
                          + (f", Reference: {p['ref_mode']} {p['ref_strength']}" if p.get("ref_mode") else "")
                          + (", Turbo" if p["quality"] == "turbo" else ""))
            img.save(paths.pictures_dir() / name, pnginfo=meta)
            entry = {"file": name, "prompt": p["prompt"], "negative": p["user_negative"], "model": p["model"],
                     "style": p["style"], "aspect": p["aspect"], "quality": p["quality"], "seed": seed,
                     "width": p["width"], "height": p["height"], "steps": job.steps,
                     "seconds": round(took, 1), "time": time.time(),
                     "ref_mode": p.get("ref_mode"), "ref_strength": p.get("ref_strength"),
                     "user_style": p.get("user_style"), "user_style_strength": p.get("user_style_strength")}
            gallery.add(entry)
            job.images.append(entry)
            log.info("made %s in %.1fs [%s, %s, %d steps: drawing %.1fs = %.2f steps/s, finishing %.1fs, "
                     "peak %.1f GB, plan %s]", name, took, p["quality"], p.get("ref_mode") or "normal",
                     job.steps, drawing, job.steps / max(drawing, 0.01), took - drawing, peak, self.mode)
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
