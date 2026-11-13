"""Teaches the AI a new style from your pictures (makes a "LoRA" style pack).

How it works, in plain English:
1. Every picture is squished into the AI's own "language" (latents) once, up front.
2. The AI's main drawing part is frozen; we only train a small add-on (~90 MB) on top of it.
3. Over and over, we blur one of your pictures with noise and ask the AI to un-blur it while
   saying the style's trigger word. Each time it gets it a bit wrong, the add-on is nudged.
4. After a few hundred to ~1,500 rounds, saying the trigger word makes the AI draw in your style.

Memory tricks so it fits on an 8 GB card: cached latents + text, half-precision frozen weights,
gradient checkpointing, batch size 1, and an automatic retry at a smaller size if memory runs out.
"""

from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from pathlib import Path

from . import models, mystyles

log = logging.getLogger(__name__)

LENGTHS = {"quick": ("Quick", 500), "normal": ("Normal", 1000), "thorough": ("Thorough", 1600)}


class Cancelled(Exception):
    pass


class Trainer:
    def __init__(self, generator_getter):
        self._get_generator = generator_getter
        self._lock = threading.Lock()
        self.state: dict = {"active": False}

    def status(self) -> dict:
        with self._lock:
            return dict(self.state)

    def _set(self, **kw):
        with self._lock:
            self.state.update(kw)

    def cancel(self):
        self._set(cancel=True, message="Stopping...")

    def start(self, style_id: str, base_model: str, length: str, source: str | None = None,
              resolution: int | None = None, steps: int | None = None) -> bool:
        with self._lock:
            if self.state.get("active"):
                return False
            meta = mystyles.get(style_id) or {}
            self.state = {"active": True, "style_id": style_id, "name": meta.get("name", ""),
                          "step": 0, "steps": steps or LENGTHS.get(length, LENGTHS["normal"])[1],
                          "message": "Getting ready...", "error": None, "cancel": False,
                          "eta": None, "loss": None, "started": time.time(), "done": False}
        threading.Thread(target=self._run, args=(style_id, base_model, source, resolution),
                         daemon=True).start()
        return True

    def _run(self, style_id, base_model, source, resolution):
        gen = self._get_generator()
        try:
            with gen._lock:                 # no pictures are drawn while training
                gen._unload()               # free the graphics card for training
                res = resolution
                try:
                    train_style(style_id, base_model, self, source=source, resolution=res or 1024)
                except Exception as e:
                    import torch
                    if isinstance(e, torch.cuda.OutOfMemoryError) and not res:
                        log.warning("out of graphics memory at 1024 - retrying at 768")
                        torch.cuda.empty_cache()
                        self._set(message="Not enough graphics memory - retrying at a smaller size...", step=0)
                        train_style(style_id, base_model, self, source=source, resolution=768)
                    else:
                        raise
            self._set(active=False, done=True, message="Done! Your style is ready on the Create tab.")
        except Cancelled:
            self._set(active=False, message="Training stopped.")
            if not mystyles.weights_file(style_id).exists():
                mystyles.delete(style_id)
        except Exception as e:
            log.exception("training failed")
            text = str(e) or e.__class__.__name__
            if "out of memory" in text.lower():
                text = ("Your graphics card ran out of memory. Close games, browsers with videos and other "
                        "heavy apps, then try again.")
            self._set(active=False, error=text, message="Training failed")
        finally:
            try:
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _load_square(path: Path, res: int):
    from PIL import Image, ImageOps
    img = Image.open(path).convert("RGB")
    return ImageOps.fit(img, (res, res), method=Image.LANCZOS)


def train_style(style_id: str, base_model: str, tr: Trainer, source: str | None = None,
                resolution: int = 1024, rank: int = 16, lr: float = 1e-4):
    """source: a model folder/Hugging Face name to train on instead of a downloaded model (build tests)."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    meta = mystyles.get(style_id)
    if not meta:
        raise RuntimeError("That style doesn't exist anymore.")
    folder = mystyles.folder(style_id)
    pics = sorted((folder / "train").glob("*.png"))
    if len(pics) < 3:
        raise RuntimeError("Add at least 3 pictures to train a style.")
    steps = tr.status()["steps"]
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    dtype = torch.float16 if cuda else (torch.bfloat16 if os.environ.get("AISTUDIO_CPU_DTYPE") == "bf16"
                                        else torch.float32)

    def check_cancel():
        if tr.status().get("cancel"):
            raise Cancelled()

    # ---------------------------------------------------------------- load the model
    tr._set(message="Loading the model...")
    if source is None:
        if not models.is_ready(base_model):
            raise RuntimeError(f"Download the {models.get(base_model).name} model first.")
        info = models.ready_info(base_model)
        source = str(models.model_dir(base_model))
        kw = {"use_safetensors": True, "variant": "fp16" if info.get("fp16") else None}
    else:
        kw = {}
    light_vae = None
    vae_dir = models.model_dir("speed") / "vae"
    if "use_safetensors" in kw and models.is_ready("speed") and (vae_dir / "config.json").exists():
        light_vae = AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=dtype)
        kw["vae"] = light_vae
    pipe = StableDiffusionXLPipeline.from_pretrained(source, torch_dtype=dtype, **kw)
    check_cancel()

    # ---------------------------------------------------------------- 1. the words (once)
    trigger = meta.get("trigger") or mystyles.make_trigger(meta.get("name", ""))
    caption = f"{trigger}, in the style of {trigger}"
    tr._set(message="Reading the style name...")
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    with torch.no_grad():
        pe, _n, pooled, _np = pipe.encode_prompt(prompt=caption, device=device, num_images_per_prompt=1,
                                                 do_classifier_free_guidance=False)
    pe, pooled = pe.detach().to(device, dtype), pooled.detach().to(device, dtype)
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    pipe.text_encoder = pipe.text_encoder_2 = None

    # ---------------------------------------------------------------- 2. the pictures (once)
    tr._set(message=f"Preparing your {len(pics)} pictures...")
    vae = pipe.vae
    upcast = light_vae is None and dtype == torch.float16 and getattr(vae.config, "force_upcast", False)
    vae.to(device, dtype=torch.float32 if upcast else dtype)
    try:
        vae.enable_tiling()
    except Exception:
        pass
    latents = []
    with torch.no_grad():
        for path in pics:
            check_cancel()
            img = _load_square(path, resolution)
            for flip in (False, True):      # also learn from the mirror image
                im = img.transpose(0) if flip else img   # 0 = FLIP_LEFT_RIGHT
                arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
                x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device, dtype=vae.dtype)
                lat = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
                latents.append(lat.to("cpu", dtype=dtype))
    vae.to("cpu")
    pipe.vae = None
    del vae, light_vae
    if cuda:
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------- 3. the add-on to train
    tr._set(message="Setting up training...")
    unet = pipe.unet
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)
    del pipe
    unet.requires_grad_(False)
    unet.to(device, dtype=dtype)
    unet.enable_gradient_checkpointing()
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:                                  # train the add-on in full precision
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda") if cuda else None
    time_ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]], device=device, dtype=dtype)
    added = {"text_embeds": pooled, "time_ids": time_ids}
    unet.train()
    n_train = noise_sched.config.num_train_timesteps
    warmup = min(50, steps // 10)

    # ---------------------------------------------------------------- 4. train
    t_start, losses = time.time(), []
    for step in range(steps):
        check_cancel()
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / max(1, warmup))
        lat = random.choice(latents).to(device)
        noise = torch.randn_like(lat)
        noise = noise + 0.0357 * torch.randn(lat.shape[0], lat.shape[1], 1, 1, device=device, dtype=lat.dtype)
        t = torch.randint(0, n_train, (1,), device=device).long()
        noisy = noise_sched.add_noise(lat, noise, t)
        with torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32):
            pred = unet(noisy, t, encoder_hidden_states=pe, added_cond_kwargs=added).sample
        loss = F.mse_loss(pred.float(), noise.float())
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if step % 5 == 0 or step == steps - 1:
            per = (time.time() - t_start) / (step + 1)
            avg = sum(losses[-50:]) / len(losses[-50:])
            tr._set(step=step + 1, eta=int(per * (steps - step - 1)), loss=round(avg, 4),
                    message=f"Training... ({resolution}x{resolution})")
        if step == 0:
            log.info("training %s: %d pictures, %d steps, %d px, rank %d", style_id, len(pics), steps, resolution, rank)
    if not all(map(math.isfinite, losses[-10:])):
        raise RuntimeError("Training went wrong (the numbers blew up). Try again with different pictures.")

    # ---------------------------------------------------------------- 5. save
    tr._set(message="Saving your style...")
    unet.eval()
    lora = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    lora = {k: v.to(torch.float16) for k, v in lora.items()}
    StableDiffusionXLPipeline.save_lora_weights(save_directory=str(folder), unet_lora_layers=lora,
                                                safe_serialization=True)
    meta = mystyles.get(style_id) or meta
    meta.pop("ready", None)
    meta.pop("has_thumb", None)
    meta.update(trigger=trigger, base_model=base_model, steps=steps, resolution=resolution,
                minutes=round((time.time() - t_start) / 60, 1), pictures=len(pics))
    mystyles.save_meta(style_id, meta)
    log.info("trained %s in %.1f min (last loss %.4f)", style_id, (time.time() - t_start) / 60, losses[-1])
    del unet, opt, latents
