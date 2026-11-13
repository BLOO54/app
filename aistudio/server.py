"""A tiny web server that runs only on your PC (127.0.0.1) and shows the app in your browser.

Nothing is reachable from the internet or other computers - "127.0.0.1" means "this PC only".
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import APP_NAME, APP_VERSION, engine_setup, gallery, models, mystyles, paths, settings, styles, updater

log = logging.getLogger(__name__)


class App:
    """Everything the web page talks to."""

    def __init__(self, watchdog: bool = True):
        self.engine = engine_setup.EngineInstaller()
        self.downloads = models.Downloads()
        self._generator = None
        self._gen_lock = threading.Lock()
        self.last_ping = time.time() + 120     # extra time for the browser to open
        self.watchdog = watchdog
        self.nvidia = None
        self.gpu_names: list[str] = []
        self.update = {"checking": False, "available": None, "error": None, "downloading": False,
                       "progress": 0.0, "checked": False}
        self.shutdown_cb = None
        from .trainer import Trainer
        self.trainer = Trainer(lambda: self.generator)
        threading.Thread(target=self._detect_gpu, daemon=True).start()

    def _detect_gpu(self):
        self.nvidia = engine_setup.has_nvidia()
        if self.nvidia:
            self.gpu_names = engine_setup.gpu_names()

    @property
    def generator(self):
        with self._gen_lock:
            if self._generator is None:
                from .generator import Generator
                self._generator = Generator()
            return self._generator

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        eng = self.engine.status()
        eng["installed"] = (not eng["active"]) and engine_setup.is_installed()
        eng["nvidia"] = self.nvidia
        eng["gpus"] = self.gpu_names
        gen = self._generator
        return {
            "app": {"name": APP_NAME, "version": APP_VERSION, "mode": updater.install_mode()},
            "engine": eng,
            "models": [{"id": m.id, "name": m.name, "about": m.about, "approx_gb": m.approx_gb,
                        "ready": models.is_ready(m.id), "download": self.downloads.status(m.id)}
                       for m in models.CATALOG],
            "addons": [{"id": a.id, "name": a.name, "about": a.about, "approx_gb": a.approx_gb,
                        "ready": models.is_ready(a.id), "download": self.downloads.status(a.id)}
                       for a in models.ADDONS],
            "generator": {"loaded": gen.loaded_model if gen else None,
                          "gpu": gen.gpu if gen else "", "offload": gen.offload if gen else False,
                          "mode": gen.mode if gen else "",
                          "busy": gen.busy() if gen else False},
            "settings": settings.load(),
            "choices": {"styles": {k: v[0] for k, v in styles.STYLES.items()},
                        "aspects": {k: f"{v[0]} ({v[1]}x{v[2]})" for k, v in styles.ASPECTS.items()},
                        "quality": styles.QUALITY},
            "update": dict(self.update),
            "styles": mystyles.all_styles(),
            "training": self.trainer.status(),
            "pictures_folder": str(paths.pictures_dir()),
        }

    def is_busy(self) -> bool:
        gen = self._generator
        if gen and gen.busy():
            return True
        if self.engine.status()["active"] or self.update["downloading"] or self.trainer.status().get("active"):
            return True
        return any(self.downloads.status(i).get("active") for i in models.all_ids())

    # ------------------------------------------------------------ updates
    def check_update(self):
        if self.update["checking"]:
            return
        self.update.update(checking=True, error=None)
        try:
            info = updater.check_latest()
            self.update["available"] = None if info is None else {
                "version": info.version, "notes": updater.whats_new(info), "size": info.size}
            self._update_info = info
        except Exception as e:
            self.update["error"] = str(e)
        finally:
            self.update.update(checking=False, checked=True)

    def install_update(self):
        info = getattr(self, "_update_info", None)
        if info is None or self.update["downloading"]:
            return
        if updater.install_mode() != "installed":
            self.update["error"] = "Updates only work in the installed app."
            return

        def run():
            self.update.update(downloading=True, progress=0.0, error=None)
            try:
                path = updater.download(info, progress=lambda f: self.update.__setitem__("progress", f))
                self.update["progress"] = 1.0
                if self._generator:
                    self._generator.unload()
                updater.launch_installer(path)
                time.sleep(1.0)
                self.quit()
            except Exception as e:
                log.exception("update failed")
                self.update.update(downloading=False, error=str(e))
        threading.Thread(target=run, daemon=True).start()

    def quit(self):
        log.info("shutting down")
        if self.shutdown_cb:
            threading.Thread(target=self.shutdown_cb, daemon=True).start()
        threading.Timer(2.0, lambda: os._exit(0)).start()

    def watch(self):
        """Close the app (and free the graphics card) a little while after the window is closed."""
        while True:
            time.sleep(10)
            if self.watchdog and time.time() - self.last_ping > 75 and not self.is_busy():
                log.info("window closed - quitting")
                self.quit()
                return
            gen = self._generator
            # free graphics memory after 15 idle minutes (it reloads on the next picture)
            if gen and gen.pipe is not None and not gen.busy() and time.time() - gen.last_used > 900:
                gen.unload()


def save_reference(data_url: str) -> str:
    """Save a picture sent from the page (as a data: URL) and return its path."""
    from PIL import Image  # installed with the AI engine
    import io
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("That reference picture couldn't be read.")
    raw = base64.b64decode(data_url.split(",", 1)[1], validate=False)
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as e:
        raise ValueError("That file isn't a picture the app can read (try PNG or JPG).") from e
    img = img.convert("RGB")
    img.thumbnail((1536, 1536))
    folder = paths.data_dir() / "refs"
    folder.mkdir(exist_ok=True)
    target = folder / f"ref-{hashlib.sha1(raw).hexdigest()[:16]}.png"
    if not target.exists():
        img.save(target)
    old = sorted(folder.glob("ref-*.png"), key=lambda f: f.stat().st_mtime)
    for f in old[:-30]:
        f.unlink(missing_ok=True)
    return str(target)


def _decode_picture(data_url: str):
    from PIL import Image
    import io
    if not isinstance(data_url, str) or "," not in data_url:
        raise ValueError("A picture couldn't be read.")
    raw = base64.b64decode(data_url.split(",", 1)[1], validate=False)
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as e:
        raise ValueError("One of the files isn't a picture the app can read (try PNG or JPG).") from e
    return img.convert("RGB")


def create_training_style(name: str, images: list) -> str:
    """Save the pictures for a new style and return its id."""
    if not isinstance(images, list) or len(images) < 3:
        raise ValueError("Add at least 3 pictures (10-30 works best).")
    if len(images) > 60:
        raise ValueError("That's a lot of pictures - please use 60 or fewer.")
    sid = mystyles.new_id()
    folder = mystyles.folder(sid)
    (folder / "train").mkdir(parents=True, exist_ok=True)
    try:
        for i, data in enumerate(images):
            img = _decode_picture(data)
            img.thumbnail((1536, 1536))
            img.save(folder / "train" / f"{i:03d}.png")
            if i == 0:
                from PIL import ImageOps
                ImageOps.fit(img, (256, 256)).save(folder / "thumb.jpg", quality=88)
    except Exception:
        mystyles.delete(sid)
        raise
    name = (name or "").strip()[:40] or "My style"
    mystyles.save_meta(sid, mystyles.new_meta(sid, name, "trained", mystyles.make_trigger(name),
                                              pictures=len(images)))
    return sid


def is_safetensors(head: bytes) -> bool:
    import struct
    if len(head) < 10:
        return False
    n = struct.unpack("<Q", head[:8])[0]
    return 2 <= n < 100_000_000 and head[8:9] == b"{"


def open_folder(path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        log.exception("could not open folder")


def make_handler(app: App):
    web = paths.web_dir()

    class Handler(BaseHTTPRequestHandler):
        server_version = "AIImageStudio"

        def log_message(self, fmt, *args):  # keep the log quiet
            pass

        # -------------------------------------------------------- helpers
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code: int = 200):
            self._send(code, json.dumps(data).encode("utf-8"), "application/json")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 40_000_000:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except ValueError:
                return {}

        def _same_origin(self) -> bool:
            # stop other websites from poking the app through your browser
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if host.split(":")[0] not in ("127.0.0.1", "localhost"):
                return False
            if origin is None:
                return True
            return urllib.parse.urlparse(origin).hostname in ("127.0.0.1", "localhost")

        def _import_style(self, url):
            """Raw upload of a .safetensors style pack (LoRA) file."""
            q = urllib.parse.parse_qs(url.query)
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 2_000_000_000:
                return self._json({"error": "That file is empty or too big."}, 400)
            sid = mystyles.new_id()
            folder = mystyles.folder(sid)
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / mystyles.WEIGHTS
            left, head = n, b""
            with open(target, "wb") as fh:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    if len(head) < 16:
                        head += chunk[:16]
                    fh.write(chunk)
                    left -= len(chunk)
            if left > 0 or not is_safetensors(head):
                mystyles.delete(sid)
                return self._json({"error": "That isn't a .safetensors style pack file."}, 400)
            name = (q.get("name") or ["Imported style"])[0]
            trigger = (q.get("trigger") or [""])[0]
            mystyles.save_meta(sid, mystyles.new_meta(sid, name, "imported", trigger))
            return self._json({"ok": True, "id": sid})

        # -------------------------------------------------------- GET
        def do_GET(self):
            if not self._same_origin():
                return self._send(403, b"forbidden", "text/plain")
            url = urllib.parse.urlparse(self.path)
            route = url.path
            if route in ("/", "/index.html"):
                return self._send(200, (web / "index.html").read_bytes(), "text/html; charset=utf-8")
            if route == "/icon.png" and (web / "icon.png").exists():
                return self._send(200, (web / "icon.png").read_bytes(), "image/png")
            if route == "/api/status":
                return self._json(app.status())
            if route == "/api/gallery":
                return self._json({"items": gallery.items()})
            if route.startswith("/api/job/") and route.endswith("/preview.jpg"):
                gen = app._generator
                job = gen.get(route.split("/")[3]) if gen else None
                if job and job.preview:
                    return self._send(200, job.preview, "image/jpeg")
                return self._send(404, b"no preview yet", "text/plain")
            if route.startswith("/api/job/"):
                gen = app._generator
                job = gen.get(route.rsplit("/", 1)[1]) if gen else None
                return self._json(job.public() if job else {"state": "missing"}, 200 if job else 404)
            if route.startswith("/styles/") and route.endswith("/thumb.jpg"):
                sid = mystyles.safe_id(route.split("/")[2])
                f = mystyles.root() / sid / "thumb.jpg" if sid else None
                if f and f.is_file():
                    return self._send(200, f.read_bytes(), "image/jpeg")
                return self._send(404, b"not found", "text/plain")
            if route.startswith("/images/"):
                name = gallery.safe_name(urllib.parse.unquote(route[len("/images/"):]))
                p = paths.pictures_dir() / name if name else None
                if p and p.is_file():
                    extra = {}
                    if "download" in url.query:
                        extra["Content-Disposition"] = f'attachment; filename="{name}"'
                    return self._send(200, p.read_bytes(), mimetypes.guess_type(name)[0] or "image/png", extra)
                return self._send(404, b"not found", "text/plain")
            return self._send(404, b"not found", "text/plain")

        # -------------------------------------------------------- POST
        def do_POST(self):
            if not self._same_origin():
                return self._send(403, b"forbidden", "text/plain")
            url = urllib.parse.urlparse(self.path)
            route = url.path
            if route == "/api/styles/import":
                return self._import_style(url)
            body = self._body()
            if route == "/api/ping":
                app.last_ping = time.time()
                return self._json({"ok": True})
            if route == "/api/install-engine":
                app.engine.start(force_cpu=bool(body.get("cpu")))
                return self._json({"ok": True})
            if route == "/api/download-model":
                mid = str(body.get("id") or "")
                if mid not in models.all_ids():
                    return self._json({"error": "unknown model"}, 400)
                if not engine_setup.is_installed():
                    return self._json({"error": "Install the AI engine first."}, 400)
                app.downloads.start(mid)
                return self._json({"ok": True})
            if route == "/api/delete-model":
                mid = str(body.get("id") or "")
                gen = app._generator
                if gen and (gen.loaded_model == mid or (mid == "style" and gen.ip_loaded)):
                    if gen.busy():
                        return self._json({"error": "Wait until the pictures are finished."}, 400)
                    gen.unload()
                if mid in models.all_ids():
                    models.delete(mid)
                return self._json({"ok": True})
            if route == "/api/generate":
                from .generator import NotAllowed, make_job_params
                ref = body.pop("ref_image", None)
                body.pop("ref_path", None)
                try:
                    if ref and body.get("ref_mode") in ("style", "remix"):
                        if not engine_setup.is_installed():
                            raise ValueError("Install the AI engine first (Setup tab).")
                        if body["ref_mode"] == "style" and not models.is_ready("style"):
                            raise ValueError("Download the Style copier add-on first (Models tab).")
                        body["ref_path"] = save_reference(ref)
                    params = make_job_params(body)
                except NotAllowed as e:
                    return self._json({"error": str(e), "blocked": True}, 400)
                except ValueError as e:
                    return self._json({"error": str(e)}, 400)
                if not engine_setup.is_installed():
                    return self._json({"error": "Install the AI engine first (Setup tab)."}, 400)
                if app.trainer.status().get("active"):
                    return self._json({"error": "A style is training right now - wait until it's done "
                                                "(or stop it in the Train tab)."}, 400)
                if params["quality"] == "turbo" and not models.is_ready("speed"):
                    return self._json({"error": "Turbo needs the Speed pack add-on (about 1.1 GB). "
                                                "Download it in the Models tab."}, 400)
                if not models.is_ready(params["model"]):
                    return self._json({"error": f"Download the {models.get(params['model']).name} "
                                                "model first (Models tab)."}, 400)
                settings.update(model=params["model"], aspect=params["aspect"], quality=params["quality"],
                                style=params["style"], count=params["count"], safe_mode=params["safe_mode"],
                                negative=params["user_negative"])
                job = app.generator.submit(params)
                return self._json({"job": job.public()})
            if route == "/api/cancel":
                gen = app._generator
                if gen:
                    gen.cancel(str(body.get("id") or ""))
                return self._json({"ok": True})
            if route == "/api/train/start":
                if app.trainer.status().get("active"):
                    return self._json({"error": "A style is already training."}, 400)
                gen = app._generator
                if gen and gen.busy():
                    return self._json({"error": "Wait until the pictures are finished."}, 400)
                if not engine_setup.is_installed():
                    return self._json({"error": "Install the AI engine first (Setup tab)."}, 400)
                base = str(body.get("model") or "juggernaut")
                if base not in {m.id for m in models.CATALOG} or not models.is_ready(base):
                    return self._json({"error": "Download that model first (Models tab)."}, 400)
                try:
                    engine_setup.ensure_packages(["peft"])
                    sid = create_training_style(str(body.get("name") or ""), body.get("images"))
                except (ValueError, RuntimeError) as e:
                    return self._json({"error": str(e)}, 400)
                app.trainer.start(sid, base, str(body.get("length") or "normal"))
                return self._json({"ok": True, "id": sid})
            if route == "/api/train/cancel":
                app.trainer.cancel()
                return self._json({"ok": True})
            if route == "/api/styles/delete":
                tr = app.trainer.status()
                if tr.get("active") and tr.get("style_id") == body.get("id"):
                    return self._json({"error": "Stop the training first."}, 400)
                gen = app._generator
                if gen and gen.busy():
                    return self._json({"error": "Wait until the pictures are finished."}, 400)
                if gen and ("u" + str(body.get("id"))) in gen.loras:
                    gen.unload()
                return self._json({"ok": mystyles.delete(str(body.get("id") or ""))})
            if route == "/api/styles/rename":
                return self._json({"ok": mystyles.rename(str(body.get("id") or ""), str(body.get("name") or ""))})
            if route == "/api/clear-history":
                gen = app._generator
                if gen and gen.busy():
                    return self._json({"error": "Wait until the pictures are finished."}, 400)
                pictures = gallery.clear_all()
                refs = 0
                for f in (paths.data_dir() / "refs").glob("ref-*.png"):
                    try:
                        f.unlink()
                        refs += 1
                    except OSError:
                        pass
                settings.update(negative="")
                if gen:
                    gen.jobs.clear()
                log.info("history cleared")
                return self._json({"ok": True, "pictures": pictures, "refs": refs})
            if route == "/api/delete-image":
                return self._json({"ok": gallery.delete(str(body.get("file") or ""))})
            if route == "/api/open-folder":
                open_folder(paths.pictures_dir())
                return self._json({"ok": True})
            if route == "/api/settings":
                return self._json(settings.update(**body))
            if route == "/api/update/check":
                threading.Thread(target=app.check_update, daemon=True).start()
                return self._json({"ok": True})
            if route == "/api/update/install":
                app.install_update()
                return self._json({"ok": True})
            if route == "/api/quit":
                self._json({"ok": True})
                app.quit()
                return None
            return self._send(404, b"not found", "text/plain")

    return Handler


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    httpd.daemon_threads = True
    app.shutdown_cb = httpd.shutdown
    return httpd
