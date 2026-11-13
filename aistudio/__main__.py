"""Start AI Image Studio.

    pythonw -m aistudio                      normal start (opens the app window)
    python  -m aistudio --self-test FILE     quick check used by the build machine
    python  -m aistudio --install-engine [--cpu]
    python  -m aistudio --gen-test MODEL OUT.png [--steps N]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from logging.handlers import RotatingFileHandler

from . import APP_NAME, APP_VERSION, PORT, paths


def fix_missing_console():
    """When started without a console window (pythonw), print() and progress bars have
    nowhere to write and crash with "'NoneType' object has no attribute 'write'".
    Give them a do-nothing place to write instead."""
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")


def setup_logging():
    handlers = [RotatingFileHandler(paths.log_file(), maxBytes=1_000_000, backupCount=2, encoding="utf-8"),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def already_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=3) as r:
            return json.loads(r.read()).get("app", {}).get("name") == APP_NAME
    except Exception:
        return False


def open_window(url: str):
    """Open as its own app-style window (Microsoft Edge 'app mode'), or a normal browser tab."""
    if sys.platform == "win32":
        candidates = [
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                         r"Microsoft\Edge\Application\msedge.exe"),
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                         r"Microsoft\Edge\Application\msedge.exe"),
        ]
        for edge in candidates:
            if os.path.exists(edge):
                try:
                    subprocess.Popen([edge, f"--app={url}", "--window-size=1400,900"], close_fds=True)
                    return
                except OSError:
                    pass
    webbrowser.open(url)


def self_test(report: str) -> int:
    """Start the server, ask it for its status, and write what we found."""
    from .server import App, make_server
    lines, ok = [], True

    def check(name, fn):
        nonlocal ok
        try:
            detail = fn()
            lines.append(f"✔ {name}: {detail}")
        except Exception as e:
            ok = False
            lines.append(f"✘ {name}: {e!r}")

    check("Python", lambda: sys.version.split()[0] + " at " + sys.executable)
    check("Web page", lambda: f"{(paths.web_dir() / 'index.html').stat().st_size} bytes")
    check("Data folder", lambda: str(paths.data_dir()))
    check("Pictures folder", lambda: str(paths.pictures_dir()))
    check("pip", lambda: subprocess.run([paths.python_exe(), "-m", "pip", "--version"], capture_output=True,
                                        text=True, check=True, timeout=60).stdout.strip())

    port = 7999
    app = App(watchdog=False)
    httpd = make_server(app, port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    def status():
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=10) as r:
            d = json.loads(r.read())
        assert d["app"]["version"] == APP_VERSION
        return f"v{d['app']['version']}, {len(d['models'])} models, mode={d['app']['mode']}"
    check("Server answers", status)

    def page():
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as r:
            html = r.read().decode("utf-8")
        assert APP_NAME in html
        return "ok"
    check("Page loads", page)
    httpd.shutdown()
    lines.append("RESULT: " + ("PASS" if ok else "FAIL"))
    text = "\n".join(lines)
    with open(report, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    return 0 if ok else 1


def install_engine(cpu: bool) -> int:
    from .engine_setup import EngineInstaller
    inst = EngineInstaller()
    inst.run(force_cpu=cpu)
    st = inst.status()
    print(json.dumps(st, indent=2))
    return 0 if not st["error"] else 1


def gen_test(model: str, out: str, steps: int) -> int:
    """Draw one small picture with any diffusers SDXL model (the build uses a tiny test model)."""
    import torch
    from .generator import Generator
    g = Generator.__new__(Generator)   # no worker thread needed
    g.offload, g.gpu = False, ""
    t0 = time.time()
    pipe = g.load_pipeline(model, fp16_files=False)
    for sched in ("euler_a", "dpmpp_2m_karras"):
        g._set_scheduler(pipe, sched)
        img = pipe(prompt="a red apple on a table", negative_prompt="blurry",
                   num_inference_steps=steps, guidance_scale=5.0,
                   generator=torch.Generator("cpu").manual_seed(1),
                   callback_on_step_end=lambda p, s, t, kw: kw).images[0]
    img.save(out)
    result = {"ok": True, "size": list(img.size), "seconds": round(time.time() - t0, 1),
              "torch": torch.__version__, "device": g.device}
    import diffusers
    import transformers
    result.update(diffusers=diffusers.__version__, transformers=transformers.__version__)
    print(json.dumps(result))
    return 0


def download_test(model_id: str, report: str) -> int:
    """Download a model exactly like the Download button does, then write the result to a file."""
    from . import models
    d = models.Downloads()
    d.start(model_id)
    while d.status(model_id).get("active"):
        time.sleep(2)
    st = d.status(model_id)
    st["ready"] = models.is_ready(model_id)
    st["info"] = models.ready_info(model_id)
    with open(report, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=2)
    return 0 if st["ready"] and not st.get("error") else 1


def main(argv=None) -> int:
    fix_missing_console()
    ap = argparse.ArgumentParser(prog="aistudio")
    ap.add_argument("--self-test", metavar="REPORT")
    ap.add_argument("--install-engine", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--gen-test", nargs=2, metavar=("MODEL", "OUT"))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--download-test", nargs=2, metavar=("MODEL_ID", "REPORT"))
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    setup_logging()
    log = logging.getLogger("aistudio")
    log.info("%s %s starting (%s)", APP_NAME, APP_VERSION, sys.executable)

    if args.self_test:
        return self_test(args.self_test)
    if args.install_engine:
        return install_engine(args.cpu)
    if args.download_test:
        return download_test(*args.download_test)
    if args.gen_test:
        return gen_test(args.gen_test[0], args.gen_test[1], args.steps)

    url = f"http://127.0.0.1:{args.port}/"
    if port_in_use(args.port):
        if already_running(args.port):
            open_window(url)
            return 0
        log.error("port %d is used by another program", args.port)
        return 1

    from .server import App, make_server
    app = App(watchdog=not args.no_browser)
    httpd = make_server(app, args.port)
    threading.Thread(target=app.watch, daemon=True).start()
    if not args.no_browser:
        threading.Timer(0.5, open_window, args=(url,)).start()
    log.info("listening on %s", url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
