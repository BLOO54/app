import json
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_tmp = tempfile.mkdtemp()
os.environ["AISTUDIO_HOME"] = os.path.join(_tmp, "home")
os.environ["AISTUDIO_PICTURES"] = os.path.join(_tmp, "pics")

from aistudio import gallery, models, safety, settings, styles, updater  # noqa: E402
from aistudio.generator import NotAllowed, make_job_params  # noqa: E402


# ---------------------------------------------------------------- safety
@pytest.mark.parametrize("prompt", [
    "a naked child", "sexy 12 year old", "nsfw teen girl", "loli hentai", "schoolgirl lingerie",
    "13yo nude", "a 15-year-old in a bikini", "Kids, NSFW",
])
def test_blocks_minor_sexual(prompt):
    assert safety.check_prompt(prompt)


@pytest.mark.parametrize("prompt", [
    "a cat wearing a hat", "kids playing soccer in a park", "a nude marble statue in a museum",
    "teenage mutant robot fighting a dragon", "sexy sports car on a mountain road",
    "a 30 year old wizard", "skeleton kid from a roblox game", "child's drawing of a house",
])
def test_allows_normal(prompt):
    assert safety.check_prompt(prompt) is None


def test_safe_negative():
    assert "nude" in safety.build_negative("", True)
    assert safety.build_negative("dogs", False) == "dogs"
    assert safety.build_negative("dogs", True).startswith("dogs, ")


# ---------------------------------------------------------------- model files
def test_select_prefers_fp16_and_skips_big_checkpoint():
    sib = [("model_index.json", 600), ("Juggernaut.safetensors", 7_000_000_000), ("README.md", 4000),
           ("unet/config.json", 1), ("unet/diffusion_pytorch_model.safetensors", 10),
           ("unet/diffusion_pytorch_model.fp16.safetensors", 5),
           ("vae/config.json", 1), ("vae/diffusion_pytorch_model.fp16.safetensors", 2),
           ("vae/diffusion_pytorch_model.safetensors", 4),
           ("text_encoder/model.fp16.safetensors", 1), ("text_encoder/model.safetensors", 2),
           ("text_encoder/pytorch_model.bin", 2),
           ("tokenizer/vocab.json", 1), ("tokenizer/merges.txt", 1), ("scheduler/scheduler_config.json", 1)]
    chosen, fp16 = models.select_files(sib)
    names = {n for n, _ in chosen}
    assert fp16
    assert "Juggernaut.safetensors" not in names and "README.md" not in names
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in names
    assert "unet/diffusion_pytorch_model.safetensors" not in names
    assert "text_encoder/pytorch_model.bin" not in names
    assert {"tokenizer/merges.txt", "scheduler/scheduler_config.json", "model_index.json"} <= names
    assert models.check_layout(chosen) is None


def test_select_without_fp16_everywhere_uses_full_files():
    sib = [("model_index.json", 1), ("unet/diffusion_pytorch_model.fp16.safetensors", 5),
           ("unet/diffusion_pytorch_model.safetensors", 10), ("vae/diffusion_pytorch_model.safetensors", 4)]
    chosen, fp16 = models.select_files(sib)
    assert not fp16
    assert {n for n, _ in chosen} == {"model_index.json", "unet/diffusion_pytorch_model.safetensors",
                                      "vae/diffusion_pytorch_model.safetensors"}


def test_layout_problems():
    assert models.check_layout([("unet/a.safetensors", 1)])
    assert models.check_layout([("model_index.json", 1)])


# ---------------------------------------------------------------- job params
def test_job_params_defaults():
    p = make_job_params({"prompt": "a castle", "style": "anime", "aspect": "portrait", "count": 99})
    assert p["width"] == 832 and p["height"] == 1216
    assert p["count"] == 8
    assert "anime" in p["full_prompt"] and p["full_prompt"].startswith("a castle")
    assert 0 <= p["seed"] < 2**31
    assert "nude" in p["negative"]          # safe mode on by default


def test_job_params_seed_and_safe_off():
    p = make_job_params({"prompt": "x", "seed": "42", "safe_mode": False, "negative": "cats", "model": "animagine"})
    assert p["seed"] == 42
    assert "nude" not in p["negative"] and p["negative"].startswith("cats")
    assert p["full_prompt"].endswith("absurdres")


def test_job_params_errors():
    with pytest.raises(ValueError):
        make_job_params({"prompt": "   "})
    with pytest.raises(NotAllowed):
        make_job_params({"prompt": "naked child"})


def test_every_model_has_all_quality_steps():
    for m in models.CATALOG:
        assert set(m.steps) == set(styles.QUALITY)


def test_style_addon_plan():
    files = models.STYLE_ADDON.files
    sib = [(f, 100) for f in files] + [("models/other.bin", 5)]
    chosen, fp16 = models.plan_download("style", sib)
    assert [n for n, _ in chosen] == files and not fp16
    with pytest.raises(RuntimeError):
        models.plan_download("style", sib[1:])
    assert models.repo_of("style") == "h94/IP-Adapter"
    assert "style" in models.all_ids() and "juggernaut" in models.all_ids()


def test_reference_params():
    p = make_job_params({"prompt": "x", "ref_path": "/tmp/r.png", "ref_mode": "style", "ref_strength": 5})
    assert p["ref_mode"] == "style" and p["ref_strength"] == 1.0
    p = make_job_params({"prompt": "x", "ref_path": "/tmp/r.png", "ref_mode": "remix", "ref_strength": 0.01})
    assert p["ref_mode"] == "remix" and p["ref_strength"] == 0.2
    p = make_job_params({"prompt": "x", "ref_mode": "style"})          # no picture -> normal job
    assert p["ref_mode"] is None and p["ref_path"] is None
    p = make_job_params({"prompt": "x", "ref_path": "/etc/passwd", "ref_mode": "hack"})
    assert p["ref_mode"] is None and p["ref_path"] is None


# ---------------------------------------------------------------- misc
def test_gallery_names():
    assert gallery.safe_name("a.png") == "a.png"
    for bad in ["../a.png", "a/b.png", "..\\x.png", "x.txt", ".hidden.png", ""]:
        assert gallery.safe_name(bad) is None


def test_gallery_add_list_delete():
    from aistudio import paths
    (paths.pictures_dir() / "t1.png").write_bytes(b"x")
    gallery.add({"file": "t1.png", "prompt": "p"})
    assert gallery.items()[0]["file"] == "t1.png"
    assert gallery.delete("t1.png")
    assert all(i["file"] != "t1.png" for i in gallery.items())


def test_settings_roundtrip():
    settings.update(aspect="wide", junk=1)
    s = settings.load()
    assert s["aspect"] == "wide" and "junk" not in s


def test_versions():
    assert updater.is_newer("v1.0.1", "1.0.0")
    assert not updater.is_newer("1.0.0", "1.0.0")


# ---------------------------------------------------------------- server (no AI engine needed)
@pytest.fixture(scope="module")
def server():
    from aistudio.server import App, make_server
    app = App(watchdog=False)
    httpd = make_server(app, 7998)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:7998"
    httpd.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def _post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}") if e.code != 403 else {}


def test_server_status_and_page(server):
    code, body = _get(server + "/api/status")
    d = json.loads(body)
    assert code == 200 and d["app"]["name"] == "AI Image Studio"
    assert {m["id"] for m in d["models"]} == {"juggernaut", "animagine"}
    code, html = _get(server + "/")
    assert b"AI Image Studio" in html


def test_server_blocks_bad_prompt_and_other_sites(server):
    code, d = _post(server + "/api/generate", {"prompt": "nude child"})
    assert code == 400 and d.get("blocked")
    code, _ = _post(server + "/api/ping", {}, headers={"Origin": "https://evil.example"})
    assert code == 403


def test_server_reference_upload(server):
    import base64
    import io
    from PIL import Image
    from aistudio import engine_setup
    buf = io.BytesIO()
    Image.new("RGB", (40, 30), (200, 10, 10)).save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    from aistudio.server import save_reference
    path = save_reference(url)
    assert os.path.exists(path) and Image.open(path).size == (40, 30)
    with pytest.raises(ValueError):
        save_reference("data:image/png;base64,bm90IGFuIGltYWdl")
    # style mode without the add-on gives a friendly error (or engine missing) - never a crash
    code, d = _post(server + "/api/generate", {"prompt": "a cat", "ref_image": url, "ref_mode": "style"})
    assert code == 400 and ("Style copier" in d["error"] or "engine" in d["error"])
    assert engine_setup is not None


def test_clear_history(server):
    from aistudio import paths
    pics = paths.pictures_dir()
    (pics / "20261113-120000-42.png").write_bytes(b"x")
    gallery.add({"file": "20261113-120000-42.png", "prompt": "secret prompt"})
    (pics / "my-own-file.png").write_bytes(b"keep me")
    refs = paths.data_dir() / "refs"
    refs.mkdir(exist_ok=True)
    (refs / "ref-abc.png").write_bytes(b"x")
    settings.update(negative="secret negative")
    code, d = _post(server + "/api/clear-history", {})
    assert code == 200 and d["pictures"] >= 1 and d["refs"] >= 1
    assert not (pics / "20261113-120000-42.png").exists()
    assert (pics / "my-own-file.png").exists()          # not ours - left alone
    assert not (pics / "gallery.jsonl").exists()
    assert not list(refs.glob("ref-*.png"))
    assert settings.load()["negative"] == ""
    assert gallery.items() == []


def test_server_image_path_traversal(server):
    with pytest.raises(urllib.error.HTTPError):
        _get(server + "/images/..%2F..%2Fsettings.json")
