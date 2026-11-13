"""Build check: is every model in the app's list still downloadable, in a format the app can load?"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("AISTUDIO_HOME", os.path.join(os.environ.get("RUNNER_TEMP", "."), "aistudio-check"))

from huggingface_hub import HfApi  # noqa: E402

from aistudio import models  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

bad = 0
for m in models.CATALOG:
    try:
        info = HfApi().model_info(m.repo, files_metadata=True)
        chosen, fp16 = models.select_files([(s.rfilename, int(s.size or 0)) for s in info.siblings])
        problem = models.check_layout(chosen)
        total = sum(s for _, s in chosen) / 1024**3
        print(f"{'✔' if not problem else '✘'} {m.name:20} {m.repo:40} {len(chosen):3} files  "
              f"{total:5.2f} GB  fp16={fp16}  {problem or ''}")
        for name, size in chosen:
            print(f"      {name}  {size / 1024**2:.1f} MB")
        if problem:
            bad += 1
    except Exception as e:
        print(f"✘ {m.name}: {e!r}")
        bad += 1
sys.exit(1 if bad else 0)
