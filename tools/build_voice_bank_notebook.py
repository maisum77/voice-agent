"""Build the Kaggle notebook that renders the emotion voice bank.

The voice bank is the piece that carries emotion at runtime. Qwen3-TTS
0.6B silently ignores an ``instruct`` string, and the clone path never
encodes one at all, so a cloned voice cannot be asked to be amused. The way
around it is to spend the instruction model once, at build time:

  * Qwen3-TTS 1.7B VoiceDesign renders the same voice description once per
    emotional delivery, each with a delivery directive appended.
  * Those clips become the reference bank. At runtime the 0.6B-Base model
    clones from whichever clip matches the turn's emotion, and the emotional
    prosody comes along with the timbre.

The consistency check matters: a voice that drifts between the clips is
worse than no voice at all, so the ECAPA speaker-embedding cosine of every
clip against the bank mean has to clear a threshold.

Usage:
    python3 tools/build_voice_bank_notebook.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "build_voice_bank_kaggle.ipynb"

MODULES = [
    ("tools/common.py", "tools/common.py"),
    ("tools/design_voice.py", "tools/design_voice.py"),
]

MD_INTRO = """# The emotion voice bank

Builds the reference clips the runtime clones from.

```
[emotion:amused] -> bank/amused.wav
[emotion:warm]   -> bank/warm.wav
[laugh]          -> bank/nv_laugh.wav
```

## Why it is built rather than prompted

Qwen3-TTS splits into three capabilities that do not combine the way you
would like:

* `1.7B-CustomVoice` accepts an `instruct` string - but only for nine preset
  timbres.
* `1.7B-VoiceDesign` accepts an `instruct` string and can invent a voice -
  but each generation re-derives it.
* `0.6B-Base` clones a voice from a reference clip - and the clone path
  never encodes an instruction at all.

So "a brand new voice" and "emotion control" cannot both live in one runtime
call. The instruction model is used here, once, and the result is baked into
the clips. In-context cloning preserves prosody better than x-vector-only
cloning, so the emotional delivery survives the clone.

## What comes out

| file | what |
|---|---|
| `voice_bank/<emotion>.wav` | one reference clip per delivery |
| `voice_bank/nv_<name>.wav` | the non-verbal clips, in the same voice |
| `voice_bank/bank.json` | the manifest: description, instructs, transcripts |

## Cost

One GPU, about ten minutes, and 4.5 GB of downloads. It is a build step: the
output is a few megabytes and ships with the OS.
"""

CODE_INSTALL = """# --- dependencies -------------------------------------------------------------
# qwen-tts pins transformers 4.57.3, which is older than what mlx-lm wants.
# This notebook does not need mlx, so the pin is fine here.
%pip install -q --upgrade pip
%pip install -q "transformers==4.57.3" "huggingface_hub<1.0" qwen-tts soundfile numpy

import subprocess, sys
print(subprocess.run([sys.executable, "-c",
      "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"],
      capture_output=True, text=True).stdout)
"""

CODE_CONFIG = """# --- configuration -------------------------------------------------------------
import os, sys, json, shutil, glob
from pathlib import Path

WORK = Path("/kaggle/working")
PIPE = WORK / "voicepipeline"
(PIPE / "tools").mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# The voice. The same description is used for every clip, so the timbre is
# identical across the bank; only the delivery directive changes.
VOICE_DESCRIPTION = (
    "A warm British male voice in his early thirties, low and unhurried, "
    "with a dry, self-aware sense of humour and a naturally gentle manner."
)

# The line every delivery clip speaks. Neutral enough to be delivered any
# way, and short enough to keep the clips small.
REFERENCE_LINE = (
    "Right, let's get this sorted. Tell me what you are seeing and we will "
    "work through it together."
)

OUT_DIR = WORK / "voice_bank"
OUT_DIR.mkdir(parents=True, exist_ok=True)

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
"""

CODE_MODULES = "# The pipeline is embedded verbatim, so it cannot drift from the repo.\n" + "\n".join(
    f'''# ---- {rel} -> {dest} -------------------------------------------------
from pathlib import Path as _P
_d = _P(WORK) / "voicepipeline" / "{dest}"
_d.parent.mkdir(parents=True, exist_ok=True)
_d.write_text(r\'\'\'
{"" if False else (ROOT / rel).read_text(encoding="utf-8")}\'\'\', encoding="utf-8")
print("wrote", _d)
'''
    for rel, dest in MODULES
)

CODE_RUN = """# --- render the bank -----------------------------------------------------------
sys.path.insert(0, str(PIPE / "tools"))
import design_voice

# The module's own defaults point at its own tree; override the two that
# matter for this run.
design_voice.VOICE_DIR = OUT_DIR
design_voice.SPEAKER_HINT = VOICE_DESCRIPTION

import time
t0 = time.time()
manifest = design_voice.render_bank(
    MODEL_ID,
    OUT_DIR,
    VOICE_DESCRIPTION,
    device="cuda",
    dtype="bfloat16",
)
print(f"rendered in {(time.time() - t0) / 60:.1f} min")
"""

CODE_CHECK = """# --- the voice has to be the same voice in every clip ---------------------------
# A voice that drifts between deliveries is worse than no voice, so the ECAPA
# speaker-embedding cosine of each clip against the bank mean has to clear
# the threshold. Below it, the clips are re-rendered with a tighter
# description.
import design_voice
import json

result = design_voice.check_consistency(OUT_DIR, threshold=0.80)
print(json.dumps(result, indent=2))

if result.get("checked") and not result.get("passed"):
    print("FAIL: the voice drifts between deliveries - re-render with a tighter description")
else:
    print("voice is consistent across the bank")
"""

CODE_SAVE = """# --- save ----------------------------------------------------------------------
import shutil, json
from pathlib import Path

# Everything the runtime needs, in one folder.
BUNDLE = WORK / "voice_bank_bundle"
if BUNDLE.exists():
    shutil.rmtree(BUNDLE)
BUNDLE.mkdir(parents=True)
for p in sorted(OUT_DIR.iterdir()):
    if p.is_file():
        shutil.copy2(p, BUNDLE / p.name)

manifest = json.loads((OUT_DIR / "bank.json").read_text())
print("bank contents:")
for name, info in manifest["clips"].items():
    print(f"  {name:10s} {info['duration_s']:.2f}s  {info['file']}")
for name, info in manifest.get("nonverbals", {}).items():
    print(f"  nv_{name:7s} {info['duration_s']:.2f}s  {info['file']}")

total = sum(p.stat().st_size for p in BUNDLE.iterdir())
print(f"\\ntotal {total / 1e6:.2f} MB in {BUNDLE}")
print("Download this folder - it is the voice bank.")
"""


def build() -> list[dict]:
    cells: list[dict] = []

    def md(src: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})

    def code(src: str) -> None:
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": src.splitlines(keepends=True),
            }
        )

    md(MD_INTRO)
    code(CODE_INSTALL)
    code(CODE_CONFIG)
    code(CODE_MODULES)
    code(CODE_RUN)
    code(CODE_CHECK)
    code(CODE_SAVE)
    return cells


def validate(cells: list[dict]) -> None:
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        python = "\n".join(
            ("#" + line[1:]) if line.lstrip()[:1] in ("!", "%") else line
            for line in src.splitlines()
        )
        try:
            ast.parse(python)
        except SyntaxError as exc:
            lines = python.splitlines()
            ln = exc.lineno or 1
            context = "\n".join(
                f"  {j + 1:4d}| {lines[j][:110]}"
                for j in range(max(0, ln - 3), min(len(lines), ln + 2))
            )
            raise SystemExit(f"cell {i} does not compile: {exc}\n{context}")

    for rel, _ in MODULES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        if "'''" in src:
            raise SystemExit(f"{rel} contains a triple-single-quote")


def main() -> int:
    cells = build()
    validate(cells)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {OUT}  ({len(cells)} cells, {n_code} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
