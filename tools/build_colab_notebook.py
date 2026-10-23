"""Build the self-contained Colab training notebook (v3).

The v2 notebook had two fatal mistakes:

1. It merged the persona adapter into the base and then attached the same
   persona adapter again in the trainer. The model started from persona^2.
2. It started from that persona adapter, not from the Oct-22 emotion adapter.
   The run therefore had to learn the emotion tags from scratch and overwrote
   the behaviour that already worked.

v3 starts from the Oct-22 emotion adapter on a clean text-only base. Stage A
continues it on a replayed, label-balanced emotion set. Stage B continues the
result on the expanded tool data plus an emotion replay. The notebook is still
self-contained: the pipeline modules and every dataset are embedded, and the
only upload is the small v1 adapter zip.

Usage:
    python3 tools/build_colab_notebook.py
"""

from __future__ import annotations

import ast
import base64
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "train_emotion_colab.ipynb"

MODULES = [
    ("tools/common.py", "tools/common.py"),
    ("tools/normalize_spoken.py", "tools/normalize_spoken.py"),
    ("tools/train_emotion_lora.py", "tools/train_emotion_lora.py"),
    ("tools/eval_emotion.py", "tools/eval_emotion.py"),
    ("tools/eval_tools.py", "tools/eval_tools.py"),
    ("tools/test_tags.py", "tools/test_tags.py"),
    ("pipeline/__init__.py", "pipeline/__init__.py"),
    ("pipeline/tts/segmenter.py", "pipeline/tts/segmenter.py"),
    ("pipeline/emotion/__init__.py", "pipeline/emotion/__init__.py"),
    ("pipeline/emotion/guard.py", "pipeline/emotion/guard.py"),
    ("pipeline/tools/__init__.py", "pipeline/tools/__init__.py"),
    ("pipeline/tools/schema.py", "pipeline/tools/schema.py"),
    ("pipeline/tools/parse.py", "pipeline/tools/parse.py"),
    ("pipeline/tools/policy.py", "pipeline/tools/policy.py"),
]

DATASETS = [
    ("data/v3/emotion_train.jsonl", "v3/emotion_train.jsonl"),
    ("data/v3/emotion_valid.jsonl", "v3/emotion_valid.jsonl"),
    ("data/v3/stage_b_train.jsonl", "v3/stage_b_train.jsonl"),
    ("data/v3/stage_b_valid.jsonl", "v3/stage_b_valid.jsonl"),
    ("data/tool/v3/eval.jsonl", "tool/v3/eval.jsonl"),
    ("data/tool/v3/eval_legacy.jsonl", "tool/v3/eval_legacy.jsonl"),
    ("data/v2/test_annotated.jsonl", "v2/test_annotated.jsonl"),
    ("data/eval/serious_probes.jsonl", "eval/serious_probes.jsonl"),
    ("data/eval/v3_clean_probes.jsonl", "eval/v3_clean_probes.jsonl"),
]


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def embed_module(rel: str, dest: str) -> str:
    src = read(rel)
    if "'''" in src:
        raise SystemExit(f"{rel} contains a triple-single-quote")
    if src.rstrip("\n").endswith("\\"):
        raise SystemExit(f"{rel} ends with a backslash")
    return f'''# ---- {rel} -> {dest} ----
from pathlib import Path as _P
_d = _P(WORK) / "{dest}"
_d.parent.mkdir(parents=True, exist_ok=True)
_d.write_text(r\'\'\'{src.rstrip()}\'\'\', encoding="utf-8")
print("wrote", _d)
'''


def blob(rel: str) -> str:
    raw = (ROOT / rel).read_bytes()
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            json.loads(line)
    return base64.b64encode(gzip.compress(raw, 9)).decode("ascii")


MD_INTRO = """# Emotion + tool adapter for Qwen3.5-0.8B (v3)

This notebook continues the **Oct-22 emotion adapter**, not the bare persona
adapter. That distinction is the whole point of v3: the Oct-22 adapter already
speaks `amused`, `warm`, and `neutral` well, so the new run only has to add
`serious`, `focused`, and the tool channel without overwriting the rest.

## What changed from v2

* v2 merged the persona into the base **and then attached the persona adapter
  again**, and it started from the persona rather than the emotion adapter.
  The result was persona^2 and a model that emitted invalid tags and
  `[emotion:serious]` almost everywhere.
* v3 starts from the Oct-22 adapter on a clean text-only base.
* Stage A adds serious/focused data with a replay of the v1 delivery for the
  labels v1 already handles, and balances every label to 100 examples.
* Stage B adds the expanded tool set (21 functions, 1,108 examples) plus an
  emotion replay, at a lower learning rate.
* The tool eval is now split in two: a **legacy** file (seen intents; 39 of
  its 75 prompts also appear in training, so it is a regression only) and a
  **32-row unseen-intent/ambiguity** file with explicit arguments. The unseen
  file is the shipping gate.

## What to do

1. Runtime -> **Change runtime type -> T4 GPU**.
2. **Run all**. When prompted, upload `v1_emotion_adapter.zip` (23 MB).
3. Wait roughly 90-120 minutes. Both stages save the latest adapter to Drive
   after every epoch. If Colab disconnects, the last saved adapter is still
   there; rerun the failed stage from it (or run Stage B directly if Stage A
   finished).
4. Download `emotion_adapter_v3.zip` at the end.

Everything else is embedded. The persona zip is **not** needed: the v1
adapter already contains the persona.
"""

MD_DATA = """## The data

| file | rows | role |
|---|---:|---|
| `v3/emotion_train.jsonl` | 500 | 100 per label; v1 replay for amused/warm/neutral, hand-written serious/focused, boundary examples |
| `v3/emotion_valid.jsonl` | 55 | held-out emotion validation |
| `v3/stage_b_train.jsonl` | 1,408 | 1,108 tool examples + 300 emotion replay |
| `v3/stage_b_valid.jsonl` | 112 | 57 tool + 55 emotion |
| `eval/v3_clean_probes.jsonl` | 30 | independent clean emotion check (also gated) |
| `tool/v3/eval.jsonl` | 32 | unseen intents + ambiguity checks; **shipping gate** |
| `tool/v3/eval_legacy.jsonl` | 75 | the v2 eval; seen intents, 39 exact prompts in train; regression only |

The replay rows keep the v1 sentence but replace the tag with the ground-truth
label. Serious and focused rows keep their hand-written targets, because those
are the behaviours v1 did not have. The hard negatives are close pairs:
frustration versus data loss, a work request versus a risky request, playful
versus dangerous.
"""

MD_AFTER = """## What just happened

1. The clean text-only base was saved with the EOS contract
   `<|im_end|>` = 248046.
2. Stage A continued the Oct-22 adapter for 3 epochs on the balanced emotion
   set at `lr=1e-5`.
3. Stage B continued the Stage A adapter for 2 epochs on the tool set plus
   emotion replay at `lr=1e-5`.
4. Both stages were evaluated against the **v1 adapter**, not the persona, so
   the regression numbers mean something.
5. The tool channel was evaluated on the unseen-intent gate and on the legacy
   v2 file.

## Bring it home

`emotion_adapter_v3.zip` contains:

```
agents3_emotion_adapter/
  adapter/                    # final PEFT adapter (Stage B)
  train_report.json
  train_report_stage_a.json
  eval_report.json            # final emotion + serious gates (98 probes)
  eval_report_clean.json      # independent clean check (30 probes)
  eval_emotion_stage_a.json
  tool_eval_report.json       # unseen-intent gate
  tool_eval_legacy_report.json
  data_report.json
  README.md
```

Convert and evaluate locally. Do **not** overwrite the v1 adapter directories:

```bash
mkdir -p models/llm/v3_upload
unzip -o ~/Downloads/emotion_adapter_v3.zip -d models/llm/v3_upload

.venv/bin/python tools/convert_peft_to_mlx_adapter.py \
    --peft models/llm/v3_upload/agents3_emotion_adapter/adapter \
    --out models/llm/agents3_emotion_mlx_adapter_v3

.venv/bin/python tools/eval_emotion_mlx.py \
    --adapter models/llm/agents3_emotion_mlx_adapter_v3 \
    --annotated data/v2/test_annotated.jsonl,data/eval/serious_probes.jsonl \
    --out data/v3/eval_emotion_v3_mlx.json --gate

.venv/bin/python tools/eval_tools.py \
    --data data/tool/v3/eval.jsonl \
    --base models/llm/agents3_text_base \
    --adapter models/llm/v3_upload/agents3_emotion_adapter/adapter \
    --out data/v3/eval_tools_v3.json --device mps --gate
```

The 98-probe emotion report is the discriminator. The clean 30 is a
fresh-prompt sanity check; v1 already passes it, so a drop there means the new
run became over-serious.

The runtime contract is `[emotion:X]<tool_call>{json}</tool_call>` followed by
speech. Policy validates before execution and holds destructive calls for
confirmation; the segmenter strips the tool block before TTS.
"""

CODE_INSTALL = """# --- dependencies -----------------------------------------------------------
%pip install -q --upgrade pip
%pip install -q -U "transformers>=5.0" peft accelerate safetensors

import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("vram: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
"""

CODE_CONFIG = """# --- configuration -----------------------------------------------------------
import os, sys, json, glob, shutil, subprocess, zipfile, re, math, time, random
import base64, gzip
from pathlib import Path

WORK = Path("/content/voicepipeline")
DATA = WORK / "data"
MODELS = WORK / "models"
for d in (WORK, DATA, MODELS):
    d.mkdir(parents=True, exist_ok=True)

STUDENT_BASE = "Qwen/Qwen3.5-0.8B"

# Stage A: emotion only, short sequences.
EPOCHS_A = 3
LR_A = 1e-5
MAX_LEN_A = 448
# Stage B: tools + replay, longer tool prompt.
EPOCHS_B = 2
LR_B = 1e-5
MAX_LEN_B = 704
BATCH = 2
GRAD_ACCUM = 8

USE_DRIVE = True
CKPT = None
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        CKPT = Path("/content/drive/MyDrive/voice_emotion")
        CKPT.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        CKPT = None
        print("Drive not mounted, using /content:", exc)

ROOT_OUT = (CKPT if CKPT else (MODELS / "llm"))
OUT_A = ROOT_OUT / "agents3_emotion_stage_a"
OUT = ROOT_OUT / "agents3_emotion_adapter"
for _d in (OUT_A, OUT):
    _d.mkdir(parents=True, exist_ok=True)
print("stage A:", OUT_A)
print("final:  ", OUT)

import torch
if not torch.cuda.is_available():
    print("*** no GPU: switch the runtime to T4 before training ***")
"""

CODE_MODULES = "# The pipeline is embedded verbatim from the repository.\n" + "\n".join(
    embed_module(rel, dest) for rel, dest in MODULES
)

CODE_DATA = """# --- embedded datasets -------------------------------------------------------
_BLOBS = {
""" + "".join(
    f'    "{dest}": "{blob(rel)}",\n' for rel, dest in DATASETS
) + """}

for _rel, _b64 in _BLOBS.items():
    _p = DATA / _rel
    _p.parent.mkdir(parents=True, exist_ok=True)
    _p.write_bytes(gzip.decompress(base64.b64decode(_b64)))
    print("wrote", _p, f"{_p.stat().st_size / 1024:.0f} KB")
"""

CODE_IMPORT = """# --- import the pipeline ----------------------------------------------------
for _p in (WORK, WORK / "tools", WORK / "pipeline", WORK / "pipeline" / "tts"):
    sys.path.insert(0, str(_p))

import common
from common import (
    DELIVERY_EMOTIONS, NONVERBALS, load_persona, read_jsonl, write_jsonl, write_json,
)
import segmenter
from segmenter import parse_tagged_reply, spoken_text

print("emotions:", DELIVERY_EMOTIONS)
print("nonverbals:", NONVERBALS)
"""

CODE_TEST = """# --- cheap sanity checks -----------------------------------------------------
sys.path.insert(0, str(WORK / "tools"))
sys.path.insert(0, str(WORK / "pipeline"))
import test_tags  # noqa: F401  (the module runs its assertions on import)
from emotion.guard import serious_reasons

for _name in ("serious_probes.jsonl", "v3_clean_probes.jsonl"):
    _probes = read_jsonl(DATA / "eval" / _name)
    _miss = [r["base_user"] for r in _probes
             if r["base_emotion"] == "serious" and not serious_reasons(r["base_user"])]
    _false = [r["base_user"] for r in _probes
              if r["base_emotion"] != "serious" and serious_reasons(r["base_user"])]
    print(f"guard {_name}: {len(_miss)} serious misses, {len(_false)} false alarms")
    if _miss or _false:
        raise SystemExit("guard coverage regression; fix pipeline/emotion/guard.py first")

_emo = read_jsonl(DATA / "v3" / "emotion_train.jsonl")
_tool = read_jsonl(DATA / "v3" / "stage_b_train.jsonl")
print("emotion train:", len(_emo), "| stage B train:", len(_tool))
print("tool gate:", len(read_jsonl(DATA / "tool" / "v3" / "eval.jsonl")),
      "| legacy:", len(read_jsonl(DATA / "tool" / "v3" / "eval_legacy.jsonl")))
"""

CODE_UPLOAD = """# --- the Oct-22 emotion adapter ----------------------------------------------
# This is the adapter v2 forgot to continue. It already contains the persona.
zip_path = None
for _root in (Path("/content"), Path("/content/drive")):
    if not _root.exists():
        continue
    for _pat in ("v1_emotion_adapter*.zip", "emotion_adapter-*.zip", "emotion_adapter*.zip"):
        _hits = sorted(_root.rglob(_pat))
        if _hits:
            zip_path = _hits[0]
            break
    if zip_path is not None:
        break

if zip_path is None:
    from google.colab import files
    print("Upload v1_emotion_adapter.zip (the 23 MB Oct-22 adapter):")
    _up = files.upload()
    if not _up:
        raise SystemExit("No file uploaded")
    zip_path = Path("/content") / next(iter(_up))

print("using", zip_path)
RAW = MODELS / "llm" / "agents3_v1_upload"
if RAW.exists():
    shutil.rmtree(RAW)
RAW.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as _z:
    _z.extractall(RAW)

# The zip may be flat, nested one directory, or be the full 49 MB report
# bundle. Find a directory that has both files.
_candidates = [
    p.parent for p in RAW.rglob("adapter_config.json")
    if (p.parent / "adapter_model.safetensors").exists()
]
if not _candidates:
    raise SystemExit("no adapter_config.json + adapter_model.safetensors in the zip")

# The 49 MB report bundle contains both the patched persona adapter and the
# trained emotion adapter. `_patched_adapter` is the starting point, not the
# result; never pick it.
_candidates.sort(key=lambda p: (
    "_patched_adapter" in p.parts,
    p.name != "adapter",
    len(p.parts),
))

from safetensors.torch import load_file
V1_DIR = None
for _c in _candidates:
    _state = load_file(str(_c / "adapter_model.safetensors"))
    if any(".model.layers." in k for k in _state):
        V1_DIR = _c
        break
if V1_DIR is None:
    raise SystemExit(
        "this looks like the multimodal persona adapter, not the Oct-22 "
        "text-only emotion adapter. Upload v1_emotion_adapter.zip."
    )
print("v1 adapter:", V1_DIR, f"({len(list(V1_DIR.iterdir()))} files)")
print("v1 keys:", len(load_file(str(V1_DIR / "adapter_model.safetensors"))))
"""

CODE_BASE = """# --- clean text-only base with the EOS contract -----------------------------
# Do NOT merge the persona here. The v1 adapter already contains it, and
# merging it into the base and then attaching it again is what broke v2.
from transformers import AutoModelForCausalLM, AutoTokenizer

print("loading", STUDENT_BASE)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_BASE, dtype=torch.float32, device_map="cuda"
)
tok = AutoTokenizer.from_pretrained(STUDENT_BASE)

TEXT_BASE = MODELS / "llm" / "agents3_text_base"
TEXT_BASE.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(TEXT_BASE), safe_serialization=True)
tok.save_pretrained(str(TEXT_BASE))

gen = {
    "eos_token_id": [248046, 248044],
    "pad_token_id": 248044,
    "bos_token_id": None,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
}
(TEXT_BASE / "generation_config.json").write_text(json.dumps(gen, indent=2))
cfg = json.loads((TEXT_BASE / "config.json").read_text())
cfg["eos_token_id"] = [248046, 248044]
cfg["pad_token_id"] = 248044
(TEXT_BASE / "config.json").write_text(json.dumps(cfg, indent=2))
print("saved clean text-only base ->", TEXT_BASE)

del model
torch.cuda.empty_cache()
"""

CODE_PREP = """# --- data report -------------------------------------------------------------
_report = {
    "emotion_train": len(read_jsonl(DATA / "v3" / "emotion_train.jsonl")),
    "emotion_valid": len(read_jsonl(DATA / "v3" / "emotion_valid.jsonl")),
    "tool_train": len(read_jsonl(DATA / "v3" / "stage_b_train.jsonl")),
    "tool_valid": len(read_jsonl(DATA / "v3" / "stage_b_valid.jsonl")),
    "tool_eval": len(read_jsonl(DATA / "tool" / "v3" / "eval.jsonl")),
    "tool_eval_legacy": len(read_jsonl(DATA / "tool" / "v3" / "eval_legacy.jsonl")),
    "epochs_a": EPOCHS_A,
    "lr_a": LR_A,
    "max_len_a": MAX_LEN_A,
    "epochs_b": EPOCHS_B,
    "lr_b": LR_B,
    "max_len_b": MAX_LEN_B,
}
write_json(OUT / "data_report.json", _report)
print(json.dumps(_report, indent=2))
"""

CODE_TRAIN_A = """# --- stage A: emotion only ---------------------------------------------------
sys.argv = [
    "train",
    "--data", str(DATA / "v3" / "emotion_train.jsonl"),
    "--valid-data", str(DATA / "v3" / "emotion_valid.jsonl"),
    "--base", str(TEXT_BASE),
    "--adapter", str(V1_DIR),
    "--out", str(OUT_A),
    "--device", "cuda",
    "--dtype", "fp16",
    "--epochs", str(EPOCHS_A),
    "--lr", str(LR_A),
    "--batch-size", str(BATCH),
    "--grad-accum", str(GRAD_ACCUM),
    "--max-seq-length", str(MAX_LEN_A),
]
import train_emotion_lora
_rc = train_emotion_lora.main()
if _rc not in (0, None):
    raise SystemExit(f"stage A failed with code {_rc}")
print("stage A adapter:", OUT_A / "adapter")
"""

CODE_EVAL_A = """# --- stage A eval against the v1 adapter -------------------------------------
# The clean set is deliberately small and independent: it is a fast check
# that Stage A learned serious/focused before spending an hour on Stage B.
sys.argv = [
    "eval",
    "--annotated",
    str(DATA / "eval" / "v3_clean_probes.jsonl"),
    "--base", str(TEXT_BASE),
    "--adapters",
    "baseline=" + str(V1_DIR) + ",emotion=" + str(OUT_A / "adapter"),
    "--out", str(DATA / "v3" / "eval_emotion_stage_a.json"),
    "--device", "cuda",
    "--gate",
]
import eval_emotion
_rc = eval_emotion.main()
print("stage A emotion gate exit code:", _rc)
if _rc != 0:
    print("*** stage A gate failed; read the FAIL rows before stage B. ***")
"""

CODE_TRAIN_B = """# --- stage B: tools + emotion replay -----------------------------------------
sys.argv = [
    "train",
    "--data", str(DATA / "v3" / "stage_b_train.jsonl"),
    "--valid-data", str(DATA / "v3" / "stage_b_valid.jsonl"),
    "--base", str(TEXT_BASE),
    "--adapter", str(OUT_A / "adapter"),
    "--out", str(OUT),
    "--device", "cuda",
    "--dtype", "fp16",
    "--epochs", str(EPOCHS_B),
    "--lr", str(LR_B),
    "--batch-size", str(BATCH),
    "--grad-accum", str(GRAD_ACCUM),
    "--max-seq-length", str(MAX_LEN_B),
]
import train_emotion_lora
_rc = train_emotion_lora.main()
if _rc not in (0, None):
    raise SystemExit(f"stage B failed with code {_rc}")
print("final adapter:", OUT / "adapter")
"""

CODE_EVAL_B = """# --- final emotion eval against v1 ------------------------------------------
sys.argv = [
    "eval",
    "--annotated",
    f"{DATA / 'v2' / 'test_annotated.jsonl'},{DATA / 'eval' / 'serious_probes.jsonl'}",
    "--base", str(TEXT_BASE),
    "--adapters",
    "baseline=" + str(V1_DIR) + ",emotion=" + str(OUT / "adapter"),
    "--out", str(DATA / "v3" / "eval_emotion.json"),
    "--device", "cuda",
    "--gate",
]
import eval_emotion
_rc = eval_emotion.main()
print("final emotion gate exit code:", _rc)
if _rc != 0:
    print("*** final emotion gate failed; read the FAIL rows. ***")
"""

CODE_EVAL_CLEAN = """# --- independent clean emotion check ----------------------------------------
# These 30 prompts are not in training and are not the original probe file.
# They are the second opinion on the final adapter.
sys.argv = [
    "eval",
    "--annotated",
    str(DATA / "eval" / "v3_clean_probes.jsonl"),
    "--base", str(TEXT_BASE),
    "--adapters",
    "baseline=" + str(V1_DIR) + ",emotion=" + str(OUT / "adapter"),
    "--out", str(DATA / "v3" / "eval_emotion_clean.json"),
    "--device", "cuda",
    "--gate",
]
import eval_emotion
_rc = eval_emotion.main()
print("clean emotion gate exit code:", _rc)
if _rc != 0:
    print("*** clean emotion gate failed; read the FAIL rows. ***")
"""

CODE_EVAL_TOOLS = """# --- tool-call gate: unseen intents -----------------------------------------
sys.argv = [
    "eval-tools",
    "--data", str(DATA / "tool" / "v3" / "eval.jsonl"),
    "--base", str(TEXT_BASE),
    "--adapter", str(OUT / "adapter"),
    "--out", str(DATA / "v3" / "eval_tools.json"),
    "--device", "cuda",
    "--gate",
]
import eval_tools
_rc = eval_tools.main()
print("unseen tool gate exit code:", _rc)
if _rc != 0:
    print("*** unseen tool gate failed; read the FAIL rows. ***")
"""

CODE_EVAL_TOOLS_LEGACY = """# --- tool-call regression: legacy v2 file ------------------------------------
sys.argv = [
    "eval-tools",
    "--data", str(DATA / "tool" / "v3" / "eval_legacy.jsonl"),
    "--base", str(TEXT_BASE),
    "--adapter", str(OUT / "adapter"),
    "--out", str(DATA / "v3" / "eval_tools_legacy.json"),
    "--device", "cuda",
    "--gate",
]
import eval_tools
_rc = eval_tools.main()
print("legacy tool gate exit code:", _rc)
"""

CODE_SAVE = """# --- package the result ------------------------------------------------------
import zipfile

_COPIES = (
    (DATA / "v3" / "eval_emotion_stage_a.json", OUT / "eval_emotion_stage_a.json"),
    (DATA / "v3" / "eval_emotion.json", OUT / "eval_report.json"),
    (DATA / "v3" / "eval_emotion_clean.json", OUT / "eval_report_clean.json"),
    (DATA / "v3" / "eval_tools.json", OUT / "tool_eval_report.json"),
    (DATA / "v3" / "eval_tools_legacy.json", OUT / "tool_eval_legacy_report.json"),
    (OUT_A / "train_report.json", OUT / "train_report_stage_a.json"),
)
for _src, _dst in _COPIES:
    if _src.exists() and _src.resolve() != _dst.resolve():
        shutil.copy2(_src, _dst)

(OUT / "README.md").write_text(
    "# agents3 emotion + tool adapter (v3)\\n\\n"
    "Continues the Oct-22 emotion adapter for Qwen3.5-0.8B.\\n"
    "Emits [emotion:...] delivery tags and <tool_call>{json}</tool_call> blocks.\\n"
    "Stage A: emotion replay + serious/focused. Stage B: tools + emotion replay.\\n"
    "Serious turns are enforced by pipeline/emotion/guard.py at runtime.\\n"
    "Merge into Qwen/Qwen3.5-0.8B, then convert with tools/convert_peft_to_mlx_adapter.py.\\n"
)

ZIP = Path("/content/emotion_adapter_v3.zip")
if ZIP.exists():
    ZIP.unlink()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as _z:
    for _p in sorted(OUT.rglob("*")):
        if _p.is_file() and "_patched_adapter" not in _p.parts:
            _z.write(_p, _p.relative_to(OUT.parent))
print("wrote", ZIP, f"{ZIP.stat().st_size / 1e6:.1f} MB")

if CKPT is not None:
    shutil.copy2(ZIP, CKPT / ZIP.name)
    print("copied to", CKPT / ZIP.name)

from google.colab import files
files.download(str(ZIP))
"""


def build() -> list[dict]:
    cells: list[dict] = []

    def md(src: str) -> None:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": src.splitlines(keepends=True),
        })

    def code(src: str) -> None:
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src.splitlines(keepends=True),
        })

    md(MD_INTRO)
    code(CODE_INSTALL)
    code(CODE_CONFIG)
    md(MD_DATA)
    code(CODE_MODULES)
    code(CODE_DATA)
    code(CODE_IMPORT)
    code(CODE_TEST)
    code(CODE_UPLOAD)
    code(CODE_BASE)
    code(CODE_PREP)
    code(CODE_TRAIN_A)
    code(CODE_EVAL_A)
    code(CODE_TRAIN_B)
    code(CODE_EVAL_B)
    code(CODE_EVAL_CLEAN)
    code(CODE_EVAL_TOOLS)
    code(CODE_EVAL_TOOLS_LEGACY)
    code(CODE_SAVE)
    md(MD_AFTER)
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

    for rel, _dest in DATASETS:
        raw = (ROOT / rel).read_bytes()
        if gzip.decompress(base64.b64decode(blob(rel))) != raw:
            raise SystemExit(f"embedded data does not round-trip: {rel}")


def main() -> int:
    cells = build()
    validate(cells)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    print(f"wrote {OUT} ({len(cells)} cells, {n_code} code, {OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
