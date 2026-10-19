"""Shared constants, paths and helpers for the emotion-training toolchain.

Everything downstream (loader, annotator, leakage check, trainer, evaluator)
imports from here so the tag vocabulary and the canonical persona can never
drift between stages.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODELS = ROOT / "models"
DATA = ROOT / "data"
ANNOTATED = DATA / "annotated"
BENCH = ROOT / "bench"

# Source artefacts that ship with the repo / LoRA zip.
LORA_DIR = MODELS / "llm" / "agents3_lora"
MERGED_DIR = MODELS / "llm" / "agents3_merged"
TEST_CONVERSATIONS = LORA_DIR / "test_conversations.json"

# Raw dataset sources.
DEEPSEEK_JSONL = ROOT / "deepseek_companion.jsonl"
HANDWRITTEN_SCRIPT = Path.home() / "Downloads" / "make_joyful_data.py"

# Canonical persona: extracted from the LoRA's own eval set so the system
# prompt is byte-identical to what the existing adapter was trained against.
PERSONA_PATH = DATA / "persona" / "system_prompt.txt"

# Derived model artefacts.
TEXT_BASE_DIR = MODELS / "llm" / "agents3_text_base"
MLX_BASE_DIR = MODELS / "mlx_base" / "Qwen3.5-0.8B-MLX-bf16"
MLX_ADAPTER_DIR = MODELS / "llm" / "agents3_mlx_adapter"
EMOTION_ADAPTER_DIR = MODELS / "llm" / "agents3_emotion_adapter"

# The teacher and the TTS voice bank.
TEACHER_DIR = MODELS / "teacher" / "Qwen3.5-9B-4bit"
VOICE_DIR = MODELS / "voice" / "bank"

# Teacher endpoint (mlx_lm server running the 9B).
TEACHER_BASE_URL = os.environ.get("TEACHER_BASE_URL", "http://127.0.0.1:8081/v1")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "mlx-community/Qwen3.5-9B-4bit")

# --------------------------------------------------------------------------
# Tag vocabulary
# --------------------------------------------------------------------------

# Delivery tags. Turn-level: selects which reference clip the TTS clones from.
# Derived from the persona prompt's own rules:
#   "be supportive when the user is frustrated [warm],
#    focused during work [focused],
#    and playful when the stakes are low [amused]"
#   "never trivialize data loss, security, or serious problems [serious]"
DELIVERY_EMOTIONS = ("neutral", "amused", "warm", "focused", "serious")

# Non-verbal tags. Positional: splices a pre-rendered clip.
NONVERBALS = ("laugh", "chuckle", "sigh", "gasp", "hm", "erm")

# The emotion that must never carry a non-verbal. Data loss / security /
# real distress is a "no jokes" situation.
SERIOUS = "serious"

TAG_RE = re.compile(
    r"\[emotion:(?P<emotion>%s)\]" % "|".join(DELIVERY_EMOTIONS)
    + r"|\[(?P<nonverbal>%s)\]" % "|".join(NONVERBALS)
)

# Same, but anchored so we can tell "tag at start of turn" from "tag mid-text".
DELIVERY_TAG_RE = re.compile(r"\[emotion:(%s)\]" % "|".join(DELIVERY_EMOTIONS))

# Affect variants used to build the augmentation set (Mode B).
AFFECT_VARIANTS = ("frustrated", "delighted", "urgent", "confused", "neutral")

# Maximum words in a spoken reply (excluding tags).
MAX_SPOKEN_WORDS = 30

# Hard cap for the length gate; a few outliers are tolerated in the median.
MAX_SPOKEN_WORDS_P95 = 40


def strip_tags(text: str) -> str:
    """Remove every emotion/non-verbal tag, leaving spoken words only."""
    return TAG_RE.sub("", text).strip()


def tags_in(text: str) -> list[tuple[str, str]]:
    """Return every (kind, value) tag in order. kind is 'emotion' or 'nonverbal'."""
    out: list[tuple[str, str]] = []
    for m in TAG_RE.finditer(text):
        if m.group("emotion"):
            out.append(("emotion", m.group("emotion")))
        else:
            out.append(("nonverbal", m.group("nonverbal")))
    return out


def emotion_of(text: str) -> str | None:
    """Return the delivery emotion of a tagged reply, or None."""
    m = DELIVERY_TAG_RE.search(text)
    return m.group(1) if m else None


def nonverbals_of(text: str) -> list[str]:
    return [v for k, v in tags_in(text) if k == "nonverbal"]


def word_count(text: str) -> int:
    return len(strip_tags(text).split())


# --------------------------------------------------------------------------
# Persona
# --------------------------------------------------------------------------

_FALLBACK_PERSONA = (
    "You are a local-first desktop AI companion integrated into a custom Linux OS. "
    "Your personality is warm, loyal, intelligent, and quietly playful, with polished "
    "British-style wit and balanced dry humour."
)


def load_persona() -> str:
    """The canonical system prompt.

    Read from the LoRA's eval set (all 72 conversations share one prompt), which
    guarantees byte-identity with what the existing adapter was trained against.
    Falls back to a short literal if the eval set is unavailable.
    """
    if PERSONA_PATH.exists():
        return PERSONA_PATH.read_text(encoding="utf-8").strip()
    if TEST_CONVERSATIONS.exists():
        convs = json.loads(TEST_CONVERSATIONS.read_text(encoding="utf-8"))
        for conv in convs:
            for msg in conv:
                if msg["role"] == "system":
                    PERSONA_PATH.parent.mkdir(parents=True, exist_ok=True)
                    PERSONA_PATH.write_text(msg["content"].strip(), encoding="utf-8")
                    return msg["content"].strip()
    return _FALLBACK_PERSONA


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------


def build_tokenized_examples(
    records: list[dict], tokenizer, max_len: int = 1024
) -> list[dict]:
    """Tokenise chat records into (input_ids, labels) with the prompt masked.

    Only assistant tokens contribute to the loss. Training on the system prompt
    would spend capacity on tokens we never want the model to produce, and
    would teach it to *predict* tags rather than *generate* them.

    The prompt and the full text are tokenised separately, which can disagree
    at the boundary, so the longest common prefix is used to find where the
    assistant's turn actually starts.
    """
    out: list[dict] = []
    for rec in records:
        msgs = rec["messages"]
        prompt_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        full_text = prompt_text + msgs[-1]["content"] + "<|im_end|>\n"

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        n = min(len(prompt_ids), len(full_ids))
        cut = 0
        while cut < n and prompt_ids[cut] == full_ids[cut]:
            cut += 1

        ids = full_ids[:max_len]
        labels = [-100] * cut + full_ids[cut:max_len]
        if all(l == -100 for l in labels):
            continue
        out.append({"input_ids": ids, "labels": labels})
    return out


def completion_mask(labels: list[list[int]]) -> list[list[int]]:
    """Turn -100 placeholders into a 0/1 mask for the loss."""
    return [[0 if v == -100 else 1 for v in row] for row in labels]
