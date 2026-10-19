"""Load and unify every dataset source into one record shape.

Sources
-------
deepseek      deepseek_companion.jsonl            31  (joyful register, long, emoji+code)
handwritten   make_joyful_data.py                 37  (joyful register, clean)
test_conv     test_conversations.json (LoRA zip)  72  (quietly playful, clean, spoken-form)

All three are normalised to:

    {
      "id":        "<source>:<index>",
      "source":    "deepseek" | "handwritten" | "test_conv",
      "system":    "<canonical persona>",   <- always the test_conv prompt
      "orig_system": "<whatever the source used>",
      "user":      "...",
      "assistant": "...",
      "messages":  [ {role,content} x3 ]
    }

The canonical system prompt is forced on every record so the model sees one
consistent (prompt, register) pair throughout training. The original prompt is
kept in ``orig_system`` for reporting only.

The split is deterministic and stratified by source, so re-running produces
byte-identical output.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ANNOTATED,
    DATA,
    DEEPSEEK_JSONL,
    HANDWRITTEN_SCRIPT,
    ROOT,
    TEST_CONVERSATIONS,
    load_persona,
    read_json,
    read_jsonl,
    word_count,
    write_json,
    write_jsonl,
)

SOURCES = ("deepseek", "handwritten", "test_conv")


# --------------------------------------------------------------------------
# Source readers
# --------------------------------------------------------------------------


def _record(source: str, idx: int, system: str, user: str, assistant: str) -> dict:
    return {
        "id": f"{source}:{idx:03d}",
        "source": source,
        "system": "",  # filled in later with the canonical persona
        "orig_system": system,
        "user": user,
        "assistant": assistant,
        "messages": [],
    }


def load_deepseek() -> list[dict]:
    if not DEEPSEEK_JSONL.exists():
        return []
    out = []
    for i, rec in enumerate(read_jsonl(DEEPSEEK_JSONL)):
        msgs = rec.get("messages", [])
        if len(msgs) < 3:
            continue
        system = msgs[0]["content"] if msgs[0]["role"] == "system" else ""
        out.append(_record("deepseek", i, system, msgs[1]["content"], msgs[2]["content"]))
    return out


def load_handwritten() -> list[dict]:
    """Re-run make_joyful_data.py's literal table without executing the writer."""
    if not HANDWRITTEN_SCRIPT.exists():
        return []
    src = HANDWRITTEN_SCRIPT.read_text(encoding="utf-8")
    try:
        start = src.index("E = [")
        end = src.index("\nwith open(")
    except ValueError:
        return []
    tree = ast.parse(src[start:end])
    pairs = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "E":
            pairs = ast.literal_eval(node.value)
    if not pairs:
        return []

    system = ""
    m = re.search(r'SYSTEM\s*=\s*"""(.*?)"""', src, re.S)
    if m:
        system = m.group(1).strip()

    out = []
    for i, (user, assistant) in enumerate(pairs):
        out.append(_record("handwritten", i, system, user, assistant))
    return out


def load_test_conv() -> list[dict]:
    if not TEST_CONVERSATIONS.exists():
        return []
    convs = read_json(TEST_CONVERSATIONS)
    out = []
    for i, conv in enumerate(convs):
        if not conv:
            continue
        system = conv[0]["content"] if conv[0]["role"] == "system" else ""
        user = next((m["content"] for m in conv if m["role"] == "user"), "")
        assistant = next((m["content"] for m in conv if m["role"] == "assistant"), "")
        if not user or not assistant:
            continue
        out.append(_record("test_conv", i, system, user, assistant))
    return out


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def _bucket(rec_id: str, salt: str = "agents3-emotion-v1") -> float:
    """Deterministic hash in [0, 1) — stable across runs and machines."""
    h = hashlib.sha256(f"{salt}:{rec_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def stratified_split(records: list[dict], valid_frac=0.10, test_frac=0.10):
    """Split by source so every split keeps the same source mix."""
    by_source: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_source[rec["source"]].append(rec)

    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for source in sorted(by_source):
        items = sorted(by_source[source], key=lambda r: _bucket(r["id"]))
        n = len(items)
        n_test = max(1, round(n * test_frac)) if n >= 5 else 0
        n_valid = max(1, round(n * valid_frac)) if n >= 8 else 0
        splits["test"].extend(items[:n_test])
        splits["valid"].extend(items[n_test : n_test + n_valid])
        splits["train"].extend(items[n_test + n_valid :])
    for key in splits:
        splits[key].sort(key=lambda r: r["id"])
    return splits


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def build_report(records: list[dict], splits: dict) -> dict:
    def stats(rows: list[dict]) -> dict:
        if not rows:
            return {"n": 0}
        words = [word_count(r["assistant"]) for r in rows]
        emoji = sum(
            1
            for r in rows
            if re.search(r"[\U0001F300-\U0001FAFF☀-➿]", r["assistant"])
        )
        code = sum(1 for r in rows if "`" in r["assistant"])
        words_sorted = sorted(words)
        return {
            "n": len(rows),
            "words_median": words_sorted[len(words_sorted) // 2],
            "words_min": min(words),
            "words_max": max(words),
            "over_30_words": sum(1 for w in words if w > 30),
            "with_emoji": emoji,
            "with_backticks": code,
            "with_question": sum(1 for r in rows if "?" in r["assistant"]),
        }

    return {
        "total": len(records),
        "by_source": {
            src: stats([r for r in records if r["source"] == src]) for src in SOURCES
        },
        "splits": {k: stats(v) for k, v in splits.items()},
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "distinct_orig_system_prompts": len(
            {r["orig_system"] for r in records if r["orig_system"]}
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA / "base")
    args = ap.parse_args()

    persona = load_persona()
    records = load_deepseek() + load_handwritten() + load_test_conv()
    if not records:
        print("no sources found", file=sys.stderr)
        return 1

    for rec in records:
        rec["system"] = persona
        rec["messages"] = [
            {"role": "system", "content": rec["system"]},
            {"role": "user", "content": rec["user"]},
            {"role": "assistant", "content": rec["assistant"]},
        ]

    records.sort(key=lambda r: r["id"])
    splits = stratified_split(records)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(args.out / f"{name}.jsonl", rows)

    report = build_report(records, splits)
    report["persona_chars"] = len(persona)
    write_json(args.out / "report.json", report)

    print(f"persona: {len(persona)} chars")
    for src in SOURCES:
        s = report["by_source"][src]
        print(
            f"  {src:12s} n={s['n']:3d} words med={s.get('words_median', 0):3d} "
            f"over30={s.get('over_30_words', 0):3d} emoji={s.get('with_emoji', 0):2d} "
            f"backticks={s.get('with_backticks', 0):2d}"
        )
    print("splits:", report["split_sizes"])
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
