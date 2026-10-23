"""Regenerate the v1 adapter's speech for the v3 replay set.

We keep the v1 delivery for amused/warm/neutral because that adapter already
does those well. The tag is replaced with the ground-truth label from the v3
training row, so the replay cannot teach a wrong tag.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "pipeline"))

from common import MLX_BASE_DIR, read_jsonl, write_jsonl  # noqa: E402
from eval_emotion_mlx import MLXStudent  # noqa: E402

ADAPTER = ROOT / "models/llm/agents3_emotion_mlx_adapter"
BASE = ROOT / "data/v3/emotion_tagged.jsonl"
OUT = ROOT / "data/v3/replay.jsonl"
TAG_RE = re.compile(r"\[emotion:(amused|warm|neutral|focused|serious)\]")


def main() -> int:
    rows = []
    for rec in read_jsonl(BASE):
        a = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
        m = TAG_RE.search(a)
        if m and m.group(1) in ("amused", "warm", "neutral"):
            rows.append(rec)

    print(f"replay prompts: {len(rows)}")
    student = MLXStudent(MLX_BASE_DIR, ADAPTER)
    out = []
    for i, rec in enumerate(rows, 1):
        system = next(m["content"] for m in rec["messages"] if m["role"] == "system")
        user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        text = student.generate(system, user).text.strip()
        out.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": text},
            ]
        })
        if i % 20 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    write_jsonl(OUT, out)
    print(f"wrote {len(out)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
