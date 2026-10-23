"""Build the Stage B mixture: tool rows plus a fixed emotion replay.

Stage B is where the adapter learns the tool channel. The replay is not there
to re-teach emotion from scratch; it is there so the low-rank update cannot
spend all of its capacity on tool JSON and forget the delivery tags. Sixty
rows per label is enough to keep the old behaviour in the loss without
doubling the run time.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from common import read_jsonl, write_jsonl  # noqa: E402

LABELS = ("amused", "warm", "neutral", "focused", "serious")
LABEL_RE = re.compile(r"\[emotion:(amused|warm|neutral|focused|serious)\]")


def label_of(row: dict) -> str | None:
    body = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
    m = LABEL_RE.search(body)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emotion", type=Path, default=ROOT / "data/v3/emotion_train.jsonl")
    ap.add_argument("--emotion-valid", type=Path, default=ROOT / "data/v3/emotion_valid.jsonl")
    ap.add_argument("--tool", type=Path, default=ROOT / "data/tool/v3/train.jsonl")
    ap.add_argument("--tool-valid", type=Path, default=ROOT / "data/tool/v3/valid.jsonl")
    ap.add_argument("--per-label", type=int, default=60)
    ap.add_argument("--out", type=Path, default=ROOT / "data/v3/stage_b_train.jsonl")
    ap.add_argument("--out-valid", type=Path, default=ROOT / "data/v3/stage_b_valid.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    by: dict[str, list[dict]] = collections.defaultdict(list)
    for row in read_jsonl(args.emotion):
        label = label_of(row)
        if label:
            by[label].append(row)

    rng = random.Random(args.seed)
    replay: list[dict] = []
    for label in LABELS:
        pool = by.get(label, [])
        if not pool:
            raise SystemExit(f"no emotion rows for {label}")
        replay += rng.sample(pool, min(args.per_label, len(pool)))

    tool = read_jsonl(args.tool)
    train = tool + replay
    rng.shuffle(train)
    write_jsonl(args.out, train)

    valid = read_jsonl(args.emotion_valid) + read_jsonl(args.tool_valid)
    write_jsonl(args.out_valid, valid)

    print(f"stage B train: {len(train)} ({len(tool)} tool + {len(replay)} emotion)")
    print("replay labels:", dict(collections.Counter(label_of(r) for r in replay)))
    print(f"stage B valid: {len(valid)} -> {args.out_valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
