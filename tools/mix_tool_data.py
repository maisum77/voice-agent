"""Mix the deterministic tool-call examples into the emotion splits.

The tool examples are generated separately because they are exact (a JSON
argument either matches the schema or it doesn't), while the emotion examples
come from the annotated dataset. Training needs both: the adapter has to keep
the persona and the delivery tags while learning the tool channel.

The tool eval split is *not* mixed in. It stays held out for
`tools/eval_tools.py`.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import read_jsonl, write_json, write_jsonl  # noqa: E402


def _load(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    out = []
    for row in rows:
        msgs = row.get("messages")
        if not msgs or msgs[-1].get("role") != "assistant":
            continue
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--tool-train", type=Path, default=Path("data/tool/train.jsonl"))
    ap.add_argument("--tool-valid", type=Path, default=Path("data/tool/valid.jsonl"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = read_jsonl(args.train)
    valid = read_jsonl(args.valid)
    tool_train = _load(args.tool_train)
    tool_valid = _load(args.tool_valid)

    with_call = 0
    for row in tool_train:
        if "<tool_call>" in row["messages"][-1]["content"]:
            with_call += 1

    train = train + tool_train
    valid = valid + tool_valid
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(valid)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    write_json(
        args.out_dir / "mix_report.json",
        {
            "train": len(train),
            "valid": len(valid),
            "emotion_train": len(train) - len(tool_train),
            "tool_train": len(tool_train),
            "tool_train_with_call": with_call,
            "tool_train_no_call": len(tool_train) - with_call,
            "tool_fraction": round(len(tool_train) / len(train), 4) if train else 0.0,
            "tool_eval_held_out": str(args.tool_valid),
        },
    )
    print(
        f"train {len(train)} ({len(tool_train)} tool, {with_call} with a call) | "
        f"valid {len(valid)} ({len(tool_valid)} tool)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
