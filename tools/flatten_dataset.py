"""Flatten the annotated file into training/validation/test splits.

Two things this gets right that a naive flattener does not.

1. **Group-aware splits.** Every affect variant of a base prompt belongs to the
   same split as that base prompt. If a variant landed in train and its base in
   test, the model would be scored on a near-duplicate of something it had
   already memorised, and the held-out number would be meaningless.

2. **A replay mixture.** A slice of plain, untagged persona turns is mixed
   into training. Without it the model learns "every reply starts with a tag",
   which is its own kind of pattern-matching, and it drifts away from the
   original persona. The replay turns are the 20% of training examples that
   carry no emotion tag at all.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ANNOTATED,
    DATA,
    load_persona,
    read_jsonl,
    strip_tags,
    write_json,
    write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=ANNOTATED / "annotated.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ANNOTATED)
    ap.add_argument("--replay-frac", type=float, default=0.20)
    ap.add_argument("--replay-source", type=Path, default=DATA / "base" / "train.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--valid-frac", type=float, default=0.15)
    args = ap.parse_args()

    persona = load_persona()
    records = read_jsonl(args.inp)
    if not records:
        print(f"no annotated records at {args.inp}", file=sys.stderr)
        return 1

    # -- group-aware split ----------------------------------------------
    ids = sorted({r["id"] for r in records})
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, round(n * args.test_frac))
    n_valid = max(1, round(n * args.valid_frac))
    split_of = {}
    for i, rid in enumerate(ids):
        if i < n_test:
            split_of[rid] = "test"
        elif i < n_test + n_valid:
            split_of[rid] = "valid"
        else:
            split_of[rid] = "train"

    def make_message(user: str, reply: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": persona},
                {"role": "user", "content": user},
                {"role": "assistant", "content": reply},
            ]
        }

    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    tagged_count = 0
    untagged_count = 0
    dropped: list[dict] = []

    def acceptable(reply: str, issues: list[str] | None) -> list[str]:
        """Problems with a training target, empty list if it is usable.

        A reply that is too long, still has markdown in it, or carries a
        non-verbal in a serious turn would teach the model to do the same.
        """
        from normalize_spoken import validate_reply

        found = list(issues or []) + [
            i for i in validate_reply(reply) if i not in (issues or [])
        ]
        return found

    for rec in records:
        split = split_of.get(rec["id"], "train")
        if rec.get("base_reply") and rec.get("base_emotion"):
            problems = acceptable(rec["base_reply"], rec.get("base_issues"))
            if not problems:
                splits[split].append(make_message(rec["base_user"], rec["base_reply"]))
                tagged_count += 1
            else:
                dropped.append({"id": rec["id"], "kind": "base", "issues": problems})
        for var in rec.get("variants", []):
            if var.get("reply") and var.get("emotion"):
                problems = acceptable(var["reply"], var.get("issues"))
                if not problems:
                    splits[split].append(make_message(var["user"], var["reply"]))
                    tagged_count += 1
                else:
                    dropped.append(
                        {
                            "id": f"{rec['id']}:{var.get('affect')}",
                            "kind": "variant",
                            "issues": problems,
                        }
                    )

    # -- replay mixture (untagged persona turns) ------------------------
    # The replay turns exist to teach "not every reply carries a tag" and to
    # stop the emotion pass from drifting the persona. They must therefore be
    # in the *canonical* register.
    #
    # Taking them from the raw sources would reintroduce the joyful register
    # of the deepseek and hand-written sets, and the model would learn to
    # answer in that voice whenever it omits a tag - the exact drift we are
    # trying to prevent. So the replay comes from the teacher's own replies
    # with the delivery tag removed.
    replay: list[dict] = []
    for rec in records:
        if rec.get("base_reply"):
            reply = strip_tags(rec["base_reply"]).strip()
            if reply:
                replay.append(make_message(rec["base_user"], reply))

    if not replay and args.replay_source.exists():
        for rec in read_jsonl(args.replay_source):
            reply = rec["assistant"].strip()
            if reply:
                replay.append(make_message(rec["user"], reply))

    rng.shuffle(replay)
    n_replay = int(len(splits["train"]) * args.replay_frac)
    splits["train"].extend(replay[:n_replay])

    for key in splits:
        rng.shuffle(splits[key])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for key, rows in splits.items():
        write_jsonl(args.out_dir / f"{key}.jsonl", rows)

    # -- report ----------------------------------------------------------
    def label_hist(rows: list[dict]) -> dict:
        from collections import Counter

        from common import emotion_of

        c = Counter(emotion_of(r["messages"][-1]["content"]) for r in rows)
        return {k or "untagged": v for k, v in sorted(c.items(), key=lambda kv: -kv[1])}

    report = {
        "source_records": len(records),
        "groups": len(ids),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "tagged_examples": tagged_count,
        "dropped_malformed": len(dropped),
        "dropped_examples": dropped[:40],
        "untagged_replay_examples": min(n_replay, len(replay)),
        "replay_pool": len(replay),
        "label_distribution": {k: label_hist(v) for k, v in splits.items()},
        "persona_chars": len(persona),
    }
    write_json(args.out_dir / "split_report.json", report)

    print(f"groups: {len(ids)}")
    for k, v in splits.items():
        print(f"  {k:6s} {len(v):4d}  {label_hist(v)}")
    print(f"replay added: {min(n_replay, len(replay))} untagged turns")
    print(f"dropped as malformed: {len(dropped)}")
    for d in dropped[:5]:
        print(f"   {d['id']}: {d.get('issues')}")
    print(f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
