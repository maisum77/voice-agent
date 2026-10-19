"""Show annotated samples for review, before the rest of the dataset is spent.

The teacher is asked for hundreds of rewrites, and the whole training set
depends on their quality. Reading twenty of them is cheap; discovering that
the teacher has been writing in the wrong register after four hundred calls
is not.

Prints each sample as the model will actually see it - the system prompt
once, then the user turn, the teacher's tagged reply, and the spoken text
with the tags stripped - so the register, the tag placement and the length
can all be judged by eye.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline" / "tts"))

from common import ANNOTATED, read_jsonl  # noqa: E402
from segmenter import parse_tagged_reply, spoken_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=ANNOTATED / "annotated.jsonl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--show-system", action="store_true")
    ap.add_argument("--only-variants", action="store_true")
    ap.add_argument("--issues-only", action="store_true")
    args = ap.parse_args()

    records = read_jsonl(args.inp)
    if not records:
        print(f"no annotated records at {args.inp}", file=sys.stderr)
        return 1

    shown = 0
    for rec in records:
        rows = []
        if rec.get("base_reply") and not args.only_variants:
            rows.append(("base", rec["base_user"], rec["base_reply"], rec.get("base_issues")))
        for var in rec.get("variants", []):
            if var.get("reply"):
                rows.append((var.get("affect", "?"), var.get("user", ""), var["reply"], var.get("issues")))
        for kind, user, reply, issues in rows:
            if args.issues_only and not issues:
                continue
            shown += 1
            print("=" * 78)
            print(f"[{rec['id']}  {rec['source']}  {kind}]")
            if args.show_system and kind == "base":
                sys_prompt = rec.get("system", "")
                if sys_prompt:
                    print(f"  SYSTEM: {sys_prompt[:300]}...")
            print(f"  USER : {user}")
            print(f"  REPLY: {reply}")
            segs = parse_tagged_reply(reply)
            print(f"  TTS  : {len(segs)} segments, delivery={segs[0].emotion if segs else '-'}")
            for s in segs:
                if s.is_speech:
                    print(f"          speak  {s.emotion:8s} {s.text!r}")
                else:
                    print(f"          clip   {s.nonverbal}")
            print(f"  SPOKEN ({len(spoken_text(reply).split())} words): {spoken_text(reply)}")
            if issues:
                print(f"  ISSUES: {issues}")
            if shown >= args.n:
                print("=" * 78)
                print(f"\n{shown} samples shown. Review them before annotating the rest.")
                return 0

    print("=" * 78)
    print(f"\n{shown} samples shown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
