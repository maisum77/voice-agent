"""Package the files the Kaggle notebook needs into one uploadable dataset.

The notebook wants exactly one thing attached: the persona LoRA zip. The
DeepSeek set, the hand-written pairs and the whole pipeline are embedded in
the notebook itself, so there is nothing else to attach and nothing to keep
in sync.

    python3 tools/make_kaggle_dataset.py
    -> dist/agents3-kaggle-dataset.zip

Upload that as a Kaggle dataset (Add Data -> New Dataset -> Upload), and the
notebook finds it under /kaggle/input automatically.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

# The persona LoRA, as it came out of the training run.
LORA_ZIP = ROOT / "agents3_lora(1).zip"

# Optional: a dataset that was annotated already, which lets the notebook
# skip the slow part.
ANNOTATED = ROOT / "data" / "annotated"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DIST / "agents3-kaggle-dataset.zip")
    ap.add_argument(
        "--with-annotated",
        action="store_true",
        help="also include the annotated dataset, so the notebook can skip "
        "the teacher and go straight to training",
    )
    args = ap.parse_args()

    if not LORA_ZIP.exists():
        raise SystemExit(f"missing {LORA_ZIP}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(LORA_ZIP, "agents3_lora(1).zip")

        if args.with_annotated:
            added = 0
            for name in (
                "annotated.jsonl",
                "annotate_report.json",
                "leakage_report.json",
                "split_report.json",
                "eval_report.json",
            ):
                p = ANNOTATED / name
                if p.exists():
                    z.write(p, f"annotated/{name}")
                    added += 1
            if not added:
                print("no annotated dataset found; attaching the LoRA only")
            else:
                print(f"attached {added} annotated files")

    size = args.out.stat().st_size
    print(f"wrote {args.out}  ({size / 1e6:.1f} MB)")
    print()
    print("Upload it at https://www.kaggle.com/datasets -> New Dataset -> Upload,")
    print("then add it to the notebook's inputs. The notebook finds it by itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
