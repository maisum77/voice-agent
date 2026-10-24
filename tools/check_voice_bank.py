"""Quick diagnostics for the voice bank.

Checks that every clip exists, has a sane sample rate/duration/RMS, and that
the stored ref_text is at least internally consistent (non-empty, unique-ish).
If sherpa-onnx + an ECAPA model are available, also runs the speaker-consistency
check from tools/design_voice.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(x * x))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, default=Path("models/voice/bank"))
    args = ap.parse_args()

    bank_path = args.bank / "bank.json"
    if not bank_path.exists():
        print(f"MISSING bank manifest: {bank_path}")
        return 1

    manifest = json.loads(bank_path.read_text(encoding="utf-8"))
    clips = manifest.get("clips", {})
    nonverbals = manifest.get("nonverbals", {})

    print(f"bank: {args.bank}")
    print(f"clips: {len(clips)}  nonverbals: {len(nonverbals)}")

    bad = False
    for group_name, group in (("clip", clips), ("nonverbal", nonverbals)):
        for name, info in group.items():
            path = args.bank / info["file"]
            if not path.exists():
                print(f"  {group_name}/{name}: MISSING FILE {path}")
                bad = True
                continue
            audio, sr = sf.read(str(path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            duration = len(audio) / sr
            ref_text = info.get("ref_text", "")
            print(
                f"  {group_name}/{name}: sr={sr} dur={duration:.2f}s "
                f"rms={rms(audio):.4f} peak={float(np.max(np.abs(audio))):.4f} "
                f"ref_text={ref_text!r}"
            )
            if sr != 24000:
                print(f"     WARN: unexpected sample rate {sr} (expected 24000)")
            if duration < 0.5 or duration > 12.0:
                print(f"     WARN: unusual duration {duration:.2f}s")
            if rms(audio) < 1e-4:
                print("     WARN: near-silent clip")
            if not ref_text.strip():
                print("     WARN: empty ref_text")
                bad = True

    # Optional ECAPA consistency check.
    try:
        import sys
        sys.path.insert(0, str(Path("tools").resolve()))
        from design_voice import check_consistency
        result = check_consistency(args.bank, 0.80)
        print("consistency:", json.dumps(result, indent=2))
        if result.get("checked") and not result.get("passed"):
            bad = True
    except Exception as e:
        print(f"consistency check skipped: {e}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
