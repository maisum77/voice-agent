"""Smoke test for the Qwen3 torch TTS engine.

Run:
    .venv/bin/python tools/test_tts_engine.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL = ROOT / "models/tts_base/Qwen3-TTS-12Hz-0.6B-Base"
BANK = ROOT / "models/voice/bank"
TEXT = "Hello, this is a quick test of the cloned voice."


def main() -> int:
    if not MODEL.exists() or not (BANK / "neutral.wav").exists():
        print("SKIP: model or voice bank missing")
        return 0

    from pipeline.tts.qwen3 import Qwen3TorchEngine
    import numpy as np
    import soundfile as sf

    engine = Qwen3TorchEngine(MODEL, device="cpu", dtype="float32")
    t0 = time.time()
    audio, sr = engine.synth(TEXT, BANK / "neutral.wav", "Right, let's get this sorted.")
    elapsed = time.time() - t0

    assert sr == 24000, sr
    assert audio.ndim == 1, audio.shape
    assert audio.size > 0, "empty audio"
    assert np.isfinite(audio).all(), "non-finite audio"
    dur = len(audio) / sr
    assert 0.5 <= dur <= 10.0, f"unexpected duration {dur:.2f}s"
    rms = float((audio ** 2).mean() ** 0.5)
    assert rms > 1e-3, f"near-silent output rms={rms:.2e}"
    peak = float(np.max(np.abs(audio)))
    assert peak <= 1.05, f"clipped output peak={peak:.3f}"

    print(f"PASS: dur={dur:.2f}s rms={rms:.4f} peak={peak:.3f} "
          f"rtf={elapsed/dur:.2f} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
