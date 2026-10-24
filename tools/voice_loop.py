#!/usr/bin/env python3
"""Real-time voice loop CLI.

Example:
    .venv/bin/python tools/voice_loop.py --interactive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.os_integration import VoiceAgent, DEFAULT_BASE, DEFAULT_ADAPTER, DEFAULT_TTS_MODEL, SYSTEM_PROMPT
from pipeline.realtime.stt import make_stt
from pipeline.realtime.voice_loop import VoiceLoop


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tts-model", type=Path, default=DEFAULT_TTS_MODEL)
    ap.add_argument("--tts-device", default="cpu")
    ap.add_argument("--tts-dtype", default="float32")
    ap.add_argument("--stt-backend", default="auto")
    ap.add_argument("--stt-model", default="base")
    ap.add_argument("--stt-device", default="cpu")
    ap.add_argument("--vad-threshold", type=float, default=0.5)
    ap.add_argument("--silence-ms", type=int, default=700)
    ap.add_argument("--min-utterance-ms", type=int, default=500)
    ap.add_argument("--max-utterance-ms", type=int, default=4000)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--input-device", type=int, default=None)
    ap.add_argument("--output-device", type=int, default=None)
    args = ap.parse_args()

    if args.debug:
        import sounddevice as sd
        print("[devices]", sd.query_devices())
    agent = VoiceAgent(
        args.base, args.adapter, device=args.device, dtype=args.dtype,
        tts_model=args.tts_model, tts_device=args.tts_device, tts_dtype=args.tts_dtype,
    )
    agent.setup_tts()

    system_prompt = SYSTEM_PROMPT.read_text(encoding="utf-8").strip() if SYSTEM_PROMPT.exists() else ""
    stt = make_stt(args.stt_backend, model=args.stt_model, device=args.stt_device)

    loop = VoiceLoop(
        agent=agent,
        system_prompt=system_prompt,
        stt=stt,
        vad_threshold=args.vad_threshold,
        silence_ms=args.silence_ms,
        min_utterance_ms=args.min_utterance_ms,
        max_utterance_ms=args.max_utterance_ms,
        debug=args.debug,
        input_device=args.input_device,
        output_device=args.output_device,
    )
    loop.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
