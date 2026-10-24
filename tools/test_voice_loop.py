#!/usr/bin/env python3
"""Smoke test for realtime voice loop imports and readiness."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.realtime.audio_io as audio_io
import pipeline.realtime.vad as vad
import pipeline.realtime.stt as stt
import pipeline.realtime.llm_stream as llm_stream
import pipeline.realtime.tts_worker as tts_worker
import pipeline.realtime.voice_loop as voice_loop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("imports: OK")
    print("audio_io:", hasattr(audio_io, "AudioPlayer"), hasattr(audio_io, "MicCapture"))
    print("vad:", hasattr(vad, "make_vad"))
    print("stt:", hasattr(stt, "make_stt"))
    print("llm_stream:", hasattr(llm_stream, "stream_generate"))
    print("tts_worker:", hasattr(tts_worker, "TTSWorker"))
    print("voice_loop:", hasattr(voice_loop, "VoiceLoop"))
    if args.dry_run:
        print("dry-run: OK (no hardware/models touched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
