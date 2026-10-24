"""Speech-to-text backends (free/open-source)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _to_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sr == 16000:
        return audio
    try:
        import librosa

        return librosa.resample(audio, orig_sr=sr, target_sr=16000).astype("float32")
    except Exception:
        return audio


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    audio = audio - np.mean(audio)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        gain = min(10.0, 0.95 / peak)
        audio = audio * gain
    return np.clip(audio, -1.0, 1.0).astype("float32")


class STTBackend:
    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        raise NotImplementedError


class PyWhisperCppSTT(STTBackend):
    def __init__(self, model: str = "tiny", language: str = "en"):
        from pywhispercpp.model import Model

        self.model = Model(model)
        self.language = language

    def transcribe(self, audio, sr=16000):
        audio = _to_16k(audio, sr)
        segments = self.model.transcribe(audio, language=self.language)
        return " ".join(getattr(s, "text", str(s)) for s in segments).strip()


class FasterWhisperSTT(STTBackend):
    def __init__(self, model: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio, sr=16000):
        audio = _normalize_audio(_to_16k(audio, sr))
        segs, _ = self.model.transcribe(audio, language="en", beam_size=5, vad_filter=True, condition_on_previous_text=False, temperature=0.0)
        return " ".join(s.text for s in segs).strip()


class OpenAIWhisperSTT(STTBackend):
    def __init__(self, model: str = "tiny", device: str = "cpu"):
        import whisper

        self.model = whisper.load_model(model, device=device)

    def transcribe(self, audio, sr=16000):
        audio = _to_16k(audio, sr)
        res = self.model.transcribe(audio, language="en")
        return (res.get("text") or "").strip()


def make_stt(backend: str = "auto", model: str = "tiny", device: str = "cpu"):
    backend = (backend or "auto").lower()
    if backend == "pywhispercpp":
        try:
            return PyWhisperCppSTT(model=model)
        except Exception:
            if backend == "pywhispercpp":
                raise
    if backend in ("auto", "faster-whisper", "faster_whisper"):
        try:
            return FasterWhisperSTT(model=model, device=device)
        except Exception:
            if backend in ("faster-whisper", "faster_whisper"):
                raise
    if backend in ("auto", "whisper", "openai-whisper"):
        try:
            return OpenAIWhisperSTT(model=model, device=device)
        except Exception:
            if backend in ("whisper", "openai-whisper"):
                raise
    raise RuntimeError(
        "No STT backend available. Install one of: pywhispercpp, faster-whisper, openai-whisper."
    )
