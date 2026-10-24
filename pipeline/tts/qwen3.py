"""Speech synthesis for the emotion pipeline.

Three things happen here, and they are the reason this is not just a call to
a TTS library.

1. **The voice is cloned from the bank clip that matches the delivery.**
   A tagged reply carries its emotion, and the emotion picks the reference
   clip, so the same voice is delivered differently without ever touching the
   model weights at runtime.

2. **Non-verbal tags splice in pre-rendered clips.** A laugh is not something
   the clone can be asked for, so it is a short recording in the same voice,
   cut in at the position the model chose.

3. **Synthesis is cancellable.** This is the part that decides whether hard
   barge-in actually works. The synthesis runs in a worker thread because the
   inference is blocking, so an interruption that arrives mid-synthesis would
   otherwise still land in the speaker a second later. Every request carries
   an epoch; an interruption bumps the epoch, and any chunk produced under a
   stale epoch is dropped on arrival.

The engines are pluggable so the same pipeline runs across the capability
tiers:

    Qwen3TorchEngine  Qwen3-TTS 0.6B-Base, clones the designed voice
    Qwen3MNNEngine    the same model on MNN - pure CPU, no torch
    KokoroEngine      sherpa-onnx, ~90 MB, always fast, no emotion
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .segmenter import Segment, parse_tagged_reply

DEFAULT_SAMPLE_RATE = 24000


# --------------------------------------------------------------------------
# Interruption guard
# --------------------------------------------------------------------------


class GenerationGuard:
    """Epoch counter for cancellable synthesis.

    A blocking inference cannot be interrupted from the outside, so instead
    every request is stamped with the current epoch. Cancelling bumps the
    epoch, and the worker checks its stamp before every chunk it emits.
    """

    def __init__(self) -> None:
        self._epoch = 0
        self._lock = threading.Lock()

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def cancel(self) -> int:
        with self._lock:
            self._epoch += 1
            return self._epoch

    def is_current(self, epoch: int) -> bool:
        with self._lock:
            return epoch == self._epoch


# --------------------------------------------------------------------------
# Voice bank
# --------------------------------------------------------------------------


@dataclass
class VoiceBank:
    """The reference clips the voice is cloned from."""

    root: Path
    manifest: dict = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> "VoiceBank":
        path = Path(root)
        manifest = json.loads((path / "bank.json").read_text(encoding="utf-8"))
        return cls(root=path, manifest=manifest)

    def reference(self, emotion: str) -> tuple[Path, str]:
        """The clip and its transcript for a delivery."""
        clips = self.manifest.get("clips", {})
        info = clips.get(emotion) or clips.get("neutral")
        if not info:
            raise KeyError(f"no reference clip for {emotion!r}")
        return self.root / info["file"], info["ref_text"]

    def nonverbal(self, name: str) -> tuple[np.ndarray, int] | None:
        """A pre-rendered non-verbal clip, if the bank has one."""
        info = self.manifest.get("nonverbals", {}).get(name)
        if not info:
            return None
        import soundfile as sf

        audio, sr = sf.read(str(self.root / info["file"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr

    def emotions(self) -> list[str]:
        return list(self.manifest.get("clips", {}))

    def nonverbals(self) -> list[str]:
        return list(self.manifest.get("nonverbals", {}))


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------


class TTSEngine:
    """Synthesises one utterance in one voice."""

    name = "base"

    def synth(self, text: str, ref_audio: Path, ref_text: str) -> tuple[np.ndarray, int]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class Qwen3TorchEngine(TTSEngine):
    """Qwen3-TTS 0.6B-Base through the official torch package."""

    name = "qwen3-torch"

    def __init__(self, model_path: Path, device: str = "cpu", dtype: str = "float32",
                 temperature: float = 0.7, max_tokens_per_word: int = 6,
                 max_new_tokens_cap: int = 256, do_sample: bool = True):
        import torch
        from qwen_tts import Qwen3TTSModel

        self._torch = torch
        self._dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]
        self.temperature = temperature
        self.max_tokens_per_word = max_tokens_per_word
        self.max_new_tokens_cap = max_new_tokens_cap
        self.do_sample = do_sample
        self.model = Qwen3TTSModel.from_pretrained(
            str(model_path), device_map=device, dtype=self._dtype
        )

    def synth(self, text, ref_audio, ref_text):
        # Bound generation length: ~6 codec tokens per word at 12 Hz,
        # capped to prevent runaway output. Default 2048 is far too high.
        # Full-text (non-streaming) mode is required for intelligible ICL
        # cloning; the streaming-simulation default misaligns ref/text.
        max_new_tokens = max(24, min(self.max_new_tokens_cap,
                                     len(text.split()) * self.max_tokens_per_word))
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language="English",
            ref_audio=str(ref_audio),
            ref_text=ref_text,
            max_new_tokens=max_new_tokens,
            non_streaming_mode=True,
            do_sample=self.do_sample,
            temperature=self.temperature,
            repetition_penalty=1.05,
        )
        audio = np.asarray(wavs[0], dtype="float32")
        if audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("TTS produced empty or non-finite audio")
        return audio, int(sr)


class Qwen3MNNEngine(TTSEngine):
    """Qwen3-TTS 0.6B-Base on MNN - pure CPU, no torch, no CUDA."""

    name = "qwen3-mnn"

    def __init__(self, model_path: Path | None = None, device: str = "cpu"):
        import qwen3_tts_mnn

        self._impl = qwen3_tts_mnn
        self.model = qwen3_tts_mnn.load(str(model_path)) if model_path else None

    def synth(self, text, ref_audio, ref_text):
        audio, sr = self.model.synthesize(
            text=text, ref_audio=str(ref_audio), ref_text=ref_text
        )
        return np.asarray(audio, dtype="float32"), int(sr)


class KokoroEngine(TTSEngine):
    """sherpa-onnx Kokoro. The floor tier: ~90 MB, always real time.

    No voice cloning and no emotion - it is the fallback that keeps the
    assistant talking on hardware that cannot hold anything better.
    """

    name = "kokoro"

    def __init__(self, model_dir: Path, voice: str = "af_heart", sample_rate: int = 24000):
        import sherpa_onnx

        self._tts = sherpa_onnx.OfflineTts(
            model=model_dir.as_posix(),
            num_threads=2,
            provider="cpu",
        )
        self.voice = voice
        self.sample_rate = sample_rate

    def synth(self, text, ref_audio, ref_text):
        audio = self._tts.generate(text, sid=0, speed=1.0)
        return np.asarray(audio.samples, dtype="float32"), int(audio.sample_rate)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


class EmotionTTS:
    """Turns a tagged reply into audio, in order, cancellably."""

    def __init__(self, engine: TTSEngine, bank: VoiceBank, guard: GenerationGuard | None = None):
        self.engine = engine
        self.bank = bank
        self.guard = guard or GenerationGuard()

    # -- one piece -------------------------------------------------------
    def render_segment(self, seg: Segment) -> tuple[np.ndarray, int]:
        if seg.is_speech:
            ref_audio, ref_text = self.bank.reference(seg.emotion)
            return self.engine.synth(seg.text, ref_audio, ref_text)
        clip = self.bank.nonverbal(seg.nonverbal)
        if clip is None:
            return np.zeros(0, dtype="float32"), DEFAULT_SAMPLE_RATE
        return clip

    # -- whole reply -----------------------------------------------------
    def render(self, tagged_reply: str) -> tuple[np.ndarray, int]:
        """Render every segment in order into one waveform."""
        pieces = []
        sr = DEFAULT_SAMPLE_RATE
        seen_sr = None
        for seg in parse_tagged_reply(tagged_reply):
            if seg.is_speech and not seg.text.strip():
                continue
            if not seg.is_speech and self.bank.nonverbal(seg.nonverbal) is None:
                continue
            audio, seg_sr = self.render_segment(seg)
            if not len(audio):
                continue
            if seen_sr is None:
                seen_sr = int(seg_sr)
                sr = seen_sr
            elif int(seg_sr) != seen_sr:
                # Never concatenate at mismatched rates (would change speed).
                try:
                    import librosa
                    audio = librosa.resample(
                        np.asarray(audio, dtype="float32"),
                        orig_sr=int(seg_sr), target_sr=seen_sr,
                    ).astype("float32")
                except Exception:
                    raise RuntimeError(
                        f"sample-rate mismatch: {seg_sr} != {seen_sr}"
                    )
            pieces.append(np.asarray(audio, dtype="float32"))
            sr = seen_sr
        if not pieces:
            return np.zeros(0, dtype="float32"), sr
        return np.concatenate(pieces), sr

    def render_streaming(self, tagged_reply: str):
        """Yield (audio, sample_rate) per segment, cancellably.

        The caller checks the guard between segments, so a barge-in stops the
        remaining pieces from ever being synthesised.
        """
        epoch = self.guard.epoch
        for seg in parse_tagged_reply(tagged_reply):
            if not self.guard.is_current(epoch):
                return
            if seg.is_speech and not seg.text.strip():
                continue
            if not seg.is_speech and self.bank.nonverbal(seg.nonverbal) is None:
                continue
            audio, sr = self.render_segment(seg)
            if not self.guard.is_current(epoch):
                return
            if len(audio):
                yield audio, sr

    # -- interruption ----------------------------------------------------
    def cancel(self) -> int:
        """Drop anything in flight. Returns the new epoch."""
        return self.guard.cancel()
