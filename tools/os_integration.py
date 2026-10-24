"""OS integration skeleton for the voice agent.

Wires together the trained emotion model, the deterministic safety guard, the
tagged-reply segmenter, and the emotion TTS pipeline into a single request /
response loop that an OS can call.

This is intentionally a *skeleton*: the STT (speech-in), audio output, and
OS permission/policy hooks are marked TODO because this repo does not ship
them. Replace the TODO sections with your OS's audio stack and permission
policy.

Usage:
    .venv/bin/python tools/os_integration.py --prompt "My build passed and all tests are green!"
    .venv/bin/python tools/os_integration.py --interactive

Dependencies already present locally:
    * models/llm/agents3_text_base            Qwen3.5-0.8B base (1.5 GB)
    * models/llm/agents3_emotion_adapter/     trained emotion LoRA (25 MB)
    * pipeline/emotion/guard.py               deterministic serious-turn guard
    * pipeline/tts/segmenter.py               tagged-reply -> speech segments
    * pipeline/tts/qwen3.py                   EmotionTTS (needs voice bank)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402
from pipeline.emotion.guard import apply  # noqa: E402
from pipeline.tts.segmenter import parse_tagged_reply, spoken_text  # noqa: E402
from pipeline.tts.qwen3 import (  # noqa: E402
    EmotionTTS,
    GenerationGuard,
    Qwen3TorchEngine,
    VoiceBank,
)

DEFAULT_BASE = ROOT / "models/llm/agents3_text_base"
DEFAULT_ADAPTER = ROOT / "models/llm/agents3_emotion_adapter/adapter"
DEFAULT_TTS_MODEL = ROOT / "models/tts_base/Qwen3-TTS-12Hz-0.6B-Base"
SYSTEM_PROMPT = ROOT / "data" / "persona" / "system_prompt.txt"
VOICE_BANK = ROOT / "models" / "voice" / "bank"


class VoiceAgent:
    """End-to-end OS-facing voice agent.

    Responsibilities:
      1. Generate a tagged reply from base + emotion LoRA.
      2. Apply the deterministic safety guard (serious-turn enforcement).
      3. Segment the reply into speech / non-verbal pieces for TTS.
      4. (Stub) Synthesize audio via EmotionTTS when a voice bank exists.
    """

    def __init__(self, base: Path, adapter: Path, device: str = "mps",
                 dtype: str = "bfloat16", tts_model: Path = DEFAULT_TTS_MODEL,
                 tts_device: str = "cpu", tts_dtype: str = "float32"):
        self.device = device
        torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                       "float16": torch.float16}[dtype]

        print(f"[integration] loading base {base} on {device} ...")
        self.tok = AutoTokenizer.from_pretrained(str(base))
        model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch_dtype)
        print(f"[integration] applying emotion adapter {adapter} ...")
        self.model = PeftModel.from_pretrained(model, str(adapter))
        self.model = self.model.to(device).eval()

        self.guard = GenerationGuard()
        self.tts_engine = None          # set once setup_tts() runs
        self.voice_bank = None
        self.tts_model_path = tts_model
        self.tts_device = tts_device
        self.tts_dtype = tts_dtype

    # ------------------------------------------------------------------ #
    # STT stub: replace with your OS's speech-to-text (e.g. Whisper, VOSK)
    # ------------------------------------------------------------------ #
    def speech_to_text(self, audio_path: Path | None = None) -> str:
        """Convert an audio file to text. TODO: wire to your OS STT."""
        raise NotImplementedError(
            "STT is not implemented in this repo. models/stt/ is empty - "
            "select an ASR model (Whisper / VOSK / paraformer) and implement "
            "this method, or pass --prompt to bypass STT."
        )

    # ------------------------------------------------------------------ #
    # TTS setup: needs a populated voice bank (see tools/design_voice.py)
    # ------------------------------------------------------------------ #
    def setup_tts(self) -> bool:
        """Build the EmotionTTS pipeline from the voice bank. Returns False if
        the bank is missing (expected until design_voice.py runs successfully)."""
        bank_json = VOICE_BANK / "bank.json"
        if not bank_json.exists():
            print("[integration] voice bank missing at "
                  f"{VOICE_BANK} - TTS disabled.")
            print("[integration] run: .venv/bin/python tools/design_voice.py")
            return False

        if not self.tts_model_path.exists():
            print(f"[integration] TTS model missing at {self.tts_model_path} - "
                  "TTS disabled.")
            return False

        print(f"[integration] loading voice bank from {VOICE_BANK} ...")
        self.voice_bank = VoiceBank.load(VOICE_BANK)

        print(f"[integration] loading TTS engine {self.tts_model_path} "
              f"on {self.tts_device} ...")
        engine = Qwen3TorchEngine(self.tts_model_path, device=self.tts_device,
                                  dtype=self.tts_dtype)
        self.tts_engine = EmotionTTS(engine, self.voice_bank, self.guard)
        print("[integration] TTS ready - emotion voice cloning enabled.")
        return True

    # ------------------------------------------------------------------ #
    # Core loop: text -> guarded, segmented reply
    # ------------------------------------------------------------------ #
    def reply(self, user_text: str, max_new_tokens: int = 110) -> dict:
        """Generate a guarded, segmented reply for a user turn.

        Returns a dict the OS can consume:
          raw          model output before the guard
          guarded      model output after the guard (safe delivery tag)
          spoken       plain text (tags stripped) for the UI text channel
          segments     list of {kind, text/nonverbal, emotion} for TTS
          serious      whether the guard flagged the turn as serious
          reasons      why the turn is serious (empty if not)
        """
        system = SYSTEM_PROMPT.read_text(encoding="utf-8")

        prompt = self.tok.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        raw = self.tok.decode(
            out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Deterministic safety guard: forces [emotion:serious] on data-loss /
        # security turns, strips non-verbals, appends "Confirm?" when needed.
        result = apply(user_text, raw)

        segments = parse_tagged_reply(result.text)

        return {
            "raw": raw,
            "guarded": result.text,
            "spoken": spoken_text(result.text),
            "segments": [
                {
                    "kind": seg.kind,
                    "text": seg.text,
                    "emotion": seg.emotion,
                    "nonverbal": seg.nonverbal,
                }
                for seg in segments
            ],
            "serious": result.serious,
            "reasons": list(result.reasons),
        }

    # ------------------------------------------------------------------ #
    # Audio output stub: replace with your OS's audio playback + barge-in
    # ------------------------------------------------------------------ #
    def speak(self, segments: list[dict]) -> None:
        """Synthesize + play segments via EmotionTTS + afplay (macOS).

        Falls back to printing if TTS isn't set up. On barge-in, cancel()
        bumps the epoch and in-flight synthesis is dropped.
        """
        if self.tts_engine is None or self.voice_bank is None:
            print("[integration] TTS not wired - segments to speak:")
            for seg in segments:
                if seg["kind"] == "speech":
                    print(f"   [{seg['emotion']}] {seg['text']}")
                else:
                    print(f"   [nonverbal:{seg['nonverbal']}]")
            return

        from pipeline.tts.qwen3 import Segment  # noqa: E402

        segs = [
            Segment(
                kind=s["kind"],
                text=s.get("text", ""),
                emotion=s.get("emotion", "neutral"),
                nonverbal=s.get("nonverbal", ""),
            )
            for s in segments
        ]

        # Build a tagged reply so EmotionTTS.render handles ordering + splice.
        tagged = self._segments_to_tagged(segs)
        print("[integration] synthesizing audio ...")
        audio, sr = self.tts_engine.render(tagged)

        if len(audio) == 0:
            print("[integration] no audio produced.")
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            wav_path = f.name

        print(f"[integration] playing {len(audio) / sr:.2f}s of audio ...")
        try:
            subprocess.run(["afplay", wav_path], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("[integration] afplay not found - saved to", wav_path)
        finally:
            Path(wav_path).unlink(missing_ok=True)

    @staticmethod
    def _segments_to_tagged(segs) -> str:
        """Render segments back into a tagged string for EmotionTTS.render."""
        from pipeline.tts.segmenter import DELIVERY_EMOTIONS, NONVERBALS

        parts = []
        current_emotion = "neutral"
        for seg in segs:
            if seg.kind == "speech":
                if seg.emotion != current_emotion:
                    parts.append(f"[emotion:{seg.emotion}]")
                    current_emotion = seg.emotion
                parts.append(seg.text)
            else:
                parts.append(f"[{seg.nonverbal}]")
        return "".join(parts)

    def run_voice_loop(self, stt_backend: str = "auto", stt_model: str = "tiny",
                       stt_device: str = "cpu", vad_threshold: float = 0.5,
                       on_user_text=None, on_assistant_text=None) -> None:
        """Start the real-time interruptible voice loop."""
        from pipeline.realtime.stt import make_stt
        from pipeline.realtime.voice_loop import VoiceLoop

        system = SYSTEM_PROMPT.read_text(encoding="utf-8").strip() if SYSTEM_PROMPT.exists() else ""
        stt = make_stt(stt_backend, model=stt_model, device=stt_device)
        loop = VoiceLoop(
            agent=self,
            system_prompt=system,
            stt=stt,
            vad_threshold=vad_threshold,
            on_user_text=on_user_text,
            on_assistant_text=on_assistant_text,
        )
        loop.run()


    def cancel(self) -> int:
        """Barge-in: bump the epoch so in-flight synthesis is dropped."""
        return self.guard.cancel()

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--device", default="mps", help="cpu | mps | cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prompt", help="single user turn to process")
    ap.add_argument("--interactive", action="store_true",
                    help="REPL: type turns, Ctrl-D to quit")
    ap.add_argument("--tts-device", default="cpu",
                    help="cpu | mps | cuda (TTS engine device, default cpu)")
    ap.add_argument("--tts-dtype", default="float32",
                    help="float32 | bfloat16 | float16 (TTS engine dtype)")
    ap.add_argument("--voice-loop", action="store_true",
                    help="start the real-time interruptible voice loop")
    ap.add_argument("--stt-backend", default="auto",
                    help="auto | pywhispercpp | faster-whisper | whisper")
    ap.add_argument("--stt-model", default="tiny",
                    help="STT model name/size")
    ap.add_argument("--setup-tts", action="store_true",
                    help="check for a voice bank and report TTS readiness")


    args = ap.parse_args()

    agent = VoiceAgent(args.base, args.adapter, device=args.device,
                       dtype=args.dtype, tts_device=args.tts_device,
                       tts_dtype=args.tts_dtype)
    agent.setup_tts()

    if args.voice_loop:
        agent.run_voice_loop(stt_backend=args.stt_backend, stt_model=args.stt_model)
        return 0

    if args.setup_tts:
        return 0




    turns = [args.prompt] if args.prompt else None

    if args.interactive:
        print("[integration] interactive mode - type a turn, Ctrl-D to quit")
        try:
            while True:
                try:
                    user_text = input("\nuser> ").strip()
                except EOFError:
                    break
                if not user_text:
                    continue
                out = agent.reply(user_text)
                agent.speak(out["segments"])
        except KeyboardInterrupt:
            pass
        print("\n[integration] goodbye")
        return 0

    if turns is None:
        ap.error("provide --prompt or --interactive")

    for user_text in turns:
        out = agent.reply(user_text)
        print(f"\nuser: {user_text}")
        print(f"serious={out['serious']} reasons={out['reasons']}")
        print(f"model (raw): {out['raw']}")
        print(f"guarded:     {out['guarded']}")
        print(f"spoken:      {out['spoken']}")
        agent.speak(out["segments"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())