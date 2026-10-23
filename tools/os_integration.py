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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402
from pipeline.emotion.guard import apply  # noqa: E402
from pipeline.tts.segmenter import parse_tagged_reply, spoken_text  # noqa: E402
from pipeline.tts.qwen3 import GenerationGuard  # noqa: E402


DEFAULT_BASE = ROOT / "models/llm/agents3_text_base"
DEFAULT_ADAPTER = ROOT / "models/llm/agents3_emotion_adapter/adapter"
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
                 dtype: str = "bfloat16"):
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
        self.tts_engine = None          # set once a voice bank is available
        self.voice_bank = None

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
        from qwen3 import Qwen3TorchEngine  # noqa: F401
        # Qwen3TorchEngine / Qwen3MNNEngine / KokoroEngine are pluggable.
        # Instantiate the engine your hardware supports, then:
        #   self.tts_engine = EmotionTTS(engine, VoiceBank.load(VOICE_BANK), self.guard)
        print("[integration] voice bank found - wire up your TTS engine here.")
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
        """Synthesize + play segments. TODO: wire to your OS audio output.

        When a voice bank + TTS engine are set up (see setup_tts), replace
        this with self.tts_engine.render_streaming(...) and check
        self.guard.is_current(epoch) between segments for barge-in.
        """
        print("[integration] audio output not wired - segments to speak:")
        for seg in segments:
            if seg["kind"] == "speech":
                print(f"   [{seg['emotion']}] {seg['text']}")
            else:
                print(f"   [nonverbal:{seg['nonverbal']}]")

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
    ap.add_argument("--setup-tts", action="store_true",
                    help="check for a voice bank and report TTS readiness")
    args = ap.parse_args()

    agent = VoiceAgent(args.base, args.adapter, device=args.device,
                       dtype=args.dtype)
    agent.setup_tts()

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