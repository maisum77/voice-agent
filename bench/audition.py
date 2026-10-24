"""Render the model's replies to audio so the emotion can be heard.

This is the last link in the loop. The model emits a tagged reply, the
segmenter splits it into speech and non-verbal pieces, and each piece is
synthesised with Qwen3-TTS cloning from the bank clip that matches the
delivery. Non-verbal tags splice in a pre-rendered clip.

Writes one wav per prompt plus a manifest recording the reply, its tags, the
time to first audio and the real-time factor, so the numbers and the audio
can be checked side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "pipeline" / "tts"))

from segmenter import parse_tagged_reply, spoken_text  # noqa: E402


def load_prompts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("prompts", [])


class ReplyGenerator:
    def __init__(self, base: Path, adapter: Path | None, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(base)
        self.model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter))
        self.model = self.model.to(device).eval()

    def reply(self, system: str, user: str, max_new_tokens: int = 110) -> str:
        import torch

        prompt = self.tok.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
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
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(
            out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "models/llm/agents3_text_base")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--tts", type=Path, default=ROOT / "models/tts_base/Qwen3-TTS-12Hz-0.6B-Base")
    ap.add_argument("--bank", type=Path, default=ROOT / "models/voice/bank")
    ap.add_argument("--prompts", type=Path, default=ROOT / "bench/audio/prompts.json")
    ap.add_argument("--out", type=Path, default=ROOT / "bench/audition")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-audio", action="store_true", help="text only, skip the TTS")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    if not prompts:
        print(f"no prompts at {args.prompts}", file=sys.stderr)
        return 1
    if args.limit:
        prompts = prompts[: args.limit]

    system = (ROOT / "data/persona/system_prompt.txt").read_text(encoding="utf-8").strip()

    print(f"generating {len(prompts)} replies...")
    gen = ReplyGenerator(args.base, args.adapter)
    replies = []
    for i, p in enumerate(prompts):
        text = gen.reply(system, p["user"] if isinstance(p, dict) else p)
        replies.append(text)
        print(f"  [{i + 1}/{len(prompts)}] {text[:110]}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []

    if not args.no_audio:
        import numpy as np
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel

        bank = json.loads((args.bank / "bank.json").read_text(encoding="utf-8"))
        print(f"loading TTS from {args.tts} on {args.device}")
        tts = Qwen3TTSModel.from_pretrained(
            str(args.tts), device_map=args.device, dtype=torch.float32
        )

        def render(text: str, emotion: str):
            clip = bank["clips"].get(emotion) or bank["clips"]["neutral"]
            max_new_tokens = max(24, min(256, len(text.split()) * 6))
            wavs, sr = tts.generate_voice_clone(
                text=text,
                language="English",
                ref_audio=str(args.bank / clip["file"]),
                ref_text=clip["ref_text"],
                max_new_tokens=max_new_tokens,
                non_streaming_mode=True,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.05,
            )
            audio = np.asarray(wavs[0], dtype="float32")
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError("TTS produced empty or non-finite audio")
            return audio, int(sr)

        for i, (p, reply) in enumerate(zip(prompts, replies)):
            segments = parse_tagged_reply(reply)
            pieces = []
            ttfa = None
            t0 = time.time()
            total_audio = 0.0

            for seg in segments:
                if seg.is_speech:
                    if not seg.text.strip():
                        continue
                    audio, sr = render(seg.text, seg.emotion)
                else:
                    info = bank.get("nonverbals", {}).get(seg.nonverbal)
                    if not info:
                        continue
                    audio, sr = sf.read(
                        str(args.bank / info["file"]), dtype="float32"
                    )
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                if ttfa is None:
                    ttfa = time.time() - t0
                total_audio += len(audio) / sr
                pieces.append((audio, sr))

            if not pieces:
                continue
            target_sr = int(pieces[0][1])
            merged_audio = []
            for audio, seg_sr in pieces:
                audio = np.asarray(audio, dtype="float32")
                if int(seg_sr) != target_sr:
                    try:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=int(seg_sr),
                                                target_sr=target_sr).astype("float32")
                    except Exception:
                        raise RuntimeError(f"sample-rate mismatch: {seg_sr} != {target_sr}")
                merged_audio.append(audio)
            sr = target_sr
            merged = np.concatenate(merged_audio)
            path = args.out / f"{i:03d}.wav"
            sf.write(str(path), merged, sr)
            elapsed = time.time() - t0

            manifest.append(
                {
                    "file": path.name,
                    "prompt": p["user"] if isinstance(p, dict) else p,
                    "reply": reply,
                    "spoken": spoken_text(reply),
                    "emotion": next(
                        (s.emotion for s in segments if s.is_speech), "neutral"
                    ),
                    "nonverbals": [s.nonverbal for s in segments if not s.is_speech],
                    "audio_seconds": round(total_audio, 3),
                    "render_seconds": round(elapsed, 3),
                    "rtf": round(elapsed / total_audio, 3) if total_audio else None,
                    "ttfa_seconds": round(ttfa, 3) if ttfa else None,
                }
            )
            print(
                f"  {path.name}: {total_audio:.1f}s audio, "
                f"rtf {manifest[-1]['rtf']}, ttfa {manifest[-1]['ttfa_seconds']}s"
            )
    else:
        for i, reply in enumerate(replies):
            manifest.append({"file": None, "reply": reply, "spoken": spoken_text(reply)})

    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"-> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
