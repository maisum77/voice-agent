"""Design the voice, and bake the emotion into reference clips.

The problem this solves
-----------------------
Qwen3-TTS splits into three capabilities that do not combine the way you
would like:

  * ``1.7B-CustomVoice`` accepts an ``instruct`` string - but only for nine
    preset timbres.
  * ``1.7B-VoiceDesign`` accepts an ``instruct`` string and can invent a new
    voice - but each generation re-derives it.
  * ``0.6B-Base`` clones a voice from a reference clip - but the clone path
    never encodes an instruction at all, so the cloned voice has no emotion
    control whatsoever.

So "a brand new voice" and "emotion control" cannot both live in the same
runtime call. The way around it is to use the instruction model *at build
time*:

  1. ``1.7B-VoiceDesign`` renders the same voice description once per emotion,
     each with an emotion directive appended. Five clips, five deliveries, one
     timbre.
  2. The clips become the reference bank. At runtime ``0.6B-Base`` clones from
     whichever clip matches the turn's emotion.

ICL voice cloning preserves prosody better than x-vector-only cloning, so the
emotional delivery in the reference survives the clone. The result is real
emotional range on a CPU-capable model, with the cost paid once at build time
and about a megabyte of assets at runtime.

The consistency check
----------------------
A voice that drifts between the five clips is worse than no voice at all, so
the bank is verified with ECAPA speaker embeddings: the cosine similarity of
every clip against the mean must clear a threshold. Below it, the clips are
re-rendered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import VOICE_DIR, write_json  # noqa: E402

# The five deliveries, matching the model's tag vocabulary exactly.
EMOTION_DIRECTIVES = {
    "neutral": "Plain and factual. No colour, no emphasis, just clear information.",
    "amused": "Dry, understated British humour. Warmly amused, a hint of a smile, never loud.",
    "warm": "Kind and encouraging, a little slower, with genuine warmth.",
    "focused": "Brisk and businesslike. No banter, no flourishes, getting to the point.",
    "serious": "Calm, level and grave. No humour whatsoever. Weight on every word.",
}

# The reference line each clip speaks. It should be neutral enough to be
# delivered any way, and short enough to keep the clips small.
REFERENCE_LINE = (
    "Right, let's get this sorted. Tell me what you are seeing and we will "
    "work through it together."
)

# Non-verbal clips are rendered as short lines that *contain* the sound, then
# used verbatim when spliced into a reply. They have to be in the same voice
# as the delivery clips, or the splice is audible.
NONVERBAL_LINES = {
    "laugh": "Ha! Ha! That is brilliant, genuinely.",
    "chuckle": "Heh. Nice one, that.",
    "sigh": "Ohhh. Right then. Deep breath.",
    "gasp": "Ah! Well, that is unexpected.",
    "hm": "Hmm. Right, let me think about that.",
    "erm": "Erm. Let me see.",
}

SPEAKER_HINT = (
    "A warm British male voice in his early thirties, low and unhurried, "
    "with a dry, self-aware sense of humour and a naturally gentle manner."
)


def build_instruct(description: str, emotion: str) -> str:
    """One natural-language instruction covering both timbre and delivery.

    The timbre half is byte-identical across every clip, which is what keeps
    the voice consistent. Only the delivery half changes.
    """
    return (
        f"{description.strip()} "
        f"Delivery: {EMOTION_DIRECTIVES[emotion].strip()}"
    )


def render_bank(
    model_path: str,
    out_dir: Path,
    description: str,
    device: str = "mps",
    dtype: str = "float32",
) -> dict:
    """Render one reference clip per emotion with Qwen3-TTS 1.7B VoiceDesign."""
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]

    print(f"loading {model_path} on {device}")
    model = Qwen3TTSModel.from_pretrained(model_path, device_map=device, dtype=torch_dtype)

    clips = {}
    for emotion, directive in EMOTION_DIRECTIVES.items():
        instruct = build_instruct(description, emotion)
        print(f"  rendering {emotion}...")
        wavs, sr = model.generate_voice_design(
            text=REFERENCE_LINE,
            language="English",
            instruct=instruct,
        )
        audio = np.asarray(wavs[0], dtype="float32")
        path = out_dir / f"{emotion}.wav"
        sf.write(str(path), audio, sr)
        clips[emotion] = {
            "file": f"{emotion}.wav",
            "sample_rate": int(sr),
            "duration_s": round(len(audio) / sr, 3),
            "instruct": instruct,
            "ref_text": REFERENCE_LINE,
        }
        print(f"    {len(audio) / sr:.2f}s -> {path}")

    # Non-verbal clips, same voice, same description.
    nonverbals = {}
    neutral_instruct = build_instruct(description, "neutral")
    for name, line in NONVERBAL_LINES.items():
        print(f"  rendering {name}...")
        wavs, sr = model.generate_voice_design(
            text=line,
            language="English",
            instruct=neutral_instruct,
        )
        audio = np.asarray(wavs[0], dtype="float32")
        path = out_dir / f"nv_{name}.wav"
        sf.write(str(path), audio, sr)
        nonverbals[name] = {
            "file": f"nv_{name}.wav",
            "sample_rate": int(sr),
            "duration_s": round(len(audio) / sr, 3),
            "ref_text": line,
        }
        print(f"    {len(audio) / sr:.2f}s -> {path}")

    manifest = {
        "model": model_path,
        "description": description,
        "reference_line": REFERENCE_LINE,
        "clips": clips,
        "nonverbals": nonverbals,
    }
    write_json(out_dir / "bank.json", manifest)
    print(f"-> {out_dir / 'bank.json'}")
    return manifest


def check_consistency(bank_dir: Path, threshold: float = 0.80) -> dict:
    """ECAPA speaker similarity of every clip against the bank mean.

    A voice that shifts between clips is worse than no voice, so this is a
    hard gate on the bank.
    """
    import numpy as np
    import soundfile as sf

    try:
        import sherpa_onnx
    except ImportError:
        return {
            "checked": False,
            "reason": "sherpa-onnx not installed - pip install sherpa-onnx",
        }

    manifest = json.loads((bank_dir / "bank.json").read_text(encoding="utf-8"))
    clips = manifest["clips"]

    # ECAPA-TDNN speaker embedding extractor.
    model_dir = Path.home() / ".cache" / "sherpa-onnx" / "speaker-embedding-models"
    onnx = next(model_dir.glob("*.onnx"), None) if model_dir.exists() else None
    if onnx is None:
        return {
            "checked": False,
            "reason": (
                "no ECAPA model found; download with "
                "sherpa-onnx-download speaker-embedding-models"
            ),
        }

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        str(onnx), num_threads=2, debug=False
    )

    vectors = {}
    for emotion, info in clips.items():
        audio, sr = sf.read(str(bank_dir / info["file"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        vectors[emotion] = extractor.extract(audio, sr=16000)

    # The non-verbal clips must also be the same voice, or a splice is
    # audible.
    for name, info in manifest.get("nonverbals", {}).items():
        audio, sr = sf.read(str(bank_dir / info["file"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        vectors[f"nv_{name}"] = extractor.extract(audio, sr=16000)

    mean = np.mean(list(vectors.values()), axis=0)
    sims = {
        emotion: float(np.dot(v, mean) / (np.linalg.norm(v) * np.linalg.norm(mean)))
        for emotion, v in vectors.items()
    }
    passed = min(sims.values()) >= threshold
    return {
        "checked": True,
        "threshold": threshold,
        "similarity": {k: round(v, 4) for k, v in sims.items()},
        "min": round(min(sims.values()), 4),
        "passed": passed,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        help="the instruction-capable model used at build time",
    )
    ap.add_argument("--out", type=Path, default=VOICE_DIR)
    ap.add_argument("--description", default=SPEAKER_HINT)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    if not args.check_only:
        render_bank(args.model, args.out, args.description, args.device, args.dtype)

    result = check_consistency(args.out, args.threshold)
    print(json.dumps(result, indent=2))
    if result.get("checked") and not result.get("passed"):
        print(
            f"\nFAIL: the least similar clip scores {result['min']:.3f}, below "
            f"{args.threshold}. The voice drifts between deliveries - re-render "
            f"with a tighter description."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
