"""Local inference: load base + emotion LoRA, generate a tagged reply, apply the guard.

Usage:
    .venv/bin/python tools/infer_reply.py --prompt "My build passed and all tests are green!"
    .venv/bin/python tools/infer_reply.py --prompt "I deleted my project folder, no backup" --adapter models/llm/v3_upload/agents3_emotion_adapter/adapter

This validates the trained model end-to-end: LLM emits a tagged reply, the
deterministic guard enforces the no-jokes-on-serious-turns rule, and the
segmenter (pipeline/tts/segmenter.py) can then split it into speech pieces.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "pipeline" / "emotion"))
sys.path.insert(0, str(ROOT / "pipeline" / "tts"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402
from guard import apply  # noqa: E402
from segmenter import parse_tagged_reply, spoken_text  # noqa: E402


def load_system_prompt() -> str:
    return (ROOT / "data" / "persona" / "system_prompt.txt").read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "models/llm/agents3_text_base")
    ap.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "models/llm/agents3_emotion_adapter/adapter",
        help="PEFT adapter dir (v1 by default; v3_upload also available)",
    )
    ap.add_argument("--prompt", required=True, help="user turn to reply to")
    ap.add_argument("--system", type=Path, default=None, help="override system prompt")
    ap.add_argument("--device", default="mps", help="cpu | mps | cuda")
    ap.add_argument("--dtype", default="bfloat16", help="float32 | bfloat16 | float16")
    ap.add_argument("--max-new-tokens", type=int, default=110)
    args = ap.parse_args()

    system = args.system.read_text(encoding="utf-8") if args.system else load_system_prompt()

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        args.dtype
    ]

    print(f"loading base {args.base} on {args.device} ({args.dtype}) ...")
    tok = AutoTokenizer.from_pretrained(str(args.base))
    model = AutoModelForCausalLM.from_pretrained(str(args.base), dtype=torch_dtype)
    print(f"applying adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.to(args.device).eval()

    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": args.prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok(prompt, return_tensors="pt").to(args.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    reply = tok.decode(
        out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    # Deterministic safety guard (forces [emotion:serious] on serious turns,
    # strips non-verbals, appends Confirm? when required).
    result = apply(args.prompt, reply)

    segments = parse_tagged_reply(result.text)

    print("\n=== raw model reply ===")
    print(reply)
    print("\n=== guarded reply ===")
    print(result.text)
    print(f"\n=== serious={result.serious} reasons={result.reasons} ===")
    print("segments:")
    for seg in segments:
        if seg.is_speech:
            print(f"  speech   [{seg.emotion:8s}] {seg.text!r}")
        else:
            print(f"  nonverbal [{seg.nonverbal}]")
    print(f"spoken: {spoken_text(result.text)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
