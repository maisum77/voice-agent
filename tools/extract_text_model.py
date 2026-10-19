"""Extract a text-only Qwen3_5ForCausalLM from the merged multimodal checkpoint.

``agents3_merged`` is a ``Qwen3_5ForConditionalGeneration``: 320 language-model
tensors plus 153 vision-tower tensors. A voice agent never feeds an image, so
the vision tower is 200 MB of dead weight that would otherwise be loaded,
moved to the accelerator and carried through every forward pass.

Layout of the merged checkpoint
-------------------------------
    model.language_model.embed_tokens.weight
    model.language_model.layers.{0..23}.linear_attn.*    (18 DeltaNet layers)
    model.language_model.layers.{3,7,11,15,19,23}.self_attn.*  (6 full-attn)
    model.language_model.layers.{0..23}.mlp.*
    model.language_model.norm.weight
    model.visual.*                                       <- dropped

Target layout (``Qwen3_5ForCausalLM``)
--------------------------------------
    model.embed_tokens.weight
    model.layers.{0..23}.*
    model.norm.weight
    (no lm_head — ``tie_word_embeddings`` is true, so the head *is* the
     embedding matrix)

The output loads with ``AutoModelForCausalLM`` and converts to GGUF for
llama.cpp with no further surgery.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import MERGED_DIR, TEXT_BASE_DIR  # noqa: E402

SRC_PREFIX = "model.language_model."
DST_PREFIX = "model."

# Files copied verbatim from the merged checkpoint.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
)


def build_text_config(merged_config: dict) -> dict:
    """Flatten ``text_config`` into a standalone ``qwen3_5_text`` config."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    text = dict(merged_config["text_config"])
    # Keys the text config class does not accept.
    for key in ("model_type", "architectures"):
        text.pop(key, None)

    cfg = Qwen3_5TextConfig(**text)
    out = cfg.to_dict()

    # Preserve the two flags that matter for correct loading.
    out["architectures"] = ["Qwen3_5ForCausalLM"]
    out["model_type"] = "qwen3_5_text"
    out["torch_dtype"] = merged_config.get("dtype", "bfloat16")
    out["tie_word_embeddings"] = True
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=MERGED_DIR)
    ap.add_argument("--out", type=Path, default=TEXT_BASE_DIR)
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    src_weights = args.src / "model.safetensors"
    if not src_weights.exists():
        print(f"missing {src_weights}", file=sys.stderr)
        return 1

    merged_config = json.loads((args.src / "config.json").read_text(encoding="utf-8"))
    text_config = build_text_config(merged_config)

    args.out.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    dropped = 0
    with safe_open(src_weights, framework="pt") as f:
        for key in f.keys():
            if not key.startswith(SRC_PREFIX):
                dropped += 1
                continue
            new_key = DST_PREFIX + key[len(SRC_PREFIX) :]
            tensors[new_key] = f.get_tensor(key)

    if "lm_head.weight" in tensors:
        raise SystemExit("unexpected lm_head.weight: embeddings are not tied")

    expected = {"model.embed_tokens.weight", "model.norm.weight"}
    missing = expected - set(tensors)
    if missing:
        raise SystemExit(f"missing expected tensors: {sorted(missing)}")

    save_file(tensors, str(args.out / "model.safetensors"), metadata={"format": "pt"})

    (args.out / "config.json").write_text(
        json.dumps(text_config, indent=2), encoding="utf-8"
    )
    for name in TOKENIZER_FILES:
        p = args.src / name
        if p.exists():
            shutil.copy2(p, args.out / name)

    total = len(tensors) + dropped
    print(f"kept {len(tensors)} text tensors, dropped {dropped} vision tensors")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
