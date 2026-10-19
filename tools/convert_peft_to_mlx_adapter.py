"""Convert a PEFT LoRA adapter into the MLX adapter format.

Why
---
MLX has native Gated DeltaNet kernels, so training the same LoRA under MLX on
an Apple Silicon machine is several times faster than the PyTorch path (whose
DeltaNet layers have to fall back to a pure-PyTorch reference implementation,
because ``fla`` / ``causal_conv1d`` / ``kernels`` are CUDA-only).

But MLX and PEFT disagree about adapter layout:

    PEFT   lora_A  [r, in]        lora_B  [out, r]
    MLX    lora_a  [in, r]        lora_b  [r, out]

Both are plain transposes of each other, and the scaling convention already
matches: PEFT uses ``alpha / r`` and MLX uses a ``scale`` parameter, so
``scale = alpha / r`` (32 / 16 = 2.0) needs no extra factor.

Naming
------
PEFT   base_model.model.model.language_model.layers.{L}.{mod}.lora_A.weight
MLX    language_model.model.layers.{L}.{mod}.lora_a

``{mod}`` is identical on both sides (``mlp.gate_proj``, ``self_attn.q_proj``),
so the mapping is a prefix strip plus a transpose.

A structural subtlety
----------------------
The HF checkpoint has ``self_attn`` only on the 6 full-attention layers
(0-indexed 3, 7, 11, 15, 19, 23); the 18 DeltaNet layers have ``linear_attn``
instead. MLX's ``DecoderLayer`` *instantiates* ``self_attn`` on every layer but
only calls it on the full-attention ones, so the 18 extras are never executed
and their weights are irrelevant. We deliberately emit adapter tensors only for
the 192 modules that actually exist in the PEFT adapter, so the two files
describe exactly the same set of trained parameters.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import LORA_DIR, MLX_ADAPTER_DIR  # noqa: E402

PEFT_PREFIX = "base_model.model.model.language_model.layers."
MLX_PREFIX = "language_model.model.layers."

PEFT_RE = re.compile(
    r"^base_model\.model\.model\.language_model\.layers\.(\d+)\.(.+)\.lora_(A|B)\.weight$"
)


def convert(peft_dir: Path, out_dir: Path, num_layers: int = 24) -> dict:
    import mlx.core as mx
    from safetensors.torch import load_file

    adapter_file = peft_dir / "adapter_model.safetensors"
    peft_config = json.loads((peft_dir / "adapter_config.json").read_text(encoding="utf-8"))

    rank = int(peft_config["r"])
    alpha = int(peft_config["lora_alpha"])
    dropout = float(peft_config.get("lora_dropout", 0.0))
    scale = alpha / rank

    state = load_file(str(adapter_file))

    out: dict[str, mx.array] = {}
    modules: set[tuple[int, str]] = set()
    for key, tensor in state.items():
        m = PEFT_RE.match(key)
        if not m:
            raise SystemExit(f"unexpected adapter key: {key}")
        layer, mod, which = int(m.group(1)), m.group(2), m.group(3)
        if layer >= num_layers:
            continue
        arr = tensor.detach().cpu().numpy()
        if which == "A":
            out[f"{MLX_PREFIX}{layer}.{mod}.lora_a"] = mx.array(arr.T)
        else:
            out[f"{MLX_PREFIX}{layer}.{mod}.lora_b"] = mx.array(arr.T)
        modules.add((layer, mod))

    mods = sorted({m for _, m in modules})
    layers = sorted({l for l, _ in modules})

    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), out)

    config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": rank,
            "scale": scale,
            "dropout": dropout,
            # Relative to a decoder layer, matching MLX's named_modules().
            "keys": [
                "mlp.gate_proj",
                "mlp.down_proj",
                "mlp.up_proj",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
            ],
        },
        "source": "peft-conversion",
        "peft_base_model": peft_config.get("base_model_name_or_path"),
    }
    (out_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    return {
        "tensors": len(out),
        "modules": len(modules),
        "layers": layers,
        "module_names": mods,
        "rank": rank,
        "alpha": alpha,
        "scale": scale,
    }


def verify(adapter_dir: Path, peft_dir: Path, prompt: str) -> dict:
    """Load the MLX base + converted adapter and check it produces output."""
    from mlx_lm.utils import load

    model, tokenizer = load(str(adapter_dir.parent / "base"), adapter_path=str(adapter_dir))
    messages = [{"role": "user", "content": prompt}]
    template = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    from mlx_lm.generate import generate

    text = generate(model, tokenizer, prompt=template, max_tokens=64, verbose=False)
    return {"output": text}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--peft", type=Path, default=LORA_DIR)
    ap.add_argument("--out", type=Path, default=MLX_ADAPTER_DIR)
    ap.add_argument("--num-layers", type=int, default=24)
    ap.add_argument("--verify", action="store_true", help="generate a sample")
    args = ap.parse_args()

    info = convert(args.peft, args.out, args.num_layers)
    print(f"converted {info['tensors']} tensors across {info['modules']} modules")
    print(f"layers touched: {info['layers']}")
    print(f"modules: {info['module_names']}")
    print(f"rank={info['rank']} alpha={info['alpha']} scale={info['scale']}")
    print(f"-> {args.out}")

    if args.verify:
        out = verify(args.out, args.peft, "The printer produced beautiful gibberish.")
        print("\n--- sample generation ---")
        print(out["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
