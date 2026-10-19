"""Train the emotion LoRA under MLX - the fast path on Apple Silicon.

Why this exists
---------------
The PyTorch path has to run the 18 Gated DeltaNet layers through the
pure-PyTorch reference implementation, because ``fla``, ``causal_conv1d`` and
``kernels`` are CUDA-only and will not install on macOS. Measured on this
machine that is about **100 seconds per optimizer step**, which is unusable.

MLX has native Gated DeltaNet kernels, so the same step runs in a fraction of
the time. The adapter itself is converted from PEFT to the MLX layout by
``tools/convert_peft_to_mlx_adapter.py`` - the two formats are plain
transposes of each other.

The training loop is written out here rather than reusing ``mlx_lm.lora``
because we need one thing that helper does not offer: a loss that is masked to
the assistant tokens only. The model's own ``default_loss`` masks by sequence
length, which would put the system prompt into the loss.

Only the LoRA parameters are trainable; the base weights are frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    MLX_ADAPTER_DIR,
    MLX_BASE_DIR,
    build_tokenized_examples,
    completion_mask,
    read_jsonl,
    write_json,
)


def build_batches(examples: list[dict], batch_size: int, seed: int) -> list[list[dict]]:
    """Sort by length into buckets, then shuffle the buckets.

    Sorting by length keeps padding low, which matters a lot here: the
    vocabulary is 248320 wide, so every padded position costs real memory in
    the logits.
    """
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    batches = [
        [examples[i] for i in order[i : i + batch_size]]
        for i in range(0, len(order), batch_size)
    ]
    random.Random(seed).shuffle(batches)
    return batches


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=MLX_BASE_DIR)
    ap.add_argument("--adapter", type=Path, default=MLX_ADAPTER_DIR)
    ap.add_argument("--out", type=Path, default=Path("models/llm/agents3_emotion_adapter_mlx"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--probe-steps", type=int, default=10)
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm.tuner.utils import load_adapters
    from mlx_lm.utils import load

    print(f"loading base: {args.base}")
    model, tokenizer = load(str(args.base))
    print(f"loading adapter: {args.adapter}")
    load_adapters(model, str(args.adapter))

    # Freeze everything, then unfreeze just the LoRA parameters.
    model.freeze()
    lora_keys = [
        k
        for k, _ in tree_flatten(model.parameters())
        if k.rsplit(".", 1)[-1] in ("lora_a", "lora_b")
    ]
    model.unfreeze(keys=lora_keys)
    n_trainable = sum(
        v.size for k, v in tree_flatten(model.parameters()) if k in set(lora_keys)
    )
    print(f"trainable parameters: {n_trainable:,} in {len(lora_keys)} tensors")

    records = read_jsonl(args.data)
    print(f"records: {len(records)}")
    examples = build_tokenized_examples(records, tokenizer, args.max_seq_length)
    lens = [len(e["input_ids"]) for e in examples]
    print(
        f"tokenised: {len(examples)} examples, len min/med/max = "
        f"{min(lens)}/{sorted(lens)[len(lens) // 2]}/{max(lens)}"
    )

    batches = build_batches(examples, args.batch_size, args.seed)
    steps_per_epoch = len(batches)
    total_steps = steps_per_epoch * args.epochs
    print(f"{steps_per_epoch} steps/epoch, {total_steps} total")

    def loss(model, inputs, targets, mask):
        logits = model(inputs)
        ce = nn.losses.cross_entropy(logits, targets) * mask
        return ce.astype(mx.float32).sum() / mx.maximum(mask.sum(), 1)

    optimizer = optim.AdamW(learning_rate=args.lr)
    loss_value_and_grad = nn.value_and_grad(model, loss)

    def make_batch(batch):
        width = max(len(b["input_ids"]) for b in batch)
        ids, labels = [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)
        ids = mx.array(ids, dtype=mx.int32)
        labels = mx.array(labels, dtype=mx.int32)
        inputs = ids[:, :-1]
        targets = labels[:, 1:]
        mask = mx.array(completion_mask(labels[:, 1:]), dtype=mx.float32)
        return inputs, targets, mask

    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in build_batches(examples, args.batch_size, args.seed + epoch):
            optimizer.learning_rate = lr_at(step)
            inputs, targets, mask = make_batch(batch)
            (lvalue, _), grad = loss_value_and_grad(model, inputs, targets, mask)
            optimizer.update(model, grad)
            mx.eval(model.parameters(), optimizer.state)
            step += 1

            if step == args.probe_steps:
                rate = (time.time() - t0) / step
                print(
                    f"\n  SPEED PROBE: {rate:.2f} s/step -> "
                    f"{rate * total_steps / 60:.1f} min for {total_steps} steps\n"
                )
                if args.probe_only:
                    return 0
            if step % args.log_every == 0:
                el = time.time() - t0
                print(
                    f"  epoch {epoch + 1}/{args.epochs} step {step}/{total_steps} "
                    f"loss {lvalue.item():.4f} lr {optimizer.learning_rate:.2e} "
                    f"{el / 60:.1f}m elapsed",
                    flush=True,
                )

    args.out.mkdir(parents=True, exist_ok=True)
    weights = {
        k: v
        for k, v in tree_flatten(model.parameters())
        if k.rsplit(".", 1)[-1] in ("lora_a", "lora_b")
    }
    mx.save_safetensors(str(args.out / "adapters.safetensors"), dict(weights))

    src_cfg = json.loads((args.adapter / "adapter_config.json").read_text())
    (args.out / "adapter_config.json").write_text(
        json.dumps(src_cfg, indent=2), encoding="utf-8"
    )
    write_json(
        args.out / "train_report.json",
        {
            "backend": "mlx",
            "records": len(records),
            "examples": len(examples),
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "total_steps": total_steps,
            "minutes": round((time.time() - t0) / 60, 2),
            "trainable_parameters": n_trainable,
        },
    )
    print(f"\ntraining done in {(time.time() - t0) / 60:.1f} min")
    print(f"saved adapter -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
