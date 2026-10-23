"""Continue training the agents3 LoRA so the model learns to emit emotion tags.

What "continue" means
----------------------
The existing ``agents3`` adapter already encodes the persona. We do not train a
second adapter and stack them - we load that adapter in trainable mode and keep
going, so the result is a single adapter that carries both the persona and the
emotion behaviour. One artefact, one merge, no stacking.

Target modules
--------------
The original adapter was trained against the *multimodal* checkpoint, where
the language model lives under ``model.language_model.layers.*``. The text-only
base we train against is named ``model.layers.*``, so the original
``target_modules`` regex no longer matches anything.

The replacement regex is ``.*\\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)``.
On the text-only base that resolves to exactly the same 96 modules the
original adapter touched:

  * ``mlp.{gate,up,down}_proj`` on all 24 layers                    (72)
  * ``self_attn.{q,k,v,o}_proj`` on the 6 full-attention layers      (24)

The 18 DeltaNet layers have no ``self_attn`` at all, and ``linear_attn`` uses
``in_proj_*`` / ``out_proj`` names that do not match the pattern, so they stay
frozen exactly as they were. ``--assert-targets`` re-derives the module set
and compares it to the original adapter's tensor list, and aborts on any
mismatch.

Loss masking
------------
Only assistant tokens contribute to the loss. Training on the system prompt
would spend capacity on tokens we never want the model to produce, and would
teach it to *predict* tags rather than *generate* them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    LORA_DIR,
    TEXT_BASE_DIR,
    read_json,
    read_jsonl,
    write_json,
)

TARGET_REGEX = r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


# --------------------------------------------------------------------------
# Adapter plumbing
# --------------------------------------------------------------------------


def peft_module_names(adapter_dir: Path) -> set[str]:
    """The (module, ab) pairs the original adapter actually contains."""
    from safetensors.torch import load_file

    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    names = set()
    for key in state:
        # Accept both the multimodal key layout (the adapter as shipped) and
        # the text-only layout (the pre-patched copy the Colab notebook builds
        # before merging). Passing either one to --adapter must work.
        m = re.match(
            r"^base_model\.model\.model\.(?:language_model\.)?layers\.(\d+)\.(.+)\.lora_([AB])\.weight$",
            key,
        )
        if m:
            names.add((int(m.group(1)), m.group(2)))
    return names


OLD_PATH = "base_model.model.model.language_model.layers."
NEW_PATH = "base_model.model.model.layers."


def patched_adapter_dir(src: Path, dst: Path) -> Path:
    """Copy the adapter, retargeted at the text-only base.

    Two things have to change, and missing either one is silent:

    1. ``target_modules`` in the config. The original regex is anchored on
       ``language_model``, which no longer appears in the text-only base's
       module names, so it would match nothing at all.

    2. The *weight key names* inside ``adapter_model.safetensors``. They
       encode the module path as it was in the multimodal checkpoint
       (``base_model.model.model.language_model.layers.0.mlp.gate_proj...``).
       PEFT resolves saved weights by that path, so against the text-only base
       every tensor is reported as a missing key and the adapter loads as
       random values - which then trains to nothing useful.
    """
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    cfg = dict(read_json(src / "adapter_config.json"))
    cfg["target_modules"] = TARGET_REGEX
    (dst / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    state = load_file(str(src / "adapter_model.safetensors"))
    renamed = {key.replace(OLD_PATH, NEW_PATH): tensor for key, tensor in state.items()}
    save_file(renamed, str(dst / "adapter_model.safetensors"), metadata={"format": "pt"})

    for name in ("tokenizer.json", "tokenizer_config.json"):
        p = src / name
        if p.exists():
            (dst / name).write_bytes(p.read_bytes())
    return dst


def assert_targets(model, expected: set[tuple[int, str]], verbose: bool = True) -> None:
    """The LoRA modules PEFT actually created must match the original set."""
    created = set()
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        m = re.match(r"^.*\.layers\.(\d+)\.(.+)$", name)
        if m:
            created.add((int(m.group(1)), m.group(2)))
    if created != expected:
        missing = sorted(expected - created)
        extra = sorted(created - expected)
        raise SystemExit(
            "target module mismatch:\n"
            f"  missing: {missing[:8]}{' ...' if len(missing) > 8 else ''}\n"
            f"  extra:   {extra[:8]}{' ...' if len(extra) > 8 else ''}\n"
            f"  expected {len(expected)} modules, PEFT created {len(created)}"
        )
    if verbose:
        print(f"  target modules OK: {len(created)} modules match the original adapter")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def build_examples(records: list[dict], tokenizer, max_len: int) -> list[dict]:
    """Tokenise into (input_ids, labels) with the prompt masked out."""
    from common import build_tokenized_examples

    return build_tokenized_examples(records, tokenizer, max_len)


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def make_batches(examples: list[dict], batch_size: int, seed: int) -> list[list[dict]]:
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    return [
        [examples[i] for i in idx[i : i + batch_size]]
        for i in range(0, len(idx), batch_size)
    ]


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------


def train(args) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda":
        # T4 (sm_75) has no native bf16; PyTorch emulates it and it is slow.
        # fp16 + GradScaler is the correct path on free Colab. The explicit
        # flag matters because `is_bf16_supported()` can lie on a virtualised
        # runtime and silently put the T4 on the slow emulated path.
        if args.dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif args.dtype == "fp16":
            dtype = torch.float16
        elif args.dtype == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32
    amp_dtype = dtype if device == "cuda" else torch.float32
    use_scaler = device == "cuda" and dtype == torch.float16

    print(f"device: {device}  dtype: {dtype}")
    if device == "cuda":
        print(
            f"gpu: {torch.cuda.get_device_name(0)}  "
            f"capability: {torch.cuda.get_device_capability(0)}"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        attn_implementation="eager" if device == "mps" else None,
    )
    model.to(device)

    # The vocabulary is 248320 wide, so the logits for a batch of B sequences
    # of length L is B*L*248320*4 bytes - 670 MB for a single sequence at 340
    # tokens, and the backward pass needs several copies of it. Gradient
    # checkpointing trades compute for that memory, which is the right trade
    # here: MPS is already slow because of the DeltaNet reference path, and
    # running out of memory is worse than running slow.
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        print("  gradient checkpointing: on")

    patched = patched_adapter_dir(args.adapter, args.out / "_patched_adapter")
    print("loading existing adapter in trainable mode...")
    model = PeftModel.from_pretrained(model, str(patched), is_trainable=True)

    if args.assert_targets:
        expected = peft_module_names(args.adapter)
        assert_targets(model, expected)
    model.train()

    records = read_jsonl(args.data)
    print(f"training records: {len(records)}")
    examples = build_examples(records, tokenizer, args.max_seq_length)
    lens = [len(e["input_ids"]) for e in examples]
    print(
        f"tokenised: {len(examples)} examples, len min/med/max = "
        f"{min(lens)}/{sorted(lens)[len(lens) // 2]}/{max(lens)}"
    )

    batches = make_batches(examples, args.batch_size, args.seed)
    steps_per_epoch = max(1, len(batches) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    print(
        f"{len(batches)} micro-batches/epoch, {steps_per_epoch} optimizer steps/epoch, "
        f"{total_steps} total"
    )

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    print(f"trainable parameters: {n_trainable:,}")
    if n_trainable == 0:
        raise SystemExit("no trainable parameters - check the target regex")

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = get_cosine_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    history: list[dict] = []

    pad_id = tokenizer.pad_token_id
    step = 0
    t0 = time.time()
    probe_done = False
    probe_times: list[float] = []

    val_examples = (
        build_examples(read_jsonl(args.valid_data), tokenizer, args.max_seq_length)
        if args.valid_data and args.valid_data.exists()
        else []
    )
    if val_examples:
        print(f"validation examples: {len(val_examples)}")
    else:
        print("no validation set - skipping validation loss")

    def evaluate() -> float | None:
        if not val_examples:
            return None
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_examples), args.batch_size):
                batch = collate(val_examples[i : i + args.batch_size], pad_id)
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                total += out.loss.item() * len(batch["input_ids"])
                count += len(batch["input_ids"])
        model.train()
        return total / count if count else None

    for epoch in range(args.epochs):
        epoch_batches = make_batches(examples, args.batch_size, args.seed + epoch)
        optim.zero_grad(set_to_none=True)
        for i, batch in enumerate(epoch_batches):
            batch = collate(batch, pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")
            ):
                out = model(**batch)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()

            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                step += 1

                if step == args.probe_steps and not probe_done:
                    probe_done = True
                    probe_times.append(time.time() - t0)
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
                        f"loss {loss.item() * args.grad_accum:.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e} "
                        f"{el / 60:.1f}m elapsed",
                        flush=True,
                    )

        val = evaluate()
        if val is not None:
            print(f"  epoch {epoch + 1}/{args.epochs} validation loss {val:.4f}")
            history.append({"epoch": epoch + 1, "step": step, "val_loss": val})
        if args.save_each_epoch:
            # Free Colab can drop the session at any time. Saving after every
            # epoch means a disconnect costs one epoch, not the whole run.
            model.save_pretrained(str(args.out / "adapter"))
            print(f"  checkpoint saved -> {args.out / 'adapter'}")

    print(f"\ntraining done in {(time.time() - t0) / 60:.1f} min")
    model.save_pretrained(str(args.out / "adapter"))
    print(f"saved adapter -> {args.out / 'adapter'}")

    write_json(
        args.out / "train_report.json",
        {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None,
            "dtype": str(dtype),
            "amp_dtype": str(amp_dtype),
            "use_grad_scaler": use_scaler,
            "records": len(records),
            "examples": len(examples),
            "validation_examples": len(val_examples),
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "total_steps": total_steps,
            "minutes": round((time.time() - t0) / 60, 2),
            "trainable_parameters": n_trainable,
            "validation_history": history,
        },
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument(
        "--valid-data",
        type=Path,
        default=None,
        help="validation split, for the per-epoch loss curve",
    )
    ap.add_argument("--base", type=Path, default=TEXT_BASE_DIR)
    ap.add_argument("--adapter", type=Path, default=LORA_DIR)
    ap.add_argument("--out", type=Path, default=Path("models/llm/agents3_emotion_adapter"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="force fp16 on a T4; auto trusts is_bf16_supported()",
    )
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--probe-steps", type=int, default=10)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--save-each-epoch", action="store_true", default=True)
    ap.add_argument("--no-save-each-epoch", dest="save_each_epoch", action="store_false")
    ap.add_argument("--assert-targets", action="store_true", default=True)
    return train(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
