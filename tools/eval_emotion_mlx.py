"""Evaluate the emotion adapter locally with MLX.

This is the fast, no-CUDA path: Apple Silicon has native Gated DeltaNet
kernels, so the same probes that take minutes under PyTorch's fallback run in
seconds here. The metrics and gates are imported from `eval_emotion.py`, so
the local number and the Colab number are directly comparable.

Usage:
    python3 tools/eval_emotion_mlx.py \
        --adapter models/llm/agents3_emotion_mlx_adapter \
        --annotated data/annotated/test_annotated.jsonl,data/eval/serious_probes.jsonl \
        --gate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "pipeline"))

from common import ANNOTATED, MLX_BASE_DIR, TEXT_BASE_DIR, write_json  # noqa: E402
from eval_emotion import (  # noqa: E402
    Generation,
    artifact_markers,
    evaluate,
    load_examples,
    run_gates,
    stop_token_ids,
)


class MLXStudent:
    """Greedy generation on an MLX base plus adapter, id-level stop control."""

    def __init__(self, base: Path, adapter: Path | None):
        import mlx.core as mx
        from mlx_lm.utils import load

        self.mx = mx
        self.model, self.tok = load(
            str(base), adapter_path=str(adapter) if adapter else None
        )
        self.stop_ids = set(stop_token_ids(self.tok))
        self.stop_ids.update(int(x) for x in getattr(self.tok, "eos_token_ids", []) or [])
        self.im_start = self._special_id("<|im_start|>")
        print(f"  mlx stop ids: {sorted(self.stop_ids)}")

    def _special_id(self, token: str) -> int | None:
        try:
            tid = self.tok.convert_tokens_to_ids(token)
        except Exception:
            return None
        if tid is None or tid == getattr(self.tok, "unk_token_id", None):
            return None
        return int(tid)

    def generate(self, system: str, user: str, max_new_tokens: int = 110) -> Generation:
        from mlx_lm.generate import generate_step

        prompt = self.tok.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = self.tok.encode(prompt)
        out_ids: list[int] = []
        greedy = lambda logits: self.mx.argmax(logits, axis=-1)  # noqa: E731
        for token, _ in generate_step(
            self.mx.array(prompt_ids),
            self.model,
            max_tokens=max_new_tokens,
            sampler=greedy,
        ):
            tid = int(token)
            if tid in self.stop_ids:
                break
            out_ids.append(tid)
            if self.im_start is not None and tid == self.im_start:
                break

        raw = self.tok.decode(out_ids, skip_special_tokens=False)
        text = self.tok.decode(out_ids, skip_special_tokens=True).strip()
        return Generation(text=text, raw=raw, artifacts=artifact_markers(raw))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--annotated",
        default=str(ANNOTATED / "annotated.jsonl"),
        help="annotated file(s), comma separated",
    )
    ap.add_argument("--base", type=Path, default=MLX_BASE_DIR)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ANNOTATED / "eval_mlx_report.json")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()

    examples = []
    for part in args.annotated.split(","):
        if part.strip():
            examples.extend(load_examples(Path(part.strip())))
    if not examples:
        print(f"no probes in {args.annotated}", file=sys.stderr)
        return 1
    print(f"probes: {len(examples)}")

    student = MLXStudent(args.base, args.adapter)
    result = evaluate(student, examples, args.max_examples)
    print(
        f"agreement {result['teacher_agreement']}  tag {result['tag_validity']}  "
        f"artifacts {result['artifact_rate']}  entropy {result['label_entropy']}  "
        f"len med/p95 {result['length_median']}/{result['length_p95']}  "
        f"distinct2 {result['distinct_2']}  rep {result['repetition_rate']}"
    )
    print("labels:", result["label_distribution"])
    print("per-label:", result["per_label"])
    print(
        f"serious raw/guarded: {result.get('serious_recall_raw')}/"
        f"{result.get('serious_recall_guarded')}  support {result.get('serious_support')}  "
        f"false-alarm {result.get('serious_false_alarm')}"
    )

    report = {"result": result}
    if args.gate:
        g = run_gates(result)
        report["gates"] = g
        print("gates:")
        for name, row in g["checks"].items():
            print(
                f"  {'PASS' if row['pass'] else 'FAIL':4s} {name:24s} "
                f"{row.get('value')} {row.get('op', '')} {row.get('threshold', '')}"
            )
    write_json(args.out, report)
    print(f"-> {args.out}")
    if args.gate and not report["gates"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
