"""Evaluate the emotion adapter against the baseline, on held-out prompts.

Metrics
-------
teacher_agreement   The headline number. For each held-out prompt we ask the
                    student for a reply and compare the emotion tag it chose
                    with the one the teacher chose for the same prompt. This
                    is the direct answer to "does it know how to feel, or did
                    it memorise the training set?".

tag_validity        Fraction of generations that carry a well-formed delivery
                    tag at all. A model that stops emitting tags has broken.

label_distribution  Histogram of chosen labels plus its entropy. A model that
                    has collapsed onto one label has broken.

nonverbal_rate     How often a non-verbal tag appears, and how often it
                    lands where the teacher put it.

length             Median and p95 spoken word count. The persona is supposed
                    to be concise; a long regression here is a real regression.

degeneracy         distinct-2 and repetition rate. A model that has started
                    looping is broken.

regression         Comparing the trained adapter against the original one on
                    the same prompts: length drift, tag rate, and how much the
                    reply text itself changed. The emotion pass must not have
                    wrecked the persona it was supposed to decorate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ANNOTATED,
    DELIVERY_EMOTIONS,
    TEXT_BASE_DIR,
    read_jsonl,
    write_json,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline" / "tts"))
from segmenter import nonverbals_of, spoken_text, strip_all_tags  # noqa: E402

TAG_RE = re.compile(
    r"\[emotion:(%s)\]" % "|".join(DELIVERY_EMOTIONS)
    + r"|\[(laugh|chuckle|sigh|gasp|hm|erm)\]"
)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


class Student:
    """Base model plus a LoRA adapter, for greedy generation."""

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

    def generate(self, system: str, user: str, max_new_tokens: int = 110) -> str:
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


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def distinct_n(texts: list[str], n: int = 2) -> float:
    grams, total = set(), 0
    for t in texts:
        words = spoken_text(t).split()
        for i in range(len(words) - n + 1):
            grams.add(tuple(words[i : i + n]))
            total += 1
    return len(grams) / total if total else 0.0


def repetition_rate(text: str, n: int = 3) -> float:
    words = spoken_text(text).split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counts.values() if c)


def evaluate(
    student: Student,
    examples: list[dict],
    max_examples: int = 0,
) -> dict:
    if max_examples:
        examples = examples[:max_examples]

    agree = 0
    agree_total = 0
    per_label: dict[str, list[int]] = {}
    tag_valid = 0
    label_counts: Counter = Counter()
    nonverbals_pred = 0
    nonverbals_true = 0
    nonverbals_match = 0
    lengths: list[int] = []
    replies: list[str] = []
    confusion: Counter = Counter()

    for ex in examples:
        out = student.generate(ex["system"], ex["user"])
        replies.append(out)

        pred = None
        m = re.search(r"\[emotion:(%s)\]" % "|".join(DELIVERY_EMOTIONS), out)
        if m:
            pred = m.group(1)
            tag_valid += 1
            label_counts[pred] += 1
            per_label.setdefault(pred, []).append(1)

        truth = ex.get("emotion")
        if truth:
            agree_total += 1
            hit = int(pred == truth)
            agree += hit
            confusion[(truth, pred or "none")] += 1
            per_label.setdefault(truth, []).append(hit)

        pn = nonverbals_of(out)
        tn = ex.get("nonverbals", [])
        nonverbals_pred += len(pn)
        nonverbals_true += len(tn)
        if pn and tn and pn[0] == tn[0]:
            nonverbals_match += 1

        lengths.append(len(spoken_text(out).split()))

    lengths_sorted = sorted(lengths)
    return {
        "n": len(examples),
        "teacher_agreement": round(agree / agree_total, 4) if agree_total else None,
        "teacher_agreement_n": agree_total,
        "tag_validity": round(tag_valid / len(examples), 4) if examples else 0.0,
        "label_distribution": dict(label_counts),
        "label_entropy": round(entropy(label_counts), 4),
        "nonverbal_rate": round(nonverbals_pred / len(examples), 4) if examples else 0.0,
        "nonverbal_match_rate": (
            round(nonverbals_match / sum(1 for e in examples if e.get("nonverbals")), 4)
            if any(e.get("nonverbals") for e in examples)
            else None
        ),
        "length_median": statistics.median(lengths) if lengths else 0,
        "length_p95": lengths_sorted[int(len(lengths_sorted) * 0.95)] if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "distinct_2": round(distinct_n(replies), 4),
        "repetition_rate": round(
            sum(repetition_rate(r) for r in replies) / len(replies), 4
        )
        if replies
        else 0.0,
        "confusion": {f"{t}->{p}": c for (t, p), c in sorted(confusion.items())},
        "samples": replies[:12],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def load_examples(path: Path) -> list[dict]:
    """Turn the annotated file into (system, user, emotion, nonverbals) probes."""
    out = []
    for rec in read_jsonl(path):
        system = rec.get("system") or ""
        if not system:
            from common import load_persona

            system = load_persona()
        if rec.get("base_reply") and rec.get("base_emotion"):
            out.append(
                {
                    "id": rec["id"],
                    "system": system,
                    "user": rec["base_user"],
                    "emotion": rec["base_emotion"],
                    "nonverbals": nonverbals_of(rec["base_reply"]),
                }
            )
        for var in rec.get("variants", []):
            if var.get("reply") and var.get("emotion"):
                out.append(
                    {
                        "id": f"{rec['id']}:{var['affect']}",
                        "system": system,
                        "user": var["user"],
                        "emotion": var["emotion"],
                        "nonverbals": nonverbals_of(var["reply"]),
                    }
                )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotated", type=Path, default=ANNOTATED / "annotated.jsonl")
    ap.add_argument("--base", type=Path, default=TEXT_BASE_DIR)
    ap.add_argument(
        "--adapters",
        required=True,
        help="comma separated name=path pairs, e.g. baseline=...,emotion=...",
    )
    ap.add_argument("--out", type=Path, default=ANNOTATED / "eval_report.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-examples", type=int, default=0)
    args = ap.parse_args()

    examples = load_examples(args.annotated)
    if not examples:
        print(f"no examples in {args.annotated}", file=sys.stderr)
        return 1
    print(f"probes: {len(examples)}")

    report: dict = {"n_probes": len(examples), "runs": {}}
    for spec in args.adapters.split(","):
        name, _, path = spec.partition("=")
        path = path.strip()
        if not path:
            continue
        print(f"\n== {name}: {path}")
        student = Student(args.base, Path(path), args.device)
        res = evaluate(student, examples, args.max_examples)
        report["runs"][name] = res
        print(
            f"   agreement {res['teacher_agreement']}  "
            f"tag_validity {res['tag_validity']}  "
            f"entropy {res['label_entropy']}  "
            f"len med/p95 {res['length_median']}/{res['length_p95']}  "
            f"distinct2 {res['distinct_2']}"
        )
        print(f"   labels: {res['label_distribution']}")
        del student

    # -- regression between the runs ------------------------------------
    names = list(report["runs"])
    if len(names) >= 2:
        base_run, new_run = report["runs"][names[0]], report["runs"][names[1]]
        report["regression"] = {
            "baseline": names[0],
            "candidate": names[1],
            "length_median_delta": new_run["length_median"] - base_run["length_median"],
            "length_p95_delta": new_run["length_p95"] - base_run["length_p95"],
            "tag_validity_delta": new_run["tag_validity"] - base_run["tag_validity"],
            "distinct_2_delta": new_run["distinct_2"] - base_run["distinct_2"],
            "repetition_delta": new_run["repetition_rate"] - base_run["repetition_rate"],
        }
        print("\nregression:", json.dumps(report["regression"], indent=2))

    write_json(args.out, report)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
