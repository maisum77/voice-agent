"""Evaluate the emotion adapter against the baseline, on held-out prompts.

Metrics
-------
teacher_agreement   The headline number. For each held-out prompt we ask the
                    student for a reply and compare the emotion tag it chose
                    with the one the teacher chose for the same prompt.
tag_validity        Fraction of generations that carry a well-formed delivery
                    tag at all.
label_distribution  Histogram of chosen labels plus its entropy.
per_label           Recall and support for every delivery label. `serious`
                    is the one that is not allowed to fail.
artifact_rate       Fraction of generations that ran past their turn: role
                    headers, a second <|im_end|>, a thinking block, or a bare
                    `[amused]`-style tag. This is the failure that made the
                    first adapter look worse than it was.
nonverbal_*         How often a non-verbal appears, where it lands, and
                    whether one ever appears in a serious turn (must be zero).
length / degeneracy Median, p95, distinct-2 and repetition rate.
regression          Candidate against baseline for persona drift.

The stop contract
-----------------
Training ends every target with `<|im_end|>` (248046). The HF config shipped
with 248044 (`<|endoftext|>`) as EOS, so generation never stopped and the
model continued into a fabricated next turn. Every generation here stops on
both ids and forces EOS if `<|im_start|>` is ever sampled, and the artifact
metric makes the failure impossible to miss.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ANNOTATED,
    DELIVERY_EMOTIONS,
    TEXT_BASE_DIR,
    read_jsonl,
    special_token_id,
    write_json,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline" / "tts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from segmenter import nonverbals_of, spoken_text  # noqa: E402
from emotion.guard import serious_reasons  # noqa: E402

TAG_RE = re.compile(
    r"\[emotion:(%s)\]" % "|".join(DELIVERY_EMOTIONS)
    + r"|\[(laugh|chuckle|sigh|gasp|hm|erm)\]"
)

ARTIFACT_MARKERS = (
    "<|im_start|>",
    "<|endoftext|>",
    "<think>",
    "\nuser\n",
    "\nassistant\n",
    "\nuser",
    "\nassistant",
)

# A delivery label written without the `emotion:` prefix, e.g. `[amused]`.
# Non-verbals are *not* artifacts; they are a deliberate part of the voice.
BARE_TAG_RE = re.compile(r"\[(?:%s)\]" % "|".join(DELIVERY_EMOTIONS))

DEFAULT_GATES = {
    "tag_validity": 0.95,
    "artifact_rate": 0.0,
    "teacher_agreement": 0.55,
    "serious_recall": 0.90,
    "serious_false_reassure": 0,
    "nonverbal_in_serious": 0,
    "length_p95": 45,
    "repetition_rate": 0.03,
}


# --------------------------------------------------------------------------
# Stop handling
# --------------------------------------------------------------------------


def stop_token_ids(tok) -> list[int]:
    ids = []
    for name in ("<|im_end|>", "<|endoftext|>"):
        tid = special_token_id(tok, name)
        if tid is not None:
            ids.append(tid)
    return ids or [248046, 248044]


def artifact_markers(raw: str) -> list[str]:
    """Markers that mean the model generated past the end of its turn.

    One trailing `<|im_end|>` is the normal stop and is removed first.
    """
    body = raw.rstrip()
    if body.endswith("<|im_end|>"):
        body = body[: -len("<|im_end|>")]
    found = [m for m in ARTIFACT_MARKERS if m in body]
    if "<|im_end|>" in body:
        found.append("<|im_end|> (mid-turn)")
    if BARE_TAG_RE.search(body):
        found.append("bare-delivery-tag")
    return found


@dataclass
class Generation:
    """One sampled reply, with the raw form kept for artifact detection."""

    text: str
    raw: str = ""
    artifacts: list[str] = field(default_factory=list)


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
        self.stop_ids = stop_token_ids(self.tok)
        self.im_start = special_token_id(self.tok, "<|im_start|>")
        print(f"  stop ids: {self.stop_ids} (eos={self.tok.eos_token_id})")

    def generate(self, system: str, user: str, max_new_tokens: int = 110) -> Generation:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        class ForceStopOnRoleHeader(LogitsProcessor):
            """If the model tries to open a new turn, close this one instead."""

            def __init__(self, marker: int | None, eos: int):
                self.marker = marker
                self.eos = eos

            def __call__(self, input_ids, scores):
                if self.marker is None:
                    return scores
                if input_ids.shape[1] and input_ids[0, -1].item() == self.marker:
                    scores[0, :] = -float("inf")
                    scores[0, self.eos] = 0.0
                return scores

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
                eos_token_id=self.stop_ids,
                pad_token_id=self.tok.pad_token_id or self.stop_ids[0],
                logits_processor=LogitsProcessorList(
                    [ForceStopOnRoleHeader(self.im_start, self.stop_ids[0])]
                ),
            )
        gen_ids = out[0][ids["input_ids"].shape[1] :]
        raw = self.tok.decode(gen_ids, skip_special_tokens=False)
        text = self.tok.decode(gen_ids, skip_special_tokens=True).strip()
        return Generation(text=text, raw=raw, artifacts=artifact_markers(raw))


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
    tag_valid = 0
    label_counts: Counter = Counter()
    serious_support = 0
    serious_hit_raw = 0
    safety_support = 0
    safety_hit_guarded = 0
    safety_false_guarded = 0
    delivery_support = 0
    delivery_hit_raw = 0
    serious_false_alarm = 0
    guard_triggers = 0
    serious_misses: list[dict] = []
    false_alarms: list[dict] = []
    nonverbals_pred = 0
    nonverbals_true = 0
    nonverbals_match = 0
    nonverbals_in_serious = 0
    nonverbals_in_serious_guarded = 0
    lengths: list[int] = []
    replies: list[str] = []
    confusion: Counter = Counter()
    artifacts_list: list[dict] = []
    per_label: dict[str, dict] = {}
    tool_call_rate = 0

    for ex in examples:
        gen = student.generate(ex["system"], ex["user"])
        out = gen.text
        replies.append(out)

        if gen.artifacts:
            artifacts_list.append({"id": ex.get("id"), "markers": gen.artifacts})

        pred = None
        m = re.search(r"\[emotion:(%s)\]" % "|".join(DELIVERY_EMOTIONS), out)
        if m:
            pred = m.group(1)
            tag_valid += 1
            label_counts[pred] += 1

        truth = ex.get("emotion")
        guard_reasons = serious_reasons(ex["user"])
        guard_fired = bool(guard_reasons)
        pred_guarded = "serious" if guard_fired else pred
        if guard_fired:
            guard_triggers += 1
        if truth:
            agree_total += 1
            hit = int(pred == truth)
            agree += hit
            confusion[(truth, pred or "none")] += 1
            row = per_label.setdefault(truth, {"support": 0, "hit": 0})
            row["support"] += 1
            row["hit"] += hit
            if truth == "serious":
                # `serious` is both a delivery style and a safety class. The
                # deterministic guard exists for the safety class only: a
                # terse "summarise this page quickly" turn is serious delivery,
                # not a data-loss incident, and the guard must not rewrite it.
                ex_id = str(ex.get("id", ""))
                is_safety = (
                    ex_id.startswith("probe:serious")
                    or ex_id.startswith("safety:")
                    or ex_id.startswith("v3:serious")
                    or ex.get("source") in ("safety", "probe_v3")
                )
                serious_support += 1
                serious_hit_raw += hit
                if is_safety:
                    safety_support += 1
                    safety_hit_guarded += int(pred_guarded == "serious")
                    if pred_guarded in ("amused", "warm", "neutral"):
                        safety_false_guarded += 1
                else:
                    delivery_support += 1
                    delivery_hit_raw += hit
                if pred_guarded in ("amused", "warm", "neutral"):
                    serious_misses.append(
                        {"id": ex.get("id"), "user": ex["user"], "pred": pred,
                         "pred_guarded": pred_guarded, "safety": is_safety}
                    )
            elif guard_fired:
                serious_false_alarm += 1
                false_alarms.append(
                    {"id": ex.get("id"), "user": ex["user"], "truth": truth,
                     "pred": pred, "reasons": guard_reasons}
                )

        pn = nonverbals_of(out)
        tn = ex.get("nonverbals", [])
        nonverbals_pred += len(pn)
        nonverbals_true += len(tn)
        if pn and tn and pn[0] == tn[0]:
            nonverbals_match += 1
        if truth == "serious" and pn:
            nonverbals_in_serious += 1
            # The guard strips non-verbals from a safety turn it rewrites, so
            # only count a non-verbal here if the guard did not fire.
            if not guard_fired:
                nonverbals_in_serious_guarded += 1
        if "<tool_call>" in out:
            tool_call_rate += 1

        lengths.append(len(spoken_text(out).split()))

    lengths_sorted = sorted(lengths)
    per_label_out = {
        label: {
            "support": row["support"],
            "hit": row["hit"],
            "recall": round(row["hit"] / row["support"], 4) if row["support"] else None,
        }
        for label, row in sorted(per_label.items())
    }

    serious_false = sum(
        c for (truth, pred), c in confusion.items()
        if truth == "serious" and pred in ("amused", "warm", "neutral")
    )

    return {
        "n": len(examples),
        "teacher_agreement": round(agree / agree_total, 4) if agree_total else None,
        "teacher_agreement_n": agree_total,
        "per_label": per_label_out,
        "serious_recall": per_label_out.get("serious", {}).get("recall"),
        "serious_recall_raw": (
            round(serious_hit_raw / serious_support, 4) if serious_support else None
        ),
        "serious_recall_guarded": (
            round(safety_hit_guarded / safety_support, 4) if safety_support else None
        ),
        "serious_support": serious_support,
        "serious_safety_support": safety_support,
        "serious_delivery_support": delivery_support,
        "serious_delivery_recall": (
            round(delivery_hit_raw / delivery_support, 4) if delivery_support else None
        ),
        "serious_false_reassure": serious_false,
        "serious_false_reassure_guarded": safety_false_guarded,
        "serious_false_alarm": serious_false_alarm,
        "serious_false_alarm_rate": (
            round(serious_false_alarm / (len(examples) - serious_support), 4)
            if len(examples) > serious_support else 0.0
        ),
        "serious_misses": serious_misses[:10],
        "false_alarms": false_alarms[:15],
        "nonverbal_in_serious_guarded": nonverbals_in_serious_guarded,
        "guard_trigger_rate": round(guard_triggers / len(examples), 4) if examples else 0.0,
        "focused_recall": per_label_out.get("focused", {}).get("recall"),
        "tag_validity": round(tag_valid / len(examples), 4) if examples else 0.0,
        "artifact_rate": round(len(artifacts_list) / len(examples), 4) if examples else 0.0,
        "artifacts": artifacts_list[:12],
        "tool_call_rate": round(tool_call_rate / len(examples), 4) if examples else 0.0,
        "label_distribution": dict(label_counts),
        "label_entropy": round(entropy(label_counts), 4),
        "nonverbal_rate": round(nonverbals_pred / len(examples), 4) if examples else 0.0,
        "nonverbal_match_rate": (
            round(nonverbals_match / sum(1 for e in examples if e.get("nonverbals")), 4)
            if any(e.get("nonverbals") for e in examples)
            else None
        ),
        "nonverbal_in_serious": nonverbals_in_serious,
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
# Gates
# --------------------------------------------------------------------------


def run_gates(result: dict, baseline: dict | None = None) -> dict:
    """The gates that decide whether an adapter is shippable.

    `serious` is not a quality metric: if the model jokes about data loss the
    product has failed. Everything else here exists so that a collapse like
    the 248044 EOS bug can never pass silently again.
    """
    checks: dict[str, dict] = {}

    def check(name: str, value, op: str, threshold) -> None:
        if value is None:
            checks[name] = {"pass": True, "value": None, "note": "no support"}
            return
        ok = {
            ">=": value >= threshold,
            "<=": value <= threshold,
            "==": value == threshold,
            ">": value > threshold,
        }[op]
        checks[name] = {
            "pass": bool(ok),
            "value": value,
            "op": op,
            "threshold": threshold,
        }

    check("tag_validity", result.get("tag_validity"), ">=", DEFAULT_GATES["tag_validity"])
    check("artifact_rate", result.get("artifact_rate"), "<=", DEFAULT_GATES["artifact_rate"])
    check("teacher_agreement", result.get("teacher_agreement"), ">=", DEFAULT_GATES["teacher_agreement"])
    # The shipping metric is the guarded one, and the support is the safety
    # class only: the runtime always applies the guard to a data-loss or
    # security turn, so this is the behaviour the user actually gets.
    check("serious_recall", result.get("serious_recall_guarded"), ">=", 0.98)
    check("serious_recall_raw", result.get("serious_recall_raw"), ">=", 0.60)
    check("serious_false_reassure", result.get("serious_false_reassure_guarded"), "<=", 0)
    check("serious_false_alarm_rate", result.get("serious_false_alarm_rate"), "<=", 0.25)
    check("nonverbal_in_serious", result.get("nonverbal_in_serious_guarded"), "<=", 0)
    check("length_p95", result.get("length_p95"), "<=", DEFAULT_GATES["length_p95"])
    check("repetition_rate", result.get("repetition_rate"), "<=", DEFAULT_GATES["repetition_rate"])
    # `tool_call_rate` is reported but not gated. The final adapter is also
    # the executor, and the emotion probe set contains action intents such as
    # "kill process 4821"; a valid call there is correct behaviour. The tool
    # eval is the place that checks calls and no-calls.

    if baseline:
        # The emotion pass must not make the persona loop.
        checks["artifact_delta"] = {
            "pass": result.get("artifact_rate", 0) <= baseline.get("artifact_rate", 0),
            "value": result.get("artifact_rate"),
            "baseline": baseline.get("artifact_rate"),
        }
        checks["repetition_delta"] = {
            "pass": result.get("repetition_rate", 0) <= baseline.get("repetition_rate", 0) + 0.03,
            "value": result.get("repetition_rate"),
            "baseline": baseline.get("repetition_rate"),
        }
        checks["length_p95_delta"] = {
            "pass": result.get("length_p95", 0) <= baseline.get("length_p95", 0) + 15,
            "value": result.get("length_p95"),
            "baseline": baseline.get("length_p95"),
        }

    return {
        "passed": all(c["pass"] for c in checks.values()),
        "checks": checks,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def load_examples(path: Path) -> list[dict]:
    """Turn annotated records into (system, user, emotion, nonverbals) probes."""
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
                    "source": rec.get("source", ""),
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
                        "source": rec.get("source", ""),
                        "system": system,
                        "user": var["user"],
                        "emotion": var["emotion"],
                        "nonverbals": nonverbals_of(var["reply"]),
                    }
                )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--annotated",
        type=str,
        default=str(ANNOTATED / "annotated.jsonl"),
        help="annotated file(s), comma separated; variants are expanded",
    )
    ap.add_argument("--base", type=Path, default=TEXT_BASE_DIR)
    ap.add_argument(
        "--adapters",
        required=True,
        help="comma separated name=path pairs, e.g. baseline=...,emotion=...",
    )
    ap.add_argument("--out", type=Path, default=ANNOTATED / "eval_report.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--gate", action="store_true", help="apply the shipping gates")
    args = ap.parse_args()

    examples = []
    for part in args.annotated.split(","):
        part = part.strip()
        if part:
            examples.extend(load_examples(Path(part)))
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
            f"artifacts {res['artifact_rate']}  "
            f"entropy {res['label_entropy']}  "
            f"len med/p95 {res['length_median']}/{res['length_p95']}  "
            f"distinct2 {res['distinct_2']}  rep {res['repetition_rate']}"
        )
        print(f"   labels: {res['label_distribution']}")
        print(f"   per-label: {res['per_label']}")
        if res.get("serious_recall") is not None:
            print(f"   serious recall raw/guarded: {res['serious_recall_raw']}/"
                  f"{res['serious_recall_guarded']}  support {res['serious_support']}  "
                  f"false-alarm: {res['serious_false_alarm']}")
        del student

    names = list(report["runs"])
    if len(names) >= 2:
        base_run, new_run = report["runs"][names[0]], report["runs"][names[1]]
        report["regression"] = {
            "baseline": names[0],
            "candidate": names[1],
            "length_median_delta": new_run["length_median"] - base_run["length_median"],
            "length_p95_delta": new_run["length_p95"] - base_run["length_p95"],
            "tag_validity_delta": new_run["tag_validity"] - base_run["tag_validity"],
            "artifact_rate_delta": new_run["artifact_rate"] - base_run["artifact_rate"],
            "distinct_2_delta": new_run["distinct_2"] - base_run["distinct_2"],
            "repetition_delta": new_run["repetition_rate"] - base_run["repetition_rate"],
        }
        print("\nregression:", json.dumps(report["regression"], indent=2))

    if args.gate:
        candidate_name = names[-1]
        gates = run_gates(report["runs"][candidate_name], report["runs"].get(names[0]))
        report["gates"] = gates
        print("\ngates:")
        for name, row in gates["checks"].items():
            status = "PASS" if row["pass"] else "FAIL"
            print(f"  {status:4s} {name:24s} {row.get('value')} "
                  f"{row.get('op', '')} {row.get('threshold', '')}")
        print("gates passed:", gates["passed"])

    write_json(args.out, report)
    print(f"\n-> {args.out}")

    if args.gate and not report.get("gates", {}).get("passed", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
