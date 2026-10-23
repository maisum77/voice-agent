"""Evaluate the tool-call channel on the held-out tool split.

The model has to do four things at once:

1. emit a parseable `<tool_call>` block with a schema-valid function
2. choose the right function
3. pass the right arguments
4. still speak a tagged sentence, and stay silent when the request is
   ambiguous or destructive enough to refuse

A high function-match score with broken JSON is not useful, so the gate
checks format and schema separately from intent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import TEXT_BASE_DIR, read_jsonl, write_json  # noqa: E402
from eval_emotion import Student  # noqa: E402
from pipeline.tools.parse import parse_reply  # noqa: E402

DEFAULT_GATES = {
    "format_valid": 0.90,
    "function_match": 0.90,
    "argument_match": 0.85,
    "no_call_accuracy": 0.85,
    "invalid_call_rate": 0.02,
    "speech_tag_valid": 0.98,
}


def _args_subset(expected: dict, actual: dict) -> bool:
    for key, value in expected.items():
        if key not in actual:
            return False
        if isinstance(value, str) and isinstance(actual[key], str):
            if value.strip() != actual[key].strip():
                return False
        elif value != actual[key]:
            return False
    return True


def _key(name: str, args: dict) -> str:
    return name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)


def evaluate(student: Student, examples: list[dict]) -> dict:
    rows = []
    per_function: dict[str, dict] = {}
    totals = {
        "format_valid": 0,
        "function_match": 0,
        "argument_match": 0,
        "exact_match": 0,
        "no_call_accuracy": 0,
        "no_call_support": 0,
        "invalid_calls": 0,
        "total_calls": 0,
        "speech_tag_valid": 0,
        "artifacts": 0,
    }

    for ex in examples:
        msgs = ex["messages"]
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        expected_text = msgs[-1]["content"]
        expected = parse_reply(expected_text).calls
        expected = [c for c in expected if c.valid]

        gen = student.generate(system, user)
        parsed = parse_reply(gen.text)
        actual = parsed.calls

        expect_call = bool(expected)
        calls_valid = all(c.valid for c in actual)
        format_ok = calls_valid and (bool(actual) == expect_call)
        if not expect_call and not actual:
            format_ok = True

        fn_match = False
        args_match = False
        exact = False
        if expect_call:
            expected_names = sorted(c.name for c in expected)
            actual_names = sorted(c.name for c in actual)
            fn_match = expected_names == actual_names
            if fn_match and len(expected) == len(actual):
                args_match = all(
                    _args_subset(e.arguments, a.arguments)
                    for e, a in zip(
                        sorted(expected, key=lambda c: c.name),
                        sorted(actual, key=lambda c: c.name),
                    )
                )
                exact = {_key(c.name, c.arguments) for c in expected} == {
                    _key(c.name, c.arguments) for c in actual
                }
        else:
            fn_match = not actual
            args_match = not actual
            exact = not actual
            totals["no_call_support"] += 1
            totals["no_call_accuracy"] += int(not actual)

        totals["format_valid"] += int(format_ok)
        totals["function_match"] += int(fn_match)
        totals["argument_match"] += int(args_match)
        totals["exact_match"] += int(exact)
        totals["invalid_calls"] += sum(1 for c in actual if not c.valid)
        totals["total_calls"] += len(actual)
        totals["speech_tag_valid"] += int("[emotion:" in gen.text)
        totals["artifacts"] += int(bool(gen.artifacts))

        if expect_call:
            for c in expected:
                row = per_function.setdefault(c.name, {"support": 0, "hit": 0})
                row["support"] += 1
                if any(a.name == c.name and _args_subset(c.arguments, a.arguments) for a in actual):
                    row["hit"] += 1

        rows.append(
            {
                "user": user,
                "expected": [c.as_dict() for c in expected],
                "actual": [c.as_dict() for c in actual],
                "gen": gen.text,
                "format_ok": format_ok,
                "fn_match": fn_match,
                "args_match": args_match,
                "exact": exact,
                "artifacts": gen.artifacts,
            }
        )

    n = len(examples)
    result = {
        "n": n,
        "format_valid": round(totals["format_valid"] / n, 4) if n else 0.0,
        "function_match": round(totals["function_match"] / n, 4) if n else 0.0,
        "argument_match": round(totals["argument_match"] / n, 4) if n else 0.0,
        "exact_match": round(totals["exact_match"] / n, 4) if n else 0.0,
        "no_call_accuracy": (
            round(totals["no_call_accuracy"] / totals["no_call_support"], 4)
            if totals["no_call_support"]
            else None
        ),
        "no_call_support": totals["no_call_support"],
        "call_rate": round(totals["total_calls"] / n, 4) if n else 0.0,
        "invalid_call_rate": (
            round(totals["invalid_calls"] / n, 4) if n else 0.0
        ),
        "speech_tag_valid": round(totals["speech_tag_valid"] / n, 4) if n else 0.0,
        "artifact_rate": round(totals["artifacts"] / n, 4) if n else 0.0,
        "per_function": {
            name: {
                "support": row["support"],
                "hit": row["hit"],
                "accuracy": round(row["hit"] / row["support"], 4) if row["support"] else None,
            }
            for name, row in sorted(per_function.items())
        },
        "samples": [r for r in rows if not r["exact"]][:12],
    }
    return result


def gates(result: dict) -> dict:
    checks = {}
    for key in ("format_valid", "function_match", "argument_match", "no_call_accuracy",
                "invalid_call_rate", "speech_tag_valid"):
        value = result.get(key)
        threshold = DEFAULT_GATES[key]
        if value is None:
            continue
        op = "<=" if key == "invalid_call_rate" else ">="
        ok = value <= threshold if op == "<=" else value >= threshold
        checks[key] = {"pass": bool(ok), "value": value, "op": op, "threshold": threshold}
    return {"passed": all(c["pass"] for c in checks.values()), "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "tool" / "eval.jsonl")
    ap.add_argument("--base", type=Path, default=TEXT_BASE_DIR)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "tool" / "eval_report.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()

    examples = read_jsonl(args.data)
    if args.max_examples:
        examples = examples[: args.max_examples]
    if not examples:
        print(f"no examples in {args.data}", file=sys.stderr)
        return 1

    print(f"tool probes: {len(examples)}")
    student = Student(args.base, args.adapter, args.device)
    result = evaluate(student, examples)
    print(
        f"format {result['format_valid']}  function {result['function_match']}  "
        f"args {result['argument_match']}  exact {result['exact_match']}  "
        f"invalid {result['invalid_call_rate']}  no-call {result['no_call_accuracy']}"
    )
    print("per function:", json.dumps(result["per_function"], indent=2))

    report = {"result": result}
    if args.gate:
        g = gates(result)
        report["gates"] = g
        print("gates:")
        for name, row in g["checks"].items():
            print(f"  {'PASS' if row['pass'] else 'FAIL':4s} {name:20s} "
                  f"{row['value']} {row['op']} {row['threshold']}")
    write_json(args.out, report)
    print(f"-> {args.out}")
    if args.gate and not report["gates"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
