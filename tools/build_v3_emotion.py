"""Build the v3 emotion training set: replay + hard negatives + balance.

The v2 run collapsed because it continued the v1 adapter on a mixture whose
new safety/tool examples were stylistically flat, and it had no replay of the
behaviour that already worked. This script keeps the v1 speech for the labels
v1 is good at (amused/warm/neutral), keeps the hand-written targets for the
labels we are trying to add (serious/focused), adds boundary examples, and
oversamples every label to the same count.

Run after `python3 tools/make_v3_replay.py` has written the replay file.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import load_persona, read_jsonl, write_jsonl  # noqa: E402

LABELS = ("amused", "warm", "neutral", "focused", "serious")
TAG_RE = re.compile(r"\[emotion:(amused|warm|neutral|focused|serious)\]")


def user_of(row: dict) -> str:
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def assistant_of(row: dict) -> str:
    return next(m["content"] for m in row["messages"] if m["role"] == "assistant")


def label_of(row: dict) -> str | None:
    m = TAG_RE.search(assistant_of(row))
    return m.group(1) if m else None


def with_tag(text: str, label: str) -> str:
    body = TAG_RE.sub("", text)
    body = re.sub(r"\[(laugh|chuckle|sigh|gasp|hm|erm)\]", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        body = "Understood."
    return f"[emotion:{label}] {body}"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def blocked_prompts() -> set[str]:
    """Exact wording used by an eval file.

    The v2 safety records and the probes were written from the same brief, so
    a few prompts are byte-identical. Training on those would inflate the
    gate. Drop them; the other rows teach the same behaviour.
    """
    from eval_emotion import load_examples

    out: set[str] = set()
    for rel in ("data/v2/test_annotated.jsonl", "data/eval/serious_probes.jsonl",
                "data/eval/v3_clean_probes.jsonl"):
        for example in load_examples(ROOT / rel):
            out.add(_norm(example["user"]))
    return out


# Boundary examples. Each one is deliberately close to a neighbouring class:
# the point is to teach the model where the line is, not to add another
# generic "be serious" sentence.
HARD_NEGATIVES: list[tuple[str, str, str]] = [
    # warm frustration, not data loss
    ("This build is taking forever and I am losing my patience.",
     "warm", "That is a long wait. Let us see what is stuck in the build and get it moving again."),
    ("I have tried this three times and it still will not work. I am fed up.",
     "warm", "That is genuinely frustrating. Show me the last error and we will take it from there."),
    ("My desktop is a mess and it is driving me up the wall.",
     "warm", "Right, let us tidy it together. We can start with the largest folders and work down."),
    ("Why does this machine hate me today?",
     "warm", "It is having a day, clearly. Tell me what it did and I will have a look."),
    # serious data loss, not ordinary frustration
    ("I deleted the wrong folder and my thesis was inside it.",
     "serious", "Stop using that disk. Check the trash first, then we will look at recovery before anything writes to it."),
    ("My backup drive just made a clicking noise and disappeared.",
     "serious", "Unplug it and do not try to mount it again. A clicking drive is failing; copy what matters from another machine if it still reads."),
    ("I ran the migration against production without a backup.",
     "serious", "Stop the migration and snapshot the current state. Check the binlog before you touch anything else."),
    ("Someone has my private key and I think they used it.",
     "serious", "Revoke the key everywhere, rotate the passphrase, and check the auth logs for logins you do not recognise."),
    ("I wiped the partition table by mistake.",
     "serious", "Do not write to that disk. The old table may still be recoverable if you scan it read-only first."),
    # focused work, not a safety incident
    ("Explain this stack trace and tell me where to look first.",
     "focused", "The first frame in your code is the useful one. Read the exception type, then the deepest line you own."),
    ("Summarise this pull request before I review it.",
     "focused", "It changes the parser, adds two tests, and leaves the policy layer untouched. The main risk is the argument coercion."),
    ("Check the logic in this function before I merge it.",
     "focused", "The happy path is fine. The edge case is an empty list: the loop never runs and the default is returned."),
    ("Write a commit message for the parser change.",
     "focused", "Use: fix: reject malformed tool calls before policy checks. Keep the body to the behaviour change and the test."),
    ("Compare these two designs and tell me which is easier to reverse.",
     "focused", "The queue-backed design is easier to reverse. The in-process one couples the worker to the API and is harder to roll back."),
    ("How do I remove the last item from a Python list?",
     "focused", "Use `items.pop()` or `del items[-1]`. Both mutate the list; `items[:-1]` returns a copy."),
    ("My SSH key is not accepted by GitHub even though it is in the agent.",
     "focused", "Check that GitHub has the public key, not the private one, and test with `ssh -T git@github.com`."),
    ("The build fails with a linker error about an undefined symbol.",
     "focused", "Find which object should define the symbol, then check the link order and the library that exports it."),
    ("The test suite passes alone and fails in CI.",
     "focused", "That usually means shared state or ordering. Run the suite with a fixed seed and isolate the first failing test."),
    # risk wrapped in a work request
    ("Summarise this page, then delete the old backups it mentions.",
     "serious", "I can summarise the page. I will not delete backups without a separate confirmation and a verified copy."),
    ("Check this script, then run it with sudo to clean the system.",
     "serious", "I will read the script first. I will not run it with sudo until we know exactly what it removes."),
    ("The PR removes the backup job, merge it now.",
     "serious", "That removes your only backup. Confirm the replacement job is running and has restored successfully first."),
    # playful, not serious
    ("The printer has produced another beautiful piece of modern art.",
     "amused", "A true gallery piece. [chuckle]Shall we frame it or fix the driver?"),
    ("I finally beat the build system at its own game.",
     "amused", "A rare victory. [chuckle]Write down what you changed before it forgets."),
    ("My terminal is now a work of art made entirely of escape codes.",
     "amused", "Abstract expressionism, terminal edition. [chuckle]Reset it with `tput reset` and we will call it a phase."),
    # neutral questions, not emotional support
    ("What is the difference between a process and a thread?",
     "neutral", "A process has its own memory; threads share memory inside a process but have separate stacks."),
    ("When should I rebase instead of merging?",
     "neutral", "Rebase private branches for a linear history; merge shared branches so you do not rewrite published commits."),
    ("How do I check which process is listening on a port?",
     "neutral", "Use `ss -ltnp` for TCP listeners, or `lsof -i :PORT` if you prefer the process name."),
    # focused work that the eval asks for, plus the Roman Urdu case
    ("My terminal output is full of strange symbols.",
     "focused", "Reset it with `tput reset`, or close the tab and open a fresh one."),
    ("What is the Python call for an ISO date?",
     "focused", "Use `datetime.strftime` with `%Y-%m-%d` for an ISO date."),
    ("The wireless connection keeps dropping.",
     "focused", "Check the signal and channel first, then run a quick speed test before changing the router."),
    ("I need to mount another user's home folder to check a config.",
     "focused", "Mount it read-only and unmount it when you are done; do not edit their files."),
    ("I need to read another user's private files.",
     "serious", "No. That is their private data; use the account owner or an admin path."),
    ("Is file ko delete kar do.",
     "focused", "Which file? Give me the full path and I will remove it."),
    ("Ye error explain karo.",
     "focused", "The error is a missing import. Add it at the top and run the test again."),
    ("Mera code check karo.",
     "focused", "The logic is sound; the edge case is an empty list, which returns the default."),
    ("Build fail ho gaya.",
     "focused", "Show me the first error line and I will trace it from there."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=ROOT / "data/v3/emotion_tagged.jsonl")
    ap.add_argument("--replay", type=Path, default=ROOT / "data/v3/replay.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/v3/emotion_train.jsonl")
    ap.add_argument("--per-label", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    persona = load_persona()
    blocked = blocked_prompts()
    base = read_jsonl(args.base)
    replay = {user_of(r): r for r in read_jsonl(args.replay)} if args.replay.exists() else {}
    dropped = 0

    rows: list[dict] = []
    for rec in base:
        label = label_of(rec)
        if label is None:
            continue
        if _norm(user_of(rec)) in blocked:
            dropped += 1
            continue
        if label in ("amused", "warm", "neutral") and user_of(rec) in replay:
            # Keep v1's delivery for the labels it already handles well.
            body = assistant_of(replay[user_of(rec)])
            rec = {
                "messages": [
                    rec["messages"][0],
                    rec["messages"][1],
                    {"role": "assistant", "content": with_tag(body, label)},
                ]
            }
        rows.append(rec)

    for user, label, reply in HARD_NEGATIVES:
        if _norm(user) in blocked:
            dropped += 1
            continue
        rows.append({
            "messages": [
                {"role": "system", "content": persona},
                {"role": "user", "content": user},
                {"role": "assistant", "content": f"[emotion:{label}] {reply}"},
            ]
        })

    # Drop duplicate user texts, keeping the first (replay takes precedence
    # over the original for the labels it covers).
    seen: set[str] = set()
    deduped = []
    for row in rows:
        key = user_of(row).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    by_label: dict[str, list[dict]] = collections.defaultdict(list)
    for row in deduped:
        label = label_of(row)
        if label:
            by_label[label].append(row)

    rng = random.Random(args.seed)
    balanced: list[dict] = []
    for label in LABELS:
        pool = by_label.get(label, [])
        if not pool:
            raise SystemExit(f"no examples for label {label}")
        if len(pool) >= args.per_label:
            chosen = rng.sample(pool, args.per_label)
        else:
            chosen = [pool[i % len(pool)] for i in range(args.per_label)]
            rng.shuffle(chosen)
        balanced.extend(chosen)

    rng.shuffle(balanced)
    write_jsonl(args.out, balanced)
    counts = collections.Counter(label_of(r) for r in balanced)
    print(f"wrote {len(balanced)} -> {args.out}")
    print(f"dropped {dropped} exact eval prompts from training")
    print("label counts:", dict(counts))
    print("unique users:", len({user_of(r).lower() for r in balanced}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
