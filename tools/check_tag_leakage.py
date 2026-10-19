"""Gate: is the annotated data going to teach emotion, or just teach a lookup?

The failure mode this exists to catch
--------------------------------------
If every prompt has exactly one reply, then *emotion is a function of what was
asked*. A model trained on that learns a lookup table. It will look like it
learned emotion, but it has learned a classifier, and it will fall over the
moment the user phrases the same request differently.

The fix is to make the same intent arrive in several moods, so emotion becomes
a function of *how the user was speaking*. This tool measures whether that
actually happened, before we spend a training run on it.

Checks
------
G1  topic-only predictability. THE GATE.

    A bag-of-words classifier is trained on the *base* prompts (one reply per
    intent, the thing that is always topic-determined) and then asked to
    predict the labels of the *affect variants* - same topic, different
    wording, different emotional colouring.

    If it can, then the topic alone determines the label, and a model trained
    on this data will learn topic -> emotion and ignore how the user actually
    sounded. That is the "learning the pattern" failure.

    If it cannot, the label genuinely depends on the mood in the user's
    wording, and the model has to read it. That is the "learning" we want.

    An earlier version of this check measured prompt -> emotion accuracy
    directly and was wrong: a high score there just means the mood is
    *readable*, which is the goal, not the bug. The score is still reported
    below as G1_info, because it is worth knowing.

G2  response diversity. Mean pairwise Jaccard over reply 4-grams. High overlap
    means the teacher emitted one canned reply and the model will memorise it.

G3  label balance. A label above 50% of the set means the model will learn to
    emit that label and little else.

G4  within-intent variety. For each base intent that has affect variants, how
    many distinct emotions did the variants produce? If every variant got the
    same emotion, the augmentation did not work.

G5  affect naming. If the teacher's rewritten user turn says "I am frustrated"
    rather than "third attempt today, this is doing my head in", the feeling
    is being stated rather than expressed, and the model learns to
    keyword-match.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ANNOTATED,
    DELIVERY_EMOTIONS,
    read_jsonl,
    strip_tags,
    write_json,
)

# Per-label expectations. Values are the maximum cross-validated accuracy we
# accept for that label before we call the annotation broken.
PREDICTABILITY_LIMITS = {
    "serious": 0.85,  # allowed to be high - it is a safety rule
    "amused": 0.45,
    "warm": 0.45,
    "focused": 0.45,
    "neutral": 0.45,
}

JACCARD_LIMIT = 0.45
LABEL_SHARE_LIMIT = 0.50

# Words that name a feeling outright. If the teacher's rewritten user turn
# contains one of these, the "affect" is being stated rather than expressed,
# and the model can learn to keyword-match instead of reading the mood.
AFFECT_WORDS = (
    "frustrated", "frustrating", "annoyed", "annoying", "upset", "angry",
    "delighted", "excited", "happy", "sad", "worried", "anxious", "confused",
    "urgent", "panicking", "stressed", "annoyance", "delight", "frustration",
    "confusion", "excitement", "urgency", "impatient", "cheerful", "grateful",
)


def ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", strip_tags(text).lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------
# G1 - topic-only predictability
# --------------------------------------------------------------------------


def _make_clf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=1.0),
    )


def _cv_accuracy(pairs: list[tuple[str, str]], seed: int = 0):
    """Cross-validated accuracy of text -> label."""
    from sklearn.model_selection import StratifiedKFold

    if len(pairs) < 40:
        return None
    X = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    counts = Counter(y)
    usable = [lab for lab, c in counts.items() if c >= 8]
    if len(usable) < 2:
        return None
    keep = [i for i, lab in enumerate(y) if lab in usable]
    X = [X[i] for i in keep]
    y = [y[i] for i in keep]

    n_splits = min(5, min(Counter(y).values()))
    if n_splits < 2:
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = [None] * len(y)
    for train_idx, test_idx in skf.split(X, y):
        clf = _make_clf()
        clf.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        for i, p in zip(test_idx, clf.predict([X[j] for j in test_idx])):
            preds[i] = p

    overall = sum(1 for p, t in zip(preds, y) if p == t) / len(y)
    per_label = {}
    for lab in usable:
        idx = [i for i, t in enumerate(y) if t == lab]
        per_label[lab] = {
            "recall": round(sum(1 for i in idx if preds[i] == lab) / len(idx), 4),
            "support": len(idx),
        }
    return {
        "n": len(y),
        "accuracy": round(overall, 4),
        "chance": round(1 / len(usable), 4),
        "per_label": per_label,
    }


def topic_only_predictability(base_pairs, variant_pairs, seed: int = 0):
    """G1. Can a topic-only model predict an affect variant's label?

    The classifier is trained on the base prompts - where the label is by
    construction a property of the topic - and then asked about the variants,
    which share the topic but were written with a different mood.
    """
    from sklearn.model_selection import StratifiedKFold

    if len(base_pairs) < 40 or len(variant_pairs) < 20:
        return None

    X = [p[0] for p in base_pairs]
    y = [p[1] for p in base_pairs]
    counts = Counter(y)
    usable = [lab for lab, c in counts.items() if c >= 8]
    if len(usable) < 2:
        return None

    keep = [i for i, lab in enumerate(y) if lab in usable]
    X = [X[i] for i in keep]
    y = [y[i] for i in keep]

    n_splits = min(5, min(Counter(y).values()))
    if n_splits < 2:
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = [None] * len(variant_pairs)
    for train_idx, _ in skf.split(X, y):
        clf = _make_clf()
        clf.fit([X[i] for i in train_idx], [y[i] for i in train_idx])
        got = clf.predict([p[0] for p in variant_pairs])
        for i, p in enumerate(got):
            if preds[i] is None:
                preds[i] = p

    truth = [p[1] for p in variant_pairs]
    acc = sum(1 for p, t in zip(preds, truth) if p == t) / len(truth)

    per_label = {}
    for lab in sorted(set(truth)):
        idx = [i for i, t in enumerate(truth) if t == lab]
        per_label[lab] = {
            "recall": round(sum(1 for i in idx if preds[i] == lab) / len(idx), 4),
            "support": len(idx),
        }

    return {
        "n_variants": len(variant_pairs),
        "accuracy": round(acc, 4),
        "chance": round(1 / len(usable), 4),
        "labels": sorted(set(truth)),
        "per_label": per_label,
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def collect(records: list[dict]):
    """Flatten the annotated file.

    Returns (base_pairs, variant_pairs, replies, variety). Base pairs are
    (user, emotion) for the original prompt; variant pairs are the same for
    the affect variants, which is what G1 is measured on.
    """
    base_pairs: list[tuple[str, str]] = []
    variant_pairs: list[tuple[str, str]] = []
    replies: list[str] = []
    groups: dict[str, list[str]] = defaultdict(list)
    affect_named = 0
    variant_turns = 0

    for rec in records:
        if rec.get("base_reply") and rec.get("base_emotion"):
            base_pairs.append((rec["base_user"], rec["base_emotion"]))
            replies.append(rec["base_reply"])
            groups[rec["id"]].append(rec["base_emotion"])
        for var in rec.get("variants", []):
            if var.get("reply") and var.get("emotion"):
                variant_pairs.append((var["user"], var["emotion"]))
                replies.append(var["reply"])
                groups[rec["id"]].append(var["emotion"])
                # Only the augmented turns matter here: Mode B exists to make
                # the *user's* wording carry the feeling.
                variant_turns += 1
                if _names_affect(var["user"]):
                    affect_named += 1

    variety = {
        "groups": len(groups),
        "mean_distinct_emotions_per_group": (
            round(sum(len(set(v)) for v in groups.values()) / len(groups), 3)
            if groups
            else 0
        ),
        "groups_with_one_emotion": sum(1 for v in groups.values() if len(set(v)) == 1),
        "distribution_of_group_sizes": dict(
            Counter(len(set(v)) for v in groups.values())
        ),
        "variant_turns": variant_turns,
        "variant_turns_naming_the_affect": affect_named,
        "affect_named_rate": (
            round(affect_named / variant_turns, 4) if variant_turns else 0.0
        ),
    }
    return base_pairs, variant_pairs, replies, variety


def _names_affect(text: str) -> bool:
    words = set(re.findall(r"[a-z']+", text.lower()))
    return bool(words & set(AFFECT_WORDS))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=ANNOTATED / "annotated.jsonl")
    ap.add_argument("--out", type=Path, default=ANNOTATED / "leakage_report.json")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any failure")
    args = ap.parse_args()

    records = read_jsonl(args.inp)
    if not records:
        print(f"no annotated records at {args.inp}", file=sys.stderr)
        return 1

    base_pairs, variant_pairs, replies, variety = collect(records)
    topic_only = topic_only_predictability(base_pairs, variant_pairs)
    full_prompt = _cv_accuracy(base_pairs + variant_pairs)

    # G2 - response diversity
    grams = [ngrams(r) for r in replies]
    jaccards = []
    for i, j in itertools.combinations(range(len(grams)), 2):
        jaccards.append(jaccard(grams[i], grams[j]))
    mean_jac = sum(jaccards) / len(jaccards) if jaccards else 0.0

    # G3 - label balance
    label_counts = Counter(lab for _, lab in base_pairs + variant_pairs)
    total = sum(label_counts.values())
    shares = {k: round(v / total, 4) for k, v in label_counts.items()}

    failures: list[str] = []

    if topic_only:
        for lab, lim in PREDICTABILITY_LIMITS.items():
            info = topic_only["per_label"].get(lab)
            if not info:
                continue
            if info["recall"] > lim:
                failures.append(
                    f"G1 {lab}: a topic-only model predicts this label "
                    f"{info['recall']:.1%} of the time (limit {lim:.0%}). The "
                    f"emotion follows the topic, not the user's mood - the model "
                    f"would learn a lookup table"
                )
        if topic_only["accuracy"] > topic_only["chance"] + 0.20:
            failures.append(
                f"G1: topic-only accuracy {topic_only['accuracy']:.3f} is well "
                f"above chance ({topic_only['chance']:.3f}) - the affect variants "
                f"did not change the emotion, so the mood is not being read"
            )
    else:
        failures.append(
            "G1: not enough data to run the topic-only test "
            "(need >= 40 base examples and >= 20 variants)"
        )

    if mean_jac > JACCARD_LIMIT:
        failures.append(
            f"G2: mean pairwise 4-gram Jaccard {mean_jac:.3f} exceeds "
            f"{JACCARD_LIMIT} - the teacher is emitting near-identical replies"
        )

    for lab, share in shares.items():
        if share > LABEL_SHARE_LIMIT:
            failures.append(
                f"G3 {lab}: {share:.1%} of the dataset - the model will learn to "
                f"emit that label and little else"
            )

    if variety["groups"] and variety["mean_distinct_emotions_per_group"] < 1.5:
        failures.append(
            f"G4: only {variety['mean_distinct_emotions_per_group']} distinct "
            f"emotions per intent on average - the affect augmentation did not "
            f"produce variety, so emotion is still a function of the prompt"
        )

    if variety["variant_turns"] and variety["affect_named_rate"] > 0.15:
        failures.append(
            f"G5: {variety['affect_named_rate']:.1%} of augmented user turns name "
            f"the feeling outright (frustrated, delighted, ...). The model would "
            f"learn to keyword-match rather than read the mood - ask the teacher "
            f"to express the feeling in the wording instead"
        )

    report = {
        "records": len(records),
        "base_examples": len(base_pairs),
        "variant_examples": len(variant_pairs),
        "G1_topic_only_predictability": topic_only,
        "G1_info_full_prompt_predictability": full_prompt,
        "G2_mean_pairwise_jaccard": round(mean_jac, 4),
        "G3_label_distribution": {
            "counts": dict(label_counts),
            "shares": shares,
        },
        "G4_within_intent_variety": variety,
        "failures": failures,
        "passed": not failures,
    }
    write_json(args.out, report)

    print(f"base examples: {len(base_pairs)}  variants: {len(variant_pairs)}")
    if topic_only:
        print(
            f"G1 topic-only accuracy on variants: {topic_only['accuracy']:.3f} "
            f"(chance {topic_only['chance']:.3f})"
        )
        for lab, info in sorted(topic_only["per_label"].items()):
            lim = PREDICTABILITY_LIMITS.get(lab, 0.45)
            flag = "OK " if info["recall"] <= lim else "FAIL"
            print(
                f"   {flag} {lab:8s} recall={info['recall']:.3f} "
                f"(limit {lim:.2f}, n={info['support']})"
            )
    if full_prompt:
        print(
            f"G1_info full-prompt accuracy: {full_prompt['accuracy']:.3f} "
            f"(chance {full_prompt['chance']:.3f}) - how readable the mood is"
        )
    print(f"G2 mean pairwise jaccard: {mean_jac:.4f} (limit {JACCARD_LIMIT})")
    print(f"G3 label shares: {shares}")
    print(
        f"G4 distinct emotions per intent: "
        f"{variety['mean_distinct_emotions_per_group']} (sizes {variety['distribution_of_group_sizes']})"
    )
    print(
        f"G5 augmented turns naming the affect: {variety['affect_named_rate']:.1%} "
        f"({variety['variant_turns_naming_the_affect']}/{variety['variant_turns']})"
    )
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nall gates passed")
    print(f"-> {args.out}")

    return 1 if (failures and args.strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
