"""Round-trip tests for the tag vocabulary.

Small, but they earn their place. The alternation in a tag regex has to be
parenthesised, or ``\\[emotion:%s\\]`` binds the prefix to the first
alternative only and silently stops matching the rest of the vocabulary. That
bug shipped once already and it is invisible until you validate a corpus and
watch every well-formed reply get rejected.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline" / "tts"))

from common import DELIVERY_EMOTIONS, NONVERBALS, emotion_of, nonverbals_of  # noqa: E402
from normalize_spoken import validate_reply  # noqa: E402
from segmenter import parse_tagged_reply, spoken_text, strip_all_tags  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        failures.append(name)


print("every delivery emotion round-trips")
for emotion in DELIVERY_EMOTIONS:
    text = f"[emotion:{emotion}]Hello there."
    check(
        f"emotion_of({emotion})",
        emotion_of(text) == emotion,
        f"-> {emotion_of(text)}",
    )
    issues = validate_reply(text)
    check(
        f"validate_reply({emotion}) has no tag issue",
        "missing_or_misplaced_delivery_tag" not in issues,
        f"-> {issues}",
    )

print("\nevery non-verbal round-trips")
for nv in NONVERBALS:
    text = f"[emotion:amused]Hi. [{nv}] There."
    check(
        f"nonverbals_of({nv})",
        nonverbals_of(text) == [nv],
        f"-> {nonverbals_of(text)}",
    )

print("\nsegments split correctly")
segs = parse_tagged_reply("[emotion:amused]One. [laugh]Two.")
check("3 segments", len(segs) == 3, f"-> {len(segs)}")
check(
    "segment kinds",
    [s.kind for s in segs] == ["speech", "nonverbal", "speech"],
    f"-> {[s.kind for s in segs]}",
)
check(
    "emotion carried forward",
    segs[2].emotion == "amused",
    f"-> {segs[2].emotion}",
)

print("\nout-of-vocabulary tags are never spoken")
check(
    "unknown emotion stripped from speech",
    "excited" not in spoken_text("[emotion:excited]Hello."),
    f"-> {spoken_text('[emotion:excited]Hello.')}",
)
check(
    "known tags stripped from speech",
    spoken_text("[emotion:amused]Hi. [laugh]There.") == "Hi. There.",
    f"-> {spoken_text('[emotion:amused]Hi. [laugh]There.')}",
)

print("\nvalidation catches real problems")
check(
    "too long",
    any(
        i.startswith("too_long")
        for i in validate_reply("[emotion:amused]" + "word " * 40)
    ),
)
check(
    "emoji",
    "emoji_present" in validate_reply("[emotion:amused]Nice 🎉"),
)
check(
    "inline code",
    "inline_code_present" in validate_reply("[emotion:amused]Run `df -h`."),
)
check(
    "nonverbal in a serious turn",
    "nonverbal_in_serious_turn" in validate_reply("[emotion:serious]Bad. [laugh]"),
)
check(
    "two delivery tags",
    "multiple_delivery_tags" in validate_reply("[emotion:amused]A [emotion:warm]B"),
)

print("\nstrip_all_tags leaves ordinary brackets alone")
check(
    "square brackets in prose survive",
    "[OS runtime] done" in strip_all_tags("[OS runtime] done"),
    f"-> {strip_all_tags('[OS runtime] done')}",
)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    raise SystemExit(1)
print("all tag tests passed")
