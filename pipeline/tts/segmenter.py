"""Turn a tagged reply into the segments the TTS engine synthesises.

A tagged reply looks like this::

    [emotion:amused]Ah, the classic 'where did my space go' question - love it. [laugh]Run df -h.

and parses into::

    speech    emotion=amused   "Ah, the classic 'where did my space go' question - love it."
    nonverbal                laugh
    speech    emotion=amused   "Run df -h."

Why the two families are separate
--------------------------------
The delivery tag is a property of the whole utterance: it selects which
reference clip the voice is cloned from, so every speech segment in a reply
shares it (unless a later delivery tag overrides it).

A non-verbal is an *event at a position*. It is not something the voice can
be cloned into - it is a short clip spliced between two synthesised segments.
That is why it needs its own kind, and why the segment order matters: a laugh
placed after "love it." must land there, not at the end of the reply.

No dependencies: this module is imported by the runtime pipeline, so it stays
importable without torch, mlx or anything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DELIVERY_EMOTIONS = ("neutral", "amused", "warm", "focused", "serious")
NONVERBALS = ("laugh", "chuckle", "sigh", "gasp", "hm", "erm")

DEFAULT_EMOTION = "neutral"

TAG_RE = re.compile(
    r"\[emotion:(?P<emotion>%s)\]" % "|".join(DELIVERY_EMOTIONS)
    + r"|\[(?P<nonverbal>%s)\]" % "|".join(NONVERBALS)
)


@dataclass
class Segment:
    """One piece of the reply, ready to synthesise or splice."""

    kind: str  # "speech" | "nonverbal"
    text: str = ""
    emotion: str = DEFAULT_EMOTION
    nonverbal: str = ""
    start: int = 0  # character offset in the source text
    end: int = 0

    @property
    def is_speech(self) -> bool:
        return self.kind == "speech"


def parse_tagged_reply(text: str) -> list[Segment]:
    """Split a tagged reply into speech and non-verbal segments.

    The delivery tag that opens the reply applies to every speech segment
    until another delivery tag appears.
    """
    segments: list[Segment] = []
    emotion = DEFAULT_EMOTION
    pos = 0
    pending: list[str] = []

    def flush() -> None:
        if not pending:
            return
        body = "".join(pending).strip()
        pending.clear()
        if body:
            segments.append(
                Segment(
                    kind="speech",
                    text=body,
                    emotion=emotion,
                    start=pos,
                    end=pos,
                )
            )

    for match in TAG_RE.finditer(text):
        chunk = text[pos : match.start()]
        if chunk.strip() or pending:
            pending.append(chunk)
        flush()
        pos = match.end()
        if match.group("emotion"):
            emotion = match.group("emotion")
        else:
            segments.append(
                Segment(
                    kind="nonverbal",
                    nonverbal=match.group("nonverbal"),
                    emotion=emotion,
                    start=match.start(),
                    end=match.end(),
                )
            )

    pending.append(text[pos:])
    flush()
    return segments


def strip_tags(text: str) -> str:
    """The plain spoken words, for the text channel of the UI."""
    return TAG_RE.sub("", text).strip()


# A tag with an out-of-vocabulary value (``[emotion:excited]``) is still a tag
# and must never be read aloud, but it is not in the vocabulary we know how to
# render. Strip it for speech while leaving the text channel untouched.
LOOSE_TAG_RE = re.compile(
    r"\[emotion:[^\]]*\]"
    + r"|\[(?:%s)\]" % "|".join(NONVERBALS)
)


def strip_all_tags(text: str) -> str:
    """Remove every tag, including ones with an unknown emotion value."""
    return LOOSE_TAG_RE.sub("", text).strip()


def emotion_of(text: str) -> str:
    """The delivery emotion of a reply, or the default."""
    match = re.search(r"\[emotion:(%s)\]" % "|".join(DELIVERY_EMOTIONS), text)
    return match.group(1) if match else DEFAULT_EMOTION


def nonverbals_of(text: str) -> list[str]:
    return [m.group("nonverbal") for m in TAG_RE.finditer(text) if m.group("nonverbal")]


def spoken_text(text: str) -> str:
    """The text as it will be spoken: tags removed, whitespace tidied."""
    out = re.sub(r"[ \t]+", " ", strip_all_tags(text))
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return re.sub(r" +([,.;:!?])", r"\1", out).strip()


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    sample = (
        "[emotion:amused]Ah, the classic 'where did my space go' question - love it. "
        "[laugh]Run df -h."
    )
    for seg in parse_tagged_reply(sample):
        if seg.is_speech:
            print(f"speech    {seg.emotion:8s} {seg.text!r}")
        else:
            print(f"nonverbal {seg.nonverbal}")
    print("spoken:", spoken_text(sample))
