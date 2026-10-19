"""Normalise replies into speakable, tag-friendly plain text.

Two jobs:

``input``   Clean a *source* reply before the teacher sees it. Emoji, code
            fences and markdown are stripped so the teacher is not wasting
            tokens on noise, and so the final output is clean even if the
            teacher is sloppy.

``output``  Validate a *teacher-produced* tagged reply. Returns a list of
            issues rather than silently editing, because a hard failure in the
            validator should surface as a rejected sample, not a quietly
            mangled training target.

Deliberately NOT done here: shortening. Cutting a reply at 30 words would
land mid-sentence. Compression is the teacher's job; this module only flags
over-length replies.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DATA,
    DELIVERY_EMOTIONS,
    MAX_SPOKEN_WORDS,
    NONVERBALS,
    read_jsonl,
    strip_tags,
    word_count,
    write_jsonl,
)

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

# Emoji: pictographs, emoticons, flags, dingbats, plus variation selectors
# and zero-width joiner so we do not leave half a glyph behind.
EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"  # emoticons & pictographs
    "\U0001f1e6-\U0001f1ff"  # flags
    "☀-➿"  # dingbats, misc symbols
    "←-⇿"  # arrows
    "⌀-⏿"  # misc technical
    "①-⓿"  # circled
    "■-◿"  # geometric
    "️"  # variation selector
    "‍"  # zero width joiner
    "]+",
    flags=re.UNICODE,
)

FENCE_RE = re.compile(r"```.*?```", re.S)
FENCE_OPEN_RE = re.compile(r"```.*$", re.S)

INLINE_CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1", re.S)
ITALIC_RE = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", re.S)
ITALIC_UNDER_RE = re.compile(r"(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])", re.S)
STRIKE_RE = re.compile(r"~~(.+?)~~", re.S)
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
BARE_URL_RE = re.compile(r"https?://\S+")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
QUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.M)
HR_RE = re.compile(r"^\s*([-*_])\s*\1\s*\1[\s\1]*$", re.M)
BULLET_RE = re.compile(r"^\s{0,6}(?:[-*+•]|\d{1,2}[.)])\s+", re.M)

# Characters the TTS reads badly.
SYMBOL_MAP = {
    "&": " and ",
    "%": " percent ",
    "+": " plus ",
    "=": " equals ",
    "@": " at ",
    "$": " dollars ",
    "€": " euros ",
    "£": " pounds ",
    "©": " copyright ",
    "®": "",
    "→": " to ",
    "←": " from ",
    "×": " times ",
    "÷": " divided by ",
    "~": " tilde ",
    "°": " degrees ",
}

# A fenced code block we replace wholesale before the teacher sees it.
CODE_PLACEHOLDER = " [shows a command on screen] "


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------


def strip_emoji(text: str) -> str:
    return EMOJI_RE.sub("", text)


def strip_code(text: str, placeholder: str | None = CODE_PLACEHOLDER) -> str:
    """Remove fenced blocks (and bare fences) from text.

    ``placeholder`` replaces a fenced block; pass ``None`` to delete it.
    """
    text = FENCE_RE.sub(placeholder or " ", text)
    text = FENCE_OPEN_RE.sub(placeholder or " ", text)
    return text


def strip_markdown(text: str) -> str:
    text = INLINE_CODE_RE.sub(r"\1", text)
    text = LINK_RE.sub(r"\1", text)
    text = BARE_URL_RE.sub("the link", text)
    text = BOLD_RE.sub(r"\2", text)
    text = ITALIC_RE.sub(r"\1", text)
    text = ITALIC_UNDER_RE.sub(r"\1", text)
    text = STRIKE_RE.sub(r"\1", text)
    text = HEADING_RE.sub("", text)
    text = QUOTE_RE.sub("", text)
    text = HR_RE.sub("", text)
    text = BULLET_RE.sub("", text)
    return text


def speakable_symbols(text: str) -> str:
    for src, dst in SYMBOL_MAP.items():
        text = text.replace(src, dst)
    return text


def tidy(text: str) -> str:
    """Collapse whitespace, fix spacing around punctuation, cap blank lines."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r" +([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
    text = re.sub(r" +([)\]])", r"\1", text)
    text = re.sub(r"([,.;:!?]){2,}", r"\1", text)
    return text.strip()


def normalize(text: str, keep_code_placeholder: bool = True) -> str:
    """Full input-side clean: emoji -> code -> markdown -> symbols -> tidy."""
    text = strip_emoji(text)
    text = strip_code(text, CODE_PLACEHOLDER if keep_code_placeholder else None)
    text = strip_markdown(text)
    text = speakable_symbols(text)
    return tidy(text)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate_reply(text: str) -> list[str]:
    """Issues with a teacher-produced tagged reply. Empty list == good."""
    issues: list[str] = []

    if not text.strip():
        issues.append("empty")
        return issues

    if not re.match(r"^\s*\[emotion:(%s)\]" % "|".join(DELIVERY_EMOTIONS), text):
        issues.append("missing_or_misplaced_delivery_tag")

    words = word_count(text)
    if words > MAX_SPOKEN_WORDS:
        issues.append(f"too_long:{words}")

    if EMOJI_RE.search(strip_tags(text)):
        issues.append("emoji_present")
    if "```" in text:
        issues.append("code_fence_present")
    if re.search(r"(?<![\w`])`", text):
        issues.append("inline_code_present")
    if re.search(r"(\*\*|__|~~)", text):
        issues.append("markdown_bold_present")

    nonverbals = re.findall(r"\[(%s)\]" % "|".join(NONVERBALS), text)
    if len(nonverbals) > 1:
        issues.append(f"too_many_nonverbals:{len(nonverbals)}")
    if "[emotion:serious]" in text and nonverbals:
        issues.append("nonverbal_in_serious_turn")

    if text.count("[emotion:") > 1:
        issues.append("multiple_delivery_tags")
    if text.rstrip().endswith("["):
        issues.append("truncated_tag")

    return issues


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=["input", "output"],
        default="input",
        help="input: clean source replies for the teacher. output: validate teacher replies.",
    )
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", dest="out", type=Path, required=True)
    ap.add_argument(
        "--drop-invalid",
        action="store_true",
        help="output mode: drop records that fail validation",
    )
    args = ap.parse_args()

    records = read_jsonl(args.inp)
    kept, dropped = [], []

    for rec in records:
        if args.mode == "input":
            rec = dict(rec)
            rec["assistant_raw"] = rec["assistant"]
            rec["user_raw"] = rec["user"]
            rec["assistant"] = normalize(rec["assistant"])
            rec["user"] = normalize(rec["user"])
            kept.append(rec)
        else:
            issues = validate_reply(rec.get("assistant", ""))
            rec = dict(rec)
            rec["issues"] = issues
            if issues and args.drop_invalid:
                dropped.append(rec)
            else:
                kept.append(rec)

    write_jsonl(args.out, kept)
    print(f"mode={args.mode} in={len(records)} kept={len(kept)} dropped={len(dropped)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
