"""Annotate the dataset with emotion tags using a strong teacher model.

The student is a 0.8B model. A 0.8B model annotating its own training data
would amplify its own biases and produce exactly the "learning the pattern"
failure we are trying to avoid, so the teacher must be substantially stronger.
Default: Qwen3.5-9B served by ``mlx_lm.server`` on localhost.

Two modes
---------
**A — re-register + tag.**  The source reply is rewritten into the canonical
persona's register, compressed to a speakable length, and given a delivery tag
and at most one non-verbal tag.

**B — affect augmentation.**  The *user's* turn is rewritten so it genuinely
carries a named feeling, and a matching tagged reply is written for it.

Mode B is the mechanism that stops the model learning a lookup table. If every
prompt has exactly one reply then emotion is a function of *what was asked* and
the model memorises it. When the same intent arrives in five moods, emotion
becomes a function of *how the user was speaking* — which is the thing a
voice companion actually has to read.

Self-consistency
----------------
The delivery label is sampled three times (one greedy, two at temperature 0.7).
A reply is only accepted when the majority agrees. Ambiguous items are dropped
and reported, because training on a fuzzy label teaches noise.

Caching
-------
Every teacher call is cached on disk keyed by the prompt hash, so re-running
the annotator costs nothing and an interrupted run resumes for free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    AFFECT_VARIANTS,
    ANNOTATED,
    DATA,
    DELIVERY_EMOTIONS,
    NONVERBALS,
    TEACHER_BASE_URL,
    TEACHER_MODEL,
    load_persona,
    read_jsonl,
    write_json,
    write_jsonl,
)

# --------------------------------------------------------------------------
# Teacher prompts
# --------------------------------------------------------------------------

VOICE_MODES = """[emotion:neutral]  factual and plain, no colour
[emotion:amused]   dry wit, playful, lightly teasing - the default
[emotion:warm]     kind and encouraging - the user is frustrated or unsure
[emotion:focused]  brisk and task-first - the user is in the middle of work
[emotion:serious]  no jokes at all - data loss, security, or real distress"""

NONVERBAL_RULES = """[laugh] [chuckle] [sigh] [gasp] [hm] [erm]
At most one, and only where a human would really do it. Never on an
[emotion:serious] reply. Never on every reply."""

SPOKEN_RULES = """- Start with exactly one voice mode tag.
- Maximum 30 words after the tag. Short.
- No emoji. No markdown. No code fences. No backticks. No bullet lists.
- If the source contains a command or code, keep only what is worth speaking.
- Plain spoken English. At most one short Roman Urdu phrase, only if it fits.
- Output only the reply. No explanation, no surrounding quotes."""

# Few-shot examples. These carry more signal than any amount of instruction:
# they show the exact tag placement, the word budget, the register shift, and
# the rule that a serious situation loses its jokes.
MODE_A_SHOTS = """
USER SAID: My build finally passed and all tests are green.
DRAFT REPLY: LET'S GOOO! 🎉 Green tests are the best kind of green. That's a proper win - go grab a chai, you've earned it! Shall I set up a little git tag to commemorate the moment?

REWRITE:
[emotion:amused]Ah, green across the board. That is a well-earned afternoon off. [laugh]Claim it.

USER SAID: I think someone might have gained unauthorized access to my machine.
DRAFT REPLY: Oh no, this is really scary! Don't panic, we need to act fast. First thing, let's change all your passwords right now, and also check your bank account.

REWRITE:
[emotion:serious]Okay, this is serious. Change nothing yet and do not reboot. Disconnect from the network first, then we go through what was touched.

USER SAID: Ye file delete kar do.
DRAFT REPLY: Aray bilkul! Pehle bata do kaunsi file hai? Delete ke baad wapas lana mushkil hota hai, toh exact path confirm kar lete hain.

REWRITE:
[emotion:focused]Ek second. Pehle exact path confirm kar lete hain - delete ke baad wapas lena mushkil hota hai.
"""

MODE_B_SHOTS = """
SITUATION: The printer produced beautiful gibberish.
FEELING: frustrated

USER: The printer has been at it for twenty minutes and I have printed exactly nothing useful.
REPLY: [emotion:warm]Right, that is a long afternoon of nothing. [sigh]Let us look at the queue before we look at the driver.

SITUATION: The printer produced beautiful gibberish.
FEELING: delighted

USER: The printer just produced a perfect page. First try, no fuss.
REPLY: [emotion:amused]Look at that. A flawless page, no drama, no edits. [laugh]I will take it.

SITUATION: The printer produced beautiful gibberish.
FEELING: neutral

USER: What is the difference between a process and a thread?
REPLY: [emotion:neutral]A process is its own little kingdom with its own memory. A thread is a room inside a house, sharing the furniture.
"""


def mode_a_system(persona: str) -> str:
    return f"""You rewrite replies for a spoken desktop AI companion. Output only the rewritten reply.

PERSONA - the system prompt the companion operates under:
{persona}

VOICE MODES - choose exactly one and put it at the very start of the reply:
{VOICE_MODES}

NON-VERBAL TAGS:
{NONVERBAL_RULES}

RULES:
{SPOKEN_RULES}
- The draft you are given may be in a louder, more jokey register than this
  persona. Bring it into this persona's register: understated British humour,
  playful but never loud, and never jokey for its own sake.
- Keep the substance of the draft - the same advice, the same facts, the same
  refusal. Change the register, the length and the delivery, not the meaning.
- If the draft jokes about something serious, that joke must go.

EXAMPLES:
{MODE_A_SHOTS}
Now rewrite the reply you are given. Output only the rewrite."""


def mode_b_system(persona: str) -> str:
    return f"""You write training pairs for a spoken desktop AI companion.

Given a situation and a named feeling, you produce two lines:

USER  - what the user says, rewritten so that it genuinely carries the feeling.
REPLY - the companion's spoken reply, with a voice mode tag.

The point is that the SAME request arrives in many moods, and the companion has
to read the mood from the user's own words. So the USER line must carry the
feeling in its wording, without ever naming it. A user says "Third failed run
today, this is doing my head in." - not "I am frustrated because the build
failed." The underlying request stays recognisable.

PERSONA - the system prompt the companion operates under:
{persona}

VOICE MODES - choose exactly one and put it at the very start of the reply:
{VOICE_MODES}

NON-VERBAL TAGS:
{NONVERBAL_RULES}

RULES:
- Output EXACTLY two lines in this format and nothing else:
USER: <the user line>
REPLY: <tagged reply>
{SPOKEN_RULES}
- Match the reply to the feeling: warm when the user is frustrated or unsure,
  amused when they are light, focused when they are mid-task, serious when it
  is genuinely serious, neutral for plain factual questions.
- If the feeling does not fit the situation, do not force it - write a USER
  line that makes the feeling natural for that request.

EXAMPLES:
{MODE_B_SHOTS}
Now write the pair for the situation you are given."""


# --------------------------------------------------------------------------
# Teacher client
# --------------------------------------------------------------------------


class TeacherError(RuntimeError):
    pass


class Teacher:
    """Base teacher: on-disk cache plus a pluggable request backend.

    Caching matters more than usual here - the same prompt is sampled three
    times for self-consistency, and re-running the annotator after a tweak to
    the prompt should not re-spend the whole budget.
    """

    backend = "base"

    def __init__(self, model: str, cache_path: Path | None = None):
        self.model = model
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        if cache_path and cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}

    # -- cache -----------------------------------------------------------
    def _key(self, messages, temperature, max_tokens) -> str:
        blob = json.dumps(
            [self.backend, self.model, messages, temperature, max_tokens], sort_keys=True
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_save(self) -> None:
        if not self.cache_path:
            return
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.cache_path)

    # -- backend ---------------------------------------------------------
    def _request(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> tuple[str, dict]:
        raise NotImplementedError

    # -- public ----------------------------------------------------------
    def chat(
        self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 320
    ) -> str:
        key = self._key(messages, temperature, max_tokens)
        with self._lock:
            if key in self.cache:
                self.cache_hits += 1
                return self.cache[key]

        text, usage = self._request(messages, temperature, max_tokens)

        with self._lock:
            self.cache[key] = text
            self.calls += 1
            for k in self.usage:
                self.usage[k] += usage.get(k, 0)
            if self.calls % 20 == 0:
                self._cache_save()
        return text


class OpenAIChatTeacher(Teacher):
    """Any OpenAI-compatible /chat/completions endpoint (mlx_lm.server)."""

    backend = "openai"

    def __init__(self, base_url: str, model: str, cache_path: Path | None = None,
                 api_key: str = "local"):
        super().__init__(model, cache_path)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _request(self, messages, temperature, max_tokens):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    body = json.loads(resp.read())
                text = body["choices"][0]["message"]["content"] or ""
                return text, body.get("usage", {}) or {}
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last_err = exc
                time.sleep(2 * (attempt + 1))
        raise TeacherError(f"teacher call failed: {last_err}")


class AnthropicTeacher(Teacher):
    """Anthropic Messages API (the opencode zen gateway and friends)."""

    backend = "anthropic"

    def __init__(self, base_url: str, model: str, cache_path: Path | None = None,
                 api_key: str | None = None, session: str | None = None):
        super().__init__(model, cache_path)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.session = session or f"voice-agent-{uuid.uuid4().hex[:12]}"

    def _request(self, messages, temperature, max_tokens):
        system = ""
        turns = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                turns.append(m)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": turns,
        }
        if system:
            payload["system"] = system
        if temperature and temperature > 0:
            payload["temperature"] = temperature

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "x-opencode-session": self.session,
            "User-Agent": "voice-agent/1.0",
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    body = json.loads(resp.read())
                text = "".join(c.get("text", "") for c in body.get("content", []))
                return text, body.get("usage", {}) or {}
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last_err = exc
                time.sleep(2 * (attempt + 1))
        raise TeacherError(f"teacher call failed: {last_err}")


class LocalTorchTeacher(Teacher):
    """Fully offline fallback: run a local HF model through transformers.

    Only useful for shaking the pipeline out and as a last-resort annotator -
    a model the same size as the student is a weak teacher. Use it to debug
    the plumbing, not to produce the training set.
    """

    backend = "local"

    def __init__(self, model_path: Path, cache_path: Path | None = None,
                 device: str = "auto"):
        super().__init__(str(model_path), cache_path)
        self.model_path = Path(model_path)
        self.device = device
        self._model = None
        self._tok = None
        self._lock2 = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, dtype=torch.float32
        ).to(device).eval()
        self.device = device

    def _request(self, messages, temperature, max_tokens):
        import torch

        with self._lock2:
            self._load()
            prompt = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = self._tok(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._model.generate(
                    **ids,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    pad_token_id=self._tok.eos_token_id,
                )
            text = self._tok.decode(
                out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True
            )
            return text, {"output_tokens": int(out.shape[1] - ids["input_ids"].shape[1])}


def build_teacher(args) -> Teacher:
    cache = ANNOTATED / "teacher_cache.json"
    if args.backend == "openai":
        return OpenAIChatTeacher(args.base_url, args.model, cache, args.api_key)
    if args.backend == "anthropic":
        return AnthropicTeacher(args.base_url, args.model, cache, args.api_key)
    if args.backend == "local":
        return LocalTorchTeacher(Path(args.model), cache, args.device)
    raise SystemExit(f"unknown backend {args.backend}")


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

TAG_RE = re.compile(
    r"\[emotion:(%s)\]" % "|".join(DELIVERY_EMOTIONS)
    + r"|\[(%s)\]" % "|".join(NONVERBALS)
)
DELIVERY_RE = re.compile(r"\[emotion:(%s)\]" % "|".join(DELIVERY_EMOTIONS))


def extract_tag(text: str) -> str | None:
    m = DELIVERY_RE.search(text or "")
    return m.group(1) if m else None


def clean_reply(text: str) -> str:
    """Strip anything the teacher wrapped around the reply."""
    text = (text or "").strip()
    # Remove a leading "REPLY:" marker.
    text = re.sub(r"^REPLY\s*:\s*", "", text, flags=re.I)
    # Remove surrounding quotes.
    text = re.sub(r'^["“](.*)["”]$', r"\1", text, flags=re.S).strip()
    # Remove trailing prose after the reply (a blank line then a sentence).
    return text.strip()


def parse_mode_b(text: str) -> tuple[str | None, str | None]:
    """Parse the two-line Mode B output, tolerating formatting drift."""
    user = reply = None
    for raw in (text or "").splitlines():
        line = raw.strip().strip("*").strip()
        if not line:
            continue
        m = re.match(r"^USER\s*[:\-]\s*(.+)$", line, re.I)
        if m and user is None:
            user = m.group(1).strip()
            continue
        m = re.match(r"^REPLY\s*[:\-]\s*(.+)$", line, re.I)
        if m and reply is None:
            reply = m.group(1).strip()
            continue
        # Unlabelled line that carries a tag is the reply.
        if reply is None and TAG_RE.search(line):
            reply = line
            continue
        # Unlabelled line without a tag, before we have a reply, is the user.
        if user is None and reply is None and "[" not in line:
            user = line
    return user, reply


# --------------------------------------------------------------------------
# Self-consistency
# --------------------------------------------------------------------------


def majority_label(
    teacher: Teacher, voter: Teacher, messages: list[dict], vote_tokens: int = 24
) -> tuple[str | None, int, int]:
    """Sample the delivery label three times. Returns (label, votes, samples).

    ``vote_tokens`` has to be large enough to reach the tag. In Mode A the tag
    is the first thing in the reply, so a couple of dozen tokens is plenty. In
    Mode B the tag sits on the second line, behind the USER line - a short
    budget would stop before reaching it and every vote would come back
    empty.
    """
    votes: list[str | None] = []
    for temp, max_tokens, t in (
        (0.0, 320, teacher),
        (0.7, vote_tokens, voter),
        (0.7, vote_tokens, voter),
    ):
        text = t.chat(messages, temperature=temp, max_tokens=max_tokens)
        votes.append(extract_tag(text))
    counts: dict[str, int] = {}
    for v in votes:
        if v:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None, 0, 3
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0], best[1], 3


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------


def clean_and_check(reply: str) -> tuple[str, list[str]]:
    """Clean a teacher reply and report anything still wrong with it.

    The teacher is asked for clean text, but a 0.8B-scale model will still
    occasionally emit an emoji, a stray backtick or a code fence. Those must
    not reach the training set, so they are stripped here and the remaining
    problems are recorded so the flattener can drop the example.
    """
    from normalize_spoken import normalize, validate_reply

    tag = ""
    m = DELIVERY_RE.search(reply or "")
    if m:
        tag = m.group(0)
    body = TAG_RE.sub("", reply or "").strip()
    body = normalize(body)
    cleaned = f"{tag} {body}".strip() if tag else body
    issues = validate_reply(cleaned)
    return cleaned, issues


def annotate_one(
    teacher: Teacher,
    voter: Teacher,
    persona: str,
    rec: dict,
    affects: tuple[str, ...],
) -> dict:
    """Produce a tagged reply for the base prompt, plus affect variants."""
    out = {
        "id": rec["id"],
        "source": rec["source"],
        "base_user": rec["user"],
        "base_reply": None,
        "base_raw": None,
        "base_emotion": None,
        "base_votes": 0,
        "variants": [],
        "errors": [],
    }

    # -- Mode A ---------------------------------------------------------
    a_msgs = [
        {"role": "system", "content": mode_a_system(persona)},
        {
            "role": "user",
            "content": f"USER SAID: {rec['user']}\n\nDRAFT REPLY:\n{rec['assistant']}",
        },
    ]
    try:
        label, votes, _ = majority_label(teacher, voter, a_msgs, vote_tokens=24)
        draft = teacher.chat(a_msgs, temperature=0.0, max_tokens=320)
        reply = clean_reply(draft)
        out["base_raw"] = draft
        if label and not reply.startswith("[emotion:"):
            reply = f"[emotion:{label}]" + reply
        reply, issues = clean_and_check(reply)
        out["base_reply"] = reply
        out["base_emotion"] = extract_tag(reply) or label
        out["base_votes"] = votes
        if issues:
            out["base_issues"] = issues
    except TeacherError as exc:
        out["errors"].append(f"mode_a: {exc}")

    # -- Mode B ---------------------------------------------------------
    for affect in affects:
        b_msgs = [
            {"role": "system", "content": mode_b_system(persona)},
            {
                "role": "user",
                "content": f"SITUATION: {rec['user']}\nFEELING: {affect}",
            },
        ]
        try:
            # Mode B puts the tag on the second line, behind the USER line,
            # so the vote budget has to cover both.
            label, votes, _ = majority_label(teacher, voter, b_msgs, vote_tokens=80)
            draft = teacher.chat(b_msgs, temperature=0.0, max_tokens=320)
            user_line, reply_line = parse_mode_b(draft)
            if not user_line or not reply_line:
                out["errors"].append(f"mode_b[{affect}]: unparseable")
                out["variants"].append(
                    {"affect": affect, "raw": draft, "unparsed": True}
                )
                continue
            reply_line = clean_reply(reply_line)
            if label and not reply_line.startswith("[emotion:"):
                reply_line = f"[emotion:{label}]" + reply_line
            reply_line, issues = clean_and_check(reply_line)
            entry = {
                "affect": affect,
                "user": user_line,
                "reply": reply_line,
                "emotion": extract_tag(reply_line) or label,
                "votes": votes,
                "raw": draft,
            }
            if issues:
                entry["issues"] = issues
            out["variants"].append(entry)
        except TeacherError as exc:
            out["errors"].append(f"mode_b[{affect}]: {exc}")

    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ANNOTATED / "annotated.jsonl")
    ap.add_argument("--report", type=Path, default=ANNOTATED / "annotate_report.json")
    ap.add_argument(
        "--backend",
        default="openai",
        choices=["openai", "anthropic", "local"],
        help="openai: any /chat/completions server. anthropic: Messages API. "
        "local: run a local HF model offline (weak teacher, for debugging).",
    )
    ap.add_argument("--base-url", default=TEACHER_BASE_URL)
    ap.add_argument("--model", default=TEACHER_MODEL)
    ap.add_argument(
        "--vote-model",
        default=None,
        help="cheaper model for the two label-vote calls (default: same as --model)",
    )
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--affects",
        default="frustrated,delighted",
        help="comma-separated affect variants for Mode B",
    )
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-votes", type=int, default=2, help="majority needed to accept")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if the teacher is not producing usable tags",
    )
    args = ap.parse_args()

    affects = tuple(a.strip() for a in args.affects.split(",") if a.strip())
    records = read_jsonl(args.inp)
    if args.limit:
        records = records[: args.limit]

    # Resume: skip ids already annotated.
    done: set[str] = set()
    if args.out.exists():
        for rec in read_jsonl(args.out):
            done.add(rec["id"])
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(records)} records, {len(done)} already done, {len(todo)} to do")

    persona = load_persona()
    cache = ANNOTATED / "teacher_cache.json"
    teacher = build_teacher(args)
    voter = teacher
    if args.vote_model and args.vote_model != args.model:
        vargs = argparse.Namespace(**vars(args))
        vargs.model = args.vote_model
        voter = build_teacher(vargs)
    print(f"teacher: {args.backend}/{args.model}  voter: {args.vote_model or args.model}")

    results: list[dict] = []
    lock = threading.Lock()
    t0 = time.time()

    def work(rec):
        try:
            res = annotate_one(teacher, voter, persona, rec, affects)
        except Exception as exc:  # keep going, record the failure
            res = {
                "id": rec["id"],
                "source": rec["source"],
                "base_user": rec["user"],
                "base_reply": None,
                "base_emotion": None,
                "base_votes": 0,
                "variants": [],
                "errors": [f"fatal: {type(exc).__name__}: {exc}"],
            }
        with lock:
            results.append(res)
            n = len(results)
            if n % 5 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / el if el > 0 else 0
                eta = (len(todo) - n) / rate if rate > 0 else 0
                print(
                    f"  {n}/{len(todo)}  {el:6.0f}s  {rate:.2f}/s  eta {eta:5.0f}s  "
                    f"(calls {teacher.calls}, cached {teacher.cache_hits})",
                    flush=True,
                )
        return res

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))

    teacher._cache_save()
    voter._cache_save()

    all_rows = read_jsonl(args.out) if args.out.exists() else []
    all_rows.extend(results)
    all_rows.sort(key=lambda r: r["id"])
    write_jsonl(args.out, all_rows)

    # -- report ---------------------------------------------------------
    accepted = [r for r in all_rows if r.get("base_reply")]
    ambiguous = [r for r in all_rows if r.get("base_votes", 0) < args.min_votes]
    label_counts: dict[str, int] = {}
    variant_counts: dict[str, int] = {}
    for r in all_rows:
        if r.get("base_emotion"):
            label_counts[r["base_emotion"]] = label_counts.get(r["base_emotion"], 0) + 1
        for v in r.get("variants", []):
            if v.get("emotion"):
                variant_counts[v["emotion"]] = variant_counts.get(v["emotion"], 0) + 1

    report = {
        "records": len(all_rows),
        "with_base_reply": len(accepted),
        "ambiguous_base": len(ambiguous),
        "variants": sum(len(r.get("variants", [])) for r in all_rows),
        "variants_parsed": sum(
            1
            for r in all_rows
            for v in r.get("variants", [])
            if not v.get("unparsed")
        ),
        "base_label_distribution": label_counts,
        "variant_label_distribution": variant_counts,
        "teacher_calls": teacher.calls,
        "teacher_cache_hits": teacher.cache_hits,
        "teacher_usage": teacher.usage,
        "voter_usage": voter.usage,
        "errors": sum(len(r.get("errors", [])) for r in all_rows),
        "affects": list(affects),
    }
    write_json(args.report, report)
    print(json.dumps(report, indent=2))

    # Fail fast on a teacher that is not following the format. Catching this
    # after 400 calls is expensive; catching it here is free.
    if all_rows:
        tagged = report["with_base_reply"] + report["variants_parsed"]
        total = len(all_rows) + report["variants"]
        rate = tagged / total if total else 0
        if rate < 0.5:
            print(
                f"\nWARNING: only {rate:.0%} of teacher outputs carried a usable "
                f"tag. The teacher is not following the format - check the "
                f"prompt, or try a different model, before annotating the rest.",
                file=sys.stderr,
            )
            if args.strict:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
