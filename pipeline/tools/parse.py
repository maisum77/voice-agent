"""Parse the tool-call channel out of a model reply.

The contract is a JSON block, optionally preceded by a delivery tag::

    [emotion:serious]<tool_call>{"name":"fs.delete","arguments":{"path":"/tmp/x"}}</tool_call>Remove it?

Qwen's native tool format is also accepted as a fallback, because a base
model that has not fully learned the JSON form will sometimes emit XML::

    <tool_call><function=fs.delete><parameter=path>/tmp/x</parameter></function></tool_call>

Parsing is intentionally strict. A call that does not match the schema is
returned as an error, never silently repaired: the policy layer decides what
to do with it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .schema import FUNCTIONS_BY_NAME, validate_call

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
XML_FUNCTION_RE = re.compile(
    r"<function=([A-Za-z_][\w.]*)>(.*?)</function>", re.DOTALL
)
XML_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z_][\w]*)>\s*(.*?)\s*</parameter>", re.DOTALL
)
EMOTION_RE = re.compile(r"\[emotion:(neutral|amused|warm|focused|serious)\]")


@dataclass
class ToolCall:
    name: str
    arguments: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    raw: str = ""

    @property
    def valid(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class ParsedReply:
    calls: list[ToolCall]
    speech: str
    emotion: str | None
    errors: list[str] = field(default_factory=list)

    @property
    def has_valid_call(self) -> bool:
        return any(c.valid for c in self.calls)


def _coerce(value, type_name: str):
    if type_name == "integer":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value
    if type_name == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    if type_name == "string":
        return value if isinstance(value, str) else str(value)
    return value


def _coerce_arguments(name: str, arguments: dict) -> dict:
    fn = FUNCTIONS_BY_NAME.get(name)
    if fn is None:
        return arguments
    types = {p.name: p.type for p in fn.params}
    return {k: _coerce(v, types.get(k, "string")) for k, v in arguments.items()}


def _parse_json_payload(payload: str) -> tuple[str, dict] | None:
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        # Tolerate a fenced block: ```json {...} ```
        payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload.strip())
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function")
    args = obj.get("arguments", obj.get("args", {}))
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return name, args


def _parse_xml_payload(payload: str) -> tuple[str, dict] | None:
    m = XML_FUNCTION_RE.search(payload)
    if not m:
        return None
    name = m.group(1)
    args = {}
    for pm in XML_PARAM_RE.finditer(m.group(2)):
        args[pm.group(1)] = pm.group(2).strip()
    return name, args


def parse_reply(text: str) -> ParsedReply:
    """Split a model reply into calls and the speech that remains."""
    calls: list[ToolCall] = []
    errors: list[str] = []

    for match in TOOL_CALL_RE.finditer(text):
        payload = match.group(1).strip()
        parsed = _parse_json_payload(payload) or _parse_xml_payload(payload)
        if parsed is None:
            errors.append(f"unparseable tool call: {payload[:120]}")
            calls.append(ToolCall(name="", raw=payload, problems=["unparseable"]))
            continue
        name, args = parsed
        args = _coerce_arguments(name, args)
        problems = validate_call(name, args)
        calls.append(ToolCall(name=name, arguments=args, problems=problems, raw=payload))

    speech = TOOL_CALL_RE.sub("", text).strip()
    emotion_match = EMOTION_RE.search(text)
    return ParsedReply(
        calls=calls,
        speech=speech,
        emotion=emotion_match.group(1) if emotion_match else None,
        errors=errors,
    )


def strip_calls(text: str) -> str:
    """The human-facing text, with every tool-call block removed."""
    return TOOL_CALL_RE.sub("", text).strip()
