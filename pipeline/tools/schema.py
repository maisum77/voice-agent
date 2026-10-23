"""The function surface the voice agent is allowed to call.

The model does not get a tool for every OS API. It gets a small set of
primitives that compose: ``shell.run`` is the escape hatch, and the rest are
the operations common enough to deserve a typed call and a policy rule.

Every function declares its risk tier. The policy layer, not the model,
decides whether a call is allowed and whether it needs confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RISK_TIERS = ("read", "modify", "execute", "admin")


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class Function:
    name: str
    description: str
    params: tuple[Param, ...]
    risk: str = "read"
    confirm: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def signature(self) -> str:
        bits = []
        for p in self.params:
            bit = f"{p.name}: {p.type}"
            if not p.required:
                bit += "?"
            bits.append(bit)
        return f"{self.name}({', '.join(bits)})"


def _p(name: str, type_: str, required: bool = True, description: str = "") -> Param:
    return Param(name=name, type=type_, required=required, description=description)


FUNCTIONS: tuple[Function, ...] = (
    # -- shell ------------------------------------------------------------
    Function(
        "shell.run",
        "Run a shell command and return its output.",
        (
            _p("command", "string", True, "The command line to execute."),
            _p("cwd", "string", False, "Working directory."),
            _p("timeout", "integer", False, "Seconds before the command is killed."),
        ),
        risk="execute",
        tags=("shell",),
    ),
    # -- filesystem -------------------------------------------------------
    Function("fs.read", "Read a text file.", (_p("path", "string"),), risk="read"),
    Function(
        "fs.write",
        "Write a text file, replacing it if it exists.",
        (_p("path", "string"), _p("content", "string")),
        risk="modify",
    ),
    Function("fs.list", "List a directory.", (_p("path", "string"),), risk="read"),
    Function(
        "fs.move",
        "Move or rename a file or directory.",
        (_p("src", "string"), _p("dst", "string")),
        risk="modify",
    ),
    Function(
        "fs.copy",
        "Copy a file or directory.",
        (_p("src", "string"), _p("dst", "string")),
        risk="modify",
    ),
    Function(
        "fs.delete",
        "Delete a file, or a directory recursively.",
        (_p("path", "string"), _p("recursive", "boolean", False)),
        risk="modify",
        confirm=True,
        tags=("destructive",),
    ),
    # -- processes --------------------------------------------------------
    Function(
        "proc.list",
        "List running processes.",
        (_p("sort_by", "string", False, "cpu, memory or pid"),),
        risk="read",
    ),
    Function(
        "proc.kill",
        "Send a signal to a process.",
        (_p("pid", "integer"), _p("signal", "string", False)),
        risk="execute",
        confirm=True,
        tags=("destructive",),
    ),
    # -- applications -----------------------------------------------------
    Function("app.launch", "Launch an installed application.", (_p("name", "string"),), risk="modify"),
    Function("app.focus", "Bring an application window to the front.", (_p("name", "string"),), risk="modify"),
    Function("app.close", "Close an application.", (_p("name", "string"),), risk="modify"),
    # -- system -----------------------------------------------------------
    Function(
        "sys.info",
        "Read CPU, memory, disk or battery information.",
        (_p("what", "string", False, "cpu, memory, disk, battery or all"),),
        risk="read",
    ),
    Function(
        "sys.power",
        "Lock, suspend, reboot or shut down the machine.",
        (_p("action", "string", description="lock, suspend, reboot or shutdown"),),
        risk="admin",
        confirm=True,
        tags=("irreversible",),
    ),
    Function(
        "sys.service",
        "Inspect or control a systemd service.",
        (_p("action", "string", description="status, start, stop, restart or disable"),
         _p("name", "string")),
        risk="admin",
        confirm=True,
        tags=("service",),
    ),
    Function(
        "sys.pkg",
        "Install, remove or update a package.",
        (_p("action", "string", description="install, remove or update"),
         _p("name", "string")),
        risk="admin",
        confirm=True,
        tags=("package",),
    ),
    # -- ui ---------------------------------------------------------------
    Function(
        "ui.notify",
        "Show a desktop notification.",
        (_p("title", "string"), _p("body", "string")),
        risk="modify",
    ),
    Function(
        "ui.clipboard",
        "Read or write the clipboard.",
        (_p("action", "string", description="get or set"), _p("text", "string", False)),
        risk="modify",
    ),
    Function(
        "ui.volume",
        "Read, set or mute the system volume.",
        (_p("action", "string", description="get, set, mute or unmute"),
         _p("level", "integer", False, "0-100")),
        risk="modify",
    ),
    # -- memory -----------------------------------------------------------
    Function("mem.read", "Read the user's local memory.", (_p("key", "string", False),), risk="read"),
    Function(
        "mem.write",
        "Store a fact in the user's local memory.",
        (_p("key", "string"), _p("value", "string")),
        risk="modify",
    ),
)

FUNCTIONS_BY_NAME = {f.name: f for f in FUNCTIONS}

# Compact enough to afford in every system prompt. The model still has to
# learn the arguments, but it does not have to guess the vocabulary.
TOOL_PROMPT = (
    "Available tools. Emit at most two calls as "
    '<tool_call>{"name": NAME, "arguments": {...}}</tool_call> before you speak.\n'
    + "\n".join(f"- {f.signature()}" for f in FUNCTIONS)
    + "\nNever claim an action ran until the runtime returns a result. "
    "Destructive actions need confirmation; ask for it in the spoken text."
)


def tool_prompt() -> str:
    return TOOL_PROMPT


def validate_call(name: str, arguments: object) -> list[str]:
    """Problems with a call, empty if it matches the schema."""
    fn = FUNCTIONS_BY_NAME.get(name)
    if fn is None:
        return [f"unknown function: {name}"]
    if not isinstance(arguments, dict):
        return ["arguments must be an object"]
    problems = []
    for p in fn.params:
        if p.required and p.name not in arguments:
            problems.append(f"missing argument: {p.name}")
    allowed = {p.name for p in fn.params}
    for key in arguments:
        if key not in allowed:
            problems.append(f"unexpected argument: {key}")
    return problems
