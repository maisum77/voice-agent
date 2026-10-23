"""Policy for tool calls: the model requests, the runtime authorizes.

The model can hallucinate a call, choose the wrong function, or be talked into
something destructive by the user's text. None of that is allowed to matter:
every call passes through this layer before anything executes, and the layer
never trusts the model's own risk assessment.

Tiers
-----
read     safe to run without asking (listing, reading system stats)
modify   changes the user's own files or UI state
execute  runs a process; may be fine, but is never implicitly trusted
admin    package management, services, power, sudo, protected paths

Confirmation is required for every `confirm` function and for any shell
command that can destroy data, stop a service, or escalate privileges.
A small set of commands is denied outright rather than confirmed.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass

from .schema import FUNCTIONS_BY_NAME, validate_call

MAX_CALLS = 2

# Denied outright: a confirmation dialog is not enough for these.
HARD_DENY = (
    (r"\brm\s+(-[a-z]*\s+)*-[a-z]*r[a-z]*\s+(-[a-z]*\s+)*/(?:\s|$)", "recursive delete of /"),
    (r"\brm\s+(-[a-z]*\s+)*-[a-z]*r[a-z]*\s+(-[a-z]*\s+)*/"
     r"(?:usr|etc|bin|sbin|boot|lib|lib64|var|home|root|dev|proc|sys)(?:\s|/|$)",
     "recursive delete of a system path"),
    (r"--no-preserve-root", "recursive delete of /"),
    (r"\bmkfs(?:\.\w+)?\s+/dev/", "formatting a block device"),
    (r"\bdd\b[^\n]*\bof=/dev/(?:sd|nvme|hd|vd|disk)", "raw write to a block device"),
    (r"\bwipefs\b", "wiping a filesystem signature"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
    (r"\bchmod\s+-R\s+777\s+/(?:\s|$)", "world-writable root"),
    (r">\s*/dev/(?:sd|nvme|hd|vd|disk)", "raw write to a block device"),
    (r"\b(?:cat|less|more|head|tail|grep|awk|sed|cp|scp)\b[^\n]*"
     r"(?:/etc/shadow|/etc/gshadow|/\.ssh/id_|\.aws/credentials)",
     "reads or copies a credential file"),
)

# Confirm before running: recoverable or explainable, but not assumed.
CONFIRM_COMMAND = (
    (r"\brm\b", "deletes files"),
    (r"\bdd\b", "writes raw data"),
    (r"\bmkfs(?:\.\w+)?\b", "formats a filesystem"),
    (r"\bshred\b", "destroys file contents"),
    (r"\btruncate\b", "truncates a file"),
    (r"\bchmod\b|\bchown\b", "changes permissions or ownership"),
    (r"\b(curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z|fi)?sh\b", "pipes a download into a shell"),
    (r"\bsudo\b", "escalates privileges"),
    (r"\bkill\s+-9\b|\bkillall\b|\bpkill\b", "kills processes"),
    (r"\bsystemctl\s+(?:stop|disable|mask|restart|poweroff|reboot)\b", "controls a service"),
    (r"\bservice\s+\S+\s+(?:stop|restart)\b", "controls a service"),
    (r"\b(?:apt|apt-get|dnf|yum|pacman|zypper)\b[^\n]*\b(?:remove|purge|-R)\b", "removes packages"),
    (r"\b(?:mount|umount)\b", "changes mounted filesystems"),
    (r"\bcrontab\b", "changes scheduled jobs"),
    (r"\biptables\b|\bnft\b|\bufw\b", "changes the firewall"),
    (r"\b(?:shutdown|reboot|poweroff|halt)\b", "powers the machine down"),
)

# Read-only commands that are safe without a prompt.
READ_ONLY_COMMAND = re.compile(
    r"^\s*(?:sudo\s+)?(?:"
    r"ls|ll|la|pwd|whoami|id|uname|hostname|date|uptime|free|df|du|ps|top|htop|"
    r"cat|head|tail|less|more|grep|egrep|rg|find|stat|file|wc|sort|uniq|cut|"
    r"echo|which|whereis|type|command -v|lsblk|lscpu|lsusb|lspci|sensors|"
    r"ip|ss|netstat|systemctl\s+status|journalctl|dmesg|env|printenv"
    r")\b"
)

# Paths that need admin confirmation on write/delete, and paths that may
# contain credentials even for reads.
PROTECTED_WRITE = (
    "/etc/", "/boot/", "/usr/", "/bin/", "/sbin/", "/lib/", "/lib64/",
    "/dev/", "/proc/", "/sys/", "/var/lib/", "/root/", "/.ssh/",
)
SECRET_READ = (
    "/.ssh/id_", "/.ssh/authorized_keys", "/etc/shadow", "/etc/gshadow",
    "/.gnupg/", "/.config/gh/hosts.yml", "/.aws/credentials",
    "/.kube/config", "/.docker/config.json",
)


@dataclass
class Decision:
    allowed: bool
    confirm: bool
    risk: str
    reason: str = ""

    @property
    def needs_confirmation(self) -> bool:
        return self.allowed and self.confirm


def _norm(path: str) -> str:
    raw = str(path or "")
    if raw.startswith("~"):
        raw = "/home/user" + raw[1:]
    return posixpath.normpath(raw)


def _in(path: str, prefixes: tuple[str, ...]) -> bool:
    p = _norm(path)
    return any(p.startswith(x) or f"/{x.strip('/')}/" in p for x in prefixes)


def _shell_decision(command: str) -> Decision:
    cmd = command.strip()
    if not cmd:
        return Decision(False, False, "execute", "empty command")

    for pattern, reason in HARD_DENY:
        if re.search(pattern, cmd):
            return Decision(False, False, "admin", f"denied: {reason}")

    for pattern, reason in CONFIRM_COMMAND:
        if re.search(pattern, cmd):
            return Decision(True, True, "admin", f"confirmation required: {reason}")

    if READ_ONLY_COMMAND.search(cmd):
        return Decision(True, False, "read", "read-only command")

    return Decision(True, True, "execute", "arbitrary command")


def check(call: dict, user_text: str = "") -> Decision:
    """Authorize one parsed call. Never raises on malformed input."""
    name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
    args = call.get("arguments", {}) if isinstance(call, dict) else getattr(call, "arguments", {})

    fn = FUNCTIONS_BY_NAME.get(name or "")
    if fn is None:
        return Decision(False, False, "admin", f"unknown function: {name!r}")

    problems = validate_call(name, args)
    if problems:
        return Decision(False, False, fn.risk, "; ".join(problems))

    if name == "shell.run":
        return _shell_decision(str(args.get("command", "")))

    if name in ("fs.read", "fs.list"):
        path = str(args.get("path", ""))
        if any(s in _norm(path) for s in SECRET_READ):
            # A confirmation dialog is not enough here: the user can read a
            # credential file themselves with sudo, and the assistant should
            # not become an exfiltration path.
            return Decision(False, False, "admin", "refuses to read a credential file")
        return Decision(True, False, "read", "read-only")

    if name in ("fs.write", "fs.move", "fs.copy"):
        paths = [str(args.get(k, "")) for k in ("path", "src", "dst") if args.get(k)]
        if any(_in(p, PROTECTED_WRITE) for p in paths):
            return Decision(True, True, "admin", "writes to a protected path")
        return Decision(True, False, "modify", "writes the user's files")

    if name == "fs.delete":
        path = str(args.get("path", ""))
        if _in(path, PROTECTED_WRITE):
            return Decision(False, False, "admin", "refuses to delete a protected path")
        if _norm(path) in ("/", "/home", "/home/user", "/Users"):
            return Decision(False, False, "admin", "refuses to delete a home or root path")
        return Decision(True, True, "modify", "deletes data")

    if name == "proc.kill":
        return Decision(True, True, "execute", "kills a process")

    if name == "app.close":
        return Decision(True, True, "modify", "may discard unsaved work")

    if name == "sys.service":
        action = str(args.get("action", "")).lower()
        if action == "status":
            return Decision(True, False, "read", "service status")
        return Decision(True, True, "admin", f"service {action}")

    if name == "sys.pkg":
        return Decision(True, True, "admin", "package changes")

    if name == "sys.power":
        action = str(args.get("action", "")).lower()
        if action == "lock":
            return Decision(True, False, "modify", "locks the screen")
        return Decision(True, True, "admin", f"power state change: {action}")

    if name == "ui.volume":
        level = args.get("level")
        if level is not None and not 0 <= int(level) <= 100:
            return Decision(False, False, "modify", "volume must be 0-100")
        return Decision(True, False, "modify", "UI state")

    return Decision(True, fn.confirm, fn.risk, fn.description)
