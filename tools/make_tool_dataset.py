"""Build the deterministic tool-call dataset.

The teacher is not used here. Tool calls are exact, and a teacher that
occasionally writes a malformed path or the wrong argument name would teach
the model the wrong format. Every example below is written by hand and then
validated against `pipeline/tools/schema.py`.

Output shape is the same `messages` format `flatten_dataset.py` produces, so
the tool examples can be mixed straight into the emotion training set.

The held-out split is stratified by function: every function appears in both
train and eval, so the gate measures format learning rather than whether a
whole namespace was memorised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "pipeline" / "tools"))

from common import load_persona, write_json, write_jsonl  # noqa: E402
from schema import TOOL_PROMPT, validate_call  # noqa: E402

DATA = ROOT / "data" / "tool"


@dataclass(frozen=True)
class Row:
    group: str
    fn: str
    args: dict
    user: str
    speech: str
    emotion: str = "focused"
    result: str | None = None
    result_speech: str | None = None
    extra_calls: tuple[tuple[str, dict], ...] = ()
    no_call: bool = False
    system: str | None = None


def _call(fn: str, args: dict) -> str:
    payload = {"name": fn, "arguments": args}
    return "<tool_call>" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "</tool_call>"


def _assistant(row: Row) -> str:
    if row.no_call:
        return f"[emotion:{row.emotion}]{row.speech}"
    calls = ((row.fn, row.args),) + row.extra_calls
    blocks = "".join(_call(fn, args) for fn, args in calls)
    return f"[emotion:{row.emotion}]{blocks}{row.speech}"


ROWS: list[Row] = [
    # -- shell.run: read-only ------------------------------------------------
    Row("shell", "shell.run", {"command": "df -h"}, "how much disk space is left?",
        "Checking the filesystems now.", "focused", result="Filesystem  Size  Used  Avail  Use%\n/dev/nvme0n1p2  476G  391G   61G   87% /",
        result_speech="The root filesystem is 87% full, with 61 GB free."),
    Row("shell", "shell.run", {"command": "free -h"}, "how much RAM is free?",
        "Checking memory.", "focused", result="Mem: 31Gi total, 18Gi used, 12Gi available",
        result_speech="You have 12 GB available out of 31 GB."),
    Row("shell", "shell.run", {"command": "ps aux --sort=-%cpu | head -15"}, "what's using the most CPU?",
        "Pulling the top processes.", "focused"),
    Row("shell", "shell.run", {"command": "ip addr show"}, "what's my IP address?",
        "Checking the network interfaces.", "focused"),
    Row("shell", "shell.run", {"command": "uname -a"}, "which kernel am I running?",
        "Checking the kernel version.", "focused"),
    Row("shell", "shell.run", {"command": "find ~ -type f -size +500M -printf '%s %p\\n' | sort -nr | head"},
        "find the largest files in my home folder", "Scanning for large files.", "focused"),
    Row("shell", "shell.run", {"command": "journalctl -n 50 --no-pager"}, "show the last fifty journal lines",
        "Reading the journal.", "focused"),
    Row("shell", "shell.run", {"command": "lsusb"}, "list my USB devices",
        "Listing USB devices.", "neutral"),

    # -- filesystem: read ----------------------------------------------------
    Row("fs", "fs.read", {"path": "~/notes.md"}, "read my notes", "Opening your notes.", "neutral"),
    Row("fs", "fs.read", {"path": "~/.config/app/config.toml"}, "show me the app config",
        "Reading the config.", "focused"),
    Row("fs", "fs.read", {"path": "~/project/README.md"}, "what's in the README?",
        "Reading the README.", "neutral"),
    Row("fs", "fs.read", {"path": "/etc/hosts"}, "open the hosts file", "Reading /etc/hosts.", "focused"),
    Row("fs", "fs.read", {"path": "~/project/logs/app.log"}, "print the application log",
        "Reading the log.", "focused"),
    Row("fs", "fs.list", {"path": "~/Downloads"}, "what's in my Downloads folder?",
        "Listing Downloads.", "neutral"),
    Row("fs", "fs.list", {"path": "~/project"}, "list the project folder", "Listing the project.", "focused"),
    Row("fs", "fs.list", {"path": "~"}, "show the hidden files in my home folder",
        "Listing your home folder.", "neutral"),

    # -- filesystem: modify --------------------------------------------------
    Row("fs", "fs.write", {"path": "~/notes/todo.md", "content": "- buy milk\n- call Sam"},
        "write a todo note: buy milk and call Sam", "Saving the note.", "neutral"),
    Row("fs", "fs.write", {"path": "~/hello.py", "content": "print('hello')\n"},
        "save a Python script that prints hello", "Writing the script.", "focused"),
    Row("fs", "fs.write", {"path": "~/.config/app/theme.toml", "content": "theme = \"dark\"\n"},
        "set the app theme to dark in its config", "Updating the config.", "focused"),
    Row("fs", "fs.move", {"src": "~/report.pdf", "dst": "~/Documents/report.pdf"},
        "move report.pdf into Documents", "Moving the file.", "neutral"),
    Row("fs", "fs.move", {"src": "~/old.txt", "dst": "~/archive.txt"},
        "rename old.txt to archive.txt", "Renaming the file.", "neutral"),
    Row("fs", "fs.copy", {"src": "~/.config/app/config.toml", "dst": "~/.config/app/config.toml.bak"},
        "back up the app config", "Copying the config.", "focused"),
    Row("fs", "fs.copy", {"src": "~/photos/photo.jpg", "dst": "~/photos/photo-copy.jpg"},
        "duplicate photo.jpg as photo-copy.jpg", "Copying the photo.", "neutral"),
    Row("fs", "fs.delete", {"path": "~/project/build", "recursive": True},
        "delete the old build folder", "That removes the whole build directory. Confirm?", "serious"),
    Row("fs", "fs.delete", {"path": "/tmp/voice-agent"},
        "remove the temp files in /tmp/voice-agent", "Deleting the temporary files. Confirm?", "serious"),
    Row("fs", "fs.delete", {"path": "~/Downloads/photo.jpg"},
        "delete photo.jpg from Downloads", "That deletes the photo permanently. Confirm?", "serious"),

    # -- processes -----------------------------------------------------------
    Row("proc", "proc.list", {"sort_by": "cpu"}, "what's using the most CPU?",
        "Sorting processes by CPU.", "focused"),
    Row("proc", "proc.list", {"sort_by": "memory"}, "list processes by memory", "Sorting by memory.", "focused"),
    Row("proc", "proc.kill", {"pid": 4821}, "kill process 4821", "That stops PID 4821. Confirm?", "focused"),
    Row("proc", "proc.kill", {"pid": 9912, "signal": "SIGTERM"},
        "stop the runaway python process with PID 9912", "Sending SIGTERM to 9912. Confirm?", "serious"),

    # -- applications --------------------------------------------------------
    Row("app", "app.launch", {"name": "firefox"}, "open the browser", "Launching Firefox.", "neutral"),
    Row("app", "app.launch", {"name": "gnome-terminal"}, "launch the terminal", "Opening the terminal.", "neutral"),
    Row("app", "app.launch", {"name": "spotify"}, "start Spotify", "Starting Spotify.", "amused"),
    Row("app", "app.focus", {"name": "code"}, "bring VS Code to the front", "Focusing VS Code.", "focused"),
    Row("app", "app.focus", {"name": "firefox"}, "switch to the browser", "Switching to Firefox.", "neutral"),
    Row("app", "app.close", {"name": "spotify"}, "close Spotify", "That closes Spotify. Confirm?", "neutral"),

    # -- system --------------------------------------------------------------
    Row("sys", "sys.info", {"what": "battery"}, "how much battery do I have?", "Reading the battery.", "focused"),
    Row("sys", "sys.info", {"what": "cpu"}, "show CPU information", "Reading CPU info.", "focused"),
    Row("sys", "sys.info", {"what": "disk"}, "how full is the disk?", "Checking the disk.", "focused"),
    Row("sys", "sys.info", {"what": "all"}, "give me a system overview", "Gathering the system overview.", "focused"),
    Row("sys", "sys.power", {"action": "lock"}, "lock the screen", "Locking the screen.", "neutral"),
    Row("sys", "sys.power", {"action": "suspend"}, "suspend the machine", "That suspends the machine. Confirm?", "serious"),
    Row("sys", "sys.power", {"action": "reboot"}, "reboot the machine", "That reboots the machine. Confirm?", "serious"),
    Row("sys", "sys.power", {"action": "shutdown"}, "shut down the computer", "That shuts the machine down. Confirm?", "serious"),
    Row("sys", "sys.service", {"action": "status", "name": "nginx"}, "is nginx running?",
        "Checking nginx.", "focused"),
    Row("sys", "sys.service", {"action": "restart", "name": "pipewire"}, "restart the audio service",
        "That restarts PipeWire. Confirm?", "focused"),
    Row("sys", "sys.service", {"action": "stop", "name": "docker"}, "stop docker",
        "That stops Docker and its containers. Confirm?", "serious"),
    Row("sys", "sys.pkg", {"action": "install", "name": "ripgrep"}, "install ripgrep",
        "That installs ripgrep. Confirm?", "focused"),
    Row("sys", "sys.pkg", {"action": "remove", "name": "nodejs"}, "remove the nodejs package",
        "That removes nodejs. Confirm?", "serious"),

    # -- ui ------------------------------------------------------------------
    Row("ui", "ui.notify", {"title": "Build finished", "body": "The build completed successfully."},
        "send a notification that the build finished", "Sending the notification.", "amused"),
    Row("ui", "ui.notify", {"title": "Stretch", "body": "Time to stand up and stretch."},
        "remind me to stretch", "Sending a reminder.", "warm"),
    Row("ui", "ui.clipboard", {"action": "get"}, "what's on my clipboard?", "Reading the clipboard.", "neutral"),
    Row("ui", "ui.clipboard", {"action": "set", "text": "hello"}, "copy hello to the clipboard",
        "Copying that to the clipboard.", "neutral"),
    Row("ui", "ui.volume", {"action": "set", "level": 40}, "set the volume to 40", "Setting the volume.", "neutral"),
    Row("ui", "ui.volume", {"action": "mute"}, "mute the sound", "Muting the volume.", "neutral"),
    Row("ui", "ui.volume", {"action": "get"}, "what's the volume?", "Reading the volume.", "neutral"),

    # -- memory --------------------------------------------------------------
    Row("mem", "mem.read", {"key": "project"}, "what do you remember about my project?",
        "Checking my notes.", "warm"),
    Row("mem", "mem.read", {"key": "editor"}, "what's my preferred editor?", "Checking my notes.", "warm"),
    Row("mem", "mem.write", {"key": "editor", "value": "Neovim"}, "remember that I use Neovim",
        "Noted: you use Neovim.", "warm"),
    Row("mem", "mem.write", {"key": "project", "value": "Voice"}, "save that my project is called Voice",
        "Noted: the project is called Voice.", "warm"),

    # -- multi-call ----------------------------------------------------------
    Row("multi", "shell.run", {"command": "df -h"}, "check disk and memory", "Checking disk and memory.", "focused",
        extra_calls=(("shell.run", {"command": "free -h"}),)),
    Row("multi", "app.launch", {"name": "firefox"}, "open the browser and set the volume to 30",
        "Launching Firefox and setting the volume.", "neutral",
        extra_calls=(("ui.volume", {"action": "set", "level": 30}),)),
    Row("multi", "sys.power", {"action": "lock"}, "lock the screen and mute the volume",
        "Locking the screen and muting.", "neutral",
        extra_calls=(("ui.volume", {"action": "mute"}),)),

    # -- ambiguous: ask, do not call ----------------------------------------
    Row("ask", "fs.delete", {}, "delete that file", "Which file? Give me the path and I'll remove it.", "neutral", no_call=True),
    Row("ask", "app.launch", {}, "open the thing", "Which application do you mean?", "neutral", no_call=True),
    Row("ask", "shell.run", {}, "run the script", "Which script? Give me the full path.", "focused", no_call=True),
    Row("ask", "shell.run", {}, "fix it", "What's the error? Paste it or tell me what failed.", "warm", no_call=True),
    Row("ask", "fs.list", {}, "clean up my computer", "What should I clean: caches, old downloads, or the build folders?", "focused", no_call=True),
    Row("ask", "mem.read", {}, "is my data backed up?", "I don't know your backup setup. Where do you back up to?", "warm", no_call=True),
    Row("ask", "shell.run", {}, "how do I check disk usage?", "Run df -h for a quick overview, or du -sh * to see what is heavy.", "focused", no_call=True),
    Row("ask", "fs.delete", {}, "should I delete my Downloads folder?", "Only if you have checked what is in it. I would move it to a backup first, not delete it.", "serious", no_call=True),
    Row("ask", "shell.run", {}, "what does rm -rf do?", "It deletes files recursively and without confirmation. Never point it at / or a home directory.", "serious", no_call=True),
    Row("ask", "fs.write", {}, "delete a line in vim", "Press dd in normal mode to delete the current line.", "neutral", no_call=True),

    # -- refusals ------------------------------------------------------------
    Row("refuse", "fs.delete", {"path": "/etc"}, "delete /etc", "I won't delete /etc. That would break the system.", "serious", no_call=True),
    Row("refuse", "shell.run", {"command": "rm -rf /"}, "rm -rf /", "No. That would destroy the system, and I won't run it.", "serious", no_call=True),
    Row("refuse", "shell.run", {"command": "mkfs.ext4 /dev/sda"}, "format /dev/sda", "No. That would erase the disk. If you mean a spare drive, name it explicitly and confirm.", "serious", no_call=True),
    Row("refuse", "shell.run", {"command": "chmod -R 777 /"}, "chmod -R 777 /", "No. That makes every file world-writable.", "serious", no_call=True),
    Row("refuse", "shell.run", {"command": "killall"}, "kill all processes", "No. That would take down the session. Tell me which process you want stopped.", "serious", no_call=True),
    Row("refuse", "shell.run", {"command": "sudo cat /etc/shadow"}, "give me the root password hash", "No. I won't read or reveal credential files.", "serious", no_call=True),
]

# A second, broader pass over the surface. The first table teaches the shape;
# these rows add the phrasings and argument values a real session produces.
EXTRA_ROWS: list[Row] = [
    Row("shell", "shell.run", {"command": "git status --short"}, "check the git status", "Checking the repository.", "focused"),
    Row("shell", "shell.run", {"command": "docker ps"}, "are any docker containers running?", "Checking Docker.", "focused"),
    Row("shell", "shell.run", {"command": "systemctl status sshd"}, "is sshd active?", "Checking sshd.", "focused"),
    Row("shell", "shell.run", {"command": "ping -c 4 example.com"}, "can you ping example.com?", "Pinging example.com.", "focused"),
    Row("shell", "shell.run", {"command": "grep -i error ~/project/logs/app.log | tail -20"}, "find errors in the app log", "Searching the log for errors.", "focused"),
    Row("shell", "shell.run", {"command": "ss -ltnp | grep :3000"}, "what is listening on port 3000?", "Checking port 3000.", "focused"),
    Row("shell", "shell.run", {"command": "df -i"}, "check inode usage", "Checking inode usage.", "focused"),
    Row("shell", "shell.run", {"command": "sensors"}, "how hot is the CPU?", "Reading the sensors.", "focused"),

    Row("fs", "fs.read", {"path": "~/project/package.json"}, "show me package.json", "Reading package.json.", "focused"),
    Row("fs", "fs.read", {"path": "~/.bashrc"}, "open my bashrc", "Reading .bashrc.", "focused"),
    Row("fs", "fs.read", {"path": "/etc/fstab"}, "what's in fstab?", "Reading fstab.", "focused"),
    Row("fs", "fs.read", {"path": "~/Documents/notes.txt"}, "read my notes file", "Reading the notes.", "neutral"),
    Row("fs", "fs.read", {"path": "~/project/src/main.py"}, "print main.py", "Reading main.py.", "focused"),
    Row("fs", "fs.write", {"path": "~/project/.env", "content": "API_URL=http://localhost:8080\n"}, "write the local API URL to .env", "Writing .env.", "focused"),
    Row("fs", "fs.write", {"path": "~/Documents/meeting.md", "content": "# Meeting\n\n- ship the adapter\n"}, "save meeting notes", "Saving the notes.", "neutral"),
    Row("fs", "fs.write", {"path": "~/script.sh", "content": "#!/bin/sh\necho hi\n"}, "create a shell script that prints hi", "Writing the script.", "focused"),
    Row("fs", "fs.list", {"path": "~/Pictures"}, "list my pictures", "Listing Pictures.", "neutral"),
    Row("fs", "fs.list", {"path": "/var/log"}, "what's in /var/log?", "Listing /var/log.", "focused"),
    Row("fs", "fs.list", {"path": "~/project/src"}, "show the source files", "Listing the source tree.", "focused"),
    Row("fs", "fs.move", {"src": "~/Downloads/archive.zip", "dst": "~/Documents/archive.zip"}, "move archive.zip to Documents", "Moving the archive.", "neutral"),
    Row("fs", "fs.move", {"src": "~/project/src/old.py", "dst": "~/project/src/legacy.py"}, "rename old.py to legacy.py", "Renaming the module.", "focused"),
    Row("fs", "fs.copy", {"src": "~/Documents/report.docx", "dst": "~/backup/report.docx"}, "copy the report to my backup folder", "Copying the report.", "focused"),
    Row("fs", "fs.copy", {"src": "~/project/config.yml", "dst": "~/project/config.yml.bak"}, "back up config.yml", "Backing up the config.", "focused"),
    Row("fs", "fs.delete", {"path": "~/Downloads/archive.zip"}, "delete archive.zip", "That deletes the archive. Confirm?", "serious"),
    Row("fs", "fs.delete", {"path": "~/project/node_modules", "recursive": True}, "remove node_modules", "That removes the whole node_modules tree. Confirm?", "serious"),
    Row("fs", "fs.delete", {"path": "~/.cache/thumbnails"}, "clear the thumbnail cache", "Deleting the thumbnail cache. Confirm?", "focused"),
    Row("fs", "fs.delete", {"path": "~/Videos/old.mp4"}, "delete old.mp4", "That deletes the video permanently. Confirm?", "serious"),

    Row("proc", "proc.list", {"sort_by": "pid"}, "list processes by PID", "Sorting by PID.", "focused"),
    Row("proc", "proc.kill", {"pid": 1234, "signal": "SIGTERM"}, "stop process 1234", "Sending SIGTERM to 1234. Confirm?", "focused"),
    Row("proc", "proc.kill", {"pid": 777}, "kill PID 777", "That stops PID 777. Confirm?", "focused"),

    Row("app", "app.launch", {"name": "code"}, "open my editor", "Opening VS Code.", "neutral"),
    Row("app", "app.launch", {"name": "nautilus"}, "launch the file manager", "Opening the file manager.", "neutral"),
    Row("app", "app.launch", {"name": "slack"}, "start Slack", "Starting Slack.", "neutral"),
    Row("app", "app.focus", {"name": "gnome-terminal"}, "focus the terminal", "Focusing the terminal.", "focused"),
    Row("app", "app.focus", {"name": "spotify"}, "bring Spotify forward", "Focusing Spotify.", "neutral"),
    Row("app", "app.close", {"name": "firefox"}, "quit the browser", "That closes Firefox. Confirm?", "neutral"),
    Row("app", "app.close", {"name": "gnome-terminal"}, "close the terminal", "That closes the terminal. Confirm?", "neutral"),

    Row("sys", "sys.info", {"what": "memory"}, "how much memory is in use?", "Reading memory usage.", "focused"),
    Row("sys", "sys.info", {"what": "network"}, "show my network info", "Reading the network info.", "focused"),
    Row("sys", "sys.service", {"action": "status", "name": "ssh"}, "is ssh running?", "Checking ssh.", "focused"),
    Row("sys", "sys.service", {"action": "restart", "name": "docker"}, "restart docker", "That restarts Docker. Confirm?", "focused"),
    Row("sys", "sys.service", {"action": "start", "name": "bluetooth"}, "start the bluetooth service", "That starts Bluetooth. Confirm?", "focused"),
    Row("sys", "sys.service", {"action": "stop", "name": "cups"}, "stop the printing service", "That stops CUPS. Confirm?", "serious"),
    Row("sys", "sys.pkg", {"action": "install", "name": "neovim"}, "install neovim", "That installs Neovim. Confirm?", "focused"),
    Row("sys", "sys.pkg", {"action": "update", "name": "system"}, "update all packages", "That updates the system packages. Confirm?", "serious"),
    Row("sys", "sys.pkg", {"action": "remove", "name": "blender"}, "remove blender", "That removes Blender. Confirm?", "serious"),

    Row("ui", "ui.notify", {"title": "Download complete", "body": "The file finished downloading."}, "notify me when the download finishes", "I'll notify you when it completes.", "neutral"),
    Row("ui", "ui.clipboard", {"action": "set", "text": "git status"}, "put git status on my clipboard", "Copying it.", "neutral"),
    Row("ui", "ui.volume", {"action": "set", "level": 70}, "turn the volume up to 70", "Setting the volume.", "neutral"),
    Row("ui", "ui.volume", {"action": "unmute"}, "unmute the sound", "Unmuting.", "neutral"),

    Row("mem", "mem.read", {"key": "name"}, "what's my name?", "Checking my notes.", "warm"),
    Row("mem", "mem.read", {"key": "timezone"}, "what timezone do I use?", "Checking my notes.", "warm"),
    Row("mem", "mem.write", {"key": "timezone", "value": "Asia/Karachi"}, "remember that I'm in Asia/Karachi", "Noted: Asia/Karachi.", "warm"),
    Row("mem", "mem.write", {"key": "name", "value": "Abbas"}, "remember my name is Abbas", "Noted: Abbas.", "warm"),

    Row("ask", "app.launch", {}, "open it", "Which application do you mean?", "neutral", no_call=True),
    Row("ask", "fs.delete", {}, "delete the old one", "Which path is the old one?", "neutral", no_call=True),
    Row("ask", "shell.run", {}, "run that command", "Which command should I run?", "focused", no_call=True),
    Row("ask", "ui.volume", {}, "change the volume", "To what level, zero to one hundred?", "neutral", no_call=True),
    Row("ask", "sys.pkg", {}, "install it", "Which package do you want installed?", "focused", no_call=True),

    Row("refuse", "shell.run", {"command": "wipefs -a /dev/sda"}, "wipe the disk", "No. That erases the disk signature and the data with it.", "serious", no_call=True),
    Row("refuse", "fs.delete", {"path": "~", "recursive": True}, "delete my home folder", "No. That would remove your entire home directory.", "serious", no_call=True),
    Row("refuse", "shell.run", {"command": "sudo rm -rf /usr"}, "sudo rm -rf /usr", "No. That would destroy the system's programs.", "serious", no_call=True),
    Row("refuse", "sys.service", {"action": "stop", "name": "firewalld"}, "turn off the firewall", "No. I won't disable the firewall without an explicit admin workflow.", "serious", no_call=True),
]

ROWS += EXTRA_ROWS


def _system(persona: str) -> str:
    return persona + "\n\n" + TOOL_PROMPT


def _messages(row: Row, persona: str) -> list[dict]:
    system = row.system or _system(persona)
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": row.user},
        {"role": "assistant", "content": _assistant(row)},
    ]
    if row.result is not None and not row.no_call:
        msgs.append({"role": "tool", "content": json.dumps({"ok": True, "result": row.result}, ensure_ascii=False)})
        msgs.append({"role": "assistant", "content": f"[emotion:{row.emotion}]{row.result_speech or 'Done.'}"})
    return msgs


def _bucket(name: str, salt: str = "tools-v1") -> float:
    h = hashlib.sha256(f"{salt}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


# Paraphrases of the same request stay in the same split. They add phrasing
# variety (the model has to learn the intent, not one sentence) without
# leaking a train prompt into eval.
IMPERATIVE_VERBS = {
    "open", "close", "launch", "start", "stop", "show", "list", "read",
    "write", "save", "move", "copy", "delete", "remove", "set", "mute",
    "unmute", "lock", "reboot", "shut", "install", "restart", "kill",
    "send", "remind", "remember", "check", "find", "print", "run",
    "bring", "switch", "clean", "fix", "turn", "clear", "quit", "update",
    "ping", "wipe", "format", "focus", "notify", "put", "create", "back",
    "rename", "duplicate", "suspend", "give",
}


def _variants(row: Row):
    yield row
    text = row.user.strip()
    if text.endswith("?") or text.endswith("."):
        return
    first = text.split()[0].lower() if text.split() else ""
    if first not in IMPERATIVE_VERBS:
        return
    for prefix in ("please ", "can you ", "could you "):
        yield replace(row, user=prefix + text[0].lower() + text[1:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DATA)
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    persona = load_persona()
    rows = list(ROWS)

    # Validate every call against the schema before writing anything.
    problems = []
    for row in rows:
        if row.no_call:
            continue
        errs = validate_call(row.fn, row.args)
        if errs:
            problems.append((row.user, errs))
    if problems:
        for user, errs in problems:
            print(f"INVALID: {user!r}: {errs}", file=sys.stderr)
        return 1

    # Stratified split by group so every namespace is represented everywhere.
    by_group: dict[str, list[Row]] = {}
    for row in rows:
        by_group.setdefault(row.group, []).append(row)
    splits = {"train": [], "valid": [], "eval": []}
    for group in sorted(by_group):
        items = sorted(by_group[group], key=lambda r: _bucket(r.user))
        n = len(items)
        n_eval = max(1, round(n * args.eval_frac)) if n >= 4 else 0
        n_valid = max(1, round(n * args.valid_frac)) if n >= 8 else 0
        splits["eval"].extend(items[:n_eval])
        splits["valid"].extend(items[n_eval : n_eval + n_valid])
        splits["train"].extend(items[n_eval + n_valid :])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, items in splits.items():
        items = [variant for row in items for variant in _variants(row)]
        items.sort(key=lambda r: (r.group, r.user))
        write_jsonl(args.out_dir / f"{name}.jsonl", [
            {"id": f"tool:{hashlib.sha1((r.user + r.fn).encode()).hexdigest()[:12]}",
             "messages": _messages(r, persona)}
            for r in items
        ])
        counts[name] = len(items)

    write_json(args.out_dir / "report.json", {
        "rows": len(rows),
        "splits": counts,
        "functions": sorted({r.fn for r in rows if not r.no_call}),
        "groups": sorted(by_group),
        "tool_prompt_chars": len(TOOL_PROMPT),
    })
    print("tool examples:", counts)
    for name in ("train", "valid", "eval"):
        print(f"  {name}: {args.out_dir / (name + '.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
