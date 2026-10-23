"""Expand the tool dataset for v3 and build a fair unseen-intent eval.

The v2 tool eval was a memorisation test: the split held out the one row that
defined the canonical argument for an intent (for example ``main.py`` ->
``~/project/src/main.py``), so the model could not possibly know the path.
This script keeps that file as a legacy regression, adds differently-worded
training rows for every intent it contains, and builds a separate eval from
intents and slot values that are not in the training set. The unseen eval
mostly uses explicit arguments, so it measures the function choice and the
copying of arguments rather than a hidden convention.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "pipeline" / "tools"))

from common import load_persona, read_jsonl, write_json, write_jsonl  # noqa: E402
from schema import TOOL_PROMPT, validate_call  # noqa: E402

PREFIXES = ("", "can you ", "could you ", "please ", "would you ", "kindly ")
OUT = ROOT / "data" / "tool" / "v3"


def call(fn: str, args: dict) -> str:
    return "<tool_call>" + json.dumps(
        {"name": fn, "arguments": args}, ensure_ascii=False, separators=(",", ":")
    ) + "</tool_call>"


def row(user: str, fn: str | None, args: dict | None, speech: str, emotion: str) -> dict:
    if fn is None:
        body = f"[emotion:{emotion}]{speech}"
    else:
        body = f"[emotion:{emotion}]{call(fn, args or {})}{speech}"
    return {
        "messages": [
            {"role": "system", "content": ""},  # filled in main
            {"role": "user", "content": user},
            {"role": "assistant", "content": body},
        ]
    }


PREFIXABLE = {
    "open", "close", "launch", "start", "stop", "show", "list", "read",
    "write", "save", "move", "copy", "delete", "remove", "set", "mute",
    "unmute", "lock", "install", "restart", "kill", "notify", "remember",
    "run", "bring", "switch", "clear", "focus", "put", "create", "turn",
    "check", "find", "print", "clean", "fix", "format", "wipe", "update",
    "rename", "duplicate", "send", "remind", "give",
}


def variants(user: str, fn: str | None, args: dict | None, speech: str, emotion: str) -> list[dict]:
    out = []
    text = user.strip()
    first = text.split()[0].lower() if text.split() else ""
    # A prefix only makes sense on an imperative. "can you how much disk
    # space is left?" is the kind of noise that teaches the wrong thing.
    if first not in PREFIXABLE or text.endswith("?"):
        return [row(text, fn, args, speech, emotion)]
    for prefix in PREFIXES:
        if prefix and first in {"can", "could", "please", "would", "kindly"}:
            continue
        u = (prefix + text[0].lower() + text[1:]) if prefix else text
        out.append(row(u, fn, args, speech, emotion))
    return out


def intent_rows() -> tuple[list[dict], list[dict], list[dict]]:
    """Return (train, valid, unseen_eval) rows before the system prompt."""
    train: list[dict] = []
    valid: list[dict] = []
    unseen: list[dict] = []

    # ---- explicit-argument coverage for every function -------------------
    # The user text contains the path/pid/level, so the model has to copy it.
    focus_apps = ["gnome-terminal", "spotify", "code", "firefox", "slack", "nautilus"]
    for app in focus_apps:
        train += variants(f"bring {app} to the front", "app.focus", {"name": app},
                          f"Focusing {app}.", "focused")
    launch_apps = ["slack", "gimp", "thunderbird", "gimp", "calculator", "obs"]
    for app in launch_apps:
        train += variants(f"open {app}", "app.launch", {"name": app},
                          f"Opening {app}.", "neutral")
    close_apps = ["spotify", "firefox", "slack", "thunderbird"]
    for app in close_apps:
        train += variants(f"close {app}", "app.close", {"name": app},
                          f"Closing {app}. Confirm?", "neutral")
    read_paths = [
        "~/project/src/main.py", "~/project/logs/app.log", "~/notes.md",
        "~/.config/app/config.toml", "~/Documents/budget.csv", "/etc/hosts",
        "~/project/README.md", "/var/log/syslog",
    ]
    for path in read_paths:
        train += variants(f"read {path}", "fs.read", {"path": path},
                          f"Reading {path}.", "focused")
    copy_pairs = [
        ("~/project/config.yml", "~/project/config.yml.bak"),
        ("~/photos/a.jpg", "~/photos/a-copy.jpg"),
        ("~/Documents/report.docx", "~/backup/report.docx"),
        ("~/project/main.py", "~/project/main.py.bak"),
        ("~/Downloads/archive.zip", "~/Documents/archive.zip"),
    ]
    for src, dst in copy_pairs:
        train += variants(f"copy {src} to {dst}", "fs.copy", {"src": src, "dst": dst},
                          "Copying the file.", "focused")
    delete_paths = [
        "~/.cache/thumbnails", "/tmp/voice-agent", "~/Downloads/old.iso",
        "~/project/node_modules", "~/Videos/old.mp4", "~/Downloads/archive.zip",
    ]
    for path in delete_paths:
        train += variants(f"delete {path}", "fs.delete", {"path": path},
                          "Deleting it. Confirm?", "focused")
    write_rows = [
        ("~/script.sh", "#!/bin/sh\necho hi\n", "create a shell script that prints hi"),
        ("~/notes/todo.md", "- buy milk\n- call Sam", "write a todo note: buy milk and call Sam"),
        ("~/hello.py", "print('hello')\n", "save a Python script that prints hello"),
        ("~/.config/app/theme.toml", 'theme = "dark"\n', "set the app theme to dark"),
    ]
    for path, content, user in write_rows:
        train += variants(user, "fs.write", {"path": path, "content": content},
                          "Writing the file.", "focused")
    for pid in (777, 4821, 9912, 1234, 4242, 5050):
        train += variants(f"kill PID {pid}", "proc.kill", {"pid": pid},
                          f"Stopping PID {pid}. Confirm?", "focused")
    services = [
        ("restart", "pipewire"), ("restart", "docker"), ("status", "sshd"),
        ("start", "bluetooth"), ("stop", "cups"), ("restart", "nginx"),
    ]
    for action, name in services:
        train += variants(f"{action} the {name} service", "sys.service",
                          {"action": action, "name": name},
                          f"{action.title()}ing {name}. Confirm?", "focused")
    for level in (10, 25, 40, 55, 70, 85):
        train += variants(f"set the volume to {level}", "ui.volume",
                          {"action": "set", "level": level},
                          "Setting the volume.", "neutral")
    train += variants("mute the sound", "ui.volume", {"action": "mute"},
                      "Muting the volume.", "neutral")
    train += variants("unmute the sound", "ui.volume", {"action": "unmute"},
                      "Unmuting.", "neutral")
    mem_rows = [
        ("name", "Abbas", "remember my name is Abbas"),
        ("editor", "Neovim", "remember that I use Neovim"),
        ("timezone", "Asia/Karachi", "remember that I'm in Asia/Karachi"),
        ("project", "Voice", "remember that my project is called Voice"),
        ("favorite_color", "blue", "remember my favourite colour is blue"),
    ]
    for key, value, user in mem_rows:
        train += variants(user, "mem.write", {"key": key, "value": value},
                          f"Noted: {value}.", "warm")
    commands = [
        ("df -h", "how much disk space is left?"),
        ("free -h", "how much RAM is free?"),
        ("uname -a", "which kernel am I running?"),
        ("docker ps", "are any docker containers running?"),
        ("uptime", "how long has the machine been up?"),
        ("git status --short", "check the git status"),
    ]
    for command, user in commands:
        train += variants(user, "shell.run", {"command": command},
                          "Running the command.", "focused")
    for what in ("cpu", "memory", "disk", "battery", "network", "gpu"):
        train += variants(f"show {what} information", "sys.info", {"what": what},
                          f"Reading {what} info.", "focused")

    # ---- underrepresented functions -------------------------------------
    for path in ["~/Downloads", "~/project", "~/Pictures", "/var/log", "~/Music",
                 "~/Documents", "~/.config", "/tmp"]:
        train += variants(f"list {path}", "fs.list", {"path": path},
                          f"Listing {path}.", "neutral")
    for src, dst in [
        ("~/report.pdf", "~/Documents/report.pdf"),
        ("~/old.txt", "~/archive.txt"),
        ("~/Downloads/a.zip", "~/Documents/a.zip"),
        ("~/project/src/old.py", "~/project/src/legacy.py"),
        ("~/a.txt", "~/b.txt"),
    ]:
        train += variants(f"move {src} to {dst}", "fs.move", {"src": src, "dst": dst},
                          "Moving the file.", "neutral")
    for sort_by in ["cpu", "memory", "pid", "name", "time"]:
        train += variants(f"list processes by {sort_by}", "proc.list", {"sort_by": sort_by},
                          f"Sorting by {sort_by}.", "focused")
    for action in ["get", "set"]:
        for text in (["clipboard"] if action == "get" else ["hello", "git status", "https://example.com"]):
            args = {"action": action}
            if action == "set":
                args["text"] = text
            train += variants(
                "what's on my clipboard?" if action == "get" else f"copy {text} to the clipboard",
                "ui.clipboard", args, "Clipboard updated." if action == "set" else "Reading the clipboard.",
                "neutral")
    for title, body, user in [
        ("Build finished", "The build completed successfully.", "notify me when the build finishes"),
        ("Download complete", "The file finished downloading.", "tell me when the download is done"),
        ("Tea", "The tea is ready.", "remind me that the tea is ready"),
        ("Meeting", "The meeting starts in five minutes.", "notify me before the meeting"),
    ]:
        train += variants(user, "ui.notify", {"title": title, "body": body},
                          "Sending the notification.", "neutral")
    for action in ["lock", "suspend", "reboot", "shutdown"]:
        train += variants(f"{action} the machine", "sys.power", {"action": action},
                          f"{action.title()}ing the machine. Confirm?", "neutral" if action == "lock" else "serious")
    for action, name in [("install", "ripgrep"), ("install", "neovim"), ("remove", "nodejs"),
                         ("update", "system"), ("install", "htop"), ("remove", "blender")]:
        train += variants(f"{action} {name}", "sys.pkg", {"action": action, "name": name},
                          f"{action.title()}ing {name}. Confirm?", "focused")
    for key in ["name", "editor", "timezone", "project", "favourite_colour", "shell"]:
        train += variants(f"what do you remember about my {key}?", "mem.read", {"key": key},
                          "Checking my notes.", "warm")

    # ---- legacy eval intents, differently worded -------------------------
    # The old eval file stays in the repo as a regression, but every intent in
    # it is represented here by a different sentence.
    legacy = {
        "app.focus": ("switch to gnome-terminal", {"name": "gnome-terminal"}, "focused"),
        "app.launch": ("open slack", {"name": "slack"}, "neutral"),
        "fs.copy": ("make a backup of ~/project/config.yml", {"src": "~/project/config.yml", "dst": "~/project/config.yml.bak"}, "focused"),
        "fs.read": ("show me ~/project/src/main.py", {"path": "~/project/src/main.py"}, "focused"),
        "fs.write": ("write a script at ~/script.sh that prints hi", {"path": "~/script.sh", "content": "#!/bin/sh\necho hi\n"}, "focused"),
        "mem.write": ("save that my name is Abbas", {"key": "name", "value": "Abbas"}, "warm"),
        "proc.kill": ("stop process 777", {"pid": 777}, "focused"),
        "shell.run": ("run docker ps", {"command": "docker ps"}, "focused"),
        "sys.info": ("what is the cpu doing?", {"what": "cpu"}, "focused"),
        "sys.service": ("restart docker for me", {"action": "restart", "name": "docker"}, "focused"),
        "ui.volume": ("turn the volume to 40", {"action": "set", "level": 40}, "neutral"),
    }
    for fn, (user, args, emo) in legacy.items():
        train += variants(user, fn, args, "Done.", emo)

    # ---- no-call, ambiguous, and refusals --------------------------------
    no_calls = [
        ("run the script", "Which script? Give me the full path.", "focused"),
        ("execute the script for me", "Which script? Give me the full path.", "focused"),
        ("delete a line in vim", "Press dd in normal mode to delete the current line.", "neutral"),
        ("how do I remove a line in vim?", "Press dd in normal mode to delete the current line.", "neutral"),
        ("delete my home folder", "No. That would remove your entire home directory.", "serious"),
        ("erase my home directory", "No. That would remove your entire home directory.", "serious"),
        ("sudo rm -rf /usr", "No. That would destroy the system's programs.", "serious"),
        ("rm -rf /usr with sudo", "No. That would destroy the system's programs.", "serious"),
        ("clean up my computer", "What should I clean: caches, old downloads, or the build folders?", "focused"),
        ("fix it", "What's the error? Paste it or tell me what failed.", "warm"),
        ("open it", "Which application do you mean?", "neutral"),
        ("install it", "Which package do you want installed?", "focused"),
        ("run the tests", "Which test suite, and in which project?", "focused"),
        ("delete that", "Which path do you mean?", "neutral"),
        ("format the disk", "No. That would erase the disk. If you mean a spare drive, name it explicitly.", "serious"),
    ]
    for user, speech, emo in no_calls:
        train += variants(user, None, None, speech, emo)

    # ---- valid: a small held-out slice of the same intents --------------
    valid += variants("bring inkscape to the front", "app.focus", {"name": "inkscape"},
                      "Focusing Inkscape.", "focused")
    valid += variants("read ~/Documents/notes.txt", "fs.read", {"path": "~/Documents/notes.txt"},
                      "Reading the notes.", "focused")
    valid += variants("stop process 6060", "proc.kill", {"pid": 6060},
                      "Stopping 6060. Confirm?", "focused")
    valid += variants("set the volume to 30", "ui.volume", {"action": "set", "level": 30},
                      "Setting the volume.", "neutral")
    valid += variants("run the script", None, None, "Which script? Give me the full path.", "focused")

    # ---- unseen eval: new slots and new wording, mostly explicit -------
    # None of these prompts appears in train/valid. The point is to check
    # function choice and argument copying, not a hidden convention: most
    # rows put the path/PID/level directly in the user text.
    unseen_specs = [
        ("open the file at ~/Documents/taxes-2025.csv", "fs.read",
         {"path": "~/Documents/taxes-2025.csv"}, "focused", "Reading the tax file."),
        ("save the text 'draft agenda' in ~/notes/meeting-2026.md", "fs.write",
         {"path": "~/notes/meeting-2026.md", "content": "draft agenda\n"}, "focused", "Writing the meeting note."),
        ("duplicate ~/Pictures/logo.svg as ~/Pictures/logo-final.svg", "fs.copy",
         {"src": "~/Pictures/logo.svg", "dst": "~/Pictures/logo-final.svg"}, "focused", "Copying the logo."),
        ("remove ~/Downloads/ubuntu-24.04.iso", "fs.delete",
         {"path": "~/Downloads/ubuntu-24.04.iso"}, "focused", "Deleting the ISO. Confirm?"),
        ("what is inside /srv/data?", "fs.list",
         {"path": "/srv/data"}, "neutral", "Listing /srv/data."),
        ("relocate ~/Documents/draft.docx to ~/Archive/draft.docx", "fs.move",
         {"src": "~/Documents/draft.docx", "dst": "~/Archive/draft.docx"}, "neutral", "Moving the draft."),
        ("start inkscape", "app.launch",
         {"name": "inkscape"}, "neutral", "Opening Inkscape."),
        ("bring the calculator window forward", "app.focus",
         {"name": "calculator"}, "focused", "Focusing the calculator."),
        ("quit gimp", "app.close",
         {"name": "gimp"}, "neutral", "Closing GIMP. Confirm?"),
        ("terminate PID 7331", "proc.kill",
         {"pid": 7331}, "focused", "Stopping 7331. Confirm?"),
        ("show me the processes using the most memory", "proc.list",
         {"sort_by": "memory"}, "focused", "Sorting by memory."),
        ("execute `ip addr`", "shell.run",
         {"command": "ip addr"}, "focused", "Running the command."),
        ("how much battery is left?", "sys.info",
         {"what": "battery"}, "neutral", "Reading battery info."),
        ("stop the postgresql service", "sys.service",
         {"action": "stop", "name": "postgresql"}, "focused", "Stopping PostgreSQL. Confirm?"),
        ("remove gimp", "sys.pkg",
         {"action": "remove", "name": "gimp"}, "focused", "Removing GIMP. Confirm?"),
        ("reboot the computer", "sys.power",
         {"action": "reboot"}, "serious", "Rebooting. Confirm?"),
        ("set the output volume to 15", "ui.volume",
         {"action": "set", "level": 15}, "neutral", "Setting the volume."),
        ("ping me when the render finishes", "ui.notify",
         {"title": "Render finished", "body": "The render completed successfully."}, "warm", "Sending the notification."),
        ("copy 'kubectl get pods' to the clipboard", "ui.clipboard",
         {"action": "set", "text": "kubectl get pods"}, "neutral", "Copying it."),
        ("remember that my shell is zsh", "mem.write",
         {"key": "shell", "value": "zsh"}, "warm", "Noted: zsh."),
        ("what is my keyboard layout?", "mem.read",
         {"key": "keyboard_layout"}, "warm", "Checking my notes."),
    ]
    for user, fn, args, emo, speech in unseen_specs:
        unseen.append(row(user, fn, args, speech, emo))
    unseen_nocall = [
        ("launch the app", "Which application do you mean?", "neutral"),
        ("remove the old file", "Which path do you mean?", "neutral"),
        ("reinstall everything", "That would reinstall every package. Tell me which package you actually need.", "serious"),
        ("wipe the drive", "No. That would erase the drive. Name the exact device if you mean a spare one.", "serious"),
        ("run the migration", "Which migration, and against which database? I will not run it without the target and a backup.", "focused"),
        ("optimize the system", "What should I optimize: startup, memory, or disk?", "focused"),
        ("delete the temporary files", "Which directory, and which files? I will not guess at a delete.", "focused"),
        ("install the dependencies", "For which project and which package manager?", "focused"),
        ("restart the service", "Which service do you mean?", "focused"),
        ("open the file", "Which file? Give me the path.", "neutral"),
        ("change the setting", "Which setting, and what value should it have?", "focused"),
    ]
    for user, speech, emo in unseen_nocall:
        unseen.append(row(user, None, None, speech, emo))
    return train, valid, unseen


def fill_system(rows: list[dict], system: str) -> None:
    for r in rows:
        r["messages"][0]["content"] = system


def check(rows: list[dict], name: str) -> None:
    problems = []
    for r in rows:
        body = r["messages"][-1]["content"]
        for m in re.finditer(r"<tool_call>(.*?)</tool_call>", body):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError as exc:
                problems.append((r["messages"][1]["content"], f"json: {exc}"))
                continue
            errs = validate_call(obj.get("name", ""), obj.get("arguments", {}))
            if errs:
                problems.append((r["messages"][1]["content"], errs))
    if problems:
        for user, errs in problems[:10]:
            print(f"INVALID {name}: {user!r}: {errs}", file=sys.stderr)
        raise SystemExit(f"{len(problems)} invalid {name} rows")
    print(f"{name}: {len(rows)} rows, all calls schema-valid")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    persona = load_persona()
    system = persona + "\n\n" + TOOL_PROMPT
    new_train, new_valid, unseen = intent_rows()
    fill_system(new_train, system)
    fill_system(new_valid, system)
    fill_system(unseen, system)

    # Keep the existing hand-written data and add the new rows. The old eval
    # is copied as a legacy regression, not used as the shipping gate.
    base_train = read_jsonl(ROOT / "data/tool/train.jsonl")
    base_valid = read_jsonl(ROOT / "data/tool/valid.jsonl")
    base_eval = read_jsonl(ROOT / "data/tool/eval.jsonl")
    train = base_train + new_train
    valid = base_valid + new_valid

    check(train, "train")
    check(valid, "valid")
    check(unseen, "unseen")

    def _user(r: dict) -> str:
        return next(m["content"] for m in r["messages"] if m["role"] == "user")

    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    train_users = {_norm(_user(r)) for r in train + valid}
    unseen_users = {_norm(_user(r)) for r in unseen}
    leaked = sorted(train_users & unseen_users)
    if leaked:
        raise SystemExit(f"unseen eval leaks into train: {leaked[:5]}")
    legacy_users = {_norm(_user(r)) for r in base_eval}
    print(f"unseen/training exact-prompt overlap: {len(leaked)}")
    print(f"legacy/training exact-prompt overlap: {len(train_users & legacy_users)} "
          "(expected: legacy is a seen-intent regression)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    write_jsonl(args.out_dir / "eval_legacy.jsonl", base_eval)
    write_jsonl(args.out_dir / "eval.jsonl", unseen)
    write_json(args.out_dir / "report.json", {
        "train": len(train),
        "valid": len(valid),
        "legacy_eval": len(base_eval),
        "unseen_eval": len(unseen),
        "new_train": len(new_train),
        "new_valid": len(new_valid),
    })
    print(f"wrote {len(train)} train / {len(valid)} valid / {len(unseen)} unseen eval -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
