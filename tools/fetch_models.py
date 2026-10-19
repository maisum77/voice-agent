"""Fetch model files from HuggingFace over a flaky link.

The standard ``huggingface_hub`` downloader stalls on this network: its
read timeouts abort a multi-gigabyte transfer part-way and the retry logic
then re-resolves the CDN URL instead of resuming the bytes it already has.
``curl -C -`` resumes from the byte offset, which is exactly what a throttled
and dropping connection needs.

For every file we know the expected size from the Hub API, so the loop can
tell "done" from "still going" and keep going until the size matches.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_REPOS = {
    "teacher": "mlx-community/Qwen3.5-9B-4bit",
    "mlx_base": "mlx-community/Qwen3.5-0.8B-MLX-bf16",
    # Runtime TTS: clones the voice from the bank clips.
    "tts_base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    # Build-time only: renders the emotion reference clips. Never shipped.
    "tts_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

# Small files we never need to bother resuming.
SKIP_PATTERNS = (".gitattributes", ".md")


def repo_files(repo: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    out = []
    for sib in data.get("siblings", []):
        name = sib["rfilename"]
        if any(name.endswith(p) for p in SKIP_PATTERNS):
            continue
        size = sib.get("size")
        if size is None:
            try:
                req = urllib.request.Request(
                    f"https://huggingface.co/{repo}/resolve/main/{name}",
                    method="HEAD",
                    headers={"User-Agent": "curl/8"},
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    size = int(resp.headers.get("Content-Length", 0))
            except Exception:
                size = 0
        out.append({"name": name, "size": size or 0})
    return out


def fetch(repo: str, name: str, dest: Path, expected: int, attempts: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    for attempt in range(attempts):
        if dest.exists() and expected and dest.stat().st_size >= expected:
            return True
        cmd = [
            # HTTP/1.1 rather than HTTP/2: the CDN keeps cancelling HTTP/2
            # streams on this link, and each cancel costs a reconnect.
            "curl", "--http1.1", "-L", "-C", "-", "--fail", "--silent",
            "--show-error", "--connect-timeout", "30", "--max-time", "7200",
            "-o", str(dest), url,
        ]
        # No --speed-limit: on a slow but steady link, aborting for being slow
        # just throws away the connection and the bytes already transferred.
        subprocess.run(cmd, check=False)
        have = dest.stat().st_size if dest.exists() else 0
        if expected and have >= expected:
            return True
        if not expected and have > 0:
            return True
        print(f"    retry {attempt + 1}/{attempts}: {name} at {have / 1e6:.1f} MB")
        time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=None, help="explicit repo id")
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("--targets", default="teacher,mlx_base", help="named targets")
    args = ap.parse_args()

    jobs = (
        [(args.repo, args.dest)]
        if args.repo
        else [(DEFAULT_REPOS[t], Path("models") / t / DEFAULT_REPOS[t].split("/")[-1])
              for t in args.targets.split(",") if t in DEFAULT_REPOS]
    )

    for repo, dest in jobs:
        print(f"== {repo} -> {dest}")
        files = repo_files(repo)
        total = sum(f["size"] for f in files)
        print(f"   {len(files)} files, {total / 1e9:.2f} GB")
        t0 = time.time()
        for f in sorted(files, key=lambda x: -x["size"]):
            name = f["name"]
            target = dest / name
            if target.exists() and f["size"] and target.stat().st_size >= f["size"]:
                print(f"   have {name} ({target.stat().st_size / 1e6:.1f} MB)")
                continue
            print(f"   get  {name} ({f['size'] / 1e6:.1f} MB)")
            ok = fetch(repo, name, target, f["size"])
            if not ok:
                print(f"   FAILED {name}")
                return 1
            el = time.time() - t0
            got = sum(x.stat().st_size for x in dest.rglob("*") if x.is_file())
            print(f"        {el / 60:.1f}m elapsed, {got / 1e6:.0f} MB total, "
                  f"{got / 1e6 / max(el, 1):.2f} MB/s")
        print(f"   done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
