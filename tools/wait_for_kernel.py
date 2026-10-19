"""Wait for a Kaggle kernel to finish, then pull its output.

    python3 tools/wait_for_kernel.py maisumabbas77/agents3-emotion-training \
        --out models/llm/agents3_emotion_adapter

Prints the kernel's log as it goes, so a failure is visible rather than
discovered twenty minutes later, and the output is downloaded the moment the
run finishes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"

FINISHED = {
    "KernelWorkerStatus.COMPLETE",
    "KernelWorkerStatus.ERROR",
    "KernelWorkerStatus.CANCELLED",
    "KernelWorkerStatus.DEAD",
}


def kaggle(*args: str, check: bool = False) -> str:
    env = dict(os.environ)
    env["PATH"] = f"{ROOT / '.venv' / 'bin'}:{env.get('PATH', '')}"
    proc = subprocess.run(
        [str(KAGGLE), *args], capture_output=True, text=True, env=env, check=check
    )
    return (proc.stdout or "") + (proc.stderr or "")


def status(kernel: str) -> str:
    out = kaggle("kernels", "status", kernel)
    for token in out.split():
        if token.startswith("KernelWorkerStatus."):
            return token.rstrip('."')
    return out.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kernel")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=60 * 60 * 4)
    args = ap.parse_args()

    t0 = time.time()
    logged = 0
    while True:
        st = status(args.kernel)
        elapsed = int(time.time() - t0)
        print(f"[{elapsed // 60:3d}m] {st}", flush=True)

        if st in FINISHED:
            break
        if elapsed > args.timeout:
            print("timed out")
            return 2
        time.sleep(args.poll)

    # Pull the log and the output.
    log = kaggle("kernels", "logs", args.kernel)
    print("---- log tail ----")
    print("\n".join(log.splitlines()[-60:]))
    print("------------------")

    outdir = args.out or (ROOT / "models" / "llm" / "agents3_emotion_adapter")
    outdir.mkdir(parents=True, exist_ok=True)
    zips = list((ROOT / "dist").glob("kernel_output_*.zip")) + list(
        Path("/tmp").glob("kernel_output_*.zip")
    )
    dl = Path("/tmp") / f"kernel_output_{int(time.time())}.zip"
    res = kaggle("kernels", "output", args.kernel, "-p", str(dl.parent), "-o", dl.name)
    if not dl.exists():
        print(res)
        print("no output archive - the kernel probably failed")
        return 1

    with zipfile.ZipFile(dl) as z:
        names = z.namelist()
        print(f"output archive: {len(names)} files")
        for n in names[:40]:
            print("   ", n)
        z.extractall(outdir)
    print(f"extracted -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
