#!/usr/bin/env bash
# Check the Kaggle runs and pull their outputs.
#
# The token is read from $KAGGLE_API_TOKEN, or from ~/.kaggle/access_token.
# Generate one at https://www.kaggle.com/settings/api - it is short lived, so
# expect to refresh it.
#
#   KAGGLE_API_TOKEN=KGAT_xxx ./tools/kaggle_results.sh
#
# Or set it once:
#   mkdir -p ~/.kaggle && echo KGAT_xxx > ~/.kaggle/access_token
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE="$ROOT/.venv/bin/kaggle"
[ -x "$KAGGLE" ] || KAGGLE="$(command -v kaggle)"

if [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -f "$HOME/.kaggle/access_token" ]; then
  export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/access_token")"
fi
if [ -z "${KAGGLE_API_TOKEN:-}" ]; then
  echo "no token. Set KAGGLE_API_TOKEN or write ~/.kaggle/access_token."
  echo "Generate one at https://www.kaggle.com/settings/api"
  exit 1
fi

echo "=== quota ==="
"$KAGGLE" quota 2>&1 | tail -3

for k in agents3-emotion-training agents3-voice-bank; do
  echo
  echo "=== $k ==="
  "$KAGGLE" kernels status "maisumabbas77/$k" 2>&1 | tail -1
done

fetch () {
  local kernel="$1" dest="$2"
  echo
  echo "=== downloading $kernel -> $dest ==="
  mkdir -p "$dest"
  local tmp
  tmp="$(mktemp -d)"
  "$KAGGLE" kernels output "maisumabbas77/$kernel" -p "$tmp" 2>&1 | tail -3
  local z
  z="$(find "$tmp" -name '*.zip' | head -1)"
  if [ -n "$z" ]; then
    unzip -o -q "$z" -d "$dest" && echo "extracted to $dest"
    find "$dest" -type f | head -30
  else
    echo "no archive - the kernel may not have produced output yet"
  fi
  rm -rf "$tmp"
}

fetch agents3-emotion-training "$ROOT/models/llm/agents3_emotion_adapter"
fetch agents3-voice-bank "$ROOT/models/voice/bank"
