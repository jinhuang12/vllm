#!/bin/bash
# PostToolUse hook — AMMO GPU reservation auto-release.
# Releases GPU reservations made by PreToolUse after commands complete.
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

GPU_RES_DIR="${AMMO_GPU_RES_DIR:-/tmp/ammo_gpu_res}"
GPU_STATE="$GPU_RES_DIR/state.json"
[ -f "$GPU_STATE" ] || exit 0

INPUT=$(cat)
# Extract command — try both field names (same as pretool hook)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // .input.command // empty' 2>/dev/null) || true

if [ -z "$COMMAND" ]; then
    echo "AMMO GPU PostToolUse: could not extract command — reservation not released. Will expire via lease." >&2
    exit 0
fi

# Only release for commands with CUDA_VISIBLE_DEVICES=<digits>
if ! echo "$COMMAND" | grep -qP 'CUDA_VISIBLE_DEVICES=[\d,]+'; then
    exit 0
fi

# Compute command hash and release via shared module
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../skills/ammo/scripts" 2>/dev/null && pwd)" || \
SCRIPTS_DIR="${CLAUDE_PROJECT_DIR:-.}/.claude/skills/ammo/scripts"

CMD_HASH=$(echo -n "$COMMAND" | python3 -c "import sys,hashlib; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()[:16])")

python3 -c "
import sys, os
sys.path.insert(0, '$SCRIPTS_DIR')
os.environ.setdefault('AMMO_GPU_RES_DIR', '$GPU_RES_DIR')
from gpu_reservation import release_by_hash
release_by_hash('$CMD_HASH')
" 2>&1 || echo "AMMO GPU PostToolUse: release failed — will expire via lease." >&2

exit 0
