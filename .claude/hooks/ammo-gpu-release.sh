#!/bin/bash
# PostToolUse hook — AMMO GPU pool auto-release.
# Detects the reservation pattern in completed commands and releases
# all GPUs held by this session.
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // .input.command // empty' 2>/dev/null) || true
[ -z "$COMMAND" ] && exit 0

# Only release for commands that used the reservation pattern
if ! echo "$COMMAND" | grep -qP '(?<!\S)gpu_reservation\.py\s+reserve(?:\s|$)'; then
    exit 0
fi

# Extract explicit --session-id from the command if present; fall back to CLAUDE_SESSION_ID
SESSION_ID=$(echo "$COMMAND" | grep -oP '(?<=--session-id\s)\S+' 2>/dev/null) || true
[ -z "$SESSION_ID" ] && SESSION_ID="${CLAUDE_SESSION_ID:-}"
[ -z "$SESSION_ID" ] && exit 0

GPU_RES_DIR="${AMMO_GPU_RES_DIR:-/tmp/ammo_gpu_res}"
GPU_STATE="$GPU_RES_DIR/state.json"
[ -f "$GPU_STATE" ] || exit 0

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../skills/ammo/scripts" 2>/dev/null && pwd)" || \
SCRIPTS_DIR="${CLAUDE_PROJECT_DIR:-.}/.claude/skills/ammo/scripts"

python3 "$SCRIPTS_DIR/gpu_reservation.py" release-session --session-id "$SESSION_ID" 2>&1 || \
    echo "AMMO GPU PostToolUse: release failed — will expire via lease." >&2

exit 0
