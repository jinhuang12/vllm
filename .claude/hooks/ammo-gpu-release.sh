#!/bin/bash
# PostToolUse hook — AMMO GPU pool auto-release.
# Detects the reservation pattern in completed commands and releases
# all GPUs held by this session.
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // .input.command // empty' 2>/dev/null) || true
[ -z "$COMMAND" ] && exit 0

# Only release for commands that used the reservation pattern.
# Pre-filter: match `gpu_reservation.py reserve` regardless of path prefix
# (`python gpu_reservation.py reserve`, `python .claude/.../gpu_reservation.py reserve`, etc.)
if ! echo "$COMMAND" | grep -qP 'gpu_reservation\.py\s+reserve(?:\s|$)'; then
    exit 0
fi

# Extract ALL --session-id values from the command (supports both `--session-id foo`
# and `--session-id=foo` forms). Bounded char class stops at shell metachars like
# `)`, `&`, `;`, `"`, `'`, so wrapping in `$(...)` doesn't corrupt the id.
# Valid id chars: alphanumerics, underscore, dot, dash.
SESSION_IDS=$(echo "$COMMAND" | grep -oPm10 '(?<=--session-id[=\s])[A-Za-z0-9_.\-]+' 2>/dev/null) || true
if [ -z "$SESSION_IDS" ]; then
    SESSION_IDS="${CLAUDE_SESSION_ID:-}"
fi
[ -z "$SESSION_IDS" ] && exit 0

GPU_RES_DIR="${AMMO_GPU_RES_DIR:-/tmp/ammo_gpu_res}"
GPU_STATE="$GPU_RES_DIR/state.json"
[ -f "$GPU_STATE" ] || exit 0

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../skills/ammo/scripts" 2>/dev/null && pwd)" || \
SCRIPTS_DIR="${CLAUDE_PROJECT_DIR:-.}/.claude/skills/ammo/scripts"

# Audit log — every release attempt is recorded so leak investigations can
# distinguish "hook never fired" from "hook fired but extraction missed the id".
RELEASE_LOG="$GPU_RES_DIR/release.log"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
mkdir -p "$GPU_RES_DIR"

# Release every extracted session id. Multiple matches can appear legitimately
# if one Bash tool call wraps multiple reserves; release them all.
# Use a while-read loop so newline-separated ids are handled safely.
while IFS= read -r SID; do
    [ -z "$SID" ] && continue
    printf '%s hook=PostToolUse sid=%s ' "$TS" "$SID" >> "$RELEASE_LOG"
    if python3 "$SCRIPTS_DIR/gpu_reservation.py" release-session --session-id "$SID" >>"$RELEASE_LOG" 2>&1; then
        printf ' ok\n' >> "$RELEASE_LOG"
    else
        printf ' fail\n' >> "$RELEASE_LOG"
        echo "AMMO GPU PostToolUse: release failed for $SID — will expire via lease." >&2
    fi
done <<< "$SESSION_IDS"

exit 0
