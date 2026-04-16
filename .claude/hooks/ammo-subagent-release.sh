#!/bin/bash
# SubagentStop hook — releases any GPU reservations owned by the departing
# subagent. This is the safety net for leaks where the PostToolUse hook
# missed a reservation (regex failure, CLAUDE_SESSION_ID fallback miss, etc.).
#
# Strategy (conservative — does not touch other subagents' reservations):
#   1. Release by $CLAUDE_SESSION_ID (covers reservations made without an
#      explicit --session-id, which default to $CLAUDE_SESSION_ID).
#   2. Release by $CLAUDE_SESSION_ID:$AGENT_ID (if agents opt-in via that
#      composite id).
#
# Does NOT reap arbitrary reservations by prefix or nvidia-smi heuristics —
# that's the 15-min lease's job.
#
# Input JSON (docs):
#   {
#     "session_id": "<main-session-uuid>",
#     "agent_id":   "<subagent uuid or empty>",
#     "agent_type": "<ammo-champion|...>",
#     "transcript_path": "<...>"
#   }
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat 2>/dev/null || echo "")
[ -z "$INPUT" ] && exit 0

SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)
AGENT_ID=$(echo "$INPUT" | jq -r '.agent_id // empty' 2>/dev/null)
AGENT_TYPE=$(echo "$INPUT" | jq -r '.agent_type // empty' 2>/dev/null)

# Only act when we have at least a session id.
[ -z "$SESSION_ID" ] && SESSION_ID="${CLAUDE_SESSION_ID:-}"
[ -z "$SESSION_ID" ] && exit 0

GPU_RES_DIR="${AMMO_GPU_RES_DIR:-/tmp/ammo_gpu_res}"
GPU_STATE="$GPU_RES_DIR/state.json"
[ -f "$GPU_STATE" ] || exit 0

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../skills/ammo/scripts" 2>/dev/null && pwd)" || \
SCRIPTS_DIR="${CLAUDE_PROJECT_DIR:-.}/.claude/skills/ammo/scripts"

RELEASE_LOG="$GPU_RES_DIR/release.log"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
mkdir -p "$GPU_RES_DIR"

# Build the list of ids to try. Order: most-specific first.
IDS=()
[ -n "$AGENT_ID" ] && IDS+=("${SESSION_ID}:${AGENT_ID}")
IDS+=("$SESSION_ID")

for SID in "${IDS[@]}"; do
    printf '%s hook=SubagentStop agent_type=%s sid=%s ' "$TS" "${AGENT_TYPE:-unknown}" "$SID" >> "$RELEASE_LOG"
    if python3 "$SCRIPTS_DIR/gpu_reservation.py" release-session --session-id "$SID" >>"$RELEASE_LOG" 2>&1; then
        printf ' ok\n' >> "$RELEASE_LOG"
    else
        printf ' fail\n' >> "$RELEASE_LOG"
    fi
done

exit 0
