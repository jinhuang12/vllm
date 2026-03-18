#!/bin/bash
# Stop hook — AMMO campaign self-assessment prompt.
# Fires ONLY for main session (SubagentStop is separate).
#
# Instead of stage-specific logic (which can get cases wrong because
# it lacks the orchestrator's full context), this hook emits a single
# generic prompt that asks the orchestrator to self-assess its workflow
# state — the same pattern as the Resume Protocol in SKILL.md.
#
# Uses file-based one-shot circuit breaker (keyed by session_id):
#   1st stop attempt: create marker file, nudge with self-assessment prompt
#   2nd stop attempt: marker file exists → allow through
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# Circuit breaker: file-based, scoped to session.
# stop_hook_active only guards against immediate re-stop (no work in between).
# Our nudge tells Claude to DO work, so the next stop is a new cycle with
# stop_hook_active=false again — creating an infinite loop. A file marker
# survives across stop cycles regardless of intervening work.
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null)
MARKER="/tmp/ammo-stop-nudged-${SESSION_ID}"

if [ -f "$MARKER" ]; then
    rm -f "$MARKER"
    exit 0
fi

# Find state.json — prefer active campaigns over completed ones.
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
STATE_FILE=""
for d in "$PROJECT_DIR"/kernel_opt_artifacts/*/; do
    [ -f "$d/state.json" ] || continue
    s=$(jq -r '.campaign.status // empty' "$d/state.json" 2>/dev/null)
    if [ "$s" = "active" ]; then
        STATE_FILE="$d/state.json"
        break
    fi
done
[ -z "$STATE_FILE" ] && exit 0  # no active AMMO campaign

STAGE=$(jq -r '.stage // "unknown"' "$STATE_FILE" 2>/dev/null)
ROUND=$(jq -r '.campaign.current_round // 1' "$STATE_FILE" 2>/dev/null)

# Drop the marker so the next stop attempt passes through.
touch "$MARKER"

cat >&2 <<EOF
AMMO: Campaign is active (stage: $STAGE, round: $ROUND).

Before stopping, self-assess:
1. Read state.json at $STATE_FILE
2. Read .claude/skills/ammo/SKILL.md (Campaign Loop + your current stage)
3. Determine: what is the next step in the workflow?
4. If an overlapped debate is active (debate.next_round_overlap.active), it must complete before stopping.
5. If you have a next step you can take, take it.
6. If you're waiting for background agents with nothing else to do, then you can stop.
EOF
exit 2
