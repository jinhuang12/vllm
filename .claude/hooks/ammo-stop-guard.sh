#!/bin/bash
# Stop hook — AMMO campaign self-assessment prompt.
# Fires ONLY for main session (SubagentStop is separate).
#
# Instead of stage-specific logic (which can get cases wrong because
# it lacks the orchestrator's full context), this hook emits a single
# generic prompt that asks the orchestrator to self-assess its workflow
# state — the same pattern as the Resume Protocol in SKILL.md.
#
# Uses stop_hook_active as one-shot circuit breaker:
#   1st stop attempt: nudge with self-assessment prompt
#   2nd stop attempt: stop_hook_active=true → allow through
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# Circuit breaker: if we already nudged once, let the orchestrator stop.
STOP_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false' 2>/dev/null)
if [ "$STOP_ACTIVE" = "true" ]; then exit 0; fi

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

cat >&2 <<EOF
AMMO: Campaign is active (stage: $STAGE, round: $ROUND).

Before stopping, self-assess:
1. Read state.json at $STATE_FILE
2. Read .claude/skills/ammo/SKILL.md (Campaign Loop + your current stage)
3. Determine: what is the next step in the workflow?
4. If you have a next step you can take, take it.
5. If you're waiting for background agents with nothing else to do, set
   campaign.status='paused' in state.json, then you can stop.
EOF
exit 2
