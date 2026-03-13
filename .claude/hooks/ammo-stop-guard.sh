#!/bin/bash
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# Circuit breaker (belt-and-suspenders: check both stop_hook_active AND filesystem flag)
STOP_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false' 2>/dev/null)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null)
BREAKER_FILE="/tmp/ammo_stop_blocked_${SESSION_ID}"
if [ "$STOP_ACTIVE" = "true" ] || [ -f "$BREAKER_FILE" ]; then
    rm -f "$BREAKER_FILE"
    exit 0  # already blocked once, let through
fi

# Find state.json
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
STATE_FILE=""
for d in "$PROJECT_DIR"/kernel_opt_artifacts/*/; do
    [ -f "$d/state.json" ] && STATE_FILE="$d/state.json" && break
done
[ -z "$STATE_FILE" ] && exit 0  # not AMMO context

STATUS=$(jq -r '.campaign.status // empty' "$STATE_FILE" 2>/dev/null)

case "$STATUS" in
    campaign_complete|campaign_exhausted) exit 0 ;;
    paused) exit 0 ;;
    active)
        STAGE=$(jq -r '.stage // "unknown"' "$STATE_FILE" 2>/dev/null)
        touch "$BREAKER_FILE"  # filesystem breaker for next attempt (session-scoped)
        echo "BLOCKED: Campaign active at stage '$STAGE'." >&2
        echo "  Complete the current stage, or update state.json: campaign.status='paused'" >&2
        exit 2 ;;
    *) exit 0 ;;
esac
