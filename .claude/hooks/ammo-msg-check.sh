#!/bin/bash
# PreToolUse hook — AMMO champion mid-turn message gate.
#
# Problem: SendMessage delivery is queued between turns. Champions are
# deaf during long tool execution chains (GPU benchmarks, E2E sweeps).
#
# Solution: Two detection strategies (needed for both agent backends):
#   1. inbox read:false  — works for in-process agents (shared transcript)
#   2. transcript counting — works for tmux agents (own transcript, read always true)
# If EITHER finds undelivered non-self messages, BLOCK (exit 2).
#
# Identity: agent_type from hook input = agent name (only for subagents).
# Team lookup: searches ~/.claude/teams/*/config.json for matching member.
# Targets: ammo-champion and ammo-impl-champion agents only.
# Skips: Read, Grep, Glob (low-latency tools).
# Behavior: fail-open (exit 0 on any error).
set -euo pipefail
trap 'exit 0' ERR

# Debug mode: set AMMO_MSG_CHECK_DEBUG=1 to trace execution
DEBUG="${AMMO_MSG_CHECK_DEBUG:-}"
DLOG="/tmp/ammo-msg-check-debug.log"
dbg() { [ -n "$DEBUG" ] && echo "[$(date +%T)] $*" >> "$DLOG"; }

if ! command -v jq &>/dev/null; then dbg "EXIT: no jq"; exit 0; fi

INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty' 2>/dev/null) || true
dbg "tool=$TOOL_NAME"

# Skip read-only / low-latency tools
case "$TOOL_NAME" in
    Read|Grep|Glob|"") exit 0;;
esac

# ── Identity from hook input ──
# For subagents, agent_type = the agent's NAME (e.g., "champion-A").
# For the lead, agent_type is absent → skip.
AGENT_NAME=$(echo "$INPUT" | jq -r '.agent_type // empty' 2>/dev/null) || true
dbg "agent_name=$AGENT_NAME"
[ -z "$AGENT_NAME" ] && { dbg "EXIT: no agent_type"; exit 0; }

# ── Find team containing this agent ──
TEAM_DIR=""
TEAM_CFG=""
for cfg in "$HOME/.claude/teams"/*/config.json; do
    [ -f "$cfg" ] || continue
    if jq -e --arg name "$AGENT_NAME" '.members[] | select(.name == $name)' "$cfg" &>/dev/null; then
        TEAM_CFG="$cfg"
        TEAM_DIR="$(dirname "$cfg")"
        break
    fi
done
dbg "team_cfg=$TEAM_CFG team_dir=$TEAM_DIR"
[ -z "$TEAM_CFG" ] && { dbg "EXIT: no team config found"; exit 0; }

# ── Verify champion agent type ──
AGENT_TYPE=$(jq -r --arg name "$AGENT_NAME" \
    '.members[] | select(.name == $name) | .agentType // empty' \
    "$TEAM_CFG" 2>/dev/null) || true

dbg "agent_type=$AGENT_TYPE"
case "$AGENT_TYPE" in
    ammo-champion|ammo-impl-champion) ;;
    *) dbg "EXIT: not champion ($AGENT_TYPE)"; exit 0;;
esac

# ── Read inbox ──
MY_INBOX="$TEAM_DIR/inboxes/$AGENT_NAME.json"
[ -f "$MY_INBOX" ] || exit 0

INBOX_COUNT=$(jq 'length' "$MY_INBOX" 2>/dev/null) || exit 0
dbg "inbox_count=$INBOX_COUNT"
[ "$INBOX_COUNT" -eq 0 ] && exit 0

# ── Strategy 1: inbox read:false (works for in-process agents) ──
UNREAD_OTHERS=$(jq --arg me "$AGENT_NAME" \
    '[.[] | select(.read == false and .from != $me)]' \
    "$MY_INBOX" 2>/dev/null) || true
UNREAD_COUNT=$(echo "$UNREAD_OTHERS" | jq 'length' 2>/dev/null) || true
UNREAD_COUNT=$((UNREAD_COUNT + 0))

dbg "strategy1: unread_count=$UNREAD_COUNT"
if [ "$UNREAD_COUNT" -gt 0 ]; then
    SUMMARIES=$(echo "$UNREAD_OTHERS" | jq -r '.[] |
        "  - \(.from): \(.summary // (.text[0:80] + "..."))"' 2>/dev/null) || exit 0
    cat >&2 <<EOF
AMMO MESSAGE GATE: $UNREAD_COUNT unread teammate message(s) detected.

$SUMMARIES

You MUST end your turn NOW to receive these messages.
They are queued and will be delivered when your current turn ends.
Do NOT continue executing tools — stop immediately.
EOF
    exit 2
fi

# ── Strategy 2: transcript counting (works for tmux agents) ──
# In tmux mode, read is always true, but the agent has its own transcript.
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null) || true
dbg "transcript_path=$TRANSCRIPT_PATH exists=$([ -f \"$TRANSCRIPT_PATH\" ] 2>/dev/null && echo y || echo n)"
[ -z "$TRANSCRIPT_PATH" ] || [ ! -f "$TRANSCRIPT_PATH" ] && { dbg "EXIT: no transcript"; exit 0; }

# Count <teammate-message> occurrences in transcript (grep -o for batched messages)
DELIVERED_COUNT=$(grep -o '<teammate-message teammate_id=' "$TRANSCRIPT_PATH" 2>/dev/null | wc -l)
DELIVERED_COUNT=$((DELIVERED_COUNT + 0))

UNDELIVERED_TOTAL=$((INBOX_COUNT - DELIVERED_COUNT))
dbg "strategy2: delivered=$DELIVERED_COUNT undelivered_total=$UNDELIVERED_TOTAL"
[ "$UNDELIVERED_TOTAL" -le 0 ] && { dbg "EXIT: all delivered"; exit 0; }

# Filter non-self from undelivered tail
UNDELIVERED_OTHERS=$(jq --argjson skip "$DELIVERED_COUNT" --arg me "$AGENT_NAME" \
    '[.[$skip:] | .[] | select(.from != $me)]' \
    "$MY_INBOX" 2>/dev/null) || exit 0

OTHER_COUNT=$(echo "$UNDELIVERED_OTHERS" | jq 'length' 2>/dev/null) || exit 0
[ "$OTHER_COUNT" -eq 0 ] && exit 0

SUMMARIES=$(echo "$UNDELIVERED_OTHERS" | jq -r '.[] |
    "  - \(.from): \(.summary // (.text[0:80] + "..."))"' 2>/dev/null) || exit 0

cat >&2 <<EOF
AMMO MESSAGE GATE: $OTHER_COUNT unread teammate message(s) detected.

$SUMMARIES

You MUST end your turn NOW to receive these messages.
They are queued and will be delivered when your current turn ends.
Do NOT continue executing tools — stop immediately.
EOF
exit 2
