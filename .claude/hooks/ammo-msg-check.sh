#!/bin/bash
# PreToolUse hook — AMMO champion mid-turn message gate (tmux agents).
#
# Problem: SendMessage delivery is queued between turns. Champions are
# deaf during long tool execution chains (GPU benchmarks, E2E sweeps).
#
# Detection: transcript counting — compares inbox size to delivered
# <teammate-message> tags in the agent's own transcript. If undelivered
# non-self messages exist, deny the tool call via JSON permissionDecision.
#
# Identity: agentName + teamName from the transcript JSONL (first 5 lines).
# Targets: ammo-champion and ammo-impl-champion agents only.
# Applies to all tool calls.
# Behavior: fail-open (exit 0 on any error).
set -euo pipefail
trap 'exit 0' ERR

DEBUG="${AMMO_MSG_CHECK_DEBUG:-}"
DLOG="/tmp/ammo-msg-check-debug.log"
dbg() { [ -n "$DEBUG" ] && echo "[$(date +%T)] $*" >> "$DLOG" || true; }

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# Extract all needed fields from hook input in one jq call
read -r TOOL_NAME TRANSCRIPT_PATH < <(
    echo "$INPUT" | jq -r '[.tool_name // "", .transcript_path // ""] | @tsv' 2>/dev/null
) || true
dbg "tool=$TOOL_NAME"

[ -z "$TOOL_NAME" ] && exit 0

# ── Identity from transcript ──
# tmux agents have agentName+teamName in their transcript JSONL (line 1 or 2).
AGENT_NAME=""
TEAM_NAME=""
if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
    result=$(head -5 "$TRANSCRIPT_PATH" 2>/dev/null | jq -rs '
        first(.[] | select(.agentName) | [.agentName, .teamName // ""] | @tsv)
    ' 2>/dev/null) || true
    if [ -n "$result" ]; then
        IFS=$'\t' read -r AGENT_NAME TEAM_NAME <<< "$result"
    fi
fi

dbg "agent=$AGENT_NAME team=$TEAM_NAME"
[ -z "$AGENT_NAME" ] && exit 0

# ── Verify champion by naming convention ──
case "$AGENT_NAME" in
    champion-*|impl-champion-*) ;;
    *) exit 0;;
esac

# ── Read inbox (existence = reliable team membership signal) ──
TEAM_DIR="$HOME/.claude/teams/$TEAM_NAME"
MY_INBOX="$TEAM_DIR/inboxes/$AGENT_NAME.json"
[ -f "$MY_INBOX" ] || exit 0

INBOX_COUNT=$(jq 'length' "$MY_INBOX" 2>/dev/null) || exit 0
dbg "inbox=$INBOX_COUNT"
[ "$INBOX_COUNT" -eq 0 ] && exit 0

# ── Detect undelivered messages via transcript counting ──
# Only count <teammate-message> tags in user messages where content is a string
# (actual deliveries). Tool results have content as an array — skip those to
# avoid false positives from file contents that mention the tag.
DELIVERED_COUNT=$(jq -R -r 'try fromjson |
    select(.type == "user") | .message.content
    | select(type == "string")
    | [scan("<teammate-message teammate_id=")] | length
' "$TRANSCRIPT_PATH" 2>/dev/null | awk '{s+=$1} END {print s+0}') || true
DELIVERED_COUNT=$((DELIVERED_COUNT + 0))

UNDELIVERED=$((INBOX_COUNT - DELIVERED_COUNT))
dbg "delivered=$DELIVERED_COUNT undelivered=$UNDELIVERED"
[ "$UNDELIVERED" -le 0 ] && exit 0

# Filter out self-messages from the undelivered tail
UNDELIVERED_OTHERS=$(jq --argjson skip "$DELIVERED_COUNT" --arg me "$AGENT_NAME" \
    '[.[$skip:] | .[] | select(.from != $me)]' \
    "$MY_INBOX" 2>/dev/null) || exit 0

COUNT=$(echo "$UNDELIVERED_OTHERS" | jq 'length' 2>/dev/null) || exit 0
[ "$COUNT" -eq 0 ] && exit 0

# Build output via jq to handle all JSON escaping (newlines, quotes, backslashes)
echo "$UNDELIVERED_OTHERS" | jq -c --argjson count "$COUNT" '
    [.[] | "  - \(.from): \(.summary // (.text[0:80] + "..."))"] | join("\n") as $summaries |
    "\($count) unread teammate message(s). End your turn NOW to receive them." as $reason |
    "AMMO MESSAGE GATE: \($count) unread teammate message(s) detected.\n\n\($summaries)\n\nYou MUST end your turn NOW. Messages are queued and will be delivered when your current turn ends. Do NOT continue executing tools." as $context |
    {hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: $reason,
        additionalContext: $context
    }}
' 2>/dev/null
exit 0
