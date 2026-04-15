#!/bin/bash
# PreToolUse hook — AMMO champion mid-turn message gate (tmux agents).
#
# Problem: SendMessage delivery is queued between turns. Champions are
# deaf during long tool execution chains (GPU benchmarks, E2E sweeps).
#
# Detection: timestamp comparison — checks if the newest non-self inbox
# message arrived AFTER the last teammate-message delivery in the transcript.
# If so, deny the tool call via JSON permissionDecision.
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

# ── Parse hook input (merged: tool_name + transcript_path + agent_type) ──
read -r TOOL_NAME TRANSCRIPT_PATH AGENT_TYPE < <(
    echo "$INPUT" | jq -r '[.tool_name // "", .transcript_path // "", .agent_type // ""] | @tsv' 2>/dev/null
) || true
dbg "tool=$TOOL_NAME"

[ -z "$TOOL_NAME" ] && exit 0

# ── P0 FIX: Skip in-process subagents (they cannot receive messages) ──
if [ -n "$AGENT_TYPE" ]; then
    dbg "skip: in-process subagent (agent_type=$AGENT_TYPE)"
    exit 0
fi

# ── Identity from transcript ──
AGENT_NAME=""
TEAM_NAME=""
RESOLVED_TRANSCRIPT="$TRANSCRIPT_PATH"

# ── P2 FIX: EnterWorktree breaks transcript_path ──
if [ -n "$TRANSCRIPT_PATH" ] && [ ! -f "$TRANSCRIPT_PATH" ]; then
    PARENT_DIR=$(echo "$(dirname "$TRANSCRIPT_PATH")" | sed 's/--claude-worktrees-[^/]*//')
    PARENT_TRANSCRIPT="$PARENT_DIR/$(basename "$TRANSCRIPT_PATH")"
    if [ -f "$PARENT_TRANSCRIPT" ]; then
        dbg "p2-fix: using parent transcript: $PARENT_TRANSCRIPT"
        RESOLVED_TRANSCRIPT="$PARENT_TRANSCRIPT"
    fi
fi

if [ -n "$RESOLVED_TRANSCRIPT" ] && [ -f "$RESOLVED_TRANSCRIPT" ]; then
    result=$(head -5 "$RESOLVED_TRANSCRIPT" 2>/dev/null | jq -rs '
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

# ── Read inbox ──
TEAM_DIR="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/teams/$TEAM_NAME"
MY_INBOX="$TEAM_DIR/inboxes/$AGENT_NAME.json"
[ -f "$MY_INBOX" ] || exit 0

# ── Get newest non-self inbox timestamp + count (merged: 1 jq fork) ──
IFS="|" read -r LAST_INBOX_TS NON_SELF_COUNT < <(jq -r --arg me "$AGENT_NAME" '
    [.[] | select(.from != $me)] as $others |
    ($others | [.[].timestamp // empty] | map(select(. != null and . != ""))
        | if length == 0 then "" else max end) as $max_ts |
    "\($max_ts)|\($others | length)"
' "$MY_INBOX" 2>/dev/null) || exit 0

# No non-self messages → nothing to gate
[ -z "$LAST_INBOX_TS" ] && exit 0
dbg "last_inbox_ts=$LAST_INBOX_TS non_self=$NON_SELF_COUNT"

# ── Get last delivery timestamp from transcript ──
LAST_DELIVERY_TS=$(jq -R -r 'try fromjson
    | select(.type == "user")
    | select(.message.content | type == "string")
    | select(.message.content | test("<teammate-message teammate_id="))
    | .timestamp // empty
' "$RESOLVED_TRANSCRIPT" 2>/dev/null | tail -1) || true

dbg "last_delivery_ts=${LAST_DELIVERY_TS:-(none)}"

# ── Compare timestamps ──
if [ -n "$LAST_DELIVERY_TS" ] && [ "$LAST_INBOX_TS" \< "$LAST_DELIVERY_TS" ] || [ "$LAST_INBOX_TS" = "$LAST_DELIVERY_TS" ]; then
    dbg "all delivered: inbox_ts=$LAST_INBOX_TS <= delivery_ts=$LAST_DELIVERY_TS"
    exit 0
fi

# ── P1 FIX: 5s cooldown after deny ──
MARKER="/tmp/ammo-msg-gate-${AGENT_NAME}.ts"
if [ -f "$MARKER" ]; then
    LAST_TS=$(cat "$MARKER" 2>/dev/null) || LAST_TS=0
    NOW=$(date +%s)
    if [ $((NOW - LAST_TS)) -lt 5 ]; then
        dbg "cooldown: $((NOW - LAST_TS))s since last deny, skipping"
        exit 0
    fi
fi

# ── Build undelivered list + deny output (merged: filter + count + format in 1 jq) ──
jq -c --arg me "$AGENT_NAME" --arg cutoff "${LAST_DELIVERY_TS:-}" '
    [.[] | select(.from != $me)
         | if $cutoff == "" then . else select((.timestamp // "") > $cutoff) end
    ] as $undelivered |
    ($undelivered | length) as $count |
    if $count == 0 then empty else
        [$undelivered[] | "  - \(.from): \(.summary // (.text[0:80] + "..."))"] | join("\n") as $summaries |
        "\($count) unread teammate message(s). End your turn NOW to receive them." as $reason |
        "AMMO MESSAGE GATE: \($count) unread teammate message(s) detected.\n\n\($summaries)\n\nYou MUST end your turn NOW. Messages are queued and will be delivered when your current turn ends. Do NOT continue executing tools." as $context |
        {hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "deny",
            permissionDecisionReason: $reason,
            additionalContext: $context
        }}
    end
' "$MY_INBOX" 2>/dev/null || exit 0

# Record deny timestamp for cooldown
date +%s > "$MARKER" 2>/dev/null || true
exit 0
