#!/bin/bash
# PreToolUse hook — AMMO mid-turn message injection (non-blocking).
#
# Problem: SendMessage delivery is queued between turns. Champions are
# deaf during long tool execution chains (GPU benchmarks, E2E sweeps).
#
# Approach: Inject undelivered inbox messages as additionalContext BEFORE
# each tool call. Never deny — champions keep working with full awareness.
#
# Dedup: Sidecar file tracks last-injected timestamp. Only new messages
# since last injection get injected. Includes "ignore if delivered again"
# context so turn-end delivery of the same messages doesn't confuse the agent.
#
# Cleanup: Sidecar deleted when all inbox messages have been delivered
# (inbox_ts <= delivery_ts in transcript).
#
# Identity: agentName + teamName from the transcript JSONL (first 5 lines).
# Targets: champion-*, impl-champion-*, team-lead (orchestrator),
#          monitor-champion-*, monitor-impl-champion-*.
# Applies to all tool calls. Fail-open (exit 0 on any error).
set -euo pipefail
trap 'exit 0' ERR

DEBUG="${AMMO_MSG_CHECK_DEBUG:-}"
DLOG="/tmp/ammo-msg-check-debug.log"
dbg() { [ -n "$DEBUG" ] && echo "[$(date +%T)] $*" >> "$DLOG" || true; }

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# ── Parse hook input ──
# Query each field separately — bash `read` with tab IFS collapses
# consecutive tabs (tab is a whitespace class member), which silently
# shifts fields when a middle field (e.g. agent_type) is empty.
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // ""' 2>/dev/null) || TOOL_NAME=""
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // ""' 2>/dev/null) || TRANSCRIPT_PATH=""
AGENT_TYPE=$(echo "$INPUT" | jq -r '.agent_type // ""' 2>/dev/null) || AGENT_TYPE=""
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // ""' 2>/dev/null) || SESSION_ID=""
dbg "tool=$TOOL_NAME agent_type=$AGENT_TYPE session_id=$SESSION_ID"

[ -z "$TOOL_NAME" ] && exit 0

# ── Identity from transcript ──
AGENT_NAME=""
TEAM_NAME=""
RESOLVED_TRANSCRIPT="$TRANSCRIPT_PATH"

# ── EnterWorktree breaks transcript_path — try parent ──
if [ -n "$TRANSCRIPT_PATH" ] && [ ! -f "$TRANSCRIPT_PATH" ]; then
    PARENT_DIR=$(echo "$(dirname "$TRANSCRIPT_PATH")" | sed 's/--claude-worktrees-[^/]*//')
    PARENT_TRANSCRIPT="$PARENT_DIR/$(basename "$TRANSCRIPT_PATH")"
    if [ -f "$PARENT_TRANSCRIPT" ]; then
        dbg "worktree-fix: using parent transcript: $PARENT_TRANSCRIPT"
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

    # ── Fallback: some tmux agents lack agentName — parse first message ──
    if [ -z "$AGENT_NAME" ]; then
        FALLBACK=$(head -1 "$RESOLVED_TRANSCRIPT" 2>/dev/null | jq -r '
            .message.content // "" |
            capture("You are (?<name>(champion|impl-champion)-\\S+) in the AMMO") |
            .name
        ' 2>/dev/null) || true
        if [ -n "$FALLBACK" ]; then
            AGENT_NAME="$FALLBACK"
            dbg "content-based fallback: agent=$AGENT_NAME"
        fi
    fi
fi

# ── Edit B: orchestrator detection via leadSessionId (via helper) ──
# The orchestrator (team-lead) has no agentName in its transcript. Use the
# shared _ammo_is_lead helper as the gating predicate — it already knows
# how to apply the agent_type / CLAUDE_SUBAGENT / transcript / team-config
# precedence. If helper says "is lead", scan team configs for the one
# whose leadSessionId matches SESSION_ID and adopt it as TEAM_NAME.
TEAMS_ROOT="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/teams"
if [ -z "$AGENT_NAME" ]; then
    HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    source "$HELPER_DIR/_ammo_is_lead.sh"
    if _ammo_is_lead "$INPUT"; then
        _candidate_team=""
        for tdir in "$TEAMS_ROOT"/*/; do
            [ -f "$tdir/config.json" ] || continue
            lead_sid=$(jq -r '.leadSessionId // empty' "$tdir/config.json" 2>/dev/null) || continue
            if [ "$lead_sid" = "$SESSION_ID" ]; then
                _candidate_team=$(jq -r '.name // empty' "$tdir/config.json" 2>/dev/null) || true
                [ -z "$_candidate_team" ] && _candidate_team=$(basename "${tdir%/}")
                break
            fi
        done
        if [ -n "$_candidate_team" ]; then
            AGENT_NAME="team-lead"
            TEAM_NAME="$_candidate_team"
            dbg "orchestrator via helper + team-config match → team=$TEAM_NAME"
        fi
    fi
fi

# ── Edit D: broader TEAM_NAME fallback ──
# If we have an AGENT_NAME but no TEAM_NAME (e.g. transcript lacks teamName
# field, fallback content-parse matched but the old inner scan is gone),
# scan all team configs for a member with that name.
if [ -n "$AGENT_NAME" ] && [ -z "$TEAM_NAME" ] && [ -d "$TEAMS_ROOT" ]; then
    for tdir in "$TEAMS_ROOT"/*/; do
        [ -f "$tdir/config.json" ] || continue
        if jq -e --arg n "$AGENT_NAME" '.members[]? | select(.name == $n)' "$tdir/config.json" >/dev/null 2>&1; then
            TEAM_NAME=$(jq -r '.name // empty' "$tdir/config.json" 2>/dev/null) || true
            [ -z "$TEAM_NAME" ] && TEAM_NAME=$(basename "${tdir%/}")
            dbg "broader team fallback: agent=$AGENT_NAME → team=$TEAM_NAME"
            break
        fi
    done
fi

dbg "agent=$AGENT_NAME team=$TEAM_NAME"
[ -z "$AGENT_NAME" ] && exit 0

# ── Edit A: expanded case filter ──
# Eligible: champion-*, impl-champion-*, team-lead, monitor-champion-*,
# monitor-impl-champion-*. Anything else (verifier-1, researcher, etc.)
# is out of scope and skipped.
case "$AGENT_NAME" in
    champion-*|impl-champion-*|team-lead|monitor-champion-*|monitor-impl-champion-*) ;;
    *) dbg "skip: agentName $AGENT_NAME not in eligible set"; exit 0;;
esac

# ── Read inbox ──
# Claude Code sanitizes team dir names (dots → dashes), so try both
TEAMS_ROOT="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/teams"
TEAM_DIR="$TEAMS_ROOT/$TEAM_NAME"
if [ ! -d "$TEAM_DIR" ]; then
    SANITIZED=$(echo "$TEAM_NAME" | tr '.' '-')
    TEAM_DIR="$TEAMS_ROOT/$SANITIZED"
    dbg "team dir sanitized: $SANITIZED"
fi
MY_INBOX="$TEAM_DIR/inboxes/$AGENT_NAME.json"
[ -f "$MY_INBOX" ] || exit 0

# ── Sidecar file: tracks last-injected timestamp ──
SIDECAR="/tmp/ammo-msg-injected-${AGENT_NAME}.ts"

# ── Get newest non-self inbox timestamp ──
LAST_INBOX_TS=$(jq -r --arg me "$AGENT_NAME" '
    [.[] | select(.from != $me) | .timestamp // empty]
    | map(select(. != null and . != ""))
    | if length == 0 then "" else max end
' "$MY_INBOX" 2>/dev/null) || exit 0

# No non-self messages → nothing to inject
[ -z "$LAST_INBOX_TS" ] && exit 0
dbg "last_inbox_ts=$LAST_INBOX_TS"

# ── Get last delivery timestamp from transcript ──
LAST_DELIVERY_TS=$(jq -R -r 'try fromjson
    | select(.type == "user")
    | select(.message.content | type == "string")
    | select(.message.content | test("<teammate-message teammate_id="))
    | .timestamp // empty
' "$RESOLVED_TRANSCRIPT" 2>/dev/null | tail -1) || true

dbg "last_delivery_ts=${LAST_DELIVERY_TS:-(none)}"

# ── If all messages delivered, clean up sidecar and exit ──
if [ -n "$LAST_DELIVERY_TS" ]; then
    if [ "$LAST_INBOX_TS" \< "$LAST_DELIVERY_TS" ] || [ "$LAST_INBOX_TS" = "$LAST_DELIVERY_TS" ]; then
        dbg "all delivered: inbox_ts=$LAST_INBOX_TS <= delivery_ts=$LAST_DELIVERY_TS — cleaning sidecar"
        rm -f "$SIDECAR" 2>/dev/null || true
        exit 0
    fi
fi

# ── Check sidecar: skip if we already injected up to this point ──
LAST_INJECTED_TS=""
if [ -f "$SIDECAR" ]; then
    LAST_INJECTED_TS=$(cat "$SIDECAR" 2>/dev/null) || LAST_INJECTED_TS=""
fi

if [ -n "$LAST_INJECTED_TS" ] && [ "$LAST_INBOX_TS" = "$LAST_INJECTED_TS" ]; then
    dbg "already injected up to $LAST_INJECTED_TS — skipping"
    exit 0
fi
if [ -n "$LAST_INJECTED_TS" ] && [ "$LAST_INBOX_TS" \< "$LAST_INJECTED_TS" ]; then
    dbg "inbox_ts=$LAST_INBOX_TS < injected_ts=$LAST_INJECTED_TS — skipping"
    exit 0
fi

# ── Determine cutoff: inject messages newer than max(delivery_ts, injected_ts) ──
CUTOFF=""
if [ -n "$LAST_DELIVERY_TS" ] && [ -n "$LAST_INJECTED_TS" ]; then
    if [ "$LAST_DELIVERY_TS" \> "$LAST_INJECTED_TS" ]; then
        CUTOFF="$LAST_DELIVERY_TS"
    else
        CUTOFF="$LAST_INJECTED_TS"
    fi
elif [ -n "$LAST_DELIVERY_TS" ]; then
    CUTOFF="$LAST_DELIVERY_TS"
elif [ -n "$LAST_INJECTED_TS" ]; then
    CUTOFF="$LAST_INJECTED_TS"
fi

dbg "cutoff=${CUTOFF:-(none)} (max of delivery=$LAST_DELIVERY_TS, injected=$LAST_INJECTED_TS)"

# ── Build injection output ──
INJECT_OUTPUT=$(jq -c --arg me "$AGENT_NAME" --arg cutoff "${CUTOFF:-}" '
    [.[] | select(.from != $me)
         | if $cutoff == "" then . else select((.timestamp // "") > $cutoff) end
    ] as $undelivered |
    if ($undelivered | length) == 0 then empty else
        ($undelivered | length) as $count |
        ([$undelivered[] |
            "<injected-teammate-message from=\"\(.from)\" ts=\"\(.timestamp // "unknown")\">\n\(.text)\n</injected-teammate-message>"
        ] | join("\n\n")) as $messages |
        "\($count) mid-turn teammate message(s) injected below. These are being delivered early so you have full context while working. When these same messages arrive again at your turn boundary, IGNORE the duplicates — you already have the content.\n\n\($messages)" as $context |
        {hookSpecificOutput: {
            hookEventName: "PreToolUse",
            additionalContext: $context
        }}
    end
' "$MY_INBOX" 2>/dev/null) || exit 0

# ── If nothing to inject (all filtered by cutoff), exit ──
[ -z "$INJECT_OUTPUT" ] && { dbg "no new messages after cutoff"; exit 0; }

# ── Update sidecar with newest injected timestamp ──
echo "$LAST_INBOX_TS" > "$SIDECAR" 2>/dev/null || true
dbg "injected: sidecar updated to $LAST_INBOX_TS"

# ── Emit injection (non-blocking — no permissionDecision) ──
echo "$INJECT_OUTPUT"
exit 0
