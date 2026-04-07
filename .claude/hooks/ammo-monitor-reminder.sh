#!/bin/bash
# PostToolUse hook — reminds AMMO orchestrator to spawn transcript-monitor
# after creating ammo-champion or ammo-impl-champion agents.
#
# Matcher: Agent (configured in settings.local.json)
# Only fires for orchestrator sessions (no agentName in transcript).
# Behavior: injects additionalContext, never blocks. Fail-open on any error.
set -euo pipefail
trap 'exit 0' ERR

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# Extract fields from hook input (separate calls to avoid @tsv tab-collapse)
AGENT_TYPE=$(echo "$INPUT" | jq -r '.tool_input.subagent_type // ""' 2>/dev/null) || true
AGENT_NAME=$(echo "$INPUT" | jq -r '.tool_input.name // ""' 2>/dev/null) || true
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // ""' 2>/dev/null) || true

# Only fire for champion agent types
case "$AGENT_TYPE" in
    ammo-champion|ammo-impl-champion) ;;
    *) exit 0;;
esac

# Only fire for orchestrator sessions (no agentName = top-level session).
# Champions spawning delegates should NOT get this reminder.
if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
    HAS_AGENT_NAME=$(head -5 "$TRANSCRIPT_PATH" 2>/dev/null | jq -rs '
        [.[] | select(.agentName)] | length
    ' 2>/dev/null) || true
    [ "${HAS_AGENT_NAME:-0}" -gt 0 ] && exit 0
fi

# Build reminder — works with or without a name
if [ -n "$AGENT_NAME" ]; then
    MONITOR_NAME="monitor-$AGENT_NAME"
    MSG="You just spawned $AGENT_NAME (type: $AGENT_TYPE). You MUST now spawn a corresponding transcript-monitor agent named $MONITOR_NAME with agentType=ammo-transcript-monitor to monitor this champion."
else
    MSG="You just spawned an unnamed $AGENT_TYPE agent. You MUST now spawn a corresponding transcript-monitor (agentType=ammo-transcript-monitor) to monitor this champion. NOTE: The champion was spawned without a name — consider re-spawning with a name parameter for proper team coordination."
fi

jq -c -n --arg msg "AMMO MONITOR REMINDER: $MSG Do this before spawning any other agents or doing other work." '
{
    hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: $msg
    }
}
'
exit 0
