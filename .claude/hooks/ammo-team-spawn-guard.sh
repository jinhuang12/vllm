#!/bin/bash
# PreToolUse hook — enforce the team-member vs subagent distinction for
# Agent-tool spawns.
#
# Only three ammo types are team members: ammo-champion, ammo-impl-champion,
# ammo-transcript-monitor. They MUST carry a valid round team_name (so they
# join the round team and are addressable / shutdownable). Every other type
# is a one-shot subagent and MUST NOT carry name or team_name — either would
# register it as a persistent team member that lingers in the roster.
#
# Probe-confirmed source of truth (CC 2.1.162):
#   - `.tool_input.team_name` IS populated when the orchestrator provides it.
#   - The `name` param is documented "Makes it addressable via
#     SendMessage({to: name}) while running" — presence registers a team member.
#   - `team_name` "Uses current team context if omitted" — a named spawn during
#     an active team auto-attaches even with team_name empty.
#   - Orchestrator invocations have empty top-level `.agent_type`.
#
# Matcher: Agent (Task is dead in CC 2.1.121+; see probes/PROBE_FINDINGS.md).
#
# Behavior:
#   - Runs inside a subagent (.agent_type set) → ALLOW unconditionally
#     (spawn gating / promotion is an orchestrator-only concern; a champion
#     spawning a delegate/validator as a subagent must not be blocked).
#   - Team-member type with empty team_name → DENY.
#   - Team-member type with team_name whose team dir does not exist
#     (under $CLAUDE_CONFIG_DIR/teams or $HOME/.claude/teams, with CC's
#     dots-to-dashes sanitization) → DENY.
#   - Any other type with name OR team_name set → DENY.
#   - Otherwise → ALLOW.
#
# Deny format: {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#              "permissionDecision":"deny","permissionDecisionReason":"..."}}
# Always exits 0 (the JSON drives the decision). Fail-open on any error.
set -euo pipefail
trap 'exit 0' ERR

if ! command -v jq >/dev/null 2>&1; then
    exit 0
fi

INPUT=$(cat)

# Inside-subagent short-circuit: .agent_type at top level is only populated
# when this hook fires from a subagent process. On the orchestrator it is
# empty. Spawn gating is an orchestrator-only concern.
AGENT_TYPE=$(echo "$INPUT" | jq -r '.agent_type // ""' 2>/dev/null) || exit 0
if [ -n "$AGENT_TYPE" ]; then
    exit 0
fi

SUBAGENT_TYPE=$(echo "$INPUT" | jq -r '.tool_input.subagent_type // ""' 2>/dev/null) || exit 0
NAME=$(echo "$INPUT" | jq -r '.tool_input.name // ""' 2>/dev/null) || exit 0
TEAM_NAME=$(echo "$INPUT" | jq -r '.tool_input.team_name // ""' 2>/dev/null) || exit 0

emit_deny() {
    local reason="$1"
    jq -c -n --arg reason "$reason" '{
        hookSpecificOutput: {
            hookEventName: "PreToolUse",
            permissionDecision: "deny",
            permissionDecisionReason: $reason
        }
    }'
    exit 0
}

case "$SUBAGENT_TYPE" in
    ammo-champion|ammo-impl-champion|ammo-transcript-monitor)
        # Team-member types: must carry a valid round team_name.
        if [ -z "$TEAM_NAME" ]; then
            emit_deny "AMMO spawn-guard: $SUBAGENT_TYPE must spawn as a team member. Pass tool_input.team_name referencing the round's team (e.g. 'ammo-round-1-<model>-<hw>') and create the team via TeamCreate first."
        fi
        # CC sanitizes team directory names: dots become dashes. Try both forms.
        TEAMS_ROOT="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/teams"
        SANITIZED=$(echo "$TEAM_NAME" | tr '.' '-')
        if [ -d "$TEAMS_ROOT/$TEAM_NAME" ] || [ -d "$TEAMS_ROOT/$SANITIZED" ]; then
            exit 0
        fi
        emit_deny "AMMO spawn-guard: team '$TEAM_NAME' does not exist under $TEAMS_ROOT. Run TeamCreate to materialize the round team before spawning $SUBAGENT_TYPE."
        ;;
    *)
        # Every other type is a one-shot subagent: no name, no team_name.
        if [ -n "$NAME" ] || [ -n "$TEAM_NAME" ]; then
            emit_deny "AMMO spawn-guard: $SUBAGENT_TYPE must spawn as a subagent, not a team member. Drop tool_input.name and tool_input.team_name and re-spawn with only subagent_type + prompt (+ run_in_background as needed). A name registers it as a persistent team member that lingers in the roster; team_name auto-attaches it to the active team. Only ammo-champion, ammo-impl-champion, and ammo-transcript-monitor are team members."
        fi
        exit 0
        ;;
esac
