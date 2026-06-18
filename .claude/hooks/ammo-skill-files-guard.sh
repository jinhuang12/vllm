#!/usr/bin/env bash
# PreToolUse/Edit|Write|MultiEdit|NotebookEdit — BLOCK subagents from editing
# orchestrator-owned skill/agent definition files.
#
# Protected paths (substring match):
#   .claude/skills/ammo/references/
#   .claude/skills/ammo/orchestration/
#   .claude/agents/
#
# Orchestrator (team-lead) is exempt — it can modify these files freely.
# Subagents get a deny with a message telling them to surface the change upstream.
set -euo pipefail
trap 'exit 0' ERR

command -v jq >/dev/null 2>&1 || exit 0

INPUT=$(cat)

# Extract file path (Edit/Write/MultiEdit/NotebookEdit all use file_path).
FILE_PATH=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // .tool_input.notebook_path // ""' 2>/dev/null) || exit 0
[ -z "$FILE_PATH" ] && exit 0

# Only gate paths under protected directories.
case "$FILE_PATH" in
    *".claude/skills/ammo/references/"*) ;;
    *".claude/skills/ammo/orchestration/"*) ;;
    *".claude/agents/"*) ;;
    *) exit 0 ;;
esac

# Orchestrator check — source the shared helper.
HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HELPER_DIR/_ammo_is_lead.sh"

if _ammo_is_lead "$INPUT"; then
    exit 0
fi

# Subagent → deny.
jq -nc --arg path "$FILE_PATH" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: ("Subagents may not edit \($path). Files under .claude/skills/ammo/references/, .claude/skills/ammo/orchestration/, and .claude/agents/ are orchestrator-owned. Surface the proposed change to the orchestrator via SendMessage instead.")
  }
}'
exit 0
