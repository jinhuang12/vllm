#!/usr/bin/env bash
# Stop hook — blocks auditor from ending turn after Phase 1 without completing Phase 2.
#
# Logic: if a Phase 2 sentinel exists (inject-audit-phase2.sh fired) but the
# verdict file doesn't contain a "## Phase 2" section, block with instructions.
# Only fires for auditor subagents (agent_type present in input).
set -euo pipefail

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# Only fire for subagents (auditors are always spawned as subagents)
AGENT_TYPE=$(jq -r '.agent_type // empty' <<<"$INPUT")
[[ -z "$AGENT_TYPE" ]] && exit 0

SESSION_ID=$(jq -r '.session_id // "unknown"' <<<"$INPUT")

# Check if any Phase 2 sentinel exists for this session
SENTINEL_MATCH=$(find /tmp -maxdepth 1 -name "ammo_audit_phase2_injected_${SESSION_ID}_*" 2>/dev/null | head -1)
[[ -z "$SENTINEL_MATCH" ]] && exit 0

# Extract verdict filename from sentinel
VERDICT_BASE=$(basename "$SENTINEL_MATCH" | sed "s/^ammo_audit_phase2_injected_${SESSION_ID}_//")

# Find the actual verdict file by searching kernel_opt_artifacts
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
VERDICT_FILE=$(find "$PROJECT_DIR"/kernel_opt_artifacts -path "*/audits/${VERDICT_BASE}.md" 2>/dev/null | head -1)
[[ -z "$VERDICT_FILE" ]] && exit 0

# Check if Phase 2 section exists in the verdict file
if grep -q "^## Phase 2" "$VERDICT_FILE" 2>/dev/null; then
    exit 0
fi

# Phase 2 not written — block
jq -c -n '{
  "decision": "block",
  "reason": "Phase 2 (Checklist Verification) not yet written. Read audit-invariants.md and complete the checklist section before finishing."
}'
