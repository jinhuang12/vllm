#!/usr/bin/env bash
# PostToolUse hook — injects Phase 2 (checklist verification) after auditor writes Phase 1.
#
# Fires on Write. Detects verdict file writes (rounds/N/audits/stage_*.md),
# uses decision:block to inject Phase 2 instructions (the only channel that
# reliably delivers context to subagents in Claude Code <=2.1.144).
# NOTE: PostToolUse "block" is informational only — the Write already succeeded.
# One-shot guard prevents re-firing after Phase 2 appends to the same file.
set -euo pipefail

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

FILE_PATH=$(jq -r '.tool_input.file_path // empty' <<<"$INPUT")

# Only verdict files (rounds/N/audits/stage_*.md)
case "$FILE_PATH" in
  */rounds/*/audits/stage_*.md) ;;
  *) exit 0 ;;
esac

# One-shot guard: don't re-fire after Phase 2 write
SESSION_ID=$(jq -r '.session_id // "unknown"' <<<"$INPUT")
AGENT_ID=$(jq -r '.agent_id // empty' <<<"$INPUT")
SENTINEL_KEY="${SESSION_ID}_${AGENT_ID:-noagent}_$(basename "$FILE_PATH" .md)"
SENTINEL="/tmp/ammo_audit_phase2_injected_${SENTINEL_KEY}"
if [[ -e "$SENTINEL" ]]; then exit 0; fi
touch "$SENTINEL"

# Discover invariants path. Prefer CLAUDE_PROJECT_DIR; fallback to dirname walk-up.
if [[ -n "${CLAUDE_PROJECT_DIR:-}" ]] && [[ -f "${CLAUDE_PROJECT_DIR}/.claude/skills/ammo/references/audit-invariants.md" ]]; then
  INVARIANTS="${CLAUDE_PROJECT_DIR}/.claude/skills/ammo/references/audit-invariants.md"
else
  # Walk up 6 levels: stage_*.md -> audits -> N -> rounds -> target -> kernel_opt_artifacts -> worktree
  WORKTREE=$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$FILE_PATH")")")")")")
  INVARIANTS="${WORKTREE}/.claude/skills/ammo/references/audit-invariants.md"
fi

# If invariants file doesn't exist, skip injection
[[ ! -f "$INVARIANTS" ]] && exit 0

jq -n --arg reason "IMPORTANT: Your Write SUCCEEDED — the verdict file is on disk. This is NOT an error.
This message is injected by the audit Phase 2 hook to deliver your next instructions.

Phase 1 written successfully. Now complete Phase 2: Checklist Verification.

Instructions:
1. Read the audit invariants file at: ${INVARIANTS}
2. Apply the Pre-Check section, then the section matching your stage, then Holistic Cross-Reference.
3. For each applicable invariant, fan out ammo-delegate subagents to gather evidence.
4. Apply precondition gating — skip rows whose preconditions don't hold (note as SKIPPED).
5. Reconcile Phase 1 findings with Phase 2 findings. Deduplicate. Assign blocker categories.
6. Write residual risks (mandatory): what could still be wrong even if every finding is resolved?
7. Render final verdict (PASS / BLOCKED / NEEDS_INVESTIGATION) and APPEND to this same file using Edit.

The worst severity across both phases determines the verdict. Phase 2 cannot downgrade Phase 1 BLOCKINGs.
Do NOT retry the Write. Do NOT report an error. Proceed directly to Phase 2." \
  '{
    decision: "block",
    reason: $reason
  }'
