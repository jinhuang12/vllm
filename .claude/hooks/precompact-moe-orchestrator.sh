#!/bin/bash
# PreCompact hook for MoE Monokernel Orchestrator
# This hook saves orchestrator state and injects context to guide compaction

# Find any active MoE monokernel state files
STATE_FILES=$(find "$CLAUDE_PROJECT_DIR/moe_monokernel_artifacts" -name "state.json" 2>/dev/null | head -1)

if [ -n "$STATE_FILES" ]; then
    # We have an active monokernel optimization - save checkpoint
    STATE_DIR=$(dirname "$STATE_FILES")

    # Extract key state info
    MODEL=$(jq -r '.model_short // .model_id // "unknown"' "$STATE_FILES" 2>/dev/null)
    CURRENT_PHASE=$(jq -r '.phases | to_entries | map(select(.value.status == "in_progress" or .value.status == "pending")) | .[0].key // "unknown"' "$STATE_FILES" 2>/dev/null)

    # Create checkpoint for restoration
    CHECKPOINT_FILE="$STATE_DIR/compaction_checkpoint.json"
    cat > "$CHECKPOINT_FILE" << EOF
{
  "checkpoint_type": "pre_compaction",
  "timestamp": "$(date -Iseconds)",
  "model": "$MODEL",
  "current_phase": "$CURRENT_PHASE",
  "state_file": "$STATE_FILES",
  "skill_path": ".claude/skills/moe-monokernel-optimizer/SKILL.md"
}
EOF

    # Output context to guide compaction
    cat << 'CONTEXT_EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "PreCompact",
    "additionalContext": "# MoE Monokernel Orchestrator Context\n\n## IMPORTANT: Orchestrator Role\n\nThis session was acting as an **orchestrator** for the `moe-monokernel-optimizer` skill.\n\n### Your Role After Compaction\n\n1. **You are the ORCHESTRATOR** - you spawn Task subagents, you do NOT implement directly\n2. **Re-read the skill**: `.claude/skills/moe-monokernel-optimizer/SKILL.md`\n3. **Load state**: `cat moe_monokernel_artifacts/*/state.json`\n4. **Continue from checkpoint**: Check `orchestrator.resume_hint` in state.json\n\n### DO NOT\n\n- Write CUDA code directly (spawn Tasks for that)\n- Skip reading the skill file\n- Forget your orchestrator role\n\n### Key Files\n\n- Skill: `.claude/skills/moe-monokernel-optimizer/SKILL.md`\n- Task Prompts: `.claude/skills/moe-monokernel-optimizer/orchestration/task-prompts.md`\n- Workflow: `.claude/skills/moe-monokernel-optimizer/orchestration/workflow.md`"
  }
}
CONTEXT_EOF

else
    # No active monokernel optimization - output empty context
    cat << 'EMPTY_EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "PreCompact",
    "additionalContext": ""
  }
}
EMPTY_EOF
fi
