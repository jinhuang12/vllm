#!/bin/bash
# SessionStart hook (after compaction) for MoE Monokernel Orchestrator
# This hook restores orchestrator context after compaction

# Find any compaction checkpoint files
CHECKPOINT_FILES=$(find "$CLAUDE_PROJECT_DIR/moe_monokernel_artifacts" -name "compaction_checkpoint.json" 2>/dev/null | head -1)

if [ -n "$CHECKPOINT_FILES" ]; then
    # We have a checkpoint - this is a post-compaction resume
    STATE_DIR=$(dirname "$CHECKPOINT_FILES")
    STATE_FILE="$STATE_DIR/state.json"

    # Read checkpoint info
    MODEL=$(jq -r '.model // "unknown"' "$CHECKPOINT_FILES" 2>/dev/null)
    CURRENT_PHASE=$(jq -r '.current_phase // "unknown"' "$CHECKPOINT_FILES" 2>/dev/null)

    # Read current state for more context
    if [ -f "$STATE_FILE" ]; then
        RESUME_HINT=$(jq -r '.orchestrator.resume_hint // "Read state.json and spawn next Task"' "$STATE_FILE" 2>/dev/null)
    else
        RESUME_HINT="Read state.json and spawn next Task"
    fi

    # Clean up checkpoint file (it served its purpose)
    rm -f "$CHECKPOINT_FILES"

    # Output restoration context
    cat << CONTEXT_EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "# Session Resumed After Compaction\n\n## You Are The MoE Monokernel Orchestrator\n\nThis session was compacted while orchestrating the \`moe-monokernel-optimizer\` skill.\n\n### Immediate Actions Required\n\n1. **Read the skill file**: \`.claude/skills/moe-monokernel-optimizer/SKILL.md\`\n2. **Load current state**: \`cat $STATE_FILE\`\n3. **Resume from checkpoint**: $RESUME_HINT\n\n### Your Role\n\n- **ORCHESTRATOR**: You spawn Task subagents to do the work\n- **DO NOT** implement CUDA code directly\n- **DO NOT** skip reading the skill file\n- **COPY** full task prompts from \`orchestration/task-prompts.md\`\n\n### Model Being Optimized: $MODEL\n### Current Phase: $CURRENT_PHASE\n\nPlease confirm you understand your orchestrator role and are ready to continue."
  }
}
CONTEXT_EOF

else
    # No checkpoint - check if there are any active state files
    STATE_FILES=$(find "$CLAUDE_PROJECT_DIR/moe_monokernel_artifacts" -name "state.json" 2>/dev/null | head -1)

    if [ -n "$STATE_FILES" ]; then
        # There's state but no checkpoint - might be a normal session start
        cat << CONTEXT_EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "# MoE Monokernel Optimization Detected\n\nThere is an existing MoE monokernel optimization state at:\n\`$STATE_FILES\`\n\nIf you are continuing this optimization, remember:\n- Read the skill: \`.claude/skills/moe-monokernel-optimizer/SKILL.md\`\n- You are the orchestrator - spawn Tasks, don't implement directly"
  }
}
CONTEXT_EOF
    else
        # No active optimization
        cat << 'EMPTY_EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": ""
  }
}
EMPTY_EOF
    fi
fi
