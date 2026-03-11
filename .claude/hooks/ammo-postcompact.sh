#!/bin/bash
# SessionStart hook for AMMO orchestrator
# Injects resume context after compaction

CHECKPOINT_FILES=$(find "$CLAUDE_PROJECT_DIR/kernel_opt_artifacts" -name "compaction_checkpoint.json" 2>/dev/null | head -1)

if [ -n "$CHECKPOINT_FILES" ]; then
    STATE_DIR=$(dirname "$CHECKPOINT_FILES")
    STATE_FILE="$STATE_DIR/state.json"

    MODEL=$(jq -r '.model // "unknown"' "$CHECKPOINT_FILES" 2>/dev/null)
    STAGE=$(jq -r '.stage // "unknown"' "$CHECKPOINT_FILES" 2>/dev/null)
    TEAM_NAME=$(jq -r '.team_name // "unknown"' "$CHECKPOINT_FILES" 2>/dev/null)
    DEBATE_TEAM=$(jq -r '.debate_team // ""' "$CHECKPOINT_FILES" 2>/dev/null)
    TRACK_COUNT=$(jq -r '.track_count // 0' "$CHECKPOINT_FILES" 2>/dev/null)
    CAMPAIGN_ROUND=$(jq -r '.campaign_round // 0' "$CHECKPOINT_FILES" 2>/dev/null)
    CAMPAIGN_STATUS=$(jq -r '.campaign_status // ""' "$CHECKPOINT_FILES" 2>/dev/null)
    CUMULATIVE_SPEEDUP=$(jq -r '.cumulative_speedup // 1.0' "$CHECKPOINT_FILES" 2>/dev/null)
    PENDING_QUEUE_SIZE=$(jq -r '.pending_queue_size // 0' "$CHECKPOINT_FILES" 2>/dev/null)

    # Read current state for more context
    if [ -f "$STATE_FILE" ]; then
        CURRENT_STATUS=$(jq -r '.status // "unknown"' "$STATE_FILE" 2>/dev/null)
        SUMMARY=$(jq -r '.summary // ""' "$STATE_FILE" 2>/dev/null)
    else
        CURRENT_STATUS="unknown"
        SUMMARY=""
    fi

    # Build additional resume context for v2 features
    EXTRA_CONTEXT=""
    if [ -n "$DEBATE_TEAM" ] && [ "$DEBATE_TEAM" != "" ]; then
        EXTRA_CONTEXT="${EXTRA_CONTEXT}\n6. **Debate team active**: Check debate section in state.json. Read team config to message champions."
    fi
    if [ "$TRACK_COUNT" -gt 0 ] 2>/dev/null; then
        EXTRA_CONTEXT="${EXTRA_CONTEXT}\n7. **Parallel tracks ($TRACK_COUNT active)**: Check parallel_tracks in state.json for worktree paths and GPU assignments."
    fi
    if [ -n "$CAMPAIGN_STATUS" ] && [ "$CAMPAIGN_STATUS" != "" ]; then
        EXTRA_CONTEXT="${EXTRA_CONTEXT}\n8. **Campaign loop active**: Round $CAMPAIGN_ROUND | Status: $CAMPAIGN_STATUS | Cumulative speedup: ${CUMULATIVE_SPEEDUP}x | Pending queue: $PENDING_QUEUE_SIZE candidates. Read campaign-loop.md for the iteration protocol."
    fi

    # Clean up checkpoint
    rm -f "$CHECKPOINT_FILES"

    cat << CONTEXT_EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "# Session Resumed After Compaction\n\n## You Are The AMMO Lead Orchestrator\n\nThis session was compacted while orchestrating an AMMO optimization.\n\n### Immediate Actions\n\n1. **Read the skill**: \`.claude/skills/ammo/SKILL.md\`\n2. **Read team config**: \`~/.claude/teams/$TEAM_NAME/config.json\`\n3. **Run TaskList** to see task progress\n4. **Load state**: \`cat $STATE_FILE\`\n5. **Message idle teammates** to resume work${EXTRA_CONTEXT}\n\n### Model: $MODEL | Stage: $STAGE | Status: $CURRENT_STATUS\n### Summary: $SUMMARY\n\nYou are the LEAD — manage tasks and gates, do not implement directly."
  }
}
CONTEXT_EOF

else
    # No checkpoint — check for active state
    STATE_FILES=$(find "$CLAUDE_PROJECT_DIR/kernel_opt_artifacts" -name "state.json" 2>/dev/null | head -1)

    if [ -n "$STATE_FILES" ]; then
        MODEL=$(jq -r '.target.model_id // "unknown"' "$STATE_FILES" 2>/dev/null)
        STAGE=$(jq -r '.stage // "unknown"' "$STATE_FILES" 2>/dev/null)

        cat << CONTEXT_EOF
{
  "hookSpecificOutput": {
    "hookEventName": "SessionStart",
    "additionalContext": "# AMMO Optimization Detected\n\nExisting optimization state at: \`$STATE_FILES\`\nModel: $MODEL | Stage: $STAGE\n\nIf continuing this optimization:\n- Read the skill: \`.claude/skills/ammo/SKILL.md\`\n- You are the lead orchestrator — spawn Tasks, don't implement directly"
  }
}
CONTEXT_EOF
    else
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
