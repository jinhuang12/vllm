#!/bin/bash
# PreCompact hook for AMMO orchestrator
# Saves orchestration context from kernel_opt_artifacts/*/state.json

STATE_FILES=$(find "$CLAUDE_PROJECT_DIR/kernel_opt_artifacts" -name "state.json" 2>/dev/null | head -1)

if [ -n "$STATE_FILES" ]; then
    STATE_DIR=$(dirname "$STATE_FILES")

    # Extract key state info using simplified schema
    MODEL=$(jq -r '.target.model_id // "unknown"' "$STATE_FILES" 2>/dev/null)
    STAGE=$(jq -r '.stage // "unknown"' "$STATE_FILES" 2>/dev/null)
    STATUS=$(jq -r '.status // "unknown"' "$STATE_FILES" 2>/dev/null)
    TEAM_NAME=$(jq -r '.team.name // "unknown"' "$STATE_FILES" 2>/dev/null)

    # Create checkpoint for restoration
    CHECKPOINT_FILE="$STATE_DIR/compaction_checkpoint.json"
    cat > "$CHECKPOINT_FILE" << EOF
{
  "checkpoint_type": "pre_compaction",
  "timestamp": "$(date -Iseconds)",
  "model": "$MODEL",
  "stage": "$STAGE",
  "status": "$STATUS",
  "team_name": "$TEAM_NAME",
  "state_file": "$STATE_FILES",
  "skill_path": ".claude/skills/ammo/SKILL.md"
}
EOF

    cat << CONTEXT_EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PreCompact",
    "additionalContext": "# AMMO Orchestrator Context\n\n## IMPORTANT: You are the LEAD orchestrator\n\nThis session is orchestrating the AMMO (Automated Model Micro-Optimizer) skill.\n\n### After Compaction\n\n1. **Read the skill**: \`.claude/skills/ammo/SKILL.md\`\n2. **Read team config**: \`~/.claude/teams/$TEAM_NAME/config.json\`\n3. **Run TaskList** to see task progress\n4. **Load state**: \`cat $STATE_FILES\`\n5. **Message idle teammates** to resume work\n\n### Model: $MODEL | Stage: $STAGE | Status: $STATUS\n\nDO NOT implement directly — delegate to teammates via task assignment."
  }
}
CONTEXT_EOF

else
    cat << 'EMPTY_EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "PreCompact",
    "additionalContext": ""
  }
}
EMPTY_EOF
fi
