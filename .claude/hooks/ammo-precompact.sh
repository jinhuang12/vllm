#!/bin/bash
# PreCompact hook for AMMO orchestrator
# Saves orchestration context from kernel_opt_artifacts/*/state.json

STATE_FILES=$(find "$CLAUDE_PROJECT_DIR/kernel_opt_artifacts" -name "state.json" 2>/dev/null | head -1)

if [ -n "$STATE_FILES" ]; then
    STATE_DIR=$(dirname "$STATE_FILES")

    # Extract key state info using simplified schema
    MODEL=$(jq -r '.target.model_id // "unknown"' "$STATE_FILES" 2>/dev/null)
    STAGE=$(jq -r '.stage // "unknown"' "$STATE_FILES" 2>/dev/null)
    STATUS=$(jq -r '.campaign.status // "unknown"' "$STATE_FILES" 2>/dev/null)
    TEAM_NAME=$(jq -r '.team.name // "unknown"' "$STATE_FILES" 2>/dev/null)
    DEBATE_TEAM=$(jq -r '.debate.team_name // ""' "$STATE_FILES" 2>/dev/null)
    TRACK_COUNT=$(jq -r '.parallel_tracks | length // 0' "$STATE_FILES" 2>/dev/null)
    CAMPAIGN_ROUND=$(jq -r '.campaign.current_round // 0' "$STATE_FILES" 2>/dev/null)
    CAMPAIGN_STATUS=$(jq -r '.campaign.status // ""' "$STATE_FILES" 2>/dev/null)
    CUMULATIVE_SPEEDUP=$(jq -r '.campaign.cumulative_e2e_speedup // 1.0' "$STATE_FILES" 2>/dev/null)
    PENDING_QUEUE_SIZE=$(jq '.campaign.pending_queue | length // 0' "$STATE_FILES" 2>/dev/null)

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
  "debate_team": "$DEBATE_TEAM",
  "track_count": $TRACK_COUNT,
  "campaign_round": $CAMPAIGN_ROUND,
  "campaign_status": "$CAMPAIGN_STATUS",
  "cumulative_speedup": $CUMULATIVE_SPEEDUP,
  "pending_queue_size": $PENDING_QUEUE_SIZE,
  "state_file": "$STATE_FILES",
  "skill_path": ".claude/skills/ammo/SKILL.md"
}
EOF

    # PreCompact only supports exit codes (0=allow, 2=block), not hookSpecificOutput.
    # Context injection happens in the SessionStart hook (ammo-postcompact.sh).
    exit 0

else
    # No AMMO state — allow compaction
    exit 0
fi
