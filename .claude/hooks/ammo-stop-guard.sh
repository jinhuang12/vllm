#!/bin/bash
# Stop hook — AMMO campaign self-assessment prompt.
# Fires ONLY for main session (SubagentStop is separate).
# Instead of hard-blocking, nudges the orchestrator to self-assess
# its workflow state and determine next steps.
#
# Uses stop_hook_active as one-shot circuit breaker:
#   1st stop attempt: block with stage-specific guidance
#   2nd stop attempt: stop_hook_active=true → allow through
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# Circuit breaker: if we already nudged once, let the orchestrator stop.
STOP_ACTIVE=$(echo "$INPUT" | jq -r '.stop_hook_active // false' 2>/dev/null)
if [ "$STOP_ACTIVE" = "true" ]; then exit 0; fi

# Find state.json
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
STATE_FILE=""
for d in "$PROJECT_DIR"/kernel_opt_artifacts/*/; do
    [ -f "$d/state.json" ] && STATE_FILE="$d/state.json" && break
done
[ -z "$STATE_FILE" ] && exit 0  # not AMMO context

STATUS=$(jq -r '.campaign.status // empty' "$STATE_FILE" 2>/dev/null)
case "$STATUS" in
    campaign_complete|campaign_exhausted|paused|"") exit 0 ;;
esac

# Campaign is active — provide stage-specific self-assessment prompt.
STAGE=$(jq -r '.stage // "unknown"' "$STATE_FILE" 2>/dev/null)
ROUND=$(jq -r '.campaign.current_round // 1' "$STATE_FILE" 2>/dev/null)

case "$STAGE" in
    4_5_parallel_tracks)
        INCOMPLETE=$(jq '[.parallel_tracks // {} | to_entries[] | select(.value.result == null or .value.status == null)] | length' "$STATE_FILE" 2>/dev/null || echo "0")
        PENDING=$(jq '.campaign.pending_queue // [] | length' "$STATE_FILE" 2>/dev/null || echo "0")
        if [ "$INCOMPLETE" -gt 0 ] 2>/dev/null && [ "$PENDING" -eq 0 ] 2>/dev/null; then
            cat >&2 <<'MSG'
AMMO: Before stopping, self-assess your workflow state.

Implementer tracks are still running and no async debate has been queued.
Per SKILL.md "Async Pipeline", while implementers work you should start
the next-round debate from existing bottleneck data.

Read state.json and .claude/skills/ammo/SKILL.md (Campaign Loop section).
If you genuinely cannot proceed, set campaign.status='paused' in state.json.
MSG
            exit 2
        fi
        if [ "$INCOMPLETE" -eq 0 ] 2>/dev/null; then
            cat >&2 <<'MSG'
AMMO: All implementation tracks have results but you haven't advanced
to Stage 6 (integration validation).

Read state.json parallel_tracks results, then proceed with Stage 6 per
.claude/skills/ammo/orchestration/integration-logic.md
MSG
            exit 2
        fi
        # Tracks running + async debate started — orchestrator is waiting legitimately
        exit 0
        ;;

    3_debate)
        WINNERS=$(jq '.debate.selected_winners // [] | length' "$STATE_FILE" 2>/dev/null || echo "0")
        TRACKS=$(jq '.parallel_tracks // {} | length' "$STATE_FILE" 2>/dev/null || echo "0")
        if [ "$WINNERS" -gt 0 ] 2>/dev/null && [ "$TRACKS" -eq 0 ] 2>/dev/null; then
            cat >&2 <<'MSG'
AMMO: Debate winners selected but no implementer tracks created yet.

Proceed to Stages 4-5: spawn ammo-implementer subagents per
.claude/skills/ammo/orchestration/parallel-tracks.md
MSG
            exit 2
        fi
        ;;

    6_integration)
        INT_STATUS=$(jq -r '.integration.status // "pending"' "$STATE_FILE" 2>/dev/null)
        case "$INT_STATUS" in
            validated|single_pass|combined|exhausted) ;;
            *)
                cat >&2 <<'MSG'
AMMO: Integration validation not complete.

Finish Stage 6 per .claude/skills/ammo/orchestration/integration-logic.md
MSG
                exit 2
                ;;
        esac
        ROUND_RECORDED=$(jq --argjson r "$ROUND" '[.campaign.rounds // [] | .[] | select(.round_id == $r)] | length' "$STATE_FILE" 2>/dev/null || echo "0")
        if [ "$ROUND_RECORDED" -eq 0 ] 2>/dev/null; then
            cat >&2 <<'MSG'
AMMO: Integration complete but round not recorded in campaign.rounds.

Run Stage 7 campaign evaluation: record round results and check
diminishing returns per SKILL.md "Campaign Loop" section.
MSG
            exit 2
        fi
        ;;

    7_campaign_eval)
        cat >&2 <<EOF
AMMO: Campaign evaluation shows status "active" at round $ROUND.

Read state.json. If a candidate shipped, begin re-profiling (Stages 1-2
on patched code). If exhausted, check diminishing returns threshold.
See SKILL.md "Diminishing Returns" section.
EOF
        exit 2
        ;;

    *)
        cat >&2 <<EOF
AMMO: Campaign active at stage '$STAGE' (round $ROUND).

Read state.json and .claude/skills/ammo/SKILL.md, assess where you are
in the workflow, and continue. If you cannot proceed, set
campaign.status='paused' in state.json.
EOF
        exit 2
        ;;
esac

exit 0
