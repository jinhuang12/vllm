#!/bin/bash
# Stop hook — AMMO orchestrator continuation nudge.
#
# Fires for the ORCHESTRATOR (not teammates) when campaign artifacts exist
# and work remains. Exits silently when:
#   - No state.json in kernel_opt_artifacts/
#   - Campaign is terminal (complete/exhausted) AND REPORT.md exists
#
# Nudges at:
#   - Stage 7 (active): proceed to next round or set terminal status + 7b
#   - Stage 7 (terminal, no report): spawn report subagent (closes the 7b gap)
#   - Stage 7b: spawn report subagent
#   - Stages 4-5 with missing overlapped debate (round 2+)
#   - Any stage with active overlapped debate
#
# Teammates are excluded via the agent_type JSON field (present for subagents,
# absent for the lead) with a fallback to session_id / team-config check.
#
# Uses file-based one-shot circuit breaker (keyed by session_id):
#   1st stop attempt: create marker file, nudge with stage-specific prompt
#   2nd stop attempt: marker file exists → allow through
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# ── Skip for subagents/teammates ──
# agent_type is ONLY present in the JSON for subagents (spawned via Agent tool).
# The lead orchestrator's Stop hook JSON does NOT contain this field.
AGENT_TYPE=$(echo "$INPUT" | jq -r '.agent_type // empty' 2>/dev/null)
if [ -n "$AGENT_TYPE" ]; then
    exit 0  # Subagent — campaign-level nudges are irrelevant
fi

# Transcript-based detection for tmux-backed teammates.
# tmux teammates lack agent_type but their transcript JSONL contains agentName.
# The lead orchestrator's transcript may also contain agentName="team-lead" after
# context compaction — exclude that to avoid silencing the orchestrator.
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null)
if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
    _AGENT_NAME=""
    _result=$(head -5 "$TRANSCRIPT_PATH" 2>/dev/null | jq -rs \
        'first(.[] | select(.agentName) | [.agentName, .teamName // ""] | @tsv)' 2>/dev/null) || true
    if [ -n "$_result" ]; then
        IFS=$'\t' read -r _AGENT_NAME _TEAM_NAME <<< "$_result"
    fi
    if [ -n "$_AGENT_NAME" ] && [ "$_AGENT_NAME" != "team-lead" ]; then
        exit 0  # Team member — campaign nudges are irrelevant
    fi
fi

SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"' 2>/dev/null)
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"

# Secondary: check team config as fallback (works when session_ids differ).
IS_LEAD=false
for team_cfg in "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/teams"/*/config.json; do
    [ -f "$team_cfg" ] || continue
    lead_sid=$(jq -r '.leadSessionId // empty' "$team_cfg" 2>/dev/null)
    if [ "$lead_sid" = "$SESSION_ID" ]; then
        IS_LEAD=true
        break
    fi
done

# If we found team configs but none had our session_id as lead, we're a teammate.
# If no team configs exist at all, we might be the orchestrator running solo — continue checks.
HAS_TEAMS=false
for team_cfg in "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/teams"/*/config.json; do
    [ -f "$team_cfg" ] && HAS_TEAMS=true && break
done

if [ "$IS_LEAD" = "false" ] && [ "$HAS_TEAMS" = "true" ]; then
    exit 0  # Team exists but we're not the lead — allow stop without nudge
fi

# ── Circuit breaker ──
MARKER="/tmp/ammo-stop-nudged-${SESSION_ID}"
if [ -f "$MARKER" ]; then
    rm -f "$MARKER"
    exit 0
fi

# ── Find campaign ──
STATE_FILE=""
ARTIFACT_DIR=""
for d in "$PROJECT_DIR"/kernel_opt_artifacts/*/; do
    [ -f "$d/state.json" ] || continue
    STATE_FILE="$d/state.json"
    ARTIFACT_DIR="$d"
    break
done
[ -z "$STATE_FILE" ] && exit 0  # no campaign artifacts

# ── Fully done check: terminal status + report exists → nothing left to do ──
STATUS=$(jq -r '.campaign.status // empty' "$STATE_FILE" 2>/dev/null)
if [[ "$STATUS" != "active" ]] && [[ -f "${ARTIFACT_DIR}REPORT.md" ]]; then
    exit 0
fi

# ── Paused check: user explicitly paused → allow stop ──
if [[ "$STATUS" = "paused" ]]; then
    exit 0
fi

STAGE=$(jq -r '.stage // "unknown"' "$STATE_FILE" 2>/dev/null)
ROUND=$(jq -r '.campaign.current_round // 1' "$STATE_FILE" 2>/dev/null)
OVERLAP_ACTIVE=$(jq -r '.debate.next_round_overlap.active // false' "$STATE_FILE" 2>/dev/null)
OVERLAP_PHASE=$(jq -r '.debate.next_round_overlap.phase // empty' "$STATE_FILE" 2>/dev/null)
OVERLAP_WINNERS_COUNT=$(jq -r '.debate.next_round_overlap.selected_winners | length // 0' "$STATE_FILE" 2>/dev/null)

# ── Overlap completion check (Bug 2 fix) ──
# active=false could mean "not started" OR "completed". Distinguish via:
#   - phase == "selection_complete" → debate finished normally
#   - selected_winners non-empty → winners were recorded (debate completed)
OVERLAP_COMPLETED=false
if [ "$OVERLAP_PHASE" = "selection_complete" ] || [ "$OVERLAP_WINNERS_COUNT" -gt 0 ] 2>/dev/null; then
    OVERLAP_COMPLETED=true
fi

# ── Stage-specific nudge ──
# Only nudge at stages where the orchestrator should keep going.
NUDGE=""
case "$STAGE" in
    7_campaign_eval*)
        if [ "$STATUS" = "active" ]; then
            NUDGE="You are at Stage 7 (Campaign Evaluation). Do NOT stop or ask the user.
Execute mechanical threshold check: read f (top bottleneck share) from profiling data, compare to min_e2e_improvement_pct in state.json.
- If f >= threshold: update state to next round and continue to Stage 1 (re-profile if SHIP, pivot technology if EXHAUSTED).
- If f < threshold: set campaign status to campaign_complete or campaign_exhausted, then IMMEDIATELY proceed to Stage 7b (spawn report subagent)."
        else
            # Terminal status but no REPORT.md (we passed the fully-done check above)
            NUDGE="Campaign is $STATUS but REPORT.md has not been generated.
Spawn the report subagent NOW: read .claude/skills/ammo/report/SKILL.md and spawn a
general-purpose subagent to generate REPORT.md in ${ARTIFACT_DIR}."
        fi
        ;;
    7b_report*|*report_gen*)
        NUDGE="You are at Stage 7b (Report Generation). Spawn the report subagent now:
Read .claude/skills/ammo/report/SKILL.md and spawn a general-purpose subagent to generate REPORT.md.
Do NOT stop without spawning the report subagent."
        ;;
    4_5*|*parallel_tracks*|*implementation*)
        # During implementation: check if overlapped debate should be launched
        if [ "$ROUND" -ge 2 ] && [ "$OVERLAP_ACTIVE" != "true" ]; then
            if [ "$OVERLAP_COMPLETED" = "true" ]; then
                exit 0  # Overlap already completed — nothing to nudge about
            fi
            # Round 2+ and overlapped debate genuinely not yet launched
            HAS_TRACKS=$(jq -r '.parallel_tracks | length // 0' "$STATE_FILE" 2>/dev/null)
            if [ "$HAS_TRACKS" -gt 0 ]; then
                NUDGE="You spawned implementation agents but have NOT launched the overlapped debate.
Per SKILL.md Overlapped Debate protocol: immediately launch Round $((ROUND+1)) debate
champions into the SAME team. Use existing bottleneck_analysis.md — do NOT wait for
implementation to finish or re-profiling. This saves 35-70 minutes of wall-clock time."
            fi
        elif [ "$OVERLAP_ACTIVE" = "true" ]; then
            NUDGE="Overlapped debate is active (debate.next_round_overlap.active=true).
Wait for both implementation tracks AND debate to complete before proceeding to Stage 6."
        else
            exit 0  # Round 1 implementation — no overlap needed
        fi
        ;;
    *)
        # For other stages, only nudge if overlapped debate is still active
        if [ "$OVERLAP_ACTIVE" = "true" ]; then
            NUDGE="An overlapped debate is still active (debate.next_round_overlap.active=true).
Wait for debate to complete before stopping. Check teammate status and collect results."
        else
            exit 0  # Other stages — allow stop without nudge
        fi
        ;;
esac

[ -z "$NUDGE" ] && exit 0

touch "$MARKER"

cat >&2 <<EOF
AMMO: Campaign at stage $STAGE, status $STATUS, round $ROUND.

$NUDGE
EOF
exit 2
