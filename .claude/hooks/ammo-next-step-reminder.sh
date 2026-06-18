#!/bin/bash
# PostToolUse hook — inject concise next-step reminder for AMMO orchestrator,
# plus stage-specific Socratic reasoning chains that force the orchestrator to
# DERIVE its next decision rather than autopilot through it.
#
# Fires after state-mutating tool calls (Bash|Write|Edit). Looks at current
# state.json (and the cached previous snapshot) and emits guidance via
# hookSpecificOutput.additionalContext. Non-blocking (always exit 0).
#
# Skipped for subagents (agentName in transcript or CLAUDE_SUBAGENT=1).
# Throttled to once per 15s per session — reduced from 30s so the Socratic
# nudges land at decision time rather than after the next action is taken.
# Terminal-status nudges bypass the throttle (always fire on irreversible
# transitions to campaign_complete / campaign_exhausted).
set -euo pipefail
trap 'exit 0' ERR

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# ── Skip subagents ──
# Delegated to the shared _ammo_is_lead helper — same precedence as
# ammo-stop-guard so the two hooks never disagree about who the lead is.
HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HELPER_DIR/_ammo_is_lead.sh"
if ! _ammo_is_lead "$INPUT"; then
    exit 0
fi

# ── Worktree cwd drift warning ──
# The orchestrator is supposed to drive the campaign from the main repo.
# When its cwd is inside .claude/worktrees/<basename>/, any python/git/state
# writes land in an isolated op worktree and never reach the main branch.
# Warn once per (session_id, worktree basename) BEFORE running the normal
# reminder so the orchestrator sees the drift instead of stage guidance.
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null) || true
case "$CWD" in
    */.claude/worktrees/*)
        WT_BASENAME=$(echo "$CWD" | sed -E 's#.*/\.claude/worktrees/([^/]+).*#\1#')
        _SID_FOR_WT="${CLAUDE_SESSION_ID:-$(echo "$INPUT" | jq -r '.session_id // "default"' 2>/dev/null)}"
        WT_MARKER="/tmp/ammo-worktree-warned-${_SID_FOR_WT}-${WT_BASENAME}"
        if [ ! -f "$WT_MARKER" ]; then
            touch "$WT_MARKER"
            jq -c -n --arg msg "AMMO CWD DRIFT: Orchestrator cwd is inside .claude/worktrees/${WT_BASENAME}/. Operations (state.json edits, git commits, python runs) from here stay in the isolated op worktree and will NOT reach the main branch. cd back to the main repo before resuming campaign-level work; worktrees are for Stage 4-5 parallel-track agents only." \
                '{hookSpecificOutput:{hookEventName:"PostToolUse",additionalContext:$msg}}'
            exit 0
        fi
        ;;
esac

SESSION_ID="${CLAUDE_SESSION_ID:-$(echo "$INPUT" | jq -r '.session_id // "default"' 2>/dev/null)}"

# ── Find state.json (BEFORE throttle so we can detect terminal-status bypass) ──
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
STATE_FILE=""
ARTIFACT_DIR=""
for d in "$PROJECT_DIR"/kernel_opt_artifacts/*/; do
    [ -f "$d/state.json" ] || continue
    STATE_FILE="$d/state.json"
    ARTIFACT_DIR="$d"
    break
done
[ -z "$STATE_FILE" ] && exit 0

# ── Read state ──
STAGE=$(jq -r '.campaign.current_stage // "unknown"' "$STATE_FILE" 2>/dev/null) || STAGE=unknown
STATUS=$(jq -r '.campaign.status // "active"' "$STATE_FILE" 2>/dev/null) || STATUS=active
CR=$(jq -r '.campaign.current_round // 1' "$STATE_FILE" 2>/dev/null) || CR=1
SCHEMA_VER=$(jq -r '.campaign.schema_version // "4.0"' "$STATE_FILE" 2>/dev/null) || SCHEMA_VER="4.0"
IDX=$(( CR - 1 ))

# Per-round sub-state (all jq failures fall back to empty/null defaults)
BASELINE_DONE=$(jq -r ".campaign.rounds[$IDX].baseline.completed_at // \"\"" "$STATE_FILE" 2>/dev/null) || BASELINE_DONE=""
MINING_DONE=$(jq -r ".campaign.rounds[$IDX].bottleneck_mining.completed_at // \"\"" "$STATE_FILE" 2>/dev/null) || MINING_DONE=""
TEAM_NAME=$(jq -r ".campaign.rounds[$IDX].team_name // \"\"" "$STATE_FILE" 2>/dev/null) || TEAM_NAME=""
SELECTED_COUNT=$(jq -r ".campaign.rounds[$IDX].debate.selected_candidates // [] | length" "$STATE_FILE" 2>/dev/null) || SELECTED_COUNT=0
SELECTED_WINNERS_COUNT=$(jq -r ".campaign.rounds[$IDX].debate.selected_winners // [] | length" "$STATE_FILE" 2>/dev/null) || SELECTED_WINNERS_COUNT=0
DEBATE_ROUNDS_COMPLETED=$(jq -r ".campaign.rounds[$IDX].debate.rounds_completed // 0" "$STATE_FILE" 2>/dev/null) || DEBATE_ROUNDS_COMPLETED=0
DEBATE_MAX_ROUNDS=$(jq -r ".campaign.rounds[$IDX].debate.max_rounds // 0" "$STATE_FILE" 2>/dev/null) || DEBATE_MAX_ROUNDS=0
TRACKS_STARTED=$(jq -r ".campaign.rounds[$IDX].parallel_tracks.started_at // \"\"" "$STATE_FILE" 2>/dev/null) || TRACKS_STARTED=""
INTEG_STARTED=$(jq -r ".campaign.rounds[$IDX].integration.started_at // \"\"" "$STATE_FILE" 2>/dev/null) || INTEG_STARTED=""
INTEG_STATUS=$(jq -r ".campaign.rounds[$IDX].integration.status // \"\"" "$STATE_FILE" 2>/dev/null) || INTEG_STATUS=""
SHIPPED_COUNT=$(jq -r ".campaign.rounds[$IDX].shipped // [] | length" "$STATE_FILE" 2>/dev/null) || SHIPPED_COUNT=0
MINING_INVALIDATED=$(jq -r ".campaign.rounds[$IDX].mining_invalidated // empty" "$STATE_FILE" 2>/dev/null) || MINING_INVALIDATED=""
MINING_INVALIDATED_PRESENT=$(jq -r ".campaign.rounds[$IDX] | has(\"mining_invalidated\")" "$STATE_FILE" 2>/dev/null) || MINING_INVALIDATED_PRESENT="false"

# ── Audit-gate legacy bypass ──
# When `audit` key is PRESENT in the round (even {}), the audit gate is active.
# When absent entirely (pre-feature / legacy campaign), skip audit checks.
AUDIT_EXISTS=$(jq -r ".campaign.rounds[$IDX] | has(\"audit\")" "$STATE_FILE" 2>/dev/null) || AUDIT_EXISTS="false"
if [ "$AUDIT_EXISTS" = "true" ]; then
    AUDIT_S1_PASSED=$(jq -r ".campaign.rounds[$IDX].audit.stage_1.passed_at // \"\"" "$STATE_FILE" 2>/dev/null) || AUDIT_S1_PASSED=""
    AUDIT_S2_PASSED=$(jq -r ".campaign.rounds[$IDX].audit.stage_2.passed_at // \"\"" "$STATE_FILE" 2>/dev/null) || AUDIT_S2_PASSED=""
    AUDIT_S45_PASSED=$(jq -r ".campaign.rounds[$IDX].audit.stage_45.passed_at // \"\"" "$STATE_FILE" 2>/dev/null) || AUDIT_S45_PASSED=""
    AUDIT_S67_PASSED=$(jq -r ".campaign.rounds[$IDX].audit.stage_67.passed_at // \"\"" "$STATE_FILE" 2>/dev/null) || AUDIT_S67_PASSED=""
    if [ -z "$AUDIT_S67_PASSED" ]; then
        AUDIT_S67_PASSED=$(jq -r ".campaign.rounds[$IDX].audit.stage_6.passed_at // \"\"" "$STATE_FILE" 2>/dev/null) || AUDIT_S67_PASSED=""
    fi
else
    AUDIT_S1_PASSED=""
    AUDIT_S2_PASSED=""
    AUDIT_S45_PASSED=""
    AUDIT_S67_PASSED=""
fi

# Track aggregates
TRACKS_PASSING=$(jq -r "
    .campaign.rounds[$IDX].parallel_tracks.tracks // {}
    | to_entries
    | map(select(.value.status == \"PASS\" or .value.status == \"GATED_PASS\"))
    | length
" "$STATE_FILE" 2>/dev/null) || TRACKS_PASSING=0
TRACKS_NON_TERMINAL=$(jq -r "
    .campaign.rounds[$IDX].parallel_tracks.tracks // {}
    | to_entries
    | map(select(.value.status as \$s | \$s != \"PASS\" and \$s != \"GATED_PASS\" and \$s != \"FAIL\"))
    | length
" "$STATE_FILE" 2>/dev/null) || TRACKS_NON_TERMINAL=0
TRACK_COUNT=$(jq -r ".campaign.rounds[$IDX].parallel_tracks.tracks // {} | length" "$STATE_FILE" 2>/dev/null) || TRACK_COUNT=0
TRACKS_MISSING_LAT_OPT=$(jq -r "
    .campaign.rounds[$IDX].parallel_tracks.tracks // {}
    | to_entries
    | map(select(.value.per_bs_verdict != null and .value.e2e_latency_opt == null))
    | length
" "$STATE_FILE" 2>/dev/null) || TRACKS_MISSING_LAT_OPT=0

# Sets of op_ids by verdict — used to detect NEW verdicts via PREV_STATE diff.
TRACKS_PASS_KEYS=$(jq -r "
    [ (.campaign.rounds[$IDX].parallel_tracks.tracks // {})
      | to_entries[]
      | select(.value.verdict == \"PASS\" or .value.verdict == \"GATED_PASS\")
      | .key ]
    | sort
    | join(\",\")
" "$STATE_FILE" 2>/dev/null) || TRACKS_PASS_KEYS=""
TRACKS_FAIL_KEYS=$(jq -r "
    [ (.campaign.rounds[$IDX].parallel_tracks.tracks // {})
      | to_entries[]
      | select(.value.verdict == \"FAIL\")
      | .key ]
    | sort
    | join(\",\")
" "$STATE_FILE" 2>/dev/null) || TRACKS_FAIL_KEYS=""
TRACKS_GATING_KEYS=$(jq -r "
    [ (.campaign.rounds[$IDX].parallel_tracks.tracks // {})
      | to_entries[]
      | select(.value.status == \"GATING_REQUIRED\")
      | .key ]
    | sort
    | join(\",\")
" "$STATE_FILE" 2>/dev/null) || TRACKS_GATING_KEYS=""

# ── Read mining metrics from state.json ──
F_E2E="?"
F_E2E_PREV="?"
TOP_COMPONENT="?"
TOP_COMPONENT_PREV_ROUND="?"
DECODE_SHARE="?"
AMDAHL_CEILING="?"
ROUND_IDX=$((CR - 1))
F_E2E=$(jq -r ".campaign.rounds[${ROUND_IDX}].bottleneck_mining.top_f_decode_pct // \"?\"" "$STATE_FILE" 2>/dev/null) || F_E2E="?"
TOP_COMPONENT=$(jq -r ".campaign.rounds[${ROUND_IDX}].bottleneck_mining.top_component // \"?\"" "$STATE_FILE" 2>/dev/null) || TOP_COMPONENT="?"
AMDAHL_CEILING=$(jq -r ".campaign.rounds[${ROUND_IDX}].bottleneck_mining.amdahl_ceiling // \"?\"" "$STATE_FILE" 2>/dev/null) || AMDAHL_CEILING="?"
# decode_share: try state.json first, fall back to baseline sweep results
DECODE_SHARE=$(jq -r ".campaign.rounds[${ROUND_IDX}].bottleneck_mining.top_f_decode_pct // \"?\"" "$STATE_FILE" 2>/dev/null) || DECODE_SHARE="?"
if [ "$DECODE_SHARE" = "?" ]; then
    BASELINE_E2E_FILE="${ARTIFACT_DIR}rounds/${CR}/sweeps/baseline/e2e_latency_results.json"
    if [ -f "$BASELINE_E2E_FILE" ]; then
        DECODE_SHARE=$(jq -r '
            [.results[]? | (.baseline.aggregate.decode_share_of_e2e // .baseline.decode_share_of_e2e // empty)]
            | if length > 0 then (add / length * 100 | round / 100 | tostring) else "?" end
        ' "$BASELINE_E2E_FILE" 2>/dev/null) || DECODE_SHARE="?"
    fi
fi
# Previous round's top component & f_e2e (for re-mining same-component check)
if [ "$CR" -gt 1 ]; then
    PREV_ROUND_IDX=$((CR - 2))
    TOP_COMPONENT_PREV_ROUND=$(jq -r ".campaign.rounds[${PREV_ROUND_IDX}].bottleneck_mining.top_component // \"?\"" "$STATE_FILE" 2>/dev/null) || TOP_COMPONENT_PREV_ROUND="?"
    F_E2E_PREV=$(jq -r ".campaign.rounds[${PREV_ROUND_IDX}].bottleneck_mining.top_f_decode_pct // \"?\"" "$STATE_FILE" 2>/dev/null) || F_E2E_PREV="?"
fi

# Threshold from state.json config
THRESHOLD=$(jq -r '.campaign.config.min_e2e_improvement_pct // empty' "$STATE_FILE" 2>/dev/null)
if [ -z "$THRESHOLD" ]; then
    THRESHOLD="0.25"
    THRESHOLD_WARN="⚠️ min_e2e_improvement_pct missing from state.json config — using fallback 0.25%. Run new_target.py to initialize properly."
fi

# Workload (ISL/OSL/BS) for Stage 1→2 nudge
ISL=$(jq -r '.target.input_len // .campaign.workload.input_len // "?"' "$STATE_FILE" 2>/dev/null) || ISL="?"
OSL=$(jq -r '.target.output_len // .campaign.workload.output_len // "?"' "$STATE_FILE" 2>/dev/null) || OSL="?"
BS_LIST=$(jq -r '(.target.batch_sizes // .campaign.workload.batch_sizes // []) | join(",")' "$STATE_FILE" 2>/dev/null) || BS_LIST="?"

# Selected winners (for Stage 3 winners nudge) — extract op_id + expected_e2e + classification
WINNER_LIST=$(jq -r '
    [ (.campaign.rounds['"$IDX"'].debate.selected_winners // []),
      (.campaign.rounds['"$IDX"'].debate.selected_candidates // [])
    ] | flatten
      | map(if type == "object" then (.op_id // .id // .name // tostring) else tostring end)
      | unique
      | join(", ")
' "$STATE_FILE" 2>/dev/null) || WINNER_LIST=""
WINNER_DETAIL=$(jq -r '
    [ (.campaign.rounds['"$IDX"'].debate.selected_winners // []),
      (.campaign.rounds['"$IDX"'].debate.selected_candidates // [])
    ] | flatten
      | map(select(type == "object"))
      | map((.op_id // .id // "?") + " (" + (.track_assignment // "lossless") + ", expected_e2e=" + ((.score_breakdown.expected_e2e_pct // .expected_e2e_pct // 0) | tostring) + "%)")
      | join("; ")
' "$STATE_FILE" 2>/dev/null) || WINNER_DETAIL=""

# Round 1 baseline (for SHIP cumulative-speedup nudge)
# v4.0: deprecated top-level field; derive from rounds[0].baseline.e2e_latency (smallest BS avg)
R1_BASELINE_S=$(jq -r '
    .campaign.round_1_baseline_latency_s //
    (.campaign.rounds[0].baseline.e2e_latency // {} | to_entries | sort_by(.key | tonumber) | .[0].value.avg // empty) //
    "?"
' "$STATE_FILE" 2>/dev/null) || R1_BASELINE_S="?"

# ── PREV_STATE snapshot (read for diff, then update at end) ──
PREV_STATE="/tmp/ammo-state-prev-${SESSION_ID}.json"
PREV_STATUS=""
PREV_STAGE=""
PREV_CR=0
PREV_BASELINE_DONE=""
PREV_MINING_DONE=""
PREV_INTEG_STATUS=""
PREV_TEAM_NAME=""
PREV_SELECTED_COUNT=0
PREV_SELECTED_WINNERS_COUNT=0
PREV_DEBATE_ROUNDS_COMPLETED=0
PREV_TRACKS_PASS_KEYS=""
PREV_TRACKS_FAIL_KEYS=""
PREV_TRACKS_GATING_KEYS=""
PREV_MINING_INVALIDATED_PRESENT="false"
PREV_MINING_INVALIDATED=""
PREV_IDX=0
if [ -f "$PREV_STATE" ]; then
    PREV_STATUS=$(jq -r '.campaign.status // "active"' "$PREV_STATE" 2>/dev/null) || PREV_STATUS=""
    PREV_STAGE=$(jq -r '.campaign.current_stage // "unknown"' "$PREV_STATE" 2>/dev/null) || PREV_STAGE=""
    PREV_CR=$(jq -r '.campaign.current_round // 1' "$PREV_STATE" 2>/dev/null) || PREV_CR=0
    PREV_IDX=$(( PREV_CR - 1 ))
    [ "$PREV_IDX" -lt 0 ] && PREV_IDX=0
    PREV_BASELINE_DONE=$(jq -r ".campaign.rounds[$PREV_IDX].baseline.completed_at // \"\"" "$PREV_STATE" 2>/dev/null) || PREV_BASELINE_DONE=""
    PREV_MINING_DONE=$(jq -r ".campaign.rounds[$PREV_IDX].bottleneck_mining.completed_at // \"\"" "$PREV_STATE" 2>/dev/null) || PREV_MINING_DONE=""
    PREV_INTEG_STATUS=$(jq -r ".campaign.rounds[$PREV_IDX].integration.status // \"\"" "$PREV_STATE" 2>/dev/null) || PREV_INTEG_STATUS=""
    PREV_TEAM_NAME=$(jq -r ".campaign.rounds[$PREV_IDX].team_name // \"\"" "$PREV_STATE" 2>/dev/null) || PREV_TEAM_NAME=""
    PREV_SELECTED_COUNT=$(jq -r ".campaign.rounds[$PREV_IDX].debate.selected_candidates // [] | length" "$PREV_STATE" 2>/dev/null) || PREV_SELECTED_COUNT=0
    PREV_SELECTED_WINNERS_COUNT=$(jq -r ".campaign.rounds[$PREV_IDX].debate.selected_winners // [] | length" "$PREV_STATE" 2>/dev/null) || PREV_SELECTED_WINNERS_COUNT=0
    PREV_DEBATE_ROUNDS_COMPLETED=$(jq -r ".campaign.rounds[$PREV_IDX].debate.rounds_completed // 0" "$PREV_STATE" 2>/dev/null) || PREV_DEBATE_ROUNDS_COMPLETED=0
    PREV_TRACKS_PASS_KEYS=$(jq -r "
        [ (.campaign.rounds[$PREV_IDX].parallel_tracks.tracks // {})
          | to_entries[]
          | select(.value.verdict == \"PASS\" or .value.verdict == \"GATED_PASS\")
          | .key ]
        | sort
        | join(\",\")
    " "$PREV_STATE" 2>/dev/null) || PREV_TRACKS_PASS_KEYS=""
    PREV_TRACKS_FAIL_KEYS=$(jq -r "
        [ (.campaign.rounds[$PREV_IDX].parallel_tracks.tracks // {})
          | to_entries[]
          | select(.value.verdict == \"FAIL\")
          | .key ]
        | sort
        | join(\",\")
    " "$PREV_STATE" 2>/dev/null) || PREV_TRACKS_FAIL_KEYS=""
    PREV_TRACKS_GATING_KEYS=$(jq -r "
        [ (.campaign.rounds[$PREV_IDX].parallel_tracks.tracks // {})
          | to_entries[]
          | select(.value.status == \"GATING_REQUIRED\")
          | .key ]
        | sort
        | join(\",\")
    " "$PREV_STATE" 2>/dev/null) || PREV_TRACKS_GATING_KEYS=""
    PREV_MINING_INVALIDATED_PRESENT=$(jq -r ".campaign.rounds[$PREV_IDX] | has(\"mining_invalidated\")" "$PREV_STATE" 2>/dev/null) || PREV_MINING_INVALIDATED_PRESENT="false"
    PREV_MINING_INVALIDATED=$(jq -r ".campaign.rounds[$PREV_IDX].mining_invalidated // empty" "$PREV_STATE" 2>/dev/null) || PREV_MINING_INVALIDATED=""
fi

# Helper: keys in $1 that are NOT in $2 (comma-separated sets).
_set_diff() {
    local _cur="$1" _prev="$2"
    [ -z "$_cur" ] && { echo ""; return; }
    if [ -z "$_prev" ]; then
        echo "$_cur"
        return
    fi
    local _out="" k
    IFS=',' read -ra _arr <<< "$_cur"
    for k in "${_arr[@]}"; do
        case ",$_prev," in
            *",$k,"*) ;;
            *) _out="${_out:+$_out,}$k" ;;
        esac
    done
    echo "$_out"
}

NEW_PASS_KEYS=$(_set_diff "$TRACKS_PASS_KEYS" "$PREV_TRACKS_PASS_KEYS")
NEW_FAIL_KEYS=$(_set_diff "$TRACKS_FAIL_KEYS" "$PREV_TRACKS_FAIL_KEYS")
NEW_GATING_KEYS=$(_set_diff "$TRACKS_GATING_KEYS" "$PREV_TRACKS_GATING_KEYS")

# ── Determine if this is a TERMINAL transition (bypasses throttle) ──
IS_TERMINAL_TRANSITION=0
if { [ "$STATUS" = "campaign_complete" ] || [ "$STATUS" = "campaign_exhausted" ]; } \
   && [ "$STATUS" != "$PREV_STATUS" ]; then
    IS_TERMINAL_TRANSITION=1
fi

# ── Throttle (15s per session). Terminal transitions bypass. ──
MARKER="/tmp/ammo-reminder-last-${SESSION_ID}"
NOW=$(date +%s)
if [ "$IS_TERMINAL_TRANSITION" -eq 0 ]; then
    if [ -f "$MARKER" ]; then
        LAST=$(stat -c %Y "$MARKER" 2>/dev/null || stat -f %m "$MARKER" 2>/dev/null || echo 0)
        if [ $(( NOW - LAST )) -lt 15 ]; then
            # Throttled — still update PREV_STATE so we don't lose the next diff.
            cp "$STATE_FILE" "$PREV_STATE" 2>/dev/null || true
            exit 0
        fi
    fi
fi
touch "$MARKER"

# ── Build SOCRATIC reasoning chain (edge-triggered) ─────────────────────────
SOCRATIC=""

# Stage 1 → 2: baseline.completed_at became non-null this turn
if [ -n "$BASELINE_DONE" ] && [ -z "$PREV_BASELINE_DONE" ]; then
    SOCRATIC="REASON THROUGH THIS: You're about to mine bottlenecks from this baseline. The user's target shape is ISL=${ISL}/OSL=${OSL}/BS=[${BS_LIST}]. Look at the sweep config in rounds/${CR}/sweeps/baseline/. Does it cover those exact values, or did it run on defaults (64/512)? If defaults: walk forward — the bottleneck you find next stage will reflect a workload the user doesn't care about. What would you do with that finding?

decode_share_of_e2e = ${DECODE_SHARE}. For ISL/OSL = ${ISL}/${OSL}, can you estimate from arithmetic alone what decode_share should roughly be? If your estimate and the measurement diverge significantly, which one is wrong — and how would you tell?

If you cannot cite a specific file:line from the sweep config confirming the user's ISL/OSL/BS were covered, that's not uncertainty — it's unverified. Spawn ammo-investigator to compare swept workload against target.json before mining begins."
fi

# Re-mining produces SAME top component after SHIP — checked FIRST so the more
# specific nudge wins on the shared MINING_DONE edge. Only meaningful when CR>1.
if [ -z "$SOCRATIC" ] && [ -n "$MINING_DONE" ] && [ -z "$PREV_MINING_DONE" ] \
   && [ "$CR" -gt 1 ] && [ "$TOP_COMPONENT" != "?" ] \
   && [ "$TOP_COMPONENT" = "$TOP_COMPONENT_PREV_ROUND" ]; then
    SOCRATIC="REASON THROUGH THIS: Mining on the new baseline found the same top bottleneck: ${TOP_COMPONENT} at f_e2e = ${F_E2E}% (was ${F_E2E_PREV}% last round).

Walk through why it's still #1: did the recent SHIP attack a DIFFERENT component (so ${TOP_COMPONENT} rose in relative ranking), or did it attack ${TOP_COMPONENT} itself and only partially fix it?

If the latter: are the remaining technology classes for ${TOP_COMPONENT} distinct from what just shipped? If last round shipped a Triton kernel replacement and this round will also attempt Triton kernel replacement on the same component, you'll converge on no-op winners.

Before spawning champions: have you appended the previous round's technology to exhausted_technologies[] so the waterfall function pivots? If not, why would this round produce a different outcome?

If you cannot cite which specific technologies in exhausted_technologies[] are distinct from what you're about to attempt, spawn ammo-investigator to compare pre-SHIP and post-SHIP profiling traces before champions repeat a dead path."
fi

# Stage 2 → 3: bottleneck_mining.completed_at became non-null (general fallback
# for the mining edge — shadowed above when re-mining specialization applies).
if [ -z "$SOCRATIC" ] && [ -n "$MINING_DONE" ] && [ -z "$PREV_MINING_DONE" ]; then
    SOCRATIC="REASON THROUGH THIS: Mining is complete. Top component: ${TOP_COMPONENT} at f_e2e = ${F_E2E}%. You're about to spawn champions.

Settle this now: f_e2e is the component's share of TOTAL wall time. f_decode is its share of decode-only time. If you computed f_e2e = f_decode × decode_busy × decode_share_of_e2e, plug those three factors in right now. Do you get back ${F_E2E}%? If you cannot reproduce the number from the sidecar metrics, the debate will be built on an unverified premise.

Also: what's the Amdahl ceiling for this component? (Sidecar reports amdahl_ceiling = ${AMDAHL_CEILING}.) Is a full round of champion effort justified for that ceiling, or are you chasing single-digit gains?

If you cannot cite the three dilution factors and show they multiply to ${F_E2E}%, spawn ammo-investigator to recompute f_e2e from raw traces before champions commit."
fi

# Stage 3 winners selected (selected_candidates or selected_winners went 0→N)
if [ -z "$SOCRATIC" ] \
   && { [ "$SELECTED_COUNT" -gt 0 ] && [ "$PREV_SELECTED_COUNT" -eq 0 ] \
        || [ "$SELECTED_WINNERS_COUNT" -gt 0 ] && [ "$PREV_SELECTED_WINNERS_COUNT" -eq 0 ]; }; then
    [ -z "$WINNER_LIST" ] && WINNER_LIST="(see debate.selected_candidates)"
    DETAIL_LINE=""
    [ -n "$WINNER_DETAIL" ] && DETAIL_LINE="Per-winner data: ${WINNER_DETAIL}."
    SOCRATIC="REASON THROUGH THIS: You selected [${WINNER_LIST}] as debate winners. These are the ONLY implementations this round will attempt. ${DETAIL_LINE}

For each winner: take its expected_e2e_pct (from score_breakdown) and compare it to min_e2e_improvement_pct = ${THRESHOLD}%. Which winners actually clear it? A projection only counts if it's built on f_e2e (not f_decode) and a measured micro-experiment speedup — not a roofline ceiling. If a winner's projection is at or near threshold on theoretical numbers alone, why is it on the list?

If all winners target the same component: what's your contingency when that approach hits a wall? \"Try harder\" isn't a contingency. Is the entire round's capacity riding on one bet? Why?

If you cannot cite the projected E2E number for each winner and show it exceeds ${THRESHOLD}%, the selection is unverified. Spawn ammo-investigator to assess whether better candidates were overlooked."
fi

# Debate exceeding max rounds (rounds_completed >= max_rounds, no winners)
if [ -z "$SOCRATIC" ] \
   && [ "$DEBATE_MAX_ROUNDS" -gt 0 ] \
   && [ "$DEBATE_ROUNDS_COMPLETED" -ge "$DEBATE_MAX_ROUNDS" ] \
   && [ "$SELECTED_COUNT" -eq 0 ] \
   && [ "$SELECTED_WINNERS_COUNT" -eq 0 ] \
   && [ "$DEBATE_ROUNDS_COMPLETED" -gt "$PREV_DEBATE_ROUNDS_COMPLETED" ]; then
    SOCRATIC="REASON THROUGH THIS: Debate has run ${DEBATE_ROUNDS_COMPLETED} of ${DEBATE_MAX_ROUNDS} rounds with no convergence. Before spawning another round, reason through WHY:

Are champions disagreeing on facts (e.g., \"this kernel is 30% of E2E\" vs \"no it's 8%\") — which is resolvable by checking artifacts? Or on projections (resolvable by micro-experiments)? Or on philosophy (which another round won't resolve)?

If it's facts: the missing data is the bottleneck, not more debate. If it's philosophy: you decide. You're the orchestrator.

Is the real issue that no candidate is strong enough? If so, that's a signal the routing or mining is off — not that debate needs more time. What would the next round change that this round didn't?

If you cannot cite what specific NEW information the next round would produce that this round didn't, end debate now and pick the strongest candidates. If the disagreement is factual and you cannot resolve it from existing artifacts, spawn ammo-investigator to surface the ground truth."
fi

# Stage 4-5 PASS / GATED_PASS — new verdicts since last snapshot
if [ -z "$SOCRATIC" ] && [ -n "$NEW_PASS_KEYS" ]; then
    OP_ID="${NEW_PASS_KEYS%%,*}"
    KS=$(jq -r ".campaign.rounds[$IDX].parallel_tracks.tracks[\"$OP_ID\"].kernel_speedup // \"?\"" "$STATE_FILE" 2>/dev/null) || KS="?"
    ES=$(jq -r ".campaign.rounds[$IDX].parallel_tracks.tracks[\"$OP_ID\"].e2e_speedup // \"?\"" "$STATE_FILE" 2>/dev/null) || ES="?"
    # Per-track f_e2e: look up op_id in selected_candidates to find its target component,
    # then compute f_e2e from component_breakdown. Falls back to top-component f_e2e.
    TRACK_F_E2E="$F_E2E"
    TRACK_COMPONENT="$TOP_COMPONENT"
    ALT_COMP=$(jq -r --arg op "$OP_ID" --argjson idx "$IDX" '
        (.campaign.rounds[$idx].bottleneck_mining.component_breakdown // [])
        | map(select(.name as $n | $op | test($n; "i"))) | .[0].name // empty
    ' "$STATE_FILE" 2>/dev/null) || ALT_COMP=""
    if [ -n "$ALT_COMP" ] && [ "$ALT_COMP" != "$TOP_COMPONENT" ]; then
        TRACK_COMPONENT="$ALT_COMP"
        TRACK_F_E2E=$(jq -r --arg comp "$ALT_COMP" --argjson idx "$IDX" '
            (.campaign.rounds[$idx].bottleneck_mining.component_breakdown // [])
            | map(select(.name == $comp)) | .[0].pct // 0 | tostring
        ' "$STATE_FILE" 2>/dev/null) || TRACK_F_E2E="$F_E2E"
    fi
    SOCRATIC="REASON THROUGH THIS: Track ${OP_ID} reports kernel_speedup = ${KS}x, e2e_speedup = ${ES}x. The component's f_e2e was ${TRACK_F_E2E}% (top component: ${TOP_COMPONENT}). If this track targets a DIFFERENT component, look up its f_e2e from bottleneck_analysis.md and use that instead.

Walk Amdahl: ceiling = 1 / (1 - f_e2e/100 + (f_e2e/100) / ${KS}), where f_e2e = ${TRACK_F_E2E}% for this track's component. Compute that number right now. Is ${ES}x above or below ceiling?
  - If above: the measurement is too good. What's the contamination story? (Cache bleed from prior track? Env-var leak? Different baseline than you think?)
  - If kernel is fast but E2E ≈ 1.0: the kernel ran faster in isolation but isn't dispatching in E2E. Why not? Is your optimized kernel actually executing during the sweep, or is something else serving that op?

Trace the causal chain in one sentence: \"kernel got ${KS}x faster on ${TRACK_F_E2E}% of runtime → E2E improved by ${ES}x → that's consistent with Amdahl because ___.\" If you cannot complete that sentence with a concrete number from e2e_latency_results.json, the verdict is unverified. Spawn ammo-investigator to check for contamination before accepting it."
fi

# Stage 4-5 FAIL — new FAIL verdicts since last snapshot
if [ -z "$SOCRATIC" ] && [ -n "$NEW_FAIL_KEYS" ]; then
    OP_ID="${NEW_FAIL_KEYS%%,*}"
    SOCRATIC="REASON THROUGH THIS: You're about to write FAIL for ${OP_ID}. FAIL means \"fundamentally unviable — no further attempt should be made.\"

Walk Non-Negotiable #10's ladder rung-by-rung:
  - Are there untried rungs (gating, contingency, crossover probing)? If yes, this isn't FAIL — it's \"gave up early.\" Why are you skipping those options?
  - Are there ANY batch sizes that PASSED or showed NOISE? If yes, GATED_PASS is available. Why are you ruling it out?
  - State the failure mode in one sentence. Is it \"algorithm fundamentally incompatible with this workload\" (genuinely FAIL) or \"implementation has a bug I didn't fix\" (not FAIL, just unfinished)? How do you know the difference?

If you cannot cite the specific output of each ladder rung you tried (file:line from validation_results.md or remediation log), then FAIL is undocumented. Spawn ammo-investigator to assess whether untried remediation paths exist before writing a terminal verdict."
fi

# Stage 4-5 GATING_REQUIRED — new GATING_REQUIRED status
if [ -z "$SOCRATIC" ] && [ -n "$NEW_GATING_KEYS" ]; then
    OP_ID="${NEW_GATING_KEYS%%,*}"
    SOCRATIC="REASON THROUGH THIS: Track ${OP_ID} is now GATING_REQUIRED — PASS at some batch sizes, regression at others, env-var dispatch needed.

Look at the per-BS verdicts. Where's the crossover point? Has the champion run crossover-probing sweeps (in-between BS values) to find the actual threshold, or are you guessing from the original sweep buckets? If guessing, the env-var threshold will be wrong and the gated optimization will fire at BS values where it regresses.

Also: which batch sizes regressed? Are those BS values ones the user explicitly cares about (check target.json)? If the user's primary BS is in the regressing set, GATED_PASS is harder to justify — they're paying for that workload shape.

If you cannot cite the exact BS threshold from a crossover-probing sweep (not inferred from the original buckets), the gating boundary is a guess. Spawn ammo-investigator to verify the gating story before shipping with an unvalidated env-var threshold."
fi

# Pre-TeamDelete — all tracks terminal AND team_name still set, edge-detected
# via PREV_TRACKS_NON_TERMINAL > 0 (i.e., this turn closed out the last track).
PREV_TRACKS_NON_TERMINAL=0
if [ -f "$PREV_STATE" ]; then
    PREV_TRACKS_NON_TERMINAL=$(jq -r "
        .campaign.rounds[$PREV_IDX].parallel_tracks.tracks // {}
        | to_entries
        | map(select(.value.status as \$s | \$s != \"PASS\" and \$s != \"GATED_PASS\" and \$s != \"FAIL\"))
        | length
    " "$PREV_STATE" 2>/dev/null) || PREV_TRACKS_NON_TERMINAL=0
fi
if [ -z "$SOCRATIC" ] && [ "$STAGE" = "4_5_parallel_tracks" ] \
   && [ "$TRACK_COUNT" -gt 0 ] && [ "$TRACKS_NON_TERMINAL" -eq 0 ] \
   && [ "$PREV_TRACKS_NON_TERMINAL" -gt 0 ] \
   && [ -n "$TEAM_NAME" ]; then
    SOCRATIC="REASON THROUGH THIS: You're about to TeamDelete the round team. Once deleted, track context exists only in artifact files.

Have all per-track validation_results.md been written and committed? Has every track's verdict been mirrored into state.json? If a track's e2e_latency_opt is still null but verdict is PASS, the persistent record disagrees with what the team knew — fix it before deletion or the data is lost.

(${TRACKS_MISSING_LAT_OPT} track(s) currently have per_bs_verdict but null e2e_latency_opt.)

If you cannot confirm that every track with a PASS verdict also has a non-null e2e_latency_opt in state.json, the record is incomplete. Spawn ammo-investigator to inventory rounds/${CR}/tracks/ against state.json before TeamDelete."
fi

# Stage 5 → 6: current_stage transitions to 6_integration
if [ -z "$SOCRATIC" ] && [ "$STAGE" = "6_integration" ] && [ "$PREV_STAGE" != "6_integration" ]; then
    N_PASS=$(jq -r "
        .campaign.rounds[$IDX].parallel_tracks.tracks // {}
        | to_entries
        | map(select(.value.verdict == \"PASS\"))
        | length
    " "$STATE_FILE" 2>/dev/null) || N_PASS=0
    N_GATED=$(jq -r "
        .campaign.rounds[$IDX].parallel_tracks.tracks // {}
        | to_entries
        | map(select(.value.verdict == \"GATED_PASS\"))
        | length
    " "$STATE_FILE" 2>/dev/null) || N_GATED=0
    SOCRATIC="REASON THROUGH THIS: You're entering integration with ${N_PASS} PASS and ${N_GATED} GATED_PASS tracks.

Before the integration sweep runs, predict the combined E2E using Amdahl with all passing tracks' f_e2e values. Write down that number. If the actual sweep comes back worse than your prediction, that's an interaction effect — how would you diagnose it?

  - If two tracks target the SAME component: only one ships. Which one and why? (Best E2E wins, but is the runner-up close enough to keep its branch around as a fallback?)
  - For GATED_PASS tracks: check the env-var default. Should it be 0 (opt-in) or 1 (opt-out)? If it defaults ON and there's a regressing BS, that regression ships to production. Is that the case here?

If you cannot write down the predicted combined E2E number right now (derived from each track's f_e2e via Amdahl), that prediction is unverified. Spawn ammo-investigator to inspect each track's gating block and compute the expected combined result before the integration sweep."
fi

# Stage 6 SHIP recorded — integration.status edged to combined / single_pass
if [ -z "$SOCRATIC" ] \
   && { [ "$INTEG_STATUS" = "combined" ] || [ "$INTEG_STATUS" = "single_pass" ]; } \
   && [ "$INTEG_STATUS" != "$PREV_INTEG_STATUS" ]; then
    CUM_S=$(jq -r '.campaign.cumulative_speedup_vs_round1 // "?"' "$STATE_FILE" 2>/dev/null) || CUM_S="?"
    SOCRATIC="REASON THROUGH THIS: You're about to SHIP. Once shipped, this becomes the new baseline for all future rounds.

State out loud: cumulative_speedup = round_1_baseline_latency_s / current_integrated_latency_s. Plug in the two numbers from state.json (round_1_baseline_latency_s = ${R1_BASELINE_S}, current cumulative_speedup_vs_round1 = ${CUM_S}). What value do you get? If you've been multiplying round-over-round improvements, that's wrong — drift compounds. Which method did you use?

The pre-SHIP mechanical checks (merge-conflict residue, regression in integration sweep, opt returncode, env promotion) — for EACH one, what specific output line did you read that confirmed it passed? \"I ran them\" is not the same as \"I read the output.\"

baseline_env promotion: cat target.json right now. Are the new keys present? What's the difference between \"I updated it\" and \"I verified the file contains the update\"?

If you cannot cite the specific output line for each mechanical check (not \"I ran them\" but the actual result text), that's unverified. Spawn ammo-investigator to confirm each check passed before shipping becomes the permanent new baseline."
fi

# Stage 7 terminal status written (ALWAYS fires — bypassed throttle above)
if [ -z "$SOCRATIC" ] && [ "$IS_TERMINAL_TRANSITION" -eq 1 ]; then
    F2=$(jq -r --argjson idx "$IDX" '
        (.campaign.rounds[$idx].bottleneck_mining.component_breakdown // [])[1].pct // "?"
        | tostring
    ' "$STATE_FILE" 2>/dev/null) || F2="?"
    SOCRATIC="REASON THROUGH THIS: You just set campaign.status = \"${STATUS}\". This is irreversible — no more optimization rounds will run.

The remaining top-component opportunity is f_e2e = ${F_E2E}%. The threshold is ${THRESHOLD}%. Is ${F_E2E} < ${THRESHOLD}? If not: terminating is wrong. The campaign is supposed to keep going.

Now answer this as if the user just asked you: \"Why did you stop? Aren't there other pivots?\" Give your sentence-by-sentence response. Walk through exhausted_technologies[] for this (component, shape_bucket) — count distinct technology CLASSES attempted (Triton, CuTeDSL, CUTLASS, CUDA C++, dispatch, infrastructure). Have all six been seriously tried, or just the obvious ones?

Look at the #2 component (f_e2e = ${F2}%). Is it above threshold? If yes, why aren't you starting a round targeting it?

Have you confused \"I ran out of ideas\" with \"the opportunity is exhausted\"? The first is a reason to pivot strategy, not to stop. If you cannot cite f < threshold for EACH viable component (with the specific f_e2e number per component from bottleneck_analysis.md), then termination is unverified. Spawn ammo-investigator to enumerate remaining viable paths from bottleneck_analysis.md and technology-selection.md before committing to terminal status."
fi

# mining_invalidated written — fires on either (a) key transitions absent→present
# OR (b) value changed (e.g., false→true). Spec: "rounds[$IDX].mining_invalidated
# written" includes any write that changes its visible state.
if [ -z "$SOCRATIC" ] \
   && [ "$MINING_INVALIDATED_PRESENT" = "true" ] \
   && { [ "$PREV_MINING_INVALIDATED_PRESENT" != "true" ] \
        || [ "$MINING_INVALIDATED" != "$PREV_MINING_INVALIDATED" ]; }; then
    SOCRATIC="REASON THROUGH THIS: You're setting mining_invalidated = ${MINING_INVALIDATED} after an EXHAUSTED round.

Two scenarios exist:
  (a) Diagnosis was right, every approach failed → keep mining valid, pivot technology (mining_invalidated = false)
  (b) Tracks revealed the diagnosis was WRONG (e.g., the bottleneck was actually elsewhere, or the component's f_e2e was miscomputed) → re-mine (mining_invalidated = true)

Which scenario are you in? Cite the specific track artifact (validation_results.md or track messaging) that supports your choice. If you can't cite evidence that the DIAGNOSIS (not the fix) was wrong, leave the flag false — re-mining without cause wastes a full round.

If you cannot cite the specific track artifact (file:line) that proves the DIAGNOSIS was wrong (not just the fix), then mining_invalidated=true is unjustified. Spawn ammo-investigator to read all track results and determine whether re-mining is warranted."
fi

# ── Build REMINDER (existing next-step dispatch) ─────────────────────────────
REMINDER=""

# Terminal campaign → states 13/14 (REMINDER path; SOCRATIC may also be set above)
if [ "$STATUS" = "campaign_complete" ] || [ "$STATUS" = "campaign_exhausted" ]; then
    if [ -f "${ARTIFACT_DIR}REPORT.md" ]; then
        REMINDER="Campaign complete. Session may stop."
    else
        REMINDER="Campaign terminal. Spawn ammo-report-writer (background)."
    fi
else
    case "$STAGE" in
        1_baseline)
            if [ -z "$BASELINE_DONE" ]; then
                REMINDER="Next: dispatch ammo-researcher (task_type: baseline). T2."
            else
                if [ "$AUDIT_EXISTS" = "true" ] && [ -z "$AUDIT_S1_PASSED" ]; then
                    REMINDER="AUDIT REQUIRED (T_AUDIT_S1): Stage 1 baseline complete. Spawn ammo-auditor (4-phase: inventory → reconstruction → checklist → reconciliation). Stage: stage_1."
                else
                    REMINDER="Baseline done. Set stage → 2_bottleneck_mining, then dispatch researcher (task_type: mining). T4."
                fi
            fi
            ;;
        2_bottleneck_mining)
            if [ -z "$MINING_DONE" ]; then
                REMINDER="Next: dispatch ammo-researcher (task_type: mining). Analyzes existing traces. T4."
            else
                # T_AUDIT_S2 only fires on schema v4.1+
                _S2_AUDIT_APPLICABLE=false
                _S_MAJOR=$(echo "$SCHEMA_VER" | cut -d. -f1)
                _S_MINOR=$(echo "$SCHEMA_VER" | cut -d. -f2)
                if [ "$_S_MAJOR" -gt 4 ] || ([ "$_S_MAJOR" -eq 4 ] && [ "$_S_MINOR" -ge 1 ]); then
                    _S2_AUDIT_APPLICABLE=true
                fi
                if [ "$_S2_AUDIT_APPLICABLE" = "true" ] && [ "$AUDIT_EXISTS" = "true" ] && [ -z "$AUDIT_S2_PASSED" ]; then
                    REMINDER="AUDIT REQUIRED (T_AUDIT_S2): Stage 2 mining complete. Spawn ammo-auditor (4-phase: inventory → reconstruction → checklist → reconciliation). Stage: stage_2."
                else
                    REMINDER="Mining done. Run T5 gate: python .claude/skills/ammo/scripts/verify_stage2_gate.py {artifact_dir}. Then set stage → 3_debate."
                fi
            fi
            ;;
        3_debate)
            if [ -z "$TEAM_NAME" ]; then
                REMINDER="Next: TeamCreate round team. Spawn 2-4 ammo-champion (no monitors for debate)."
            elif [ "$SELECTED_COUNT" -eq 0 ]; then
                REMINDER="Debate in progress. Min 1 round (full A/B/C). Round 2 if champions declare open items. Custom kernel mandate (Triton/CuTeDSL/CUTLASS/CUDA)."
            elif [ -z "$TRACKS_STARTED" ]; then
                REMINDER="Winners selected. Shut down each debate champion (SendMessage shutdown_request); confirm each shutdown_approved before spawning ammo-impl-champion + monitor per winner."
            fi
            ;;
        4_5_parallel_tracks)
            if [ "$TRACKS_MISSING_LAT_OPT" -gt 0 ] && [ "$TRACKS_NON_TERMINAL" -eq 0 ]; then
                REMINDER="$TRACKS_MISSING_LAT_OPT track(s) have per_bs_verdict but null e2e_latency_opt. Extract per-BS latency map from each track's rounds/{CR}/sweeps/opt/{op_id}/e2e_latency_results.json (opt label: avg_s→avg, p50_s→p50, p10_s→p10, p25_s→p25, p75_s→p75, p90_s→p90, p99_s→p99) and write to .campaign.rounds[\$IDX].parallel_tracks.tracks[op_id].e2e_latency_opt. Same shape as baseline.e2e_latency."
            elif [ "$TRACK_COUNT" -gt 0 ] && [ "$TRACKS_NON_TERMINAL" -gt 0 ]; then
                REMINDER="Tracks running. Wait for ALL to reach terminal (PASS/GATED_PASS/FAIL). Do NOT advance to Stage 6 early."
            elif [ "$TRACK_COUNT" -gt 0 ] && [ "$TRACKS_NON_TERMINAL" -eq 0 ] && [ -z "$INTEG_STARTED" ]; then
                if [ "$AUDIT_EXISTS" = "true" ] && [ -z "$AUDIT_S45_PASSED" ]; then
                    REMINDER="AUDIT REQUIRED (T_AUDIT_S45): All parallel tracks terminal. Spawn ammo-auditor (4-phase: inventory → reconstruction → checklist → reconciliation). Stage: stage_45."
                else
                    if [ "$TRACKS_PASSING" -eq 1 ]; then
                        REMINDER="All tracks terminal (1 passer). TeamDelete, then Stage 6 single-track short-circuit: copy Stage 5 results to integration slot, set status=single_pass, run Pre-SHIP checks, SHIP. See orchestration/integration-logic.md § Single-Track Short-Circuit."
                    else
                        REMINDER="All tracks terminal. TeamDelete, then Stage 6 integration (file-set conflict analysis)."
                    fi
                fi
            fi
            ;;
        6_integration)
            _INTEG_TERMINAL=0
            case "$INTEG_STATUS" in
                completed|exhausted|failed) _INTEG_TERMINAL=1 ;;
            esac
            if [ "$SHIPPED_COUNT" -gt 0 ] || [ "$_INTEG_TERMINAL" = "1" ]; then
                if [ "$AUDIT_EXISTS" = "true" ] && [ -z "$AUDIT_S67_PASSED" ]; then
                    REMINDER="AUDIT REQUIRED (T_AUDIT_S67): Spawn ammo-auditor (4-phase: inventory → reconstruction → checklist → reconciliation). Stage: stage_67. No auto-pass — auditor always runs."
                elif [ "$SHIPPED_COUNT" -gt 0 ]; then
                    REMINDER="T_AUDIT_S67 passed. Set stage → 7_campaign_eval. Mechanical check: f < min_e2e_improvement_pct → terminal. Else → new round."
                fi
            elif [ "$SHIPPED_COUNT" -eq 0 ]; then
                if [ "$TRACKS_PASSING" -eq 1 ]; then
                    REMINDER="Single passer — run short-circuit: copy Stage 5 results into rounds/{CR}/sweeps/integration/, set integration.status=single_pass, write e2e_latency_combined. Then Pre-SHIP checks + SHIP. See orchestration/integration-logic.md § Single-Track Short-Circuit."
                elif [ "$TRACKS_PASSING" -gt 1 ]; then
                    REMINDER="Multiple passers — run combined integration sweep with .venv/bin/python + --fresh-cache. Pre-SHIP checks: grep merge-conflict residue, jq dual-verdict override, jq opt returncode. If all pass → SHIP + merge + env promotion + golden-refs capture (~15s)."
                else
                    REMINDER="Run integration decision matrix. No passing tracks → round EXHAUSTED."
                fi
            fi
            ;;
        7_campaign_eval*)
            if [ "$AUDIT_EXISTS" = "true" ] && [ -z "$AUDIT_S67_PASSED" ]; then
                REMINDER="AUDIT REQUIRED (T_AUDIT_S67): audit.stage_67.passed_at not set. Spawn ammo-auditor (4-phase: inventory → reconstruction → checklist → reconciliation). Stage: stage_67."
            else
                REMINDER="Shut down any remaining round-team agents (SendMessage shutdown_request, confirm each shutdown_approved), then TeamDelete the round team. Mechanical check: f < min_e2e_improvement_pct → terminal. Else → new round. SHIP → 2_bottleneck_mining (baseline shifted). EXHAUSTED → 3_debate (reuse existing mining, pivot technology). Override: set mining_invalidated=true on previous round if mining was wrong. No user prompting."
            fi
            ;;
    esac
fi

# ── Update PREV_STATE snapshot for next turn's diff ──
cp "$STATE_FILE" "$PREV_STATE" 2>/dev/null || true

# ── Combine + emit (Socratic suppresses general reminder when triggered) ──
FULL_MSG=""
if [ -n "$SOCRATIC" ]; then
    FULL_MSG="$SOCRATIC"
elif [ -n "$REMINDER" ]; then
    FULL_MSG="AMMO NEXT STEP: $REMINDER"
fi
if [ -n "${THRESHOLD_WARN:-}" ] && [ -n "$FULL_MSG" ]; then
    FULL_MSG="$FULL_MSG\n\n$THRESHOLD_WARN"
fi

if [ -n "$FULL_MSG" ]; then
    jq -c -n --arg msg "$FULL_MSG" \
        '{hookSpecificOutput:{hookEventName:"PostToolUse",additionalContext:$msg}}'
fi
exit 0
