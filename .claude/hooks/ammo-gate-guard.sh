#!/bin/bash
# TaskCompleted hook — AMMO gate guard
# Blocks gate task completion if verification hasn't been run,
# and enforces iteration loop mechanics on KILL decisions.
#
# Receives TaskCompleted event JSON on stdin.
# Exit 0 = allow, Exit 2 = block (stderr fed back to model).

set -euo pipefail

# Dependency check — cannot enforce without jq
if ! command -v jq &>/dev/null; then
    exit 0
fi

# ── Read event payload ──
INPUT=$(cat)

# Extract task_subject directly from the event
SUBJECT=$(echo "$INPUT" | jq -r '.task_subject // empty' 2>/dev/null)

# Fast bail: if no subject or not a gate/route task, allow immediately
if [ -z "$SUBJECT" ]; then
    exit 0
fi

IS_GATE_VERIFY=false
IS_ROUTE_DECISION=false

if echo "$SUBJECT" | grep -qi "GATE:.*verify_"; then
    IS_GATE_VERIFY=true
elif echo "$SUBJECT" | grep -qi "Route.*decision\|B14"; then
    IS_ROUTE_DECISION=true
fi

IS_GATE_DEBATE=false
IS_GATE_TRACKS=false
IS_GATE_INTEGRATION=false

if echo "$SUBJECT" | grep -qi "GATE:.*debate.*winner\|GATE:.*debate.*selection\|GATE:.*summary\.md"; then
    IS_GATE_DEBATE=true
elif echo "$SUBJECT" | grep -qi "GATE:.*all.*tracks\|GATE:.*track.*validated\|GATE:.*track.*results"; then
    IS_GATE_TRACKS=true
elif echo "$SUBJECT" | grep -qi "GATE:.*integration\|GATE:.*combined.*validation"; then
    IS_GATE_INTEGRATION=true
fi

if [ "$IS_GATE_VERIFY" = "false" ] && [ "$IS_ROUTE_DECISION" = "false" ] && \
   [ "$IS_GATE_DEBATE" = "false" ] && [ "$IS_GATE_TRACKS" = "false" ] && \
   [ "$IS_GATE_INTEGRATION" = "false" ]; then
    exit 0
fi

# ── Locate artifact directory ──
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$CWD}"

ARTIFACT_DIR=""
if [ -n "$PROJECT_DIR" ]; then
    for d in "$PROJECT_DIR"/kernel_opt_artifacts/*/; do
        if [ -f "$d/state.json" ]; then
            ARTIFACT_DIR="$d"
            break
        fi
    done
fi

# If no artifact dir or no state.json, this is not an AMMO task — allow
if [ -z "$ARTIFACT_DIR" ] || [ ! -f "$ARTIFACT_DIR/state.json" ]; then
    exit 0
fi

STATE_FILE="$ARTIFACT_DIR/state.json"

# ── Helper: extract current route decision ──
# Handles multiple schema variants observed in production
get_route() {
    local sf="$1"
    local r=""
    # Try .route_decision.route (future canonical)
    r=$(jq -r '.route_decision.route // empty' "$sf" 2>/dev/null)
    if [ -n "$r" ]; then echo "$r"; return; fi
    # Try .route_decision.decision (documented schema)
    r=$(jq -r '.route_decision.decision // empty' "$sf" 2>/dev/null)
    if [ -n "$r" ]; then echo "$r"; return; fi
    # Try latest opportunity_attempts entry
    r=$(jq -r '.opportunity_attempts[-1].result // empty' "$sf" 2>/dev/null)
    if [ -n "$r" ]; then echo "$r"; return; fi
    # Try .route_decision.final
    r=$(jq -r '.route_decision.final // empty' "$sf" 2>/dev/null)
    if [ -n "$r" ]; then echo "$r"; return; fi
    echo ""
}

# ── Path 1: GATE verify tasks ──
if [ "$IS_GATE_VERIFY" = "true" ]; then

    if echo "$SUBJECT" | grep -qi "verify_phase1"; then
        # Stage 1 gate — check for verification evidence
        STAGE1_STATUS=$(jq -r '.verification_run.stage1 // empty' "$STATE_FILE" 2>/dev/null)
        if [ -n "$STAGE1_STATUS" ] && [ "$STAGE1_STATUS" != "null" ]; then
            exit 0
        fi
        if ls "$ARTIFACT_DIR"/phase1_verification_*.md 2>/dev/null | head -1 | grep -q .; then
            exit 0
        fi
        echo "BLOCKED: Gate task requires running verify_phase1_baseline.py first." >&2
        echo "Run: python .claude/skills/ammo/scripts/verify_phase1_baseline.py $ARTIFACT_DIR" >&2
        exit 2

    elif echo "$SUBJECT" | grep -qi "verify_validation"; then
        # Validation gate — check for verification evidence
        EVIDENCE_FOUND=false
        VAL_STATUS=$(jq -r '.verification_run.validation // empty' "$STATE_FILE" 2>/dev/null)
        if [ -n "$VAL_STATUS" ] && [ "$VAL_STATUS" != "null" ]; then
            EVIDENCE_FOUND=true
        fi
        if [ "$EVIDENCE_FOUND" = "false" ]; then
            if ls "$ARTIFACT_DIR"/validation_gate_*.md 2>/dev/null | head -1 | grep -q .; then
                EVIDENCE_FOUND=true
            fi
        fi

        if [ "$EVIDENCE_FOUND" = "false" ]; then
            echo "BLOCKED: Gate task requires running verify_validation_gates.py first." >&2
            echo "Run: python .claude/skills/ammo/scripts/verify_validation_gates.py $ARTIFACT_DIR" >&2
            exit 2
        fi

        # Evidence exists — enforce KILL requires recorded attempt
        ROUTE=$(get_route "$STATE_FILE")
        if [ "$ROUTE" = "KILL" ]; then
            ATTEMPTS=$(jq '.opportunity_attempts | length' "$STATE_FILE" 2>/dev/null)
            if [ "$ATTEMPTS" = "0" ] || [ -z "$ATTEMPTS" ]; then
                echo "BLOCKED: Cannot complete validation gate with KILL decision." >&2
                echo "  opportunity_attempts is empty — record the attempt first." >&2
                echo "  B14 (Route decision) handles iteration after this gate completes." >&2
                exit 2
            fi
        fi
        exit 0
    fi

    # Unknown verify_ pattern — allow
    exit 0
fi

# ── Path 2: B14 Route Decision enforcement ──
if [ "$IS_ROUTE_DECISION" = "true" ]; then

    ROUTE=$(get_route "$STATE_FILE")
    if [ "$ROUTE" = "KILL" ]; then
        # Verify iteration tasks (B15/B16) exist in the team task directory
        TEAM_NAME=$(echo "$INPUT" | jq -r '.team_name // empty' 2>/dev/null)
        TASK_DIR="$HOME/.claude/tasks/$TEAM_NAME"

        ITERATION_FOUND=false
        if [ -d "$TASK_DIR" ]; then
            for tf in "$TASK_DIR"/*.json; do
                [ -f "$tf" ] || continue
                subj=$(jq -r '.subject // empty' "$tf" 2>/dev/null)
                if echo "$subj" | grep -qi "B15\|updated.*plan\|iteration.*plan"; then
                    ITERATION_FOUND=true
                    break
                fi
            done
        fi

        if [ "$ITERATION_FOUND" = "false" ]; then
            echo "BLOCKED: Cannot complete Route decision (KILL) without iteration tasks." >&2
            echo "  Create B15, B16, and B9'-B13' chain first." >&2
            exit 2
        fi
    fi
fi

# ── Path 3: Debate gate enforcement ──
if [ "$IS_GATE_DEBATE" = "true" ]; then
    if [ ! -d "$ARTIFACT_DIR/debate" ]; then
        echo "BLOCKED: Debate gate requires debate/ directory in artifact dir." >&2
        echo "  Expected: $ARTIFACT_DIR/debate" >&2
        exit 2
    fi

    if [ ! -f "$ARTIFACT_DIR/debate/summary.md" ]; then
        echo "BLOCKED: Debate gate requires debate/summary.md." >&2
        echo "  Expected: $ARTIFACT_DIR/debate/summary.md" >&2
        exit 2
    fi

    # Check for proposals directory (Phase 0 must have run)
    if [ ! -d "$ARTIFACT_DIR/debate/proposals" ]; then
        echo "BLOCKED: Debate gate requires debate/proposals/ directory (Phase 0)." >&2
        echo "  Expected: $ARTIFACT_DIR/debate/proposals/" >&2
        exit 2
    fi

    PROPOSAL_COUNT=$(find "$ARTIFACT_DIR/debate/proposals" -maxdepth 1 -name "*.md" -type f | wc -l)
    if [ "$PROPOSAL_COUNT" -lt 2 ]; then
        echo "BLOCKED: Debate gate requires at least 2 champion proposals." >&2
        echo "  Found $PROPOSAL_COUNT proposal files in $ARTIFACT_DIR/debate/proposals/" >&2
        exit 2
    fi

    ROUND_COUNT=$(find "$ARTIFACT_DIR/debate" -maxdepth 1 -name "round_*" -type d | wc -l)
    if [ "$ROUND_COUNT" -lt 1 ]; then
        echo "BLOCKED: Debate gate requires at least 1 debate round." >&2
        echo "  Found $ROUND_COUNT round_* directories in $ARTIFACT_DIR/debate" >&2
        exit 2
    fi

    exit 0
fi

# ── Path 4: Track validation gate enforcement ──
if [ "$IS_GATE_TRACKS" = "true" ]; then
    TRACK_IDS=$(jq -r '.parallel_tracks[]?.id // empty' "$STATE_FILE" 2>/dev/null)

    if [ -z "$TRACK_IDS" ]; then
        echo "BLOCKED: Track validation gate found no parallel_tracks in state.json." >&2
        exit 2
    fi

    while IFS= read -r TRACK_ID; do
        RESULT=$(jq -r --arg tid "$TRACK_ID" '.parallel_tracks[] | select(.id == $tid) | .result // empty' "$STATE_FILE" 2>/dev/null)
        if [ -z "$RESULT" ] || [ "$RESULT" = "null" ]; then
            echo "BLOCKED: Track $TRACK_ID has no result yet." >&2
            exit 2
        fi
    done <<< "$TRACK_IDS"

    exit 0
fi

# ── Path 5: Integration gate enforcement ──
if [ "$IS_GATE_INTEGRATION" = "true" ]; then
    INTEGRATION_STATUS=$(jq -r '.integration.status // empty' "$STATE_FILE" 2>/dev/null)

    if [ "$INTEGRATION_STATUS" != "validated" ] && [ "$INTEGRATION_STATUS" != "single_pass" ] && \
       [ "$INTEGRATION_STATUS" != "combined" ] && [ "$INTEGRATION_STATUS" != "exhausted" ]; then
        echo "BLOCKED: Integration gate requires integration.status to be 'validated', 'single_pass', 'combined', or 'exhausted'." >&2
        echo "  Current integration.status: '${INTEGRATION_STATUS:-empty}'" >&2
        exit 2
    fi

    exit 0
fi

exit 0
