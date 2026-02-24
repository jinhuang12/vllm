#!/bin/bash
# PreToolUse hook on TaskUpdate
# Blocks gate task completion if verification script hasn't been run.
# Also enforces iteration loop mechanics on KILL decisions.
#
# Logic:
# 1. Read tool_input JSON from stdin
# 2. If status != "completed", allow (exit 0)
# 3. Read task subject from task file
# 4. If subject matches "GATE:.*verify_" → check verification evidence + KILL enforcement
# 5. If subject matches "Route decision" or "B14" → check iteration task creation on KILL
# 6. Otherwise, allow (exit 0)

set -euo pipefail

# Read tool input from stdin
INPUT=$(cat)

# Extract tool name — only act on TaskUpdate
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name // empty' 2>/dev/null)
if [ "$TOOL_NAME" != "TaskUpdate" ]; then
    exit 0
fi

# Extract status from tool input
STATUS=$(echo "$INPUT" | jq -r '.tool_input.status // empty' 2>/dev/null)
if [ "$STATUS" != "completed" ]; then
    exit 0
fi

# Extract task ID
TASK_ID=$(echo "$INPUT" | jq -r '.tool_input.taskId // empty' 2>/dev/null)
if [ -z "$TASK_ID" ]; then
    exit 0
fi

# Find the task file — search across all team task directories
TASK_FILE=""
for dir in "$HOME/.claude/tasks"/*/; do
    candidate="$dir${TASK_ID}.json"
    if [ -f "$candidate" ]; then
        TASK_FILE="$candidate"
        break
    fi
done

if [ -z "$TASK_FILE" ] || [ ! -f "$TASK_FILE" ]; then
    # Task file not found — not our concern
    exit 0
fi

# Read task subject
SUBJECT=$(jq -r '.subject // empty' "$TASK_FILE" 2>/dev/null)

# Find artifact directory (needed for both gate checks and B14 enforcement)
ARTIFACT_DIR=""
for d in "$CLAUDE_PROJECT_DIR"/kernel_opt_artifacts/*/; do
    if [ -d "$d" ]; then
        ARTIFACT_DIR="$d"
        break
    fi
done

# ── Path 1: GATE verify tasks ──
if echo "$SUBJECT" | grep -qi "GATE:.*verify_"; then

    if [ -z "$ARTIFACT_DIR" ]; then
        echo "BLOCKED: No artifact directory found in kernel_opt_artifacts/. Cannot verify gate." >&2
        exit 2
    fi

    if echo "$SUBJECT" | grep -qi "verify_phase1"; then
        # Stage 1 gate — check for verification evidence
        STAGE1_STATUS=$(jq -r '.verification_run.stage1 // empty' "$ARTIFACT_DIR/state.json" 2>/dev/null)
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
        VAL_STATUS=$(jq -r '.verification_run.validation // empty' "$ARTIFACT_DIR/state.json" 2>/dev/null)
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

        # Evidence exists — now enforce KILL requires recorded attempt
        ROUTE=$(jq -r '.route_decision.route // empty' "$ARTIFACT_DIR/state.json" 2>/dev/null)
        if [ "$ROUTE" = "KILL" ]; then
            ATTEMPTS=$(jq '.opportunity_attempts | length' "$ARTIFACT_DIR/state.json" 2>/dev/null)
            if [ "$ATTEMPTS" = "0" ] || [ -z "$ATTEMPTS" ]; then
                echo "BLOCKED: Cannot complete B13 with KILL decision." >&2
                echo "  opportunity_attempts is empty — record the attempt first." >&2
                echo "  B14 (Route decision) handles iteration after B13 completes." >&2
                exit 2
            fi
        fi
        exit 0
    fi

    # Unknown verify_ pattern — allow
    exit 0
fi

# ── Path 2: B14 Route Decision enforcement ──
if echo "$SUBJECT" | grep -qi "Route.*decision\|B14"; then

    if [ -z "$ARTIFACT_DIR" ]; then
        echo "BLOCKED: No artifact directory found in kernel_opt_artifacts/." >&2
        exit 2
    fi

    ROUTE=$(jq -r '.route_decision.route // empty' "$ARTIFACT_DIR/state.json" 2>/dev/null)
    if [ "$ROUTE" = "KILL" ]; then
        # Verify iteration tasks (B15/B16) exist in the team task directory
        TASK_DIR=$(dirname "$TASK_FILE")
        ITERATION_FOUND=false
        for tf in "$TASK_DIR"/*.json; do
            subj=$(jq -r '.subject // empty' "$tf" 2>/dev/null)
            if echo "$subj" | grep -qi "B15\|updated.*plan\|iteration.*plan"; then
                ITERATION_FOUND=true
                break
            fi
        done
        if [ "$ITERATION_FOUND" = "false" ]; then
            echo "BLOCKED: Cannot complete B14 (KILL) without iteration tasks." >&2
            echo "  Create B15, B16, and B9'-B13' chain first." >&2
            exit 2
        fi
    fi
fi

exit 0
