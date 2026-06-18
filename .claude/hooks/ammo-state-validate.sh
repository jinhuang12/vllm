#!/bin/bash
# PostToolUse hook — validates state.json against .claude/schemas/state.schema.json.
#
# Fires on Write|Edit (file_path match) AND on Bash (command-string detection).
# Uses python jsonschema for full validation. Blocks via decision:block on violations.
set -euo pipefail
trap 'exit 0' ERR

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""' 2>/dev/null) || true

# --- Bash detection path ---
# When fired on Bash, tool_input has .command not .file_path.
# Detect state.json writes by inspecting the command string.
if [ -z "$FILE_PATH" ]; then
    # Skip if the Bash command failed (write likely didn't complete)
    TOOL_STATUS=$(echo "$INPUT" | jq -r '.tool_response.status // "success"' 2>/dev/null) || true
    [ "$TOOL_STATUS" = "error" ] && exit 0

    COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // ""' 2>/dev/null) || true
    [ -z "$COMMAND" ] && exit 0

    # Fast filter: bail if command doesn't mention state.json at all
    case "$COMMAND" in
        *state.json*) ;;
        *) exit 0;;
    esac

    # Bail if CLAUDE_PROJECT_DIR is not set (can't locate state.json reliably)
    [ -z "${CLAUDE_PROJECT_DIR:-}" ] && exit 0

    # Exclude read-only commands (cat, grep, jq without redirect, head, tail, etc.)
    # Only proceed if the command looks like a write (>, mv, tee, open(...,'w'), json.dump)
    LOOKS_LIKE_WRITE=false
    case "$COMMAND" in
        *'> '*state.json*|*'>>'*state.json*) LOOKS_LIKE_WRITE=true;;
        *'mv '*state.json*|*'mv '*'state.json'*) LOOKS_LIKE_WRITE=true;;
        *'tee '*state.json*) LOOKS_LIKE_WRITE=true;;
        *"open("*state.json*"'w'"*) LOOKS_LIKE_WRITE=true;;
        *"open("*state.json*'"w"'*) LOOKS_LIKE_WRITE=true;;
        *'open('*state.json*'\"w\"'*) LOOKS_LIKE_WRITE=true;;
        *'json.dump'*state.json*) LOOKS_LIKE_WRITE=true;;
        *'json.dump'*'state.json'*) LOOKS_LIKE_WRITE=true;;
        *'write_text'*state.json*|*'write_bytes'*state.json*) LOOKS_LIKE_WRITE=true;;
        *'sed -i'*state.json*) LOOKS_LIKE_WRITE=true;;
        *'cp '*state.json*) LOOKS_LIKE_WRITE=true;;
    esac
    [ "$LOOKS_LIKE_WRITE" = "false" ] && exit 0

    # Extract the actual target path from the command.
    # Look for kernel_opt_artifacts/*/state.json patterns in the command text.
    # Allow path chars including those inside quotes (Python open('path/state.json','w'))
    EXTRACTED=$(echo "$COMMAND" | grep -oE '[A-Za-z0-9_./-]*kernel_opt_artifacts/[A-Za-z0-9_./-]+/state\.json' | head -1) || true

    if [ -n "$EXTRACTED" ]; then
        # Resolve relative path against CLAUDE_PROJECT_DIR
        if [[ "$EXTRACTED" = /* ]]; then
            FILE_PATH="$EXTRACTED"
        else
            FILE_PATH="$CLAUDE_PROJECT_DIR/$EXTRACTED"
        fi
    else
        # Command mentions state.json + looks like a write but path doesn't contain
        # kernel_opt_artifacts — this is an unrelated state.json write. Bail.
        exit 0
    fi

    [ -f "$FILE_PATH" ] || exit 0
else
    # --- Write/Edit path (original logic) ---
    case "$FILE_PATH" in
        */kernel_opt_artifacts/*/state.json) ;;
        *) exit 0;;
    esac
    [ -f "$FILE_PATH" ] || exit 0
fi

# Walk up from state.json to find the schema.
# Stop at .git boundary to avoid picking up unrelated schemas.
DIR=$(dirname "$FILE_PATH")
SCHEMA=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if [ -f "$DIR/.claude/schemas/state.schema.json" ]; then
        SCHEMA="$DIR/.claude/schemas/state.schema.json"
        break
    fi
    PARENT=$(dirname "$DIR")
    [ "$PARENT" = "$DIR" ] && break
    # Stop at git root — don't walk above the worktree
    [ -d "$DIR/.git" ] && break
    DIR="$PARENT"
done

[ -z "$SCHEMA" ] && exit 0

# Pass paths via env vars to avoid shell injection
ERRORS=$(STATE_FILE="$FILE_PATH" SCHEMA_FILE="$SCHEMA" python3 <<'PY'
import json, os, sys

try:
    from jsonschema import Draft202012Validator
except ImportError:
    print("jsonschema not installed — skipping validation", file=sys.stderr)
    sys.exit(0)

try:
    with open(os.environ['STATE_FILE']) as f:
        state = json.load(f)
    with open(os.environ['SCHEMA_FILE']) as f:
        schema = json.load(f)
except (json.JSONDecodeError, OSError, KeyError):
    sys.exit(0)

validator = Draft202012Validator(schema)
errors = []
for err in sorted(validator.iter_errors(state), key=lambda e: list(e.absolute_path)):
    path = '.'.join(str(p) for p in err.absolute_path) or '(root)'
    msg = err.message
    if len(msg) > 200:
        msg = msg[:200] + '...'
    errors.append(f'  - {path}: {msg}')

if errors:
    n = len(errors)
    out = '\n'.join(errors[:10])
    if n > 10:
        out += f'\n  ... and {n - 10} more'
    print(out)
PY
) || true

if [ -z "$ERRORS" ]; then
    # Cross-field: Stage 6 requires all tracks in terminal status (PASS, GATED_PASS, FAIL).
    # Prevents the orchestrator from advancing to integration while any track is still
    # IN_PROGRESS, GATING_REQUIRED, or GPU_BLOCKED (see Track A17 — session 6327c5d6 race).
    STAGE=$(jq -r '.campaign.current_stage // ""' "$FILE_PATH" 2>/dev/null) || STAGE=""
    if [ "$STAGE" = "6_integration" ]; then
        NON_TERMINAL_COUNT=$(jq -r '
            [.campaign.rounds[(.campaign.current_round - 1)].parallel_tracks.tracks // {}
             | to_entries[].value.status]
            | map(select(. != "PASS" and . != "GATED_PASS" and . != "FAIL"))
            | length
        ' "$FILE_PATH" 2>/dev/null) || NON_TERMINAL_COUNT=0
        if [ "${NON_TERMINAL_COUNT:-0}" -gt 0 ]; then
            NON_TERMINAL_LIST=$(jq -r '
                [.campaign.rounds[(.campaign.current_round - 1)].parallel_tracks.tracks // {}
                 | to_entries[]
                 | select(.value.status != "PASS" and .value.status != "GATED_PASS" and .value.status != "FAIL")
                 | "\(.key)=\(.value.status)"]
                | join(", ")
            ' "$FILE_PATH" 2>/dev/null) || NON_TERMINAL_LIST=""
            REASON=$(printf 'Stage 6 transition blocked: %s track(s) still non-terminal (%s). All tracks must reach PASS/GATED_PASS/FAIL before current_stage=6_integration.' "$NON_TERMINAL_COUNT" "$NON_TERMINAL_LIST")
            jq -c -n --arg reason "$REASON" '{
                "decision": "block",
                "reason": $reason,
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": $reason
                }
            }'
            exit 0
        fi
    fi

    # ── Audit gate: block stage transitions that skip audits ──
    # LEGACY GATE: Only enforce when `audit` key is PRESENT in the current round.
    # If entirely absent (old campaign), skip — preserves backward compat.
    CR_IDX=$(jq -r '(.campaign.current_round - 1)' "$FILE_PATH" 2>/dev/null) || CR_IDX=0
    AUDIT_EXISTS=$(jq -r ".campaign.rounds[$CR_IDX] | has(\"audit\")" "$FILE_PATH" 2>/dev/null) || AUDIT_EXISTS="false"

    if [ "$AUDIT_EXISTS" = "true" ]; then
        AUDIT_KEY=""
        case "$STAGE" in
            2_bottleneck_mining) AUDIT_KEY="stage_1" ;;
            3_debate)            AUDIT_KEY="stage_2" ;;
            6_integration)       AUDIT_KEY="stage_45" ;;
            7_campaign_eval*)    AUDIT_KEY="stage_67" ;;
        esac

        # Post-SHIP re-mine exemption: a round N>1 entering 2_bottleneck_mining is
        # a re-mine on a shifted baseline — Stage 1 (fresh baseline capture) is
        # eliminated post-SHIP (T16; SKILL.md §"Baseline promotion on SHIP"), so no
        # same-round stage_1 audit exists or is expected. Requiring it would deadlock
        # round N>1 mining.
        #
        # FAIL-CLOSED CONDITION: only drop the same-round stage_1 requirement when the
        # PREVIOUS round carries an `audit` key — that is exactly when the new-round-start
        # gate below fires and enforces the predecessor's stage_67. If the previous round
        # omits its audit key entirely (a legacy/mixed-state round), that gate is silent
        # too; dropping stage_1 here would let a round begin mining with no audit chain
        # anywhere. In that case we keep the stage_1 requirement so the write fails closed.
        if [ "$AUDIT_KEY" = "stage_1" ] && [ "${CR_IDX:-0}" -gt 0 ]; then
            PREV_HAS_AUDIT=$(jq -r ".campaign.rounds[$(( CR_IDX - 1 ))] | has(\"audit\")" "$FILE_PATH" 2>/dev/null) || PREV_HAS_AUDIT="false"
            if [ "$PREV_HAS_AUDIT" = "true" ]; then
                AUDIT_KEY=""
            fi
        fi

        # stage_2 gate only applies to schema v4.1+ campaigns
        if [ "$AUDIT_KEY" = "stage_2" ]; then
            SCHEMA_VER=$(jq -r '.campaign.schema_version // "4.0"' "$FILE_PATH" 2>/dev/null) || SCHEMA_VER="4.0"
            SCHEMA_MAJOR=$(echo "$SCHEMA_VER" | cut -d. -f1)
            SCHEMA_MINOR=$(echo "$SCHEMA_VER" | cut -d. -f2)
            if [ "$SCHEMA_MAJOR" -lt 4 ] || ([ "$SCHEMA_MAJOR" -eq 4 ] && [ "$SCHEMA_MINOR" -lt 1 ]); then
                AUDIT_KEY=""
            fi
        fi

        if [ -n "$AUDIT_KEY" ]; then
            AUDIT_PASSED=$(jq -r ".campaign.rounds[$CR_IDX].audit.${AUDIT_KEY}.passed_at // \"\"" "$FILE_PATH" 2>/dev/null) || AUDIT_PASSED=""
            # Backward compat: accept legacy stage_6 if stage_67 not set (pre-consolidation campaigns)
            if [ -z "$AUDIT_PASSED" ] && [ "$AUDIT_KEY" = "stage_67" ]; then
                AUDIT_PASSED=$(jq -r ".campaign.rounds[$CR_IDX].audit.stage_6.passed_at // \"\"" "$FILE_PATH" 2>/dev/null) || AUDIT_PASSED=""
            fi
            if [ -z "$AUDIT_PASSED" ]; then
                REASON="Audit gate (4-phase audit): transition to ${STAGE} blocked — audit.${AUDIT_KEY}.passed_at not set in current round. Spawn ammo-auditor for ${AUDIT_KEY} first (see .claude/skills/ammo/orchestration/audit-protocol.md)."
                jq -c -n --arg reason "$REASON" '{
                    "decision": "block",
                    "reason": $reason,
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": $reason
                    }
                }'
                exit 0
            fi
        fi
    fi

    # ── New-round start gate: block round N+1 without audit.stage_67 (or legacy stage_7) on round N ──
    CR=$(jq -r '.campaign.current_round // 1' "$FILE_PATH" 2>/dev/null) || CR=1
    if [ "$CR" -gt 1 ]; then
        PREV_IDX=$(( CR - 2 ))
        PREV_AUDIT_EXISTS=$(jq -r ".campaign.rounds[$PREV_IDX] | has(\"audit\")" "$FILE_PATH" 2>/dev/null) || PREV_AUDIT_EXISTS="false"
        if [ "$PREV_AUDIT_EXISTS" = "true" ]; then
            case "$STAGE" in
                1_baseline|2_bottleneck_mining|3_debate)
                    # Accept either consolidated stage_67 or legacy stage_7
                    PREV_S67_PASSED=$(jq -r ".campaign.rounds[$PREV_IDX].audit.stage_67.passed_at // \"\"" "$FILE_PATH" 2>/dev/null) || PREV_S67_PASSED=""
                    if [ -z "$PREV_S67_PASSED" ]; then
                        PREV_S67_PASSED=$(jq -r ".campaign.rounds[$PREV_IDX].audit.stage_7.passed_at // \"\"" "$FILE_PATH" 2>/dev/null) || PREV_S67_PASSED=""
                    fi
                    if [ -z "$PREV_S67_PASSED" ]; then
                        REASON="Audit gate (4-phase audit): new round start blocked — audit.stage_67.passed_at not set on previous round (round $((PREV_IDX+1))). Spawn ammo-auditor for stage_67 first."
                        jq -c -n --arg reason "$REASON" '{
                            "decision": "block",
                            "reason": $reason,
                            "hookSpecificOutput": {
                                "hookEventName": "PostToolUse",
                                "additionalContext": $reason
                            }
                        }'
                        exit 0
                    fi
                    ;;
            esac
        fi
    fi

    exit 0
fi

REASON=$(printf "state.json violates schema (%s):\n%s\nFix the values and retry the write." "$SCHEMA" "$ERRORS")

jq -c -n --arg reason "$REASON" '
{
    "decision": "block",
    "reason": $reason,
    "hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": $reason
    }
}
'
exit 0
