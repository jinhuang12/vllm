#!/bin/bash
# PostToolUse hook — non-blocking warning when files land outside the
# canonical AMMO V2 artifact layout under kernel_opt_artifacts/.
#
# Spec: docs/superpowers/specs/2026-05-12-ammo-artifact-layout-design.md
# Reference: ai_cli_session/.claude/skills/ammo/references/artifact-layout.md
#
# Matchers (configured in settings.local.json): Write, Edit, Bash.
#   - Write/Edit: extracts tool_input.file_path
#   - Bash: scans command for `> path`, `mkdir -p path`, `--out PATH`, etc.,
#     limited to paths containing kernel_opt_artifacts/.
#
# Non-blocking: only emits additionalContext warnings. Never returns
# {"decision": "block"} — layout drift is an organizational hazard, not a
# correctness violation.
#
# Fail-open: any internal error (jq missing, malformed JSON) exits 0 silently.
set -euo pipefail
trap 'exit 0' ERR

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // ""' 2>/dev/null) || true
[ -z "$TOOL" ] && exit 0

# Allowed regex patterns relative to {artifact_dir}/. Any file path under
# kernel_opt_artifacts/{target}/ that does NOT match one of these triggers a
# warning. Keep in lockstep with references/artifact-layout.md § Prohibited
# Patterns.
#
# NB: op_id pattern uses [A-Za-z0-9_-]+ to admit existing OP-001 / op-001 /
# op_001 styles.
ALLOWED_PATTERNS=(
    '^state\.json$'
    '^target\.json$'
    '^REPORT\.md$'
    '^report_assets/'
    '^blockers/'
    '^env\.(json|md)$'
    '^validation_results\.md$'
    '^validation_summary\.json$'
    '^rounds/[0-9]+/constraints\.md$'
    '^rounds/[0-9]+/profiling/(probe|nsys|ncu|torch_profile)/'
    '^rounds/[0-9]+/sweeps/(baseline|opt/[A-Za-z0-9_-]+|integration|golden_capture)/'
    '^rounds/[0-9]+/mining/'
    '^rounds/[0-9]+/debate/(proposals|round_[0-9]+|micro_experiments|monitor_audits)/'
    '^rounds/[0-9]+/debate/summary\.md$'
    '^rounds/[0-9]+/tracks/[A-Za-z0-9_-]+/(validation_results\.md|validator_tests|monitor_audits|_scratch)'
    '^rounds/[0-9]+/audits/'
    '^rounds/[0-9]+/_archive/'
)

# Helper: emit a warning for a single non-conforming relative path.
emit_warn() {
    local rel="$1"
    local msg
    msg="LAYOUT WARN: ${rel} is outside the canonical AMMO V2 layout. Expected: rounds/{N}/{profiling|sweeps|mining|debate|tracks|audits|_archive}/... See ai_cli_session/.claude/skills/ammo/references/artifact-layout.md § Prohibited Patterns."
    jq -c -n --arg msg "$msg" '
    {
        hookSpecificOutput: {
            hookEventName: "PostToolUse",
            additionalContext: $msg
        }
    }'
}

# Helper: classify one absolute path. Returns 0 (conforming) or 1 (warn-worthy).
# Prints the relative path on warning.
check_path() {
    local p="$1"
    case "$p" in
        */kernel_opt_artifacts/*) ;;
        *) return 0;;
    esac

    # Strip the longest prefix up to and including kernel_opt_artifacts/{target}/.
    # The {target} segment is whatever follows kernel_opt_artifacts/.
    local tail="${p##*/kernel_opt_artifacts/}"
    # Drop the {target}/ segment, leaving the artifact-relative path.
    local rel="${tail#*/}"
    # If tail has no slash (path was just kernel_opt_artifacts/{file}), skip.
    [ "$rel" = "$tail" ] && return 0

    # Walk allowed patterns.
    for pat in "${ALLOWED_PATTERNS[@]}"; do
        if echo "$rel" | grep -Eq "$pat"; then
            return 0
        fi
    done

    emit_warn "$rel"
    return 1
}

case "$TOOL" in
    Write|Edit|MultiEdit)
        FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""' 2>/dev/null) || true
        [ -z "$FILE_PATH" ] && exit 0
        check_path "$FILE_PATH" || true
        ;;
    Bash)
        CMD=$(echo "$INPUT" | jq -r '.tool_input.command // ""' 2>/dev/null) || true
        [ -z "$CMD" ] && exit 0
        # Extract candidate paths under kernel_opt_artifacts/. We grep for
        # tokens that match `[^ ]*kernel_opt_artifacts/[^ ]+`, dedup, and
        # check each. Quotes are handled by the tokenizer below.
        # Use Python to pull out plausible path tokens robustly.
        PATHS=$(printf '%s' "$CMD" | python3 -c '
import sys, re
text = sys.stdin.read()
# Match tokens that contain kernel_opt_artifacts/ and look like paths.
# Stop at whitespace, backticks, or quote boundaries.
seen = set()
out = []
for m in re.finditer(r"[^\s`\"'\''<>|;]*kernel_opt_artifacts/[^\s`\"'\''<>|;)]+", text):
    p = m.group(0).rstrip("/.,;:")
    if p and p not in seen:
        seen.add(p)
        out.append(p)
for p in out:
    print(p)
' 2>/dev/null) || true
        if [ -n "$PATHS" ]; then
            while IFS= read -r p; do
                [ -z "$p" ] && continue
                check_path "$p" || true
            done <<< "$PATHS"
        fi
        ;;
    *)
        exit 0
        ;;
esac

exit 0
