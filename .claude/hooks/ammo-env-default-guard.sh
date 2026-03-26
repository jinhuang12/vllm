#!/bin/bash
# PreToolUse hook — AMMO env var default-off enforcement.
#
# Fires on Edit and Write tool calls. Blocks edits to envs.py that register
# VLLM_OP* env vars with default True/1. This prevents cross-track
# contamination where a prior round's gating flag silently activates stale
# optimizations during subsequent sweeps.
#
# Convention: VLLM_OP* flags must default to False/0/"0". The sweep harness
# enables them explicitly via opt_env in target.json.
#
# Exit 0 = allow, Exit 2 = block (stderr fed back to agent).
set -euo pipefail

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)

# Extract file path from Edit or Write tool input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null) || true
[ -z "$FILE_PATH" ] && exit 0

# Only inspect edits to envs.py files
case "$FILE_PATH" in
    */envs.py) ;;
    *) exit 0 ;;
esac

# Extract the content being written
# Edit tool: new_string field; Write tool: content field
NEW_CONTENT=$(echo "$INPUT" | jq -r '.tool_input.new_string // .tool_input.content // empty' 2>/dev/null) || true
[ -z "$NEW_CONTENT" ] && exit 0

# Check for VLLM_OP\d+ with default True/1
# Patterns to catch:
#   VLLM_OP001 = True
#   VLLM_OP002: bool = True
#   "VLLM_OP003", True
#   default="1"  (in a line containing VLLM_OP)
#   default=True  (in a line containing VLLM_OP)
if echo "$NEW_CONTENT" | grep -qP 'VLLM_OP\d+.*(?:=\s*True|=\s*1\b|default\s*=\s*True|default\s*=\s*"1"|,\s*True)'; then
    cat >&2 <<'EOF'
BLOCKED: VLLM_OP* env var registered with default True/1.

Cross-track contamination prevention requires all VLLM_OP* flags to default
to False (or 0/"0"). The E2E sweep harness enables them explicitly via
opt_env in target.json.

Fix: Change the default to False (or 0/"0"). Examples:
  VLLM_OP001: bool = False
  "VLLM_OP001": lambda: bool(os.getenv("VLLM_OP001", "0") == "1")

See: integration-logic.md § Per-Track Environment Variable Isolation
EOF
    exit 2
fi

exit 0
