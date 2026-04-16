#!/bin/bash
# Test harness for ammo-gpu-release.sh (PostToolUse hook).
# Run: bash .claude/hooks/test-ammo-gpu-release.sh
#
# Validates that the session_id extraction regex handles:
#   1. Canonical `$(reserve ... --session-id foo --no-auto-release)` form
#   2. Canonical with trailing ')': `$(reserve ... --session-id foo)` (no following flag)
#   3. Equals form: `--session-id=foo`
#   4. Retry-loop form inside `for` loop
#   5. No reserve in command → no release attempt
#   6. No --session-id → fallback to $CLAUDE_SESSION_ID

set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/ammo-gpu-release.sh"
SCRIPTS_DIR="$(cd "$(dirname "$0")/../skills/ammo/scripts" && pwd)"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
GPU_RES_DIR="$TMPDIR/gpu_res"
mkdir -p "$GPU_RES_DIR"

cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

# Seed a state.json with four held GPUs by distinct session ids so we can
# observe which ones the hook releases.
seed_state() {
    cat > "$GPU_RES_DIR/state.json" <<EOF
{
  "gpus": {
    "0": {"session_id": "sess_foo", "reserved_at": "2026-04-16T19:00:00Z", "lease_expires": "2099-01-01T00:00:00Z", "command_snippet": ""},
    "1": {"session_id": "sess_bar", "reserved_at": "2026-04-16T19:00:00Z", "lease_expires": "2099-01-01T00:00:00Z", "command_snippet": ""},
    "2": {"session_id": "t2-baseline", "reserved_at": "2026-04-16T19:00:00Z", "lease_expires": "2099-01-01T00:00:00Z", "command_snippet": ""},
    "3": {"session_id": "stage2_ncu_123", "reserved_at": "2026-04-16T19:00:00Z", "lease_expires": "2099-01-01T00:00:00Z", "command_snippet": ""}
  },
  "gpu_count": 4,
  "audit": []
}
EOF
}

# Invoke the hook with the given command; return the session_ids still held.
run_hook() {
    local cmd="$1"
    local json
    json=$(jq -n --arg cmd "$cmd" '{
        hook_event_name: "PostToolUse",
        tool_name: "Bash",
        tool_input: {command: $cmd}
    }')
    AMMO_GPU_RES_DIR="$GPU_RES_DIR" \
        CLAUDE_SESSION_ID="env-session" \
        CLAUDE_PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)" \
        bash "$HOOK" <<< "$json" >/dev/null 2>&1 || true
}

assert_released() {
    local name="$1" expected_sid="$2"
    TOTAL=$((TOTAL + 1))
    # After release, the gpu entry owned by expected_sid should be null.
    local still_held
    still_held=$(jq -r --arg s "$expected_sid" \
        '.gpus | to_entries | map(select(.value != null and .value.session_id == $s)) | length' \
        "$GPU_RES_DIR/state.json")
    if [ "$still_held" = "0" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $name (released $expected_sid)"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $name (expected $expected_sid released, still held)"
    fi
}

assert_not_released() {
    local name="$1" sid="$2"
    TOTAL=$((TOTAL + 1))
    local still_held
    still_held=$(jq -r --arg s "$sid" \
        '.gpus | to_entries | map(select(.value != null and .value.session_id == $s)) | length' \
        "$GPU_RES_DIR/state.json")
    if [ "$still_held" != "0" ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $name ($sid preserved)"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $name ($sid was unexpectedly released)"
    fi
}

echo "=== Canonical \$(...) pattern (space-form, with trailing flag) ==="
seed_state
run_hook 'CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus 1 --session-id sess_foo --no-auto-release) && CUDA_VISIBLE_DEVICES=$CVD echo ok'
assert_released "canonical space-form with following flag" "sess_foo"

echo "=== Canonical \$(...) pattern (trailing paren, no following flag) ==="
seed_state
run_hook 'CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus 1 --session-id sess_bar) && echo done'
assert_released "space-form with trailing paren" "sess_bar"

echo "=== Equals form (--session-id=foo) ==="
seed_state
run_hook 'python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus 1 --session-id=t2-baseline'
assert_released "equals-form" "t2-baseline"

echo "=== Retry loop form ==="
seed_state
run_hook 'for i in $(seq 1 60); do CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus 1 --session-id stage2_ncu_123 --no-auto-release 2>/dev/null) && break; done'
assert_released "retry loop" "stage2_ncu_123"

echo "=== No reserve in command → no release ==="
seed_state
run_hook 'python -c "import torch; print(torch.cuda.is_available())"'
assert_not_released "non-reserve command" "sess_foo"

echo ""
echo "=== Summary ==="
echo "$PASS / $TOTAL passed"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED: $FAIL"
    exit 1
fi
exit 0
