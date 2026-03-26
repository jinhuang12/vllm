#!/bin/bash
# Test harness for ammo-pretool-guard.sh (PreToolUse hook)
# Run: bash .claude/hooks/test-ammo-pretool-guard.sh
#
# Tests:
#   Bug S7a — Explicit CVD bypass: CUDA_VISIBLE_DEVICES=0 should NOT skip reservation check
#   Bug S7b — Prefixed vllm/torchrun detection: CUDA_VISIBLE_DEVICES=0 vllm bench latency
#   Regression — Existing functionality preserved

set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/ammo-pretool-guard.sh"
PASS=0
FAIL=0
TOTAL=0

# Create temporary test environment
TMPDIR=$(mktemp -d)
ARTIFACT_DIR="$TMPDIR/kernel_opt_artifacts/test_target"
mkdir -p "$ARTIFACT_DIR"

# Create minimal state.json so AMMO campaign is detected as active
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{ "campaign": { "status": "active" } }
EOF

GPU_RES_DIR="$TMPDIR/gpu_res"
mkdir -p "$GPU_RES_DIR"

cleanup() {
    rm -rf "$TMPDIR"
    rm -f /tmp/hook-stderr
}
trap cleanup EXIT

run_test() {
    local test_name="$1"
    local expected_exit="$2"
    local command_str="$3"
    local actual_exit=0

    TOTAL=$((TOTAL + 1))
    # Clean warned flag before each test
    rm -f "$GPU_RES_DIR/.warned_"* 2>/dev/null || true

    # Build hook input JSON (PreToolUse Bash)
    local json_input
    json_input=$(jq -n --arg cmd "$command_str" '{
        hook_event_name: "PreToolUse",
        tool_name: "Bash",
        tool_input: { command: $cmd }
    }')

    echo "$json_input" | env \
        HOME="$TMPDIR" \
        CLAUDE_PROJECT_DIR="$TMPDIR" \
        CLAUDE_SESSION_ID="test-session" \
        AMMO_GPU_RES_DIR="$GPU_RES_DIR" \
        bash "$HOOK" 2>/tmp/hook-stderr || actual_exit=$?

    if [ "$actual_exit" -eq "$expected_exit" ]; then
        echo "  PASS [$TOTAL]: $test_name (exit=$actual_exit)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $test_name (expected=$expected_exit, got=$actual_exit)"
        echo "        stderr: $(head -3 /tmp/hook-stderr 2>/dev/null || echo '(none)')"
        FAIL=$((FAIL + 1))
    fi
}

# ══════════════════════════════════════════════════
echo "== Bug S7a: Explicit CVD bypass — should trigger reservation warning =="
# ══════════════════════════════════════════════════

# These commands hardcode CUDA_VISIBLE_DEVICES=0 without using gpu_reservation.py.
# They should trigger the one-shot warning (exit 2), NOT pass through silently (exit 0).

run_test "CUDA_VISIBLE_DEVICES=0 python benchmark.py → warn" 2 \
    'CUDA_VISIBLE_DEVICES=0 python benchmark.py --model foo --gpu'

run_test "CUDA_VISIBLE_DEVICES=0 ncu → warn" 2 \
    'CUDA_VISIBLE_DEVICES=0 ncu --set full python test_kernel.py'

run_test "CUDA_VISIBLE_DEVICES=1 python torch script → warn" 2 \
    'CUDA_VISIBLE_DEVICES=1 python run_cuda_test.py'

# ══════════════════════════════════════════════════
echo ""
echo "== Bug S7b: Prefixed vllm/torchrun — must be detected as GPU command =="
# ══════════════════════════════════════════════════

# vllm/torchrun with env prefix should be detected (the old ^\s* anchor missed these)
run_test "CUDA_VISIBLE_DEVICES=0 vllm bench latency → warn" 2 \
    'CUDA_VISIBLE_DEVICES=0 vllm bench latency --model /path/to/model'

run_test "CUDA_VISIBLE_DEVICES=0 torchrun script.py → warn" 2 \
    'CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py'

run_test "FOO=bar vllm serve model → warn" 2 \
    'FOO=bar vllm serve /path/to/model --tp 1'

# ══════════════════════════════════════════════════
echo ""
echo "== Regression: Reservation pattern still allowed =="
# ══════════════════════════════════════════════════

run_test "Reservation pattern → allow" 0 \
    'CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus 1) && CUDA_VISIBLE_DEVICES=$CVD python benchmark.py'

# ══════════════════════════════════════════════════
echo ""
echo "== Regression: Empty CVD still allowed =="
# ══════════════════════════════════════════════════

run_test "CUDA_VISIBLE_DEVICES=\"\" python script → allow" 0 \
    'CUDA_VISIBLE_DEVICES="" python benchmark.py --cuda --no-gpu'

# ══════════════════════════════════════════════════
echo ""
echo "== Regression: Non-GPU commands pass through =="
# ══════════════════════════════════════════════════

run_test "grep command → allow" 0 \
    'grep -r "kernel" .'

run_test "cat file → allow" 0 \
    'cat state.json'

run_test "git log → allow" 0 \
    'git log --oneline -10'

run_test "ls command → allow" 0 \
    'ls -la'

# ══════════════════════════════════════════════════
echo ""
echo "== Regression: Bare vllm/torchrun still detected =="
# ══════════════════════════════════════════════════

run_test "vllm bench latency (bare) → warn" 2 \
    'vllm bench latency --model /path/to/model'

run_test "torchrun script (bare) → warn" 2 \
    'torchrun --nproc_per_node=1 train.py'

# ══════════════════════════════════════════════════
echo ""
echo "== Regression: Import checks exempted =="
# ══════════════════════════════════════════════════

run_test "python -c 'import torch' → allow" 0 \
    "python -c 'import torch'"

# ══════════════════════════════════════════════════
echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
