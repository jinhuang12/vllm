#!/bin/bash
# PreToolUse hook — AMMO campaign production-parity reminders.
#
# Warns (but does NOT block) when a Bash command contains patterns that
# violate AMMO non-negotiables. Hard blocking caused false positives on
# git commits, grep searches, documentation writes, and compound commands
# where the pattern appeared in a non-violating context. Since the agent
# already knows the non-negotiables from SKILL.md, a reminder is sufficient
# — primary enforcement lives in the skill instructions and DA audits.
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)
# Try both field names (Claude Code docs ambiguous: tool_input vs input)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // .input.command // empty' 2>/dev/null) || true
[ -z "$COMMAND" ] && exit 0

# Fast bail: not AMMO context
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
ls "$PROJECT_DIR"/kernel_opt_artifacts/*/state.json &>/dev/null || exit 0

# Suppress noisy warnings on read-only / inspection commands
if echo "$COMMAND" | grep -qP '^\s*(grep|rg|cat|head|tail|less|find|ag|ack|env|printenv|echo|printf|jq|wc|python\s+-c|git\s+(log|show|diff|blame|commit|tag|stash|rebase|cherry-pick))\b'; then
    exit 0
fi

# N1: Production parity reminders
if echo "$COMMAND" | grep -qP 'TORCH_COMPILE_DISABLE\s*=\s*1'; then
    echo "AMMO REMINDER: TORCH_COMPILE_DISABLE=1 detected. AMMO non-negotiable N1 requires production parity (CUDA graphs + torch.compile). If this is a false positive (e.g., documentation or search), ignore this warning." >&2
fi
if echo "$COMMAND" | grep -qP '(--|=)enforce[_-]eager'; then
    echo "AMMO REMINDER: --enforce-eager detected. AMMO non-negotiable N1 requires CUDA graphs to be enabled. If this is a false positive (e.g., documentation or search), ignore this warning." >&2
fi
if echo "$COMMAND" | grep -qP 'VLLM_TORCH_COMPILE_LEVEL\s*=\s*[01](\s|$|")'; then
    echo "AMMO REMINDER: VLLM_TORCH_COMPILE_LEVEL < 2 detected. AMMO non-negotiable N1 requires torch.compile level ≥ 2. If this is a false positive (e.g., documentation or search), ignore this warning." >&2
fi

# N4: Sweep script mandate reminder
if echo "$COMMAND" | grep -qP 'vllm\s+bench\s+latency' && \
   ! echo "$COMMAND" | grep -q 'run_vllm_bench_latency_sweep'; then
    echo "AMMO REMINDER: Raw 'vllm bench latency' detected. AMMO non-negotiable N4 requires using the sweep script:" >&2
    echo "  python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py --artifact-dir <dir>" >&2
    echo "If this is a false positive (e.g., documentation or search), ignore this warning." >&2
fi

# ── GPU Reservation Auto-Reserve ──
GPU_RES_DIR="${AMMO_GPU_RES_DIR:-/tmp/ammo_gpu_res}"
SESSION_ID="${CLAUDE_SESSION_ID:-}"
# Scripts always live relative to THIS hook file (not CLAUDE_PROJECT_DIR)
HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${HOOK_DIR}/../skills/ammo/scripts"

# Detect if this command is GPU-heavy (conservative):
# Pattern 1: python/pytest with GPU-related keywords
# Pattern 2: python -c with GPU imports
# Pattern 3: nsys/ncu profilers, or nvidia-smi --query-compute
# Exemptions: bare import-only python -c calls
IS_GPU_CMD=false
if echo "$COMMAND" | grep -qP '\b(nsys|ncu)\b' || \
   echo "$COMMAND" | grep -qP 'nvidia-smi\s+--query-compute'; then
    IS_GPU_CMD=true
elif echo "$COMMAND" | grep -qP '\b(python3?|pytest)\b' && \
     echo "$COMMAND" | grep -qiP '(torch|cuda|triton|vllm|benchmark|kernel|gpu)'; then
    # Exemption: python -c "import vllm" / python -c "import torch" (bare import, no other GPU keywords)
    if echo "$COMMAND" | grep -qP '^\s*python3?\s+-c\s+["\x27]import\s+(vllm|torch)["\x27]\s*$'; then
        IS_GPU_CMD=false
    else
        IS_GPU_CMD=true
    fi
elif echo "$COMMAND" | grep -qP 'python3?\s+-c' && \
     echo "$COMMAND" | grep -qiP '(torch|cuda|triton)'; then
    IS_GPU_CMD=true
fi

if [ "$IS_GPU_CMD" = "false" ]; then
    exit 0
fi

# Extract CUDA_VISIBLE_DEVICES from command
CVD_VALUE=""
CVD_FOUND=false
CVD_EMPTY=false

# Check for CVD with digit IDs: CUDA_VISIBLE_DEVICES=0 or =0,1,2,3
if echo "$COMMAND" | grep -qP 'CUDA_VISIBLE_DEVICES=[0-9]'; then
    CVD_VALUE=$(echo "$COMMAND" | grep -oP 'CUDA_VISIBLE_DEVICES=\K[0-9][0-9,]*')
    CVD_FOUND=true
# Check for CVD empty string: ="" or ='' or empty-value forms
elif echo "$COMMAND" | grep -qP "CUDA_VISIBLE_DEVICES=(\"\"|\x27\x27)"; then
    CVD_EMPTY=true
    CVD_FOUND=true
fi

# ── Case A: CVD has explicit GPU IDs ──
if [ "$CVD_FOUND" = "true" ] && [ "$CVD_EMPTY" = "false" ]; then
    # Determine lease duration: 4h for nsys, 2h otherwise
    LEASE_HOURS=2
    if echo "$COMMAND" | grep -qP '\bnsys\b'; then
        LEASE_HOURS=4
    fi

    # Compute command hash
    CMD_HASH=$(echo -n "$COMMAND" | python3 -c "import sys,hashlib; print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest()[:16])")

    # Extract GPU IDs as a Python list expression
    GPU_IDS_PY=$(echo "$CVD_VALUE" | python3 -c "import sys; s=sys.stdin.read().strip(); print([int(x) for x in s.split(',') if x.strip()])")

    # Pass SNIPPET via environment to avoid shell quoting issues
    SNIPPET="${COMMAND:0:80}"

    # Call write_reservation via inline Python (snippet passed via env var _AMMO_SNIPPET)
    RES_RESULT=$(_AMMO_SNIPPET="$SNIPPET" python3 -c "
import sys, os
sys.path.insert(0, '${SCRIPTS_DIR}')
os.environ['AMMO_GPU_RES_DIR'] = '${GPU_RES_DIR}'
import gpu_reservation as gr
import pathlib
gr.STATE_DIR = pathlib.Path('${GPU_RES_DIR}')
try:
    gr.write_reservation(
        gpu_ids=${GPU_IDS_PY},
        cmd_hash='${CMD_HASH}',
        session_id='${SESSION_ID}',
        cvd_requested='${CVD_VALUE}',
        command_snippet=os.environ.get('_AMMO_SNIPPET', ''),
        lease_hours=${LEASE_HOURS},
    )
    print('OK')
except gr.ReservationError as e:
    print('RESERVATION_ERROR:' + str(e))
except gr.LockTimeoutError as e:
    print('LOCK_TIMEOUT:' + str(e))
except Exception as e:
    print('OTHER_ERROR:' + str(e))
" 2>&1)

    if echo "$RES_RESULT" | grep -q '^OK'; then
        exit 0
    elif echo "$RES_RESULT" | grep -q '^RESERVATION_ERROR:'; then
        ERR_MSG=$(echo "$RES_RESULT" | sed 's/^RESERVATION_ERROR://')
        cat >&2 <<EOF
AMMO GPU RESERVATION BLOCKED: GPU already reserved.
$ERR_MSG

To proceed: wait for the holding command to finish (it will auto-release),
or use a different GPU with explicit CUDA_VISIBLE_DEVICES=<free_gpu_id>.
To inspect current reservations: cat ${GPU_RES_DIR}/state.json
EOF
        exit 2
    elif echo "$RES_RESULT" | grep -q '^LOCK_TIMEOUT:'; then
        cat >&2 <<EOF
AMMO GPU RESERVATION BLOCKED: Lock timeout — could not acquire reservation lock.
Another process may be holding the lock. Retry in a few seconds.
To inspect: ls -la ${GPU_RES_DIR}/
EOF
        exit 2
    else
        # Unexpected error — fail-open to avoid blocking legitimate work
        echo "AMMO GPU RESERVATION WARNING: Unexpected reservation error: $RES_RESULT" >&2
        exit 0
    fi
fi

# ── Case B: CVD is empty string (agent says no GPU needed) ──
if [ "$CVD_FOUND" = "true" ] && [ "$CVD_EMPTY" = "true" ]; then
    # Sanity check: warn if command looks GPU-heavy despite CVD=""
    if echo "$COMMAND" | grep -qP '\b(run_vllm_bench|benchmark_kernel|nsys\s+profile|ncu)\b'; then
        echo "AMMO GPU WARNING: CUDA_VISIBLE_DEVICES is empty but command looks GPU-heavy. If this is intentional, ignore." >&2
    fi
    exit 0
fi

# ── Case C: No CUDA_VISIBLE_DEVICES at all ──
if [ "$CVD_FOUND" = "false" ]; then
    if [ -z "$SESSION_ID" ]; then
        # No session ID — fail-open
        exit 0
    fi

    WARNED_FLAG="${GPU_RES_DIR}/.warned_${SESSION_ID}"
    mkdir -p "$GPU_RES_DIR"

    if [ ! -f "$WARNED_FLAG" ]; then
        touch "$WARNED_FLAG"
        cat >&2 <<EOF
AMMO GPU RESERVATION: No CUDA_VISIBLE_DEVICES set on GPU command.

GPU commands MUST specify CUDA_VISIBLE_DEVICES to enable automatic reservation
and prevent benchmark collisions. Example:

  CUDA_VISIBLE_DEVICES=0 python benchmark_kernel.py ...

Re-run the command with CUDA_VISIBLE_DEVICES=<gpu_id> set.
(This warning fires only once per session — subsequent commands without CVD will be allowed through.)
EOF
        exit 2
    else
        # Already warned this session — trust agent judgment
        exit 0
    fi
fi

exit 0
