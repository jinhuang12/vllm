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

# Determine if an AMMO campaign is active
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
_CAMPAIGN_ACTIVE=false
ls "$PROJECT_DIR"/kernel_opt_artifacts/*/state.json &>/dev/null && _CAMPAIGN_ACTIVE=true

if [ "$_CAMPAIGN_ACTIVE" = "true" ]; then
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
fi

# ── GPU Pool Pattern Guard ──
# Detect GPU-heavy commands that DON'T use the reservation pattern.
# One-shot warning per session (same mechanism as ammo-stop-guard.sh).

# Detect if this command is likely GPU-heavy (conservative patterns)
IS_GPU_CMD=false
if echo "$COMMAND" | grep -qP '\b(nsys|ncu)\b' || \
   echo "$COMMAND" | grep -qP 'nvidia-smi\s+--query-compute'; then
    IS_GPU_CMD=true
elif echo "$COMMAND" | grep -qP '\b(python3?|pytest)\b' && \
     echo "$COMMAND" | grep -qiP '(torch|cuda|triton|vllm|benchmark|kernel|gpu)'; then
    # Exemption: bare import checks
    if echo "$COMMAND" | grep -qP '^\s*python3?\s+-c\s+["\x27]import\s+(vllm|torch)["\x27]\s*$'; then
        IS_GPU_CMD=false
    else
        IS_GPU_CMD=true
    fi
fi

if [ "$IS_GPU_CMD" = "false" ]; then
    exit 0
fi

# If command uses the reservation pattern, allow through
if echo "$COMMAND" | grep -q 'gpu_reservation.py reserve'; then
    exit 0
fi

# If command has explicit CUDA_VISIBLE_DEVICES=<digits>, allow through
# (backward compat for scripts that set CVD themselves)
if echo "$COMMAND" | grep -qP 'CUDA_VISIBLE_DEVICES=[0-9]'; then
    exit 0
fi

# If command has CUDA_VISIBLE_DEVICES="" (explicit no-GPU), allow through
if echo "$COMMAND" | grep -qP "CUDA_VISIBLE_DEVICES=(\"\"|\x27\x27)"; then
    exit 0
fi

# GPU command without reservation pattern — one-shot warning
SESSION_ID="${CLAUDE_SESSION_ID:-}"
if [ -z "$SESSION_ID" ]; then
    exit 0  # No session ID — fail-open
fi

GPU_RES_DIR="${AMMO_GPU_RES_DIR:-/tmp/ammo_gpu_res}"
WARNED_FLAG="${GPU_RES_DIR}/.warned_${SESSION_ID}"
mkdir -p "$GPU_RES_DIR"

if [ ! -f "$WARNED_FLAG" ]; then
    touch "$WARNED_FLAG"
    cat >&2 <<EOF
AMMO GPU POOL: GPU command detected without reservation.

Use the GPU pool pattern to acquire GPUs before running GPU commands:

  CVD=\$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus N) && CUDA_VISIBLE_DEVICES=\$CVD <your_command>

Or set CUDA_VISIBLE_DEVICES="" if no GPU is needed.
(This warning fires only once per session.)
EOF
    exit 2
else
    exit 0  # Already warned — trust agent judgment
fi

exit 0
