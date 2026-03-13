#!/bin/bash
set -euo pipefail
if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)
# Try both field names (Claude Code docs ambiguous: tool_input vs input)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // .input.command // empty' 2>/dev/null)
[ -z "$COMMAND" ] && exit 0

# Fast bail: not AMMO context
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
ls "$PROJECT_DIR"/kernel_opt_artifacts/*/state.json &>/dev/null || exit 0

# Skip read-only / inspection commands (prevent false positives)
if echo "$COMMAND" | grep -qP '^\s*(grep|rg|cat|head|tail|less|find|ag|ack|env|printenv|echo|printf|jq|wc|python\s+-c|git\s+(log|show|diff|blame|commit|tag|stash|rebase|cherry-pick))\b'; then
    exit 0
fi

# N1: Production parity
if echo "$COMMAND" | grep -qP 'TORCH_COMPILE_DISABLE\s*=\s*1'; then
    echo "BLOCKED: Production parity — TORCH_COMPILE_DISABLE=1. Remove this flag." >&2
    exit 2
fi
if echo "$COMMAND" | grep -qP '(--|=)enforce[_-]eager'; then
    echo "BLOCKED: Production parity — --enforce-eager. CUDA graphs must be enabled." >&2
    exit 2
fi
if echo "$COMMAND" | grep -qP 'VLLM_TORCH_COMPILE_LEVEL\s*=\s*[01](\s|$|")'; then
    echo "BLOCKED: Production parity — VLLM_TORCH_COMPILE_LEVEL must be ≥2." >&2
    exit 2
fi

# N4: Sweep script mandate
if echo "$COMMAND" | grep -qP 'vllm\s+bench\s+latency' && \
   ! echo "$COMMAND" | grep -q 'run_vllm_bench_latency_sweep'; then
    echo "BLOCKED: Use the sweep script, not raw 'vllm bench latency'." >&2
    echo "  python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py --artifact-dir <dir>" >&2
    exit 2
fi

exit 0
