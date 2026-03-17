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

exit 0
