#!/bin/bash
# PreToolUse hook — Block bare `python` / `python3` invocations of AMMO sweep
# scripts. The session .venv has vLLM editable-installed; system python does not.
# Only targets sweep/profiling scripts that import vllm — NOT management scripts
# like gpu_reservation.py, new_target.py, gpu_status.py which don't need vllm.
set -euo pipefail

if ! command -v jq &>/dev/null; then exit 0; fi

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null) || true
[ -z "$COMMAND" ] && exit 0

# Only fire on commands invoking vllm-dependent sweep/profiling scripts.
# These are the scripts that import vllm and MUST run under .venv/bin/python:
SWEEP_SCRIPTS="(run_vllm_bench_latency_sweep|nsys_probe|cutedsl_cudagraph_selftest)\.py"
if ! echo "$COMMAND" | grep -qP "python3?\s+\S*\.claude/skills/ammo/scripts/${SWEEP_SCRIPTS}"; then
    exit 0
fi

# Allow: .venv/bin/python prefix (relative or absolute)
if echo "$COMMAND" | grep -qP "\.venv/bin/python3?\s+\S*\.claude/skills/ammo/scripts/${SWEEP_SCRIPTS}"; then
    exit 0
fi

# Allow: source .venv/bin/activate && ... earlier in the command
if echo "$COMMAND" | grep -qP 'source\s+\S*\.venv/bin/activate\s*&&'; then
    exit 0
fi

cat >&2 <<'EOF'
BLOCKED: Use .venv/bin/python for sweep scripts (system python has no vllm)

Fix — use:
    .venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py ...
EOF
exit 2
