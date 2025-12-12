#!/usr/bin/env bash
set -euo pipefail

# Repo-root runner for the MoE monokernel optimizer prompt in *non-interactive* mode.
#
# Usage:
#   ./.codex/scripts/moe-monokernel-optimizer-exec.sh MODEL_ID="..." HARDWARE="L40S" DTYPE=fp8 TP=1 TOPK=8 MODE=full
#
# Notes:
# - This renders the same prompt used by `/prompts:moe-monokernel-optimizer` and pipes it to:
#     codex exec --cd <repo_root> --full-auto -
# - Customize CODEX_EXEC_FLAGS if you want stricter sandbox/approval settings.

# Resolve repo root
if REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(pwd)"
fi

export CODEX_HOME="${REPO_ROOT}/.codex"

# Non-interactive guardrails (avoid hangs)
export GIT_TERMINAL_PROMPT=0
export GIT_EDITOR=true
export VISUAL=true
export EDITOR=true
export PAGER=cat

PROMPT_FILE="${CODEX_HOME}/prompts/moe-monokernel-optimizer.md"
RENDERER="${CODEX_HOME}/scripts/render_prompt.py"

if [[ ! -f "${PROMPT_FILE}" ]]; then
  echo "error: prompt file not found: ${PROMPT_FILE}" >&2
  exit 2
fi
if [[ ! -x "${RENDERER}" ]]; then
  echo "error: renderer not found or not executable: ${RENDERER}" >&2
  exit 2
fi

# Default flags for `codex exec`.
# Override by exporting CODEX_EXEC_FLAGS, e.g.:
#   export CODEX_EXEC_FLAGS="--sandbox workspace-write"
CODEX_EXEC_FLAGS="${CODEX_EXEC_FLAGS:---full-auto}"

# Render prompt -> codex exec (stdin)
python3 "${RENDERER}" --prompt-file "${PROMPT_FILE}" "$@" | \
  codex exec --cd "${REPO_ROOT}" ${CODEX_EXEC_FLAGS} -
