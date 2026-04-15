#!/usr/bin/env bash
# ammo-artifact-catalog.sh — Stop hook that runs Sonnet to catalog campaign artifacts.
# Runs synchronously — async scheduling is handled by the hook config.

# Recursion guard: if we're already inside a catalog run, bail
[[ "${AMMO_CATALOG_ACTIVE:-}" == "1" ]] && exit 0

# Claude CLI required
command -v claude &>/dev/null || exit 0

# Resolve project dir
PROJECT_DIR="${CLAUDE_PROJECT_DIR:-.}"
PROMPT_FILE="$PROJECT_DIR/.claude/hooks/artifact-catalog-prompt.md"

# Prompt file must exist
[[ -f "$PROMPT_FILE" ]] || exit 0

# Campaign must be active (state.json exists)
STATE_FILE=$(find "$PROJECT_DIR/kernel_opt_artifacts" -maxdepth 2 -name "state.json" 2>/dev/null | head -1)
[[ -z "$STATE_FILE" ]] && exit 0

export AMMO_CATALOG_ACTIVE=1
export CLAUDE_CODE_USE_BEDROCK=1
cd "$PROJECT_DIR" || exit 1

claude \
  --model "global.anthropic.claude-sonnet-4-6" \
  --allowedTools "Glob,Read,Write,Edit,Bash" \
  --no-session-persistence \
  -p "$(cat "$PROMPT_FILE")" \
  >/dev/null 2>&1
