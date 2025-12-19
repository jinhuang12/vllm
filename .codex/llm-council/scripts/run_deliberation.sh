#!/bin/bash
# run_deliberation.sh - Multi-round critic deliberation (parallel by default)
#
# Usage:
#   bash run_deliberation.sh <round> [max_rounds] [--parallel|--sequential]
#   bash run_deliberation.sh <round> [max_rounds] --mode parallel|sequential
#
# Modes:
#   - parallel   (DEFAULT): Gemini and Claude Code run concurrently (independent critiques)
#   - sequential: Gemini runs first, then Claude Code runs *with Critic #1's same-round feedback*
#
# Notes:
# - Parallel mode is faster, but Critic #2 does NOT see Critic #1's same-round output.
# - Sequential mode is slower, but enables a "build on Critic #1" style review.
#
# Features:
#   - Graceful degradation: continues with available critics if one fails
#   - Full access (YOLO mode): critics can run commands and inspect/edit files
#   - Session ID tracking: robust multi-round resume using explicit session IDs
#
# Example:
#   bash run_deliberation.sh 1 3              # Round 1 of 3 (parallel default)
#   bash run_deliberation.sh 1 3 --sequential # Round 1 of 3 (sequential)
#   # Revise .llm-council/context.md based on feedback
#   bash run_deliberation.sh 2 3              # Round 2 of 3

set -e

show_help() {
  cat <<'EOF'
run_deliberation.sh - Multi-round critic deliberation (parallel by default)

Usage:
  bash run_deliberation.sh <round> [max_rounds] [--parallel|--sequential]
  bash run_deliberation.sh <round> [max_rounds] --mode parallel|sequential

Examples:
  bash run_deliberation.sh 1 3              # parallel (default)
  bash run_deliberation.sh 1 3 --sequential # sequential
  bash run_deliberation.sh 2 3 --parallel   # explicit parallel
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

ROUND=${1:?Usage: run_deliberation.sh <round> [max_rounds] [--parallel|--sequential]}
shift

MAX_ROUNDS=3
MODE="${LLM_COUNCIL_MODE:-parallel}"

# Optional second positional arg: max_rounds
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  MAX_ROUNDS="$1"
  shift
fi

# Parse remaining flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)
      MODE="parallel"
      shift
      ;;
    --sequential)
      MODE="sequential"
      shift
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1"
      echo ""
      show_help
      exit 1
      ;;
  esac
done

if [[ "$MODE" != "parallel" && "$MODE" != "sequential" ]]; then
  echo "ERROR: Invalid mode: $MODE (expected: parallel|sequential)"
  exit 1
fi

COUNCIL_DIR=".llm-council"

# Claude Code entrypoint
#
# Default behavior:
#   1) Prefer a bundled wrapper script named `claude-aws` (same directory as this file)
#   2) Otherwise, try `claude-aws` from PATH
#   3) Otherwise, fall back to `claude`
#
# You can override the command by setting:
#   export LLM_COUNCIL_CLAUDE_CMD="claude"        # direct Anthropic API / logged-in CLI
#   export LLM_COUNCIL_CLAUDE_CMD="claude-aws"    # your own wrapper/alias executable
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_CLAUDE_CMD="claude"
if [ -x "$SCRIPT_DIR/claude-aws" ]; then
  DEFAULT_CLAUDE_CMD="$SCRIPT_DIR/claude-aws"
elif command -v claude-aws &>/dev/null; then
  DEFAULT_CLAUDE_CMD="claude-aws"
fi
CLAUDE_CMD="${LLM_COUNCIL_CLAUDE_CMD:-$DEFAULT_CLAUDE_CMD}"

# If running as root/sudo, Claude Code may refuse `--dangerously-skip-permissions`.
# We only set IS_SANDBOX when running as root AND the user hasn't set it already.
CLAUDE_ENV=()
if [ "${EUID:-$(id -u)}" -eq 0 ] && [ -z "${IS_SANDBOX:-}" ]; then
  CLAUDE_ENV+=("IS_SANDBOX=true")
fi

# CLI availability flags (set by check_cli_availability)
GEMINI_AVAILABLE=false
CLAUDE_AVAILABLE=false

# ══════════════════════════════════════════════════════════════════════════════
# CLI AVAILABILITY CHECK
# ══════════════════════════════════════════════════════════════════════════════
check_cli_availability() {
  echo "Checking critic CLI availability..."

  # Test Gemini
  if command -v gemini &>/dev/null; then
    if gemini --version &>/dev/null 2>&1; then
      GEMINI_AVAILABLE=true
      echo "  ✓ Gemini CLI available"
    else
      echo "  ✗ Gemini CLI found but not functional (check authentication)"
    fi
  else
    echo "  ✗ Gemini CLI not installed"
  fi

  # Test Claude Code (via configured entrypoint)
  if [ -x "$CLAUDE_CMD" ] || command -v "$CLAUDE_CMD" &>/dev/null; then
    # `-v` prints the version without contacting the API
    if "$CLAUDE_CMD" -v &>/dev/null 2>&1; then
      CLAUDE_AVAILABLE=true
      echo "  ✓ Claude Code CLI available ($CLAUDE_CMD)"
    else
      echo "  ✗ Claude Code CLI found but not functional (check installation/auth/config)"
    fi
  else
    echo "  ✗ Claude Code CLI not installed (expected: $CLAUDE_CMD)"
  fi

  # Fail if none available
  if [ "$GEMINI_AVAILABLE" = false ] && [ "$CLAUDE_AVAILABLE" = false ]; then
    echo ""
    echo "ERROR: No critics available. Install Gemini CLI and/or Claude Code CLI and authenticate."
    exit 1
  fi

  echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
# SESSION ID MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
capture_gemini_session_id() {
  # Capture the most recent Gemini session ID.
  # `gemini --list-sessions` prints a header first; grab the last UUID listed.
  local session_id
  session_id=$(gemini --list-sessions 2>/dev/null | grep -oE '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}' | tail -1 || echo "")
  if [ -n "$session_id" ]; then
    echo "$session_id" > "$COUNCIL_DIR/tmp/session_gemini.txt"
    echo "  Session ID saved: ${session_id:0:8}..."
  else
    # Fallback to "latest" so resume still works even if UUID parsing fails.
    if [ ! -f "$COUNCIL_DIR/tmp/session_gemini.txt" ]; then
      echo "latest" > "$COUNCIL_DIR/tmp/session_gemini.txt"
    fi
    echo "  WARNING: Could not capture Gemini session ID (fallback: latest)"
  fi
}

extract_claude_json_result_and_session() {
  # Extract result + session_id from Claude Code CLI JSON output.
  # Writes the response markdown to $2 and stores the session_id for resume.
  local json_file="$1"
  local out_md="$2"
  local sid_file="$COUNCIL_DIR/tmp/session_claude.txt"

  if [ ! -f "$json_file" ]; then
    echo "  WARNING: Claude JSON output file missing: $json_file"
    return 1
  fi

  local session_id
  session_id=$(python3 - <<'PY' "$json_file" "$out_md" "$sid_file"
import json, sys, re
json_file, out_md, sid_file = sys.argv[1], sys.argv[2], sys.argv[3]
txt = open(json_file, "r", encoding="utf-8", errors="replace").read()

def try_parse(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

obj = try_parse(txt)
if obj is None:
    # stderr logs can sometimes be mixed in; recover the last JSON object if possible.
    m = re.search(r"({.*})\s*$", txt, re.S)
    if m:
        obj = try_parse(m.group(1))

sid = ""
result = None
if isinstance(obj, dict):
    sid = str(obj.get("session_id") or "")
    result = obj.get("result")

# Prefer the structured `result` field; fall back to raw output if parsing failed.
if result is not None:
    open(out_md, "w", encoding="utf-8").write(str(result).rstrip() + "\n")
else:
    open(out_md, "w", encoding="utf-8").write(txt)

if sid:
    open(sid_file, "w", encoding="utf-8").write(sid)

print(sid)
PY
)

  if [ -n "$session_id" ]; then
    echo "  Session ID saved: ${session_id:0:8}..."
    return 0
  else
    echo "  WARNING: Could not parse Claude session ID (continuing without resume capability)"
    return 1
  fi
}

get_gemini_session_id() {
  cat "$COUNCIL_DIR/tmp/session_gemini.txt" 2>/dev/null || echo ""
}

get_claude_session_id() {
  cat "$COUNCIL_DIR/tmp/session_claude.txt" 2>/dev/null || echo ""
}

# Helper: Parse VOTE line from critic output (accepts markdown emphasis).
parse_vote() {
  local file="$1"
  if [ ! -f "$file" ]; then
    echo "VOTE: UNKNOWN"
    return 0
  fi
  local raw vote
  raw=$(grep -oE 'VOTE[[:space:]:-]*[*_`[:space:]]*(ACCEPT|REJECT)' "$file" 2>/dev/null | head -1 || true)
  vote=$(printf "%s" "$raw" | tr '[:lower:]' '[:upper:]' | grep -oE 'ACCEPT|REJECT' | head -1 || true)
  if [ -n "$vote" ]; then
    echo "VOTE: $vote"
  else
    echo "VOTE: UNKNOWN"
  fi
}

# Helper function: Validate critic output
# Returns 0 (success) if valid, 1 (failure) if invalid
validate_critic_output() {
  local output_file="$1"
  local critic_name="$2"

  # Check file exists
  if [ ! -f "$output_file" ]; then
    echo "  ERROR: $critic_name output not found at $output_file"
    echo "  The critic invocation may have failed. Check API connectivity."
    return 1
  fi

  # Check file is not empty
  if [ ! -s "$output_file" ]; then
    echo "  ERROR: $critic_name output is empty at $output_file"
    return 1
  fi

  # Check file was modified recently (within last 5 minutes)
  local file_age=$(( $(date +%s) - $(stat -c %Y "$output_file" 2>/dev/null || stat -f %m "$output_file" 2>/dev/null) ))
  if [ $file_age -gt 300 ]; then
    echo "  WARNING: $critic_name output is stale ($file_age seconds old)"
    echo "  This may be from a previous deliberation. Proceeding anyway."
  fi

  return 0
}

# Helper function: Clean up old round files
cleanup_round_files() {
  local round="$1"
  rm -f "$COUNCIL_DIR/tmp/critic_1_r${round}.md" 2>/dev/null || true
  rm -f "$COUNCIL_DIR/tmp/critic_2_r${round}.md" 2>/dev/null || true
  rm -f "$COUNCIL_DIR/tmp/critic_2_r${round}.json" 2>/dev/null || true
  rm -f "$COUNCIL_DIR/tmp/gemini_prompt_r${round}.txt" 2>/dev/null || true
  rm -f "$COUNCIL_DIR/tmp/claude_prompt_r${round}.txt" 2>/dev/null || true
  rm -f "$COUNCIL_DIR/tmp/round_${round}_aggregate.md" 2>/dev/null || true
}

# Validate round number
if [ "$ROUND" -lt 1 ] || [ "$ROUND" -gt "$MAX_ROUNDS" ]; then
  echo "Error: Round must be between 1 and $MAX_ROUNDS"
  exit 1
fi

# Verify council directory exists
if [ ! -d "$COUNCIL_DIR" ]; then
  echo "Error: $COUNCIL_DIR not found. Run setup_council.sh first."
  exit 1
fi

# Verify required files exist
for f in "$COUNCIL_DIR/critic_prompt.md" "$COUNCIL_DIR/context.md" "$COUNCIL_DIR/history.md"; do
  if [ ! -f "$f" ]; then
    echo "Error: Required file $f not found"
    exit 1
  fi
done

# Ensure tmp directory exists and clean up old round files
mkdir -p "$COUNCIL_DIR/tmp"
cleanup_round_files "$ROUND"

# Check CLI availability (sets GEMINI_AVAILABLE and CLAUDE_AVAILABLE)
check_cli_availability

echo "════════════════════════════════════════════════════════════════════"
echo "  Round $ROUND of $MAX_ROUNDS - ${MODE^} Deliberation"
echo "════════════════════════════════════════════════════════════════════"
if [ "$MODE" = "parallel" ]; then
  echo "  (parallel default: independent critiques; Critic #2 does NOT see Critic #1's same-round output)"
else
  echo "  (sequential: Critic #2 sees Critic #1's same-round output)"
fi
echo ""

# Track which critics ran this round
CRITIC_1_RAN=false
CRITIC_2_RAN=false
VOTE_1="VOTE: SKIPPED"
VOTE_2="VOTE: SKIPPED"

CRITIC_1_OUTPUT="$COUNCIL_DIR/tmp/critic_1_r${ROUND}.md"
CRITIC_2_OUTPUT="$COUNCIL_DIR/tmp/critic_2_r${ROUND}.md"
CRITIC_2_JSON="$COUNCIL_DIR/tmp/critic_2_r${ROUND}.json"
GEMINI_PROMPT_FILE="$COUNCIL_DIR/tmp/gemini_prompt_r${ROUND}.txt"
CLAUDE_PROMPT_FILE="$COUNCIL_DIR/tmp/claude_prompt_r${ROUND}.txt"

# Snapshot repo guidance (optional)
REPO_GUIDANCE="$(cat AGENTS.md 2>/dev/null || cat GEMINI.md 2>/dev/null || echo 'No AGENTS.md or GEMINI.md found')"

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
write_gemini_prompt() {
  if [ "$ROUND" -eq 1 ]; then
    cat > "$GEMINI_PROMPT_FILE" << EOF
$(cat "$COUNCIL_DIR/critic_prompt.md")

$(cat "$COUNCIL_DIR/context.md")

--- DELIBERATION HISTORY ---
$(cat "$COUNCIL_DIR/history.md")
--- END HISTORY ---

You are Critic #1. Round $ROUND of $MAX_ROUNDS.
Review the proposal. Use web search to verify any technical claims if needed.
RESPONSE FORMAT REQUIRED: Start your response with a line "VOTE: ACCEPT" or "VOTE: REJECT".
EOF
  else
    cat > "$GEMINI_PROMPT_FILE" << EOF
$(cat "$COUNCIL_DIR/critic_prompt.md")

--- UPDATED CONTEXT ---
$(cat "$COUNCIL_DIR/context.md")
--- END UPDATED CONTEXT ---

--- DELIBERATION HISTORY ---
$(cat "$COUNCIL_DIR/history.md")
--- END HISTORY ---

You are Critic #1. Round $ROUND of $MAX_ROUNDS.
The proposer has updated their plan based on Round $(($ROUND - 1)) feedback.

Focus on:
1. Were blocking issues from prior rounds addressed?
2. Any new concerns with the revised proposal?

Vote: ACCEPT (no blocking issues remain) or REJECT (blocking issues still exist)
RESPONSE FORMAT REQUIRED: Start your response with a line "VOTE: ACCEPT" or "VOTE: REJECT".
EOF
  fi
}

write_claude_prompt() {
  local sees_critic1_same_round="$1"  # "true" or "false"

  if [ "$ROUND" -eq 1 ]; then
    cat > "$CLAUDE_PROMPT_FILE" << EOF
$(cat "$COUNCIL_DIR/critic_prompt.md")

$(cat "$COUNCIL_DIR/context.md")

--- DELIBERATION HISTORY ---
$(cat "$COUNCIL_DIR/history.md")
--- END HISTORY ---

Repository guidance:
$REPO_GUIDANCE

You are Critic #2. Round $ROUND of $MAX_ROUNDS.
EOF

    if [ "$sees_critic1_same_round" = "true" ]; then
      cat >> "$CLAUDE_PROMPT_FILE" << EOF
Critic #1 has already reviewed this proposal - their feedback is in the history above.
Build on their analysis: agree, disagree, or raise new concerns.
EOF
    else
      cat >> "$CLAUDE_PROMPT_FILE" << EOF
Provide an independent critique. Do NOT assume what Critic #1 said this round.
EOF
    fi

    cat >> "$CLAUDE_PROMPT_FILE" << EOF

Use web search to verify any technical claims if needed.
Provide your assessment following the response format in the critic prompt.
EOF

  else
    # Round 2+: include updated context + prior-round history; optionally include Critic #1's same-round feedback (sequential only)
    cat > "$CLAUDE_PROMPT_FILE" << EOF
$(cat "$COUNCIL_DIR/critic_prompt.md")

Round $ROUND of $MAX_ROUNDS - Revision Review

The proposer has updated their plan based on Round $(($ROUND - 1)) feedback.

--- UPDATED CONTEXT ---
$(cat "$COUNCIL_DIR/context.md")
--- END UPDATED CONTEXT ---

--- DELIBERATION HISTORY (PRIOR ROUNDS) ---
$(cat "$COUNCIL_DIR/history.md")
--- END HISTORY ---

Repository guidance:
$REPO_GUIDANCE

You are Critic #2. Round $ROUND of $MAX_ROUNDS.
EOF

    if [ "$sees_critic1_same_round" = "true" ]; then
      cat >> "$CLAUDE_PROMPT_FILE" << EOF
--- CRITIC #1's ROUND $ROUND FEEDBACK ---
$(cat "$CRITIC_1_OUTPUT" 2>/dev/null || echo "Critic #1 did not run this round")
--- END CRITIC #1 FEEDBACK ---

Review the updated proposal and Critic #1's new feedback.
Focus on:
1. Were blocking issues from prior rounds addressed?
2. Do you agree with Critic #1's assessment this round?
3. Any new concerns with the revised proposal?
EOF
    else
      cat >> "$CLAUDE_PROMPT_FILE" << EOF
Provide an independent critique of the updated proposal. Do NOT assume what Critic #1 said this round.
Focus on whether prior blocking issues are resolved and whether any new risks exist.
EOF
    fi

    cat >> "$CLAUDE_PROMPT_FILE" << EOF

Vote: ACCEPT (no blocking issues remain) or REJECT (blocking issues still exist)
RESPONSE FORMAT REQUIRED: Start your response with a line "VOTE: ACCEPT" or "VOTE: REJECT".
EOF
  fi
}

# ══════════════════════════════════════════════════════════════════════════════
# CRITIC RUNNERS
# ══════════════════════════════════════════════════════════════════════════════
run_gemini_fg() {
  local gemini_flags="-y"

  write_gemini_prompt

  if [ "$ROUND" -eq 1 ]; then
    gemini $gemini_flags -p "" < "$GEMINI_PROMPT_FILE" > "$CRITIC_1_OUTPUT" 2>&1
  else
    local sid
    sid=$(get_gemini_session_id)
    if [ -n "$sid" ]; then
      gemini $gemini_flags --resume "$sid" -p "" < "$GEMINI_PROMPT_FILE" > "$CRITIC_1_OUTPUT" 2>&1
    else
      gemini $gemini_flags -p "" < "$GEMINI_PROMPT_FILE" > "$CRITIC_1_OUTPUT" 2>&1
    fi
  fi
}

run_gemini_bg() {
  local gemini_flags="-y"

  write_gemini_prompt

  if [ "$ROUND" -eq 1 ]; then
    gemini $gemini_flags -p "" < "$GEMINI_PROMPT_FILE" > "$CRITIC_1_OUTPUT" 2>&1 &
  else
    local sid
    sid=$(get_gemini_session_id)
    if [ -n "$sid" ]; then
      gemini $gemini_flags --resume "$sid" -p "" < "$GEMINI_PROMPT_FILE" > "$CRITIC_1_OUTPUT" 2>&1 &
    else
      gemini $gemini_flags -p "" < "$GEMINI_PROMPT_FILE" > "$CRITIC_1_OUTPUT" 2>&1 &
    fi
  fi
  echo $!
}

run_claude_fg() {
  local sees_critic1_same_round="$1"  # "true" or "false"
  write_claude_prompt "$sees_critic1_same_round"

  if [ "$ROUND" -eq 1 ]; then
    env "${CLAUDE_ENV[@]}" "$CLAUDE_CMD" --dangerously-skip-permissions -p --output-format json \
      --tools "default" \
      < "$CLAUDE_PROMPT_FILE" > "$CRITIC_2_JSON" 2>&1
    extract_claude_json_result_and_session "$CRITIC_2_JSON" "$CRITIC_2_OUTPUT"
  else
    local sid
    sid=$(get_claude_session_id)
    if [ -n "$sid" ]; then
      env "${CLAUDE_ENV[@]}" "$CLAUDE_CMD" --resume "$sid" --dangerously-skip-permissions \
        -p --output-format json --tools "default" \
        < "$CLAUDE_PROMPT_FILE" > "$CRITIC_2_JSON" 2>&1
      extract_claude_json_result_and_session "$CRITIC_2_JSON" "$CRITIC_2_OUTPUT"
    else
      env "${CLAUDE_ENV[@]}" "$CLAUDE_CMD" --dangerously-skip-permissions -p --output-format json \
        --tools "default" \
        < "$CLAUDE_PROMPT_FILE" > "$CRITIC_2_JSON" 2>&1
      extract_claude_json_result_and_session "$CRITIC_2_JSON" "$CRITIC_2_OUTPUT"
    fi
  fi
}

run_claude_bg() {
  local sees_critic1_same_round="$1"  # "true" or "false"
  write_claude_prompt "$sees_critic1_same_round"

  if [ "$ROUND" -eq 1 ]; then
    env "${CLAUDE_ENV[@]}" "$CLAUDE_CMD" --dangerously-skip-permissions -p --output-format json \
      --tools "default" \
      < "$CLAUDE_PROMPT_FILE" > "$CRITIC_2_JSON" 2>&1 &
  else
    local sid
    sid=$(get_claude_session_id)
    if [ -n "$sid" ]; then
      env "${CLAUDE_ENV[@]}" "$CLAUDE_CMD" --resume "$sid" --dangerously-skip-permissions \
        -p --output-format json --tools "default" \
        < "$CLAUDE_PROMPT_FILE" > "$CRITIC_2_JSON" 2>&1 &
    else
      env "${CLAUDE_ENV[@]}" "$CLAUDE_CMD" --dangerously-skip-permissions -p --output-format json \
        --tools "default" \
        < "$CLAUDE_PROMPT_FILE" > "$CRITIC_2_JSON" 2>&1 &
    fi
  fi
  echo $!
}

append_to_history_if_ran() {
  # Append outputs in a stable order (Critic #1 then Critic #2)
  if [ "$CRITIC_1_RAN" = true ]; then
    cat >> "$COUNCIL_DIR/history.md" << EOF

## Round $ROUND - Critic #1 (Gemini)

$(cat "$CRITIC_1_OUTPUT")
EOF
  fi

  if [ "$CRITIC_2_RAN" = true ]; then
    cat >> "$COUNCIL_DIR/history.md" << EOF

## Round $ROUND - Critic #2 (Claude Code)

$(cat "$CRITIC_2_OUTPUT")
EOF
  fi
}

# ══════════════════════════════════════════════════════════════════════════════
# RUN CRITICS
# ══════════════════════════════════════════════════════════════════════════════
if [ "$MODE" = "sequential" ]; then
  # ────────────────────────────────────────────────────────────────────────────
  # Sequential: Gemini first (append), then Claude (sees Critic #1 same-round)
  # ────────────────────────────────────────────────────────────────────────────
  if [ "$GEMINI_AVAILABLE" = true ]; then
    echo "[Round $ROUND] Running Critic #1 (Gemini)..."
    echo "  Mode: full access (YOLO) + web search (Gemini)"
    if [ "$ROUND" -eq 1 ]; then
      echo "  Session: Fresh (Round 1)"
    else
      echo "  Session: Resume-by-ID if available"
    fi
    run_gemini_fg || true
    # Capture session ID even if output parsing fails; best-effort
    capture_gemini_session_id || true

    if validate_critic_output "$CRITIC_1_OUTPUT" "Critic #1 (Gemini)"; then
      CRITIC_1_RAN=true
      VOTE_1=$(parse_vote "$CRITIC_1_OUTPUT")
      echo "[Round $ROUND] Critic #1 complete. $VOTE_1"

      # Append Critic 1's feedback BEFORE Critic 2 runs
      cat >> "$COUNCIL_DIR/history.md" << EOF

## Round $ROUND - Critic #1 (Gemini)

$(cat "$CRITIC_1_OUTPUT")
EOF
      echo "[Round $ROUND] Critic #1 feedback appended to history.md"
    else
      echo "[Round $ROUND] Critic #1 failed, continuing with available critics"
    fi
  else
    echo "[Round $ROUND] Skipping Critic #1 (Gemini) - not available"
  fi
  echo ""

  if [ "$CLAUDE_AVAILABLE" = true ]; then
    echo "[Round $ROUND] Running Critic #2 (Claude Code)..."
    echo "  Mode: headless (-p) + YOLO (--dangerously-skip-permissions)"
    if [ "$ROUND" -eq 1 ]; then
      echo "  Session: Fresh (Round 1)"
    else
      echo "  Session: Resume-by-ID if available"
    fi

    # In sequential mode, Claude sees Critic #1 same-round feedback (via history and/or explicit section)
    run_claude_fg "true" || true

    if validate_critic_output "$CRITIC_2_OUTPUT" "Critic #2 (Claude Code)"; then
      CRITIC_2_RAN=true
      VOTE_2=$(parse_vote "$CRITIC_2_OUTPUT")
      echo "[Round $ROUND] Critic #2 complete. $VOTE_2"

      # Append Critic 2's feedback
      cat >> "$COUNCIL_DIR/history.md" << EOF

## Round $ROUND - Critic #2 (Claude Code)

$(cat "$CRITIC_2_OUTPUT")
EOF
      echo "[Round $ROUND] Critic #2 feedback appended to history.md"
    else
      echo "[Round $ROUND] Critic #2 failed, continuing with available critics"
    fi
  else
    echo "[Round $ROUND] Skipping Critic #2 (Claude Code) - not available"
  fi
  echo ""

else
  # ────────────────────────────────────────────────────────────────────────────
  # Parallel: Gemini + Claude run concurrently (independent critiques)
  # ────────────────────────────────────────────────────────────────────────────
  GEMINI_PID=""
  CLAUDE_PID=""

  if [ "$GEMINI_AVAILABLE" = true ]; then
    echo "[Round $ROUND] Launching Critic #1 (Gemini) in parallel..."
    echo "  Mode: full access (YOLO) + web search (Gemini)"
    GEMINI_PID=$(run_gemini_bg)
  else
    echo "[Round $ROUND] Skipping Critic #1 (Gemini) - not available"
  fi

  if [ "$CLAUDE_AVAILABLE" = true ]; then
    echo "[Round $ROUND] Launching Critic #2 (Claude Code) in parallel..."
    echo "  Mode: headless (-p) + YOLO (--dangerously-skip-permissions)"
    # In parallel mode, Claude does NOT see Critic #1 same-round output.
    CLAUDE_PID=$(run_claude_bg "false")
  else
    echo "[Round $ROUND] Skipping Critic #2 (Claude Code) - not available"
  fi

  echo ""

  # Wait for both (do not exit early on failures)
  set +e
  GEMINI_RC=0
  CLAUDE_RC=0
  if [ -n "$GEMINI_PID" ]; then
    wait "$GEMINI_PID"
    GEMINI_RC=$?
  fi
  if [ -n "$CLAUDE_PID" ]; then
    wait "$CLAUDE_PID"
    CLAUDE_RC=$?
  fi
  set -e

  # Best-effort session capture after completion
  if [ -n "$GEMINI_PID" ]; then
    capture_gemini_session_id || true
  fi
  if [ -n "$CLAUDE_PID" ]; then
    extract_claude_json_result_and_session "$CRITIC_2_JSON" "$CRITIC_2_OUTPUT" || true
  fi

  # Validate outputs (independent)
  if [ "$GEMINI_AVAILABLE" = true ]; then
    if validate_critic_output "$CRITIC_1_OUTPUT" "Critic #1 (Gemini)"; then
      CRITIC_1_RAN=true
      VOTE_1=$(parse_vote "$CRITIC_1_OUTPUT")
      echo "[Round $ROUND] Critic #1 complete. $VOTE_1"
    else
      echo "[Round $ROUND] Critic #1 failed (rc=$GEMINI_RC), continuing with available critics"
    fi
  fi

  if [ "$CLAUDE_AVAILABLE" = true ]; then
    if validate_critic_output "$CRITIC_2_OUTPUT" "Critic #2 (Claude Code)"; then
      CRITIC_2_RAN=true
      VOTE_2=$(parse_vote "$CRITIC_2_OUTPUT")
      echo "[Round $ROUND] Critic #2 complete. $VOTE_2"
    else
      echo "[Round $ROUND] Critic #2 failed (rc=$CLAUDE_RC), continuing with available critics"
    fi
  fi

  # Append to history only AFTER both are done (stable order)
  append_to_history_if_ran
  if [ "$CRITIC_1_RAN" = true ] || [ "$CRITIC_2_RAN" = true ]; then
    echo "[Round $ROUND] Round feedback appended to history.md"
  fi
  echo ""
fi

# ══════════════════════════════════════════════════════════════════════════════
# ROUND SUMMARY + AGGREGATE FILE
# ══════════════════════════════════════════════════════════════════════════════
# Re-parse votes from final outputs (defensive against partial writes)
if [ "$CRITIC_1_RAN" = true ]; then
  VOTE_1=$(parse_vote "$CRITIC_1_OUTPUT")
fi
if [ "$CRITIC_2_RAN" = true ]; then
  VOTE_2=$(parse_vote "$CRITIC_2_OUTPUT")
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  Round $ROUND Summary"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Show critic status
if [ "$GEMINI_AVAILABLE" = true ]; then
  if [ "$CRITIC_1_RAN" = true ]; then
    echo "  Critic #1 (Gemini): $VOTE_1"
  else
    echo "  Critic #1 (Gemini): FAILED"
  fi
else
  echo "  Critic #1 (Gemini): UNAVAILABLE"
fi

if [ "$CLAUDE_AVAILABLE" = true ]; then
  if [ "$CRITIC_2_RAN" = true ]; then
    echo "  Critic #2 (Claude Code): $VOTE_2"
  else
    echo "  Critic #2 (Claude Code): FAILED"
  fi
else
  echo "  Critic #2 (Claude Code): UNAVAILABLE"
fi
echo ""

# Count how many critics ran and their votes
CRITICS_RAN=0
ACCEPT_COUNT=0
REJECT_COUNT=0

if [ "$CRITIC_1_RAN" = true ]; then
  ((CRITICS_RAN++)) || true
  [[ "$VOTE_1" == "VOTE: ACCEPT" ]] && ((ACCEPT_COUNT++)) || true
  [[ "$VOTE_1" == "VOTE: REJECT" ]] && ((REJECT_COUNT++)) || true
fi

if [ "$CRITIC_2_RAN" = true ]; then
  ((CRITICS_RAN++)) || true
  [[ "$VOTE_2" == "VOTE: ACCEPT" ]] && ((ACCEPT_COUNT++)) || true
  [[ "$VOTE_2" == "VOTE: REJECT" ]] && ((REJECT_COUNT++)) || true
fi

# Determine next steps based on votes
if [ "$CRITICS_RAN" -eq 0 ]; then
  echo "❌ NO CRITICS RAN: All critic invocations failed."
  echo "   Check CLI installations, authentication, and network connectivity."
  exit 1
elif [ "$CRITICS_RAN" -eq 1 ]; then
  echo "⚠️  PARTIAL COUNCIL: Only one critic produced output."
  if [ "$ACCEPT_COUNT" -eq 1 ]; then
    echo "   The available critic ACCEPTed. Consider running again when both critics are available."
  else
    echo "   The available critic REJECTed or gave unknown verdict."
  fi
elif [ "$ACCEPT_COUNT" -eq 2 ]; then
  echo "✅ CONSENSUS: Both critics ACCEPT. Proposal approved!"
  echo ""
  echo "Next: Proceed with implementation."
elif [ "$ROUND" -ge "$MAX_ROUNDS" ]; then
  echo "⚠️  MAX ROUNDS REACHED ($MAX_ROUNDS)"
  echo "   Review history.md for all feedback and make a decision."
  echo ""
  echo "   Options:"
  echo "   - Proceed with current proposal (accepting known issues)"
  echo "   - Make final revisions without another round"
  echo "   - Escalate for human review"
else
  echo "📝 REVISION NEEDED: One or more critics rejected (or returned UNKNOWN)."
  echo ""
  echo "Next steps:"
  echo "  1. Review feedback in $COUNCIL_DIR/history.md (and the aggregate file below)"
  echo "  2. Update $COUNCIL_DIR/context.md with revised proposal"
  echo "  3. Run: bash run_deliberation.sh $(($ROUND + 1)) $MAX_ROUNDS"
fi

# Write round aggregate file (stable, single file per round)
AGG_FILE="$COUNCIL_DIR/tmp/round_${ROUND}_aggregate.md"
{
  echo "# LLM Council - Round $ROUND of $MAX_ROUNDS"
  echo ""
  echo "- Mode: $MODE"
  echo "- Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""
  echo "## Votes"
  echo "- Critic #1 (Gemini): $VOTE_1"
  echo "- Critic #2 (Claude Code): $VOTE_2"
  echo ""
  echo "## Critic #1 (Gemini)"
  if [ "$CRITIC_1_RAN" = true ]; then
    cat "$CRITIC_1_OUTPUT"
  else
    echo "_Not run / unavailable / failed_"
  fi
  echo ""
  echo "## Critic #2 (Claude Code)"
  if [ "$CRITIC_2_RAN" = true ]; then
    cat "$CRITIC_2_OUTPUT"
  else
    echo "_Not run / unavailable / failed_"
  fi
  echo ""
  echo "## Session IDs"
  GEMINI_SID=$(get_gemini_session_id)
  CLAUDE_SID=$(get_claude_session_id)
  [ -n "$GEMINI_SID" ] && echo "- Gemini: $GEMINI_SID" || echo "- Gemini: (none)"
  [ -n "$CLAUDE_SID" ] && echo "- Claude: $CLAUDE_SID" || echo "- Claude: (none)"
} > "$AGG_FILE"

echo ""
echo "Full feedback saved to:"
[ "$CRITIC_1_RAN" = true ] && echo "  - $CRITIC_1_OUTPUT (Critic #1 Round $ROUND)"
[ "$CRITIC_2_RAN" = true ] && echo "  - $CRITIC_2_OUTPUT (Critic #2 Round $ROUND)"
echo "  - $AGG_FILE (Round aggregate)"
echo "  - $COUNCIL_DIR/history.md (cumulative)"

# Show session IDs for debugging
echo ""
echo "Session IDs:"
GEMINI_SID=$(get_gemini_session_id)
CLAUDE_SID=$(get_claude_session_id)
[ -n "$GEMINI_SID" ] && echo "  - Gemini: $GEMINI_SID"
[ -n "$CLAUDE_SID" ] && echo "  - Claude: $CLAUDE_SID"
