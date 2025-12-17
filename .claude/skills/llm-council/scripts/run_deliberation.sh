#!/bin/bash
# run_deliberation.sh - Sequential multi-round critic deliberation
#
# Usage: run_deliberation.sh <round> [max_rounds]
#
# This script runs critics SEQUENTIALLY within each round:
#   1. Critic #1 (Gemini) reviews the proposal
#   2. Critic #1's feedback is appended to history
#   3. Critic #2 (Codex) reviews proposal + Critic #1's feedback
#   4. Critic #2's feedback is appended to history
#
# For Round 2+, both CLIs use session resume by ID for robustness.
#
# Features:
#   - Graceful degradation: continues with available critics if one fails
#   - Full access (YOLO mode): critics have unrestricted access to codebase
#   - Web search: critics can verify claims online
#   - Session ID tracking: robust resume by explicit session ID
#
# Example:
#   bash run_deliberation.sh 1 3   # Round 1 of 3
#   # Claude revises context.md based on feedback
#   bash run_deliberation.sh 2 3   # Round 2 of 3

set -e

ROUND=${1:?Usage: run_deliberation.sh <round> [max_rounds]}
MAX_ROUNDS=${2:-3}
COUNCIL_DIR=".llm-council"

# CLI availability flags (set by check_cli_availability)
GEMINI_AVAILABLE=false
CODEX_AVAILABLE=false

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

    # Test Codex
    if command -v codex &>/dev/null; then
        if codex --version &>/dev/null 2>&1; then
            CODEX_AVAILABLE=true
            echo "  ✓ Codex CLI available"
        else
            echo "  ✗ Codex CLI found but not functional (check authentication)"
        fi
    else
        echo "  ✗ Codex CLI not installed"
    fi

    # Fail if none available
    if [ "$GEMINI_AVAILABLE" = false ] && [ "$CODEX_AVAILABLE" = false ]; then
        echo ""
        echo "ERROR: No critics available. Install gemini or codex CLI and authenticate."
        exit 1
    fi

    echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
# SESSION ID MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
capture_gemini_session_id() {
    # Capture the most recent Gemini session ID
    local session_id
    session_id=$(gemini --list-sessions 2>/dev/null | head -1 | grep -oP '[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}' || echo "")
    if [ -n "$session_id" ]; then
        echo "$session_id" > "$COUNCIL_DIR/tmp/session_gemini.txt"
        echo "  Session ID saved: ${session_id:0:8}..."
    else
        echo "  WARNING: Could not capture Gemini session ID"
    fi
}

capture_codex_session_id() {
    # Capture the most recent Codex session ID from filesystem
    # Sessions stored at: ~/.codex/sessions/YYYY/MM/DD/rollout-TIMESTAMP-UUID.jsonl
    sleep 5  # Allow session file to be written (Codex may take a few seconds)
    local session_file
    session_file=$(find ~/.codex/sessions -name "*.jsonl" -type f 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || echo "")
    if [ -n "$session_file" ]; then
        local session_id
        session_id=$(basename "$session_file" | grep -oP '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' || echo "")
        if [ -n "$session_id" ]; then
            echo "$session_id" > "$COUNCIL_DIR/tmp/session_codex.txt"
            echo "  Session ID saved: ${session_id:0:8}..."
        else
            echo "  WARNING: Could not parse Codex session ID from $session_file"
        fi
    else
        echo "  WARNING: Could not find Codex session file"
    fi
}

get_gemini_session_id() {
    cat "$COUNCIL_DIR/tmp/session_gemini.txt" 2>/dev/null || echo ""
}

get_codex_session_id() {
    cat "$COUNCIL_DIR/tmp/session_codex.txt" 2>/dev/null || echo ""
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

# Check CLI availability (sets GEMINI_AVAILABLE and CODEX_AVAILABLE)
check_cli_availability

echo "════════════════════════════════════════════════════════════════════"
echo "  Round $ROUND of $MAX_ROUNDS - Sequential Deliberation"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Track which critics ran this round
CRITIC_1_RAN=false
CRITIC_2_RAN=false
VOTE_1="VOTE: SKIPPED"
VOTE_2="VOTE: SKIPPED"

# ══════════════════════════════════════════════════════════════════════════════
# CRITIC 1: Gemini
# ══════════════════════════════════════════════════════════════════════════════
# Gemini: full access (YOLO mode) with web search, session resume by ID for Round 2+
CRITIC_1_OUTPUT="$COUNCIL_DIR/tmp/critic_1_r${ROUND}.md"

if [ "$GEMINI_AVAILABLE" = true ]; then
    echo "[Round $ROUND] Running Critic #1 (Gemini)..."
    echo "  Mode: full access (YOLO) + web search"

    # Common Gemini flags for YOLO mode (auto-approve all actions)
    GEMINI_FLAGS="-y"

    if [ "$ROUND" -eq 1 ]; then
        echo "  Session: Fresh (Round 1)"
        echo "  Reading: critic_prompt.md, context.md, history.md"

        gemini $GEMINI_FLAGS \
          "@$COUNCIL_DIR/critic_prompt.md \
           @$COUNCIL_DIR/context.md \
           @$COUNCIL_DIR/history.md \
           You are Critic #1. Round $ROUND of $MAX_ROUNDS. \
           Review the proposal. Use web search to verify any technical claims if needed." \
          > "$CRITIC_1_OUTPUT" 2>&1

        # Capture session ID for future rounds
        capture_gemini_session_id
    else
        # Round 2+: Resume by session ID
        GEMINI_SESSION=$(get_gemini_session_id)
        if [ -n "$GEMINI_SESSION" ]; then
            echo "  Session: Resume by ID (${GEMINI_SESSION:0:8}...)"

            gemini $GEMINI_FLAGS --resume "$GEMINI_SESSION" \
              "Round $ROUND of $MAX_ROUNDS - Revision Review

The proposer has updated their plan based on Round $(($ROUND - 1)) feedback.

Review the updated proposal at @$COUNCIL_DIR/context.md
Check deliberation history at @$COUNCIL_DIR/history.md

Focus on:
1. Were blocking issues from prior rounds addressed?
2. Any new concerns with the revised proposal?

Vote: ACCEPT (no blocking issues remain) or REJECT (blocking issues still exist)
Follow the response format from the original critic prompt." \
              > "$CRITIC_1_OUTPUT" 2>&1
        else
            echo "  WARNING: No session ID found, starting fresh session"
            gemini $GEMINI_FLAGS \
              "@$COUNCIL_DIR/critic_prompt.md \
               @$COUNCIL_DIR/context.md \
               @$COUNCIL_DIR/history.md \
               You are Critic #1. Round $ROUND of $MAX_ROUNDS. \
               Review the proposal. Focus on whether blocking issues from previous rounds were addressed." \
              > "$CRITIC_1_OUTPUT" 2>&1
            capture_gemini_session_id
        fi
    fi

    # Validate Critic 1's output
    if validate_critic_output "$CRITIC_1_OUTPUT" "Critic #1 (Gemini)"; then
        CRITIC_1_RAN=true
        VOTE_1=$(grep -o 'VOTE: [A-Z]*' "$CRITIC_1_OUTPUT" 2>/dev/null | head -1 || echo "VOTE: UNKNOWN")
        echo "[Round $ROUND] Critic #1 complete. $VOTE_1"

        # Append Critic 1's feedback to history BEFORE Critic 2 runs
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

# ══════════════════════════════════════════════════════════════════════════════
# CRITIC 2: Codex
# ══════════════════════════════════════════════════════════════════════════════
# Codex: full access with web search, session resume by ID for Round 2+
CRITIC_2_OUTPUT="$COUNCIL_DIR/tmp/critic_2_r${ROUND}.md"

if [ "$CODEX_AVAILABLE" = true ]; then
    echo "[Round $ROUND] Running Critic #2 (Codex)..."
    echo "  Mode: full access + web search"

    # Common Codex flags for full access (dangerously bypass approvals and sandbox)
    CODEX_FLAGS="--dangerously-bypass-approvals-and-sandbox --search -C ."

    if [ "$ROUND" -eq 1 ]; then
        echo "  Session: Fresh (Round 1)"
        echo "  Reading: critic_prompt.md, context.md, history.md (with Critic #1 feedback)"

        # Build the full prompt for Round 1
        codex exec $CODEX_FLAGS \
          "$(cat "$COUNCIL_DIR/critic_prompt.md")

$(cat "$COUNCIL_DIR/context.md")

--- DELIBERATION HISTORY ---
$(cat "$COUNCIL_DIR/history.md")
--- END HISTORY ---

Repository guidance:
$(cat AGENTS.md 2>/dev/null || cat GEMINI.md 2>/dev/null || echo 'No AGENTS.md or GEMINI.md found')

You are Critic #2. Round 1 of $MAX_ROUNDS.
Critic #1 has already reviewed this proposal - their feedback is in the history above.
Build on their analysis: agree, disagree, or raise new concerns.
Use web search to verify any technical claims if needed.
Provide your independent assessment following the response format in the critic prompt." \
          --output-last-message "$CRITIC_2_OUTPUT"

        # Capture session ID for future rounds
        capture_codex_session_id
    else
        # Round 2+: Resume by session ID
        CODEX_SESSION=$(get_codex_session_id)
        if [ -n "$CODEX_SESSION" ]; then
            echo "  Session: Resume by ID (${CODEX_SESSION:0:8}...)"

            # Note: codex exec resume does not support --output-last-message, so capture stdout
            codex exec resume "$CODEX_SESSION" \
              "Round $ROUND of $MAX_ROUNDS - Revision Review

The proposer has updated their plan based on Round $(($ROUND - 1)) feedback.

--- UPDATED CONTEXT ---
$(cat "$COUNCIL_DIR/context.md")
--- END UPDATED CONTEXT ---

--- CRITIC #1's ROUND $ROUND FEEDBACK ---
$(cat "$CRITIC_1_OUTPUT" 2>/dev/null || echo "Critic #1 did not run this round")
--- END CRITIC #1 FEEDBACK ---

Review the updated proposal and Critic #1's new feedback.
Focus on:
1. Were blocking issues from prior rounds addressed?
2. Do you agree with Critic #1's assessment this round?
3. Any new concerns with the revised proposal?

Vote: ACCEPT (no blocking issues remain) or REJECT (blocking issues still exist)
Follow the response format from the original critic prompt." \
              > "$CRITIC_2_OUTPUT" 2>&1
        else
            echo "  WARNING: No session ID found, using --last fallback"
            # Note: codex exec resume does not support --output-last-message, so capture stdout
            codex exec resume --last \
              "Round $ROUND of $MAX_ROUNDS - Revision Review

The proposer has updated their plan based on Round $(($ROUND - 1)) feedback.

--- UPDATED CONTEXT ---
$(cat "$COUNCIL_DIR/context.md")
--- END UPDATED CONTEXT ---

--- CRITIC #1's ROUND $ROUND FEEDBACK ---
$(cat "$CRITIC_1_OUTPUT" 2>/dev/null || echo "Critic #1 did not run this round")
--- END CRITIC #1 FEEDBACK ---

Review the updated proposal and Critic #1's new feedback.
Focus on:
1. Were blocking issues from prior rounds addressed?
2. Do you agree with Critic #1's assessment this round?
3. Any new concerns with the revised proposal?

Vote: ACCEPT (no blocking issues remain) or REJECT (blocking issues still exist)
Follow the response format from the original critic prompt." \
              > "$CRITIC_2_OUTPUT" 2>&1
            capture_codex_session_id
        fi
    fi

    # Validate Critic 2's output
    if validate_critic_output "$CRITIC_2_OUTPUT" "Critic #2 (Codex)"; then
        CRITIC_2_RAN=true
        VOTE_2=$(grep -o 'VOTE: [A-Z]*' "$CRITIC_2_OUTPUT" 2>/dev/null | head -1 || echo "VOTE: UNKNOWN")
        echo "[Round $ROUND] Critic #2 complete. $VOTE_2"

        # Append Critic 2's feedback to history
        cat >> "$COUNCIL_DIR/history.md" << EOF

## Round $ROUND - Critic #2 (Codex)

$(cat "$CRITIC_2_OUTPUT")
EOF
        echo "[Round $ROUND] Critic #2 feedback appended to history.md"
    else
        echo "[Round $ROUND] Critic #2 failed, continuing with available critics"
    fi
else
    echo "[Round $ROUND] Skipping Critic #2 (Codex) - not available"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# ROUND SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
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

if [ "$CODEX_AVAILABLE" = true ]; then
    if [ "$CRITIC_2_RAN" = true ]; then
        echo "  Critic #2 (Codex):  $VOTE_2"
    else
        echo "  Critic #2 (Codex):  FAILED"
    fi
else
    echo "  Critic #2 (Codex):  UNAVAILABLE"
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
    echo "⚠️  PARTIAL COUNCIL: Only one critic was available."
    if [ "$ACCEPT_COUNT" -eq 1 ]; then
        echo "   The available critic ACCEPTed. Consider running again with full council."
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
    echo "📝 REVISION NEEDED: One or more critics rejected."
    echo ""
    echo "Next steps:"
    echo "  1. Review feedback in $COUNCIL_DIR/history.md"
    echo "  2. Update $COUNCIL_DIR/context.md with revised proposal"
    echo "  3. Run: bash run_deliberation.sh $(($ROUND + 1)) $MAX_ROUNDS"
fi

echo ""
echo "Full feedback saved to:"
[ "$CRITIC_1_RAN" = true ] && echo "  - $CRITIC_1_OUTPUT (Critic #1 Round $ROUND)"
[ "$CRITIC_2_RAN" = true ] && echo "  - $CRITIC_2_OUTPUT (Critic #2 Round $ROUND)"
echo "  - $COUNCIL_DIR/history.md (cumulative)"

# Show session IDs for debugging
echo ""
echo "Session IDs:"
GEMINI_SID=$(get_gemini_session_id)
CODEX_SID=$(get_codex_session_id)
[ -n "$GEMINI_SID" ] && echo "  - Gemini: $GEMINI_SID"
[ -n "$CODEX_SID" ] && echo "  - Codex:  $CODEX_SID"
