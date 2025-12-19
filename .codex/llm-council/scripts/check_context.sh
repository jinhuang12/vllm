#!/bin/bash
# check_context.sh - Detect and handle stale context in .llm-council/
#
# Usage: check_context.sh <new_fingerprint_string> [--force-continue|--force-clear]
#
# Outputs: fresh|continue|cleared
#   - fresh:    No existing context, ready for setup
#   - continue: Fingerprints match, continuing with existing context
#   - cleared:  Old context was cleared due to topic change
#
# Side effects:
#   - Clears .llm-council/ if fingerprints differ (unless --force-continue)
#   - Always clears with --force-clear
#   - Cleans up stale /tmp/ artifacts (test directories and legacy critic outputs)

set -e

# Helper function: Clean up stale /tmp/ artifacts
cleanup_tmp_artifacts() {
    # Remove test verification directories
    rm -rf /tmp/llm-council-verify-* 2>/dev/null || true

    # Remove legacy critic output files (backwards compatibility)
    rm -f /tmp/critic_1.md /tmp/critic_2.md 2>/dev/null || true
    rm -f /tmp/critic_1_r*.md /tmp/critic_2_r*.md 2>/dev/null || true
    rm -f /tmp/critic_prompt.md 2>/dev/null || true
    rm -f /tmp/gemini_context_check.md 2>/dev/null || true
    rm -f /tmp/claude_session_check.md 2>/dev/null || true
}

NEW_FP="$1"
FORCE="$2"
COUNCIL_DIR=".llm-council"
FP_FILE="$COUNCIL_DIR/topic_fingerprint.txt"

# Handle force flags
if [ "$FORCE" = "--force-clear" ]; then
    if [ -d "$COUNCIL_DIR" ]; then
        rm -rf "$COUNCIL_DIR"
        cleanup_tmp_artifacts
        echo "cleared"
    else
        echo "fresh"
    fi
    exit 0
fi

# No existing context directory
if [ ! -d "$COUNCIL_DIR" ] || [ ! -f "$FP_FILE" ]; then
    echo "fresh"
    exit 0
fi

# Compare fingerprints
OLD_FP=$(cat "$FP_FILE" 2>/dev/null || echo "")

if [ "$NEW_FP" = "$OLD_FP" ]; then
    echo "continue"
    exit 0
fi

# Force continue even with different fingerprint
if [ "$FORCE" = "--force-continue" ]; then
    echo "continue"
    exit 0
fi

# Fingerprints differ → auto-clear stale context
rm -rf "$COUNCIL_DIR"
cleanup_tmp_artifacts
echo "cleared"
