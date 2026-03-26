#!/bin/bash
# Test harness for ammo-stop-guard.sh (Stop hook)
# Run: bash .claude/hooks/test-ammo-stop-guard.sh
#
# Tests:
#   Bug 1 — Role filtering: only lead/solo orchestrator should get nudges
#   Bug 2 — Completed vs not-started overlap detection
#   Regression — Existing functionality preserved

set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/ammo-stop-guard.sh"
PASS=0
FAIL=0
TOTAL=0

# Create temporary test environment
TMPDIR=$(mktemp -d)
ARTIFACT_DIR="$TMPDIR/kernel_opt_artifacts/test_target"
mkdir -p "$ARTIFACT_DIR"

cleanup() {
    rm -rf "$TMPDIR"
    rm -f /tmp/hook-stderr
    rm -f "/tmp/ammo-stop-nudged-test-session"
    rm -f "/tmp/ammo-stop-nudged-unknown"
    rm -f "/tmp/ammo-stop-nudged-lead-session-123"
    rm -f "/tmp/ammo-stop-nudged-teammate-session-1"
}
trap cleanup EXIT

run_test() {
    local test_name="$1"
    local expected_exit="$2"
    local json_input="$3"
    local actual_exit=0

    TOTAL=$((TOTAL + 1))
    # Clean marker files before each test
    rm -f /tmp/ammo-stop-nudged-* 2>/dev/null || true

    echo "$json_input" | env HOME="$TMPDIR" CLAUDE_PROJECT_DIR="$TMPDIR" bash "$HOOK" 2>/tmp/hook-stderr || actual_exit=$?

    if [ "$actual_exit" -eq "$expected_exit" ]; then
        echo "  PASS [$TOTAL]: $test_name (exit=$actual_exit)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $test_name (expected=$expected_exit, got=$actual_exit)"
        echo "        stderr: $(cat /tmp/hook-stderr 2>/dev/null || echo '(none)')"
        FAIL=$((FAIL + 1))
    fi
}

# ══════════════════════════════════════════════════
echo "== Bug 1: Role Filtering — Teammates must be silenced =="
# ══════════════════════════════════════════════════

# Setup: active campaign at stages 4-5 with overlapped debate active (round 2)
# This is the scenario that produced 269 overlapped_active_wait false positives.
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": true,
      "phase": "debating",
      "selected_winners": [],
      "profiling_basis": "bottleneck_analysis.md",
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {
    "op003": { "status": "in_progress" },
    "op004": { "status": "in_progress" }
  }
}
EOF

# Setup team config: lead is "lead-session-123"
mkdir -p "$TMPDIR/.claude/teams/ammo-round-2"
cat > "$TMPDIR/.claude/teams/ammo-round-2/config.json" << 'EOF'
{
  "leadSessionId": "lead-session-123",
  "members": [
    {"name": "impl-champion-op003", "sessionId": "teammate-session-1"},
    {"name": "impl-validator-op003", "sessionId": "teammate-session-2"},
    {"name": "champion-r3-1", "sessionId": "teammate-session-3"}
  ]
}
EOF

# Test 1: Teammate with agent_type in JSON → should NOT get nudged (exit 0)
run_test "Teammate (agent_type=implementor) silenced" 0 \
    '{"session_id": "lead-session-123", "agent_type": "implementor", "agent_id": "uuid-1"}'

# Test 2: Teammate with different agent type → also silenced
run_test "Teammate (agent_type=verifier) silenced" 0 \
    '{"session_id": "lead-session-123", "agent_type": "verifier", "agent_id": "uuid-2"}'

# Test 3: Lead (no agent_type) → should get nudged (exit 2)
run_test "Lead (no agent_type) gets overlap-active nudge" 2 \
    '{"session_id": "lead-session-123"}'

# Test 4: Teammate receiving "not launched" warning (the 40 false positives)
# Setup: round 2, stages 4-5, overlap NOT launched yet
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": null,
      "selected_winners": [],
      "profiling_basis": null,
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {
    "op003": { "status": "in_progress" },
    "op004": { "status": "in_progress" }
  }
}
EOF

run_test "Teammate silenced for 'not launched' warning" 0 \
    '{"session_id": "lead-session-123", "agent_type": "implementor", "agent_id": "uuid-3"}'

# Test 5: Lead should get the "not launched" warning
run_test "Lead gets 'not launched' warning at stage 4-5 round 2" 2 \
    '{"session_id": "lead-session-123"}'

# Test 6: Teammate receiving Stage 7 report nag (the 7 false positives)
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "7_campaign_eval",
  "campaign": { "status": "campaign_complete", "current_round": 3 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": null,
      "selected_winners": [],
      "profiling_basis": null,
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {}
}
EOF

run_test "Teammate silenced for Stage 7 report nag" 0 \
    '{"session_id": "lead-session-123", "agent_type": "general-purpose", "agent_id": "uuid-4"}'

# Test 7: Lead should get Stage 7 report nag (no REPORT.md)
run_test "Lead gets Stage 7 report nag (no REPORT.md)" 2 \
    '{"session_id": "lead-session-123"}'

# Test 8: Solo orchestrator (no team configs) → should still get nudged
rm -rf "$TMPDIR/.claude/teams"
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": true,
      "phase": "debating",
      "selected_winners": [],
      "profiling_basis": "bottleneck_analysis.md",
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {
    "op003": { "status": "in_progress" }
  }
}
EOF

run_test "Solo orchestrator (no teams) gets nudge" 2 \
    '{"session_id": "unknown"}'

# Restore team config for remaining tests
mkdir -p "$TMPDIR/.claude/teams/ammo-round-2"
cat > "$TMPDIR/.claude/teams/ammo-round-2/config.json" << 'EOF'
{
  "leadSessionId": "lead-session-123",
  "members": [
    {"name": "impl-champion-op003", "sessionId": "teammate-session-1"}
  ]
}
EOF

# ══════════════════════════════════════════════════
echo ""
echo "== Bug 2: Completed overlap must NOT trigger 'not launched' =="
# ══════════════════════════════════════════════════

# Test 9: Overlap completed (phase=selection_complete, active=false) → no nudge
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": "selection_complete",
      "selected_winners": ["op005", "op006"],
      "profiling_basis": "bottleneck_analysis.md",
      "f_values_at_proposal": {"op005": 0.12, "op006": 0.08}
    }
  },
  "parallel_tracks": {
    "op003": { "status": "in_progress" },
    "op004": { "status": "in_progress" }
  }
}
EOF

run_test "Completed overlap (phase=selection_complete) → no nudge" 0 \
    '{"session_id": "lead-session-123"}'

# Test 10: Overlap completed with winners but phase cleared → no nudge
# (selected_winners non-empty proves debate completed even if phase is null)
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": null,
      "selected_winners": ["op005"],
      "profiling_basis": "bottleneck_analysis.md",
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {
    "op003": { "status": "in_progress" },
    "op004": { "status": "in_progress" }
  }
}
EOF

run_test "Completed overlap (winners non-empty, phase null) → no nudge" 0 \
    '{"session_id": "lead-session-123"}'

# Test 11: Overlap truly not started (phase=null, no winners) at round 2 → should nudge
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": null,
      "selected_winners": [],
      "profiling_basis": null,
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {
    "op003": { "status": "in_progress" },
    "op004": { "status": "in_progress" }
  }
}
EOF

run_test "Not-started overlap (no winners, no phase) → lead gets nudge" 2 \
    '{"session_id": "lead-session-123"}'

# Test 12: Overlap in progress (active=true) → lead should wait
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": true,
      "phase": "phase_0",
      "selected_winners": [],
      "profiling_basis": "bottleneck_analysis.md",
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {
    "op003": { "status": "in_progress" }
  }
}
EOF

run_test "Active overlap (phase_0) → lead gets 'wait' nudge" 2 \
    '{"session_id": "lead-session-123"}'

# ══════════════════════════════════════════════════
echo ""
echo "== Regression: Existing correct behavior preserved =="
# ══════════════════════════════════════════════════

# Test 13: No campaign artifacts → silent pass
rm -f "$ARTIFACT_DIR/state.json"

run_test "No state.json → silent pass" 0 \
    '{"session_id": "lead-session-123"}'

# Test 14: Terminal campaign + REPORT.md → silent pass
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "7_campaign_eval",
  "campaign": { "status": "campaign_complete", "current_round": 3 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": null,
      "selected_winners": [],
      "profiling_basis": null,
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {}
}
EOF
echo "# Optimization Report" > "${ARTIFACT_DIR}/REPORT.md"

run_test "Terminal campaign + REPORT.md → silent pass" 0 \
    '{"session_id": "lead-session-123"}'

rm -f "${ARTIFACT_DIR}/REPORT.md"

# Test 15: Paused campaign → silent pass
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "paused", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": null,
      "selected_winners": [],
      "profiling_basis": null,
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {}
}
EOF

run_test "Paused campaign → silent pass" 0 \
    '{"session_id": "lead-session-123"}'

# Test 16: Round 1 at stages 4-5 → no overlap check needed, silent pass
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 1 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": null,
      "selected_winners": [],
      "profiling_basis": null,
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {
    "op001": { "status": "in_progress" }
  }
}
EOF

run_test "Round 1 implementation → no overlap needed, silent pass" 0 \
    '{"session_id": "lead-session-123"}'

# Test 17: Stage 7 active → lead gets threshold-check nudge
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "7_campaign_eval",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": null,
      "selected_winners": [],
      "profiling_basis": null,
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {}
}
EOF

run_test "Stage 7 active → lead gets threshold-check nudge" 2 \
    '{"session_id": "lead-session-123"}'

# Test 18: Stage 7b → lead gets report nudge
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "7b_report_gen",
  "campaign": { "status": "campaign_complete", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": null,
      "selected_winners": [],
      "profiling_basis": null,
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {}
}
EOF

run_test "Stage 7b → lead gets report nudge" 2 \
    '{"session_id": "lead-session-123"}'

# Test 19-20: Circuit breaker — second stop attempt passes through
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "7_campaign_eval",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": null,
      "selected_winners": [],
      "profiling_basis": null,
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {}
}
EOF

# First stop → nudge (exit 2), creates marker
run_test "Circuit breaker: 1st stop → nudge" 2 \
    '{"session_id": "lead-session-123"}'

# Second stop — run manually to preserve the marker from 1st stop
TOTAL=$((TOTAL + 1))
rm -f /tmp/ammo-stop-nudged-* 2>/dev/null || true
# Run hook twice: first creates marker, second sees it
echo '{"session_id": "lead-session-123"}' | env HOME="$TMPDIR" CLAUDE_PROJECT_DIR="$TMPDIR" bash "$HOOK" 2>/dev/null || true
actual_exit=0
echo '{"session_id": "lead-session-123"}' | env HOME="$TMPDIR" CLAUDE_PROJECT_DIR="$TMPDIR" bash "$HOOK" 2>/tmp/hook-stderr || actual_exit=$?
if [ "$actual_exit" -eq 0 ]; then
    echo "  PASS [$TOTAL]: Circuit breaker: 2nd stop → pass through (exit=$actual_exit)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: Circuit breaker: 2nd stop → pass through (expected=0, got=$actual_exit)"
    echo "        stderr: $(cat /tmp/hook-stderr 2>/dev/null || echo '(none)')"
    FAIL=$((FAIL + 1))
fi

# Test 21: Other stage with no active overlap → silent pass
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "3_debate",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": false,
      "phase": null,
      "selected_winners": [],
      "profiling_basis": null,
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {}
}
EOF

run_test "Other stage, no active overlap → silent pass" 0 \
    '{"session_id": "lead-session-123"}'

# Test 22: Other stage with stale active overlap → lead gets nudge
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "stage": "6_integration",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": true,
      "phase": "debating",
      "selected_winners": [],
      "profiling_basis": "bottleneck_analysis.md",
      "f_values_at_proposal": {}
    }
  },
  "parallel_tracks": {}
}
EOF

run_test "Other stage with stale active overlap → lead gets nudge" 2 \
    '{"session_id": "lead-session-123"}'

# ══════════════════════════════════════════════════
echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
