#!/bin/bash
# Test harness for ammo-gate-guard.sh (TaskCompleted hook)
# Run: bash .claude/hooks/test-ammo-gate-guard.sh
#
# Tests all enforcement paths:
#   1. Non-gate tasks pass through
#   2. Gate tasks require verification evidence
#   3. KILL decisions require recorded attempts (B13)
#   4. KILL decisions require iteration tasks (B14)
#   5. Edge cases (empty subject, missing state.json, SHIP route)

set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/ammo-gate-guard.sh"
PASS=0
FAIL=0
TOTAL=0

# Create temporary test environment
TMPDIR=$(mktemp -d)
ARTIFACT_DIR="$TMPDIR/kernel_opt_artifacts/test_target"
TASK_DIR="$HOME/.claude/tasks/ammo-test-hookguard"
mkdir -p "$ARTIFACT_DIR"

cleanup() {
    rm -rf "$TMPDIR"
    rm -rf "$TASK_DIR"
    rm -f /tmp/hook-stderr
}
trap cleanup EXIT

run_test() {
    local test_name="$1"
    local expected_exit="$2"
    local json_input="$3"
    local actual_exit=0

    TOTAL=$((TOTAL + 1))
    echo "$json_input" | CLAUDE_PROJECT_DIR="$TMPDIR" bash "$HOOK" 2>/tmp/hook-stderr || actual_exit=$?

    if [ "$actual_exit" -eq "$expected_exit" ]; then
        echo "  PASS [$TOTAL]: $test_name (exit=$actual_exit)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $test_name (expected=$expected_exit, got=$actual_exit)"
        echo "        stderr: $(cat /tmp/hook-stderr 2>/dev/null || echo '(none)')"
        FAIL=$((FAIL + 1))
    fi
}

# ══════════════════════════════════════
echo "== Fast bail tests =="
# ══════════════════════════════════════

run_test "Non-gate task allows completion" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "B2: Capture baseline environment",
  "task_description": "Capture baseline...",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

run_test "Empty subject allows completion" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "",
  "cwd": "'"$TMPDIR"'"
}'

run_test "Missing subject field allows completion" 0 '{
  "hook_event_name": "TaskCompleted",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== No state.json (not AMMO task) =="
# ══════════════════════════════════════

# Remove state.json to simulate non-AMMO project
rm -f "$ARTIFACT_DIR/state.json"

run_test "Gate task without state.json allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: verify_validation_gates.py",
  "task_description": "Run validation",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

run_test "B14 without state.json allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "B14: Route decision",
  "task_description": "Decide route",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== Phase 1 gate (verify_phase1) =="
# ══════════════════════════════════════

# No evidence at all
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {"stage1": null, "validation": null}
}
EOF

run_test "Phase1 gate without evidence BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: verify_phase1_baseline.py",
  "task_description": "Run phase 1 verification",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# With verification_run.stage1 set
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {"stage1": "PASS", "validation": null}
}
EOF

run_test "Phase1 gate with stage1=PASS allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: verify_phase1_baseline.py",
  "task_description": "Run phase 1 verification",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== Validation gate (verify_validation) =="
# ══════════════════════════════════════

# No validation evidence
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {"stage1": "PASS", "validation": null}
}
EOF

run_test "Validation gate without evidence BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: verify_validation_gates.py",
  "task_description": "Run validation verification",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# With SHIP route + validation evidence
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {"decision": "SHIP"},
  "opportunity_attempts": [],
  "verification_run": {"stage1": "PASS", "validation": "PASS"}
}
EOF

run_test "Validation gate with SHIP allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: verify_validation_gates.py",
  "task_description": "Run validation verification",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== KILL enforcement on validation gate (B13) =="
# ══════════════════════════════════════

# KILL route, empty opportunity_attempts — should BLOCK
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {"decision": "KILL"},
  "opportunity_attempts": [],
  "verification_run": {"stage1": "PASS", "validation": "KILL"}
}
EOF

run_test "Validation gate KILL + empty attempts BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: verify_validation_gates.py",
  "task_description": "Run validation verification",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# KILL route, populated opportunity_attempts — should allow
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {"decision": "KILL"},
  "opportunity_attempts": [{"attempt": 1, "result": "KILL", "reason": "BS=8 regressed"}],
  "verification_run": {"stage1": "PASS", "validation": "KILL"}
}
EOF

run_test "Validation gate KILL + recorded attempt allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: verify_validation_gates.py",
  "task_description": "Run validation verification",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== Route from opportunity_attempts fallback =="
# ══════════════════════════════════════

# Production schema: route_decision has attempt_N + final, not .route or .decision
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {"attempt_1": "KILL", "final": "EXHAUSTED"},
  "opportunity_attempts": [{"attempt": 1, "result": "KILL"}],
  "verification_run": {"stage1": "PASS", "validation": "KILL"}
}
EOF

run_test "Route from opportunity_attempts[-1].result works" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: verify_validation_gates.py",
  "task_description": "Run validation verification",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== B14 Route Decision enforcement =="
# ══════════════════════════════════════

# KILL route, no iteration tasks — should BLOCK
mkdir -p "$TASK_DIR"
rm -f "$TASK_DIR"/*.json

cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {"decision": "KILL"},
  "opportunity_attempts": [{"attempt": 1, "result": "KILL"}],
  "verification_run": {"stage1": "PASS", "validation": "KILL"}
}
EOF

run_test "B14 KILL + no iteration tasks BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "B14: Route decision",
  "task_description": "Decide SHIP or KILL",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# KILL route, B15 exists — should allow
cat > "$TASK_DIR/task-b15.json" << 'EOF'
{"subject": "B15: Write updated optimization_plan.md", "status": "pending"}
EOF

run_test "B14 KILL + B15 exists allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "B14: Route decision",
  "task_description": "Decide SHIP or KILL",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# SHIP route — should allow regardless
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {"decision": "SHIP"},
  "opportunity_attempts": [{"attempt": 1, "result": "SHIP"}],
  "verification_run": {"stage1": "PASS", "validation": "PASS"}
}
EOF

rm -f "$TASK_DIR"/*.json

run_test "B14 SHIP allows (no iteration tasks needed)" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "B14: Route decision",
  "task_description": "Decide SHIP or KILL",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== Route decision with 'Route decision' subject variant =="
# ══════════════════════════════════════

cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {"decision": "KILL"},
  "opportunity_attempts": [{"attempt": 1, "result": "KILL"}],
  "verification_run": {"stage1": "PASS", "validation": "KILL"}
}
EOF

rm -f "$TASK_DIR"/*.json

run_test "Route decision (alt subject) KILL + no tasks BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "Route decision (attempt 2)",
  "task_description": "Decide route",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== Debate gate enforcement =="
# ══════════════════════════════════════

# No debate/ directory — should BLOCK
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {}
}
EOF
rm -rf "$ARTIFACT_DIR/debate"

run_test "Debate gate without debate/ dir BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: debate winner selection",
  "task_description": "Select debate winner",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# debate/ exists but no summary.md — should BLOCK
mkdir -p "$ARTIFACT_DIR/debate"

run_test "Debate gate without summary.md BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: debate winner selection",
  "task_description": "Select debate winner",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# debate/ with summary.md + round_1 but no proposals — should BLOCK (missing proposals)
echo "# Summary" > "$ARTIFACT_DIR/debate/summary.md"
mkdir -p "$ARTIFACT_DIR/debate/round_1"

run_test "Debate gate without proposals BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: debate summary.md",
  "task_description": "Select debate winner",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# debate/ with summary.md, 2+ round dirs, and proposals — should PASS
mkdir -p "$ARTIFACT_DIR/debate/round_2"
mkdir -p "$ARTIFACT_DIR/debate/proposals"
echo "# Proposal 1" > "$ARTIFACT_DIR/debate/proposals/champion-1_proposal.md"
echo "# Proposal 2" > "$ARTIFACT_DIR/debate/proposals/champion-2_proposal.md"

run_test "Debate gate with all evidence allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: debate winner selection",
  "task_description": "Select debate winner",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== Track validation gate enforcement =="
# ══════════════════════════════════════

# Track with null result — should BLOCK
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {},
  "parallel_tracks": [
    {"id": "track_a", "result": "improved"},
    {"id": "track_b", "result": null}
  ]
}
EOF

run_test "Track gate with null result BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: all tracks validated",
  "task_description": "Validate all tracks",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# All tracks with results — should PASS
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {},
  "parallel_tracks": [
    {"id": "track_a", "result": "improved"},
    {"id": "track_b", "result": "regressed"}
  ]
}
EOF

run_test "Track gate with all results allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: all tracks validated",
  "task_description": "Validate all tracks",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== Integration gate enforcement =="
# ══════════════════════════════════════

# integration.status is "pending" — should BLOCK
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {},
  "integration": {"status": "pending"}
}
EOF

run_test "Integration gate with pending status BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: integration check",
  "task_description": "Check integration",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# integration.status is "validated" — should PASS
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {},
  "integration": {"status": "validated"}
}
EOF

run_test "Integration gate with validated status allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: integration check",
  "task_description": "Check integration",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# integration.status is "single_pass" — should PASS
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {},
  "integration": {"status": "single_pass"}
}
EOF

run_test "Integration gate with single_pass status allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: combined validation",
  "task_description": "Combined validation",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== Campaign evaluation gate enforcement =="
# ══════════════════════════════════════

# No campaign object — should BLOCK
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {},
  "integration": {"status": "validated"}
}
EOF

run_test "Campaign gate without campaign object BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: campaign evaluation",
  "task_description": "Evaluate campaign",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# Campaign active, no round recorded — should BLOCK
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {},
  "integration": {"status": "validated"},
  "campaign": {
    "status": "active",
    "current_round": 1,
    "min_e2e_improvement_pct": 3,
    "cumulative_e2e_speedup": 1.0,
    "rounds": [],
    "shipped_optimizations": []
  }
}
EOF

run_test "Campaign gate without round recorded BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: campaign evaluation",
  "task_description": "Evaluate campaign",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# Campaign active, round recorded without top_bottleneck_share_pct — should BLOCK
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {},
  "integration": {"status": "validated"},
  "campaign": {
    "status": "active",
    "current_round": 1,
    "min_e2e_improvement_pct": 3,
    "cumulative_e2e_speedup": 1.0,
    "rounds": [{"round_id": 1, "shipped": [], "implementation_results": {}}],
    "shipped_optimizations": []
  }
}
EOF

run_test "Campaign gate without top_bottleneck_share_pct BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: campaign evaluation",
  "task_description": "Evaluate campaign",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# Campaign complete with fully recorded round — should PASS
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {},
  "integration": {"status": "validated"},
  "campaign": {
    "status": "campaign_complete",
    "current_round": 1,
    "min_e2e_improvement_pct": 3,
    "cumulative_e2e_speedup": 1.12,
    "rounds": [{"round_id": 1, "top_bottleneck_share_pct": 2.1, "shipped": ["op001"], "implementation_results": {"op001": {"status": "PASSED"}}}],
    "shipped_optimizations": ["op001"]
  }
}
EOF

run_test "Campaign gate with complete campaign allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: campaign evaluation",
  "task_description": "Evaluate campaign",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# Campaign active, round with top_bottleneck_share_pct — should PASS (no shipped candidates)
cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {},
  "integration": {"status": "exhausted"},
  "campaign": {
    "status": "active",
    "current_round": 1,
    "min_e2e_improvement_pct": 3,
    "cumulative_e2e_speedup": 1.0,
    "rounds": [{"round_id": 1, "top_bottleneck_share_pct": 8.5, "shipped": [], "implementation_results": {"op001": {"status": "FAILED"}}}],
    "shipped_optimizations": []
  }
}
EOF

run_test "Campaign gate active with exhausted round allows (no shipped = no re-profile needed)" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: diminishing returns check",
  "task_description": "Check diminishing returns",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "== Debate gate with campaign round scoping =="
# ══════════════════════════════════════

# Round 2 debate without scoped directory — should BLOCK
rm -rf "$ARTIFACT_DIR/debate"
mkdir -p "$ARTIFACT_DIR/debate/proposals"
echo "# Summary" > "$ARTIFACT_DIR/debate/summary.md"
echo "# Proposal 1" > "$ARTIFACT_DIR/debate/proposals/champion-1_proposal.md"
echo "# Proposal 2" > "$ARTIFACT_DIR/debate/proposals/champion-2_proposal.md"
mkdir -p "$ARTIFACT_DIR/debate/round_1"

cat > "$ARTIFACT_DIR/state.json" << 'EOF'
{
  "route_decision": {},
  "opportunity_attempts": [],
  "verification_run": {},
  "campaign": {
    "status": "active",
    "current_round": 2,
    "rounds": [{"round_id": 1}]
  }
}
EOF

run_test "Round 2 debate without scoped dir BLOCKS" 2 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: debate winner selection",
  "task_description": "Select debate winner",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# Round 2 debate with scoped directory — should PASS
mkdir -p "$ARTIFACT_DIR/debate/campaign_round_2/proposals"
mkdir -p "$ARTIFACT_DIR/debate/campaign_round_2/round_1"
echo "# Summary" > "$ARTIFACT_DIR/debate/campaign_round_2/summary.md"
echo "# Proposal 1" > "$ARTIFACT_DIR/debate/campaign_round_2/proposals/champion-1_proposal.md"
echo "# Proposal 2" > "$ARTIFACT_DIR/debate/campaign_round_2/proposals/champion-2_proposal.md"

run_test "Round 2 debate with scoped dir allows" 0 '{
  "hook_event_name": "TaskCompleted",
  "task_subject": "GATE: debate winner selection",
  "task_description": "Select debate winner",
  "team_name": "ammo-test-hookguard",
  "cwd": "'"$TMPDIR"'"
}'

# ══════════════════════════════════════
echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
