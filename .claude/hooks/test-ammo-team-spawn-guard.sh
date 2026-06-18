#!/bin/bash
# Test harness for ammo-team-spawn-guard.sh (PreToolUse hook, matcher=Agent)
# Run: bash .claude/hooks/test-ammo-team-spawn-guard.sh
#
# The hook DENIES Agent-tool spawns of ammo-champion / ammo-impl-champion /
# ammo-transcript-monitor subagent types when the orchestrator does not
# provide a valid team_name. All other subagent_types are ALLOWED.
#
# Key fields consumed from hook stdin (probe-confirmed):
#   .tool_input.subagent_type    — e.g. "ammo-champion"
#   .tool_input.team_name        — e.g. "ammo-round-1-qwen3.5-b200"
#   .agent_type                  — empty on orchestrator, populated in subagents
#
# Deny is emitted as:
#   {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#    "permissionDecision":"deny","permissionDecisionReason":"..."}}
# with exit 0 (Claude Code parses the JSON for the decision).

set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/ammo-team-spawn-guard.sh"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() {
    rm -rf "$TMPDIR" "$TMPDIR/hook-stdout" "$TMPDIR/hook-stderr" 2>/dev/null || true
}
trap cleanup EXIT

make_team_dir() {
    local cfg_dir="$1" team="$2"
    mkdir -p "$cfg_dir/teams/$team"
    echo "{\"name\":\"$team\",\"leadSessionId\":\"lead\",\"members\":[]}" \
        > "$cfg_dir/teams/$team/config.json"
}

# run_test:
#   $1 name
#   $2 expected_exit (0 always — deny is a JSON output, not a non-zero exit)
#   $3 expected decision: "deny" | "allow" | "no_output"
#   $4 json input
#   $5 (optional) expected substring of permissionDecisionReason (for deny)
#   Extra env: TEST_HOME, TEST_CONFIG_DIR set before call
run_test() {
    local name="$1" expected_exit="$2" expected_decision="$3"
    local json_input="$4" expected_reason="${5:-}"
    local actual_exit=0
    TOTAL=$((TOTAL + 1))

    local home_dir="${TEST_HOME:-$TMPDIR/home-default}"
    mkdir -p "$home_dir"
    local env_args=(env HOME="$home_dir")
    if [ -n "${TEST_CONFIG_DIR:-}" ]; then
        env_args+=(CLAUDE_CONFIG_DIR="$TEST_CONFIG_DIR")
    fi
    if [ -n "${TEST_PATH_OVERRIDE:-}" ]; then
        env_args+=(PATH="$TEST_PATH_OVERRIDE")
    fi

    echo "$json_input" | "${env_args[@]}" bash "$HOOK" \
        > "$TMPDIR/hook-stdout" 2>"$TMPDIR/hook-stderr" || actual_exit=$?

    local pass=true
    if [ "$actual_exit" -ne "$expected_exit" ]; then
        pass=false
    fi

    case "$expected_decision" in
        deny)
            if ! grep -q '"permissionDecision"[[:space:]]*:[[:space:]]*"deny"' "$TMPDIR/hook-stdout" 2>/dev/null; then
                pass=false
            fi
            if [ -s "$TMPDIR/hook-stdout" ] && ! jq . "$TMPDIR/hook-stdout" >/dev/null 2>&1; then
                echo "  WARN: stdout is not valid JSON"
                pass=false
            fi
            if [ -n "$expected_reason" ]; then
                if ! grep -qF "$expected_reason" "$TMPDIR/hook-stdout" 2>/dev/null; then
                    pass=false
                fi
            fi
            ;;
        allow)
            # Allow = exit 0 AND no deny decision (empty or allow JSON both ok)
            if grep -q '"permissionDecision"[[:space:]]*:[[:space:]]*"deny"' "$TMPDIR/hook-stdout" 2>/dev/null; then
                pass=false
            fi
            ;;
        no_output)
            if [ -s "$TMPDIR/hook-stdout" ]; then
                pass=false
            fi
            ;;
    esac

    if [ "$pass" = "true" ]; then
        echo "  PASS [$TOTAL]: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $name (expected_exit=$expected_exit, got=$actual_exit, want=$expected_decision)"
        echo "        stdout: $(head -3 "$TMPDIR/hook-stdout" 2>/dev/null || echo '(none)')"
        echo "        stderr: $(head -3 "$TMPDIR/hook-stderr" 2>/dev/null || echo '(none)')"
        FAIL=$((FAIL + 1))
    fi
}

# ══════════════════════════════════════════════
echo "== T1-T3: Eligible subagent types WITHOUT team_name → DENY =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/t1" TEST_CONFIG_DIR=""
run_test "T1 ammo-champion without team_name → DENY" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"champion-1"}}' \
    "team"

run_test "T2 ammo-impl-champion without team_name → DENY" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-impl-champion","name":"impl-op001"}}' \
    "team"

run_test "T3 ammo-transcript-monitor without team_name → DENY" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-transcript-monitor","name":"monitor-champion-1"}}' \
    "team"

# ══════════════════════════════════════════════
echo ""; echo "== T4-T5: Valid team_name + existing dir → ALLOW =="
# ══════════════════════════════════════════════

T4_HOME="$TMPDIR/t4"
mkdir -p "$T4_HOME/.claude"
make_team_dir "$T4_HOME/.claude" "ammo-round-1-qwen-h100"
TEST_HOME="$T4_HOME" TEST_CONFIG_DIR=""
run_test "T4 ammo-champion + ammo-round-1-qwen-h100 (dir exists) → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"champion-1","team_name":"ammo-round-1-qwen-h100"}}'

T5_HOME="$TMPDIR/t5"
mkdir -p "$T5_HOME/.claude"
make_team_dir "$T5_HOME/.claude" "ammo-round-2-foo"
TEST_HOME="$T5_HOME" TEST_CONFIG_DIR=""
run_test "T5 ammo-impl-champion + ammo-round-2-foo (dir exists) → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-impl-champion","name":"impl-1","team_name":"ammo-round-2-foo"}}'

run_test "T5 ammo-transcript-monitor + ammo-round-2-foo (dir exists) → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-transcript-monitor","name":"monitor-impl-1","team_name":"ammo-round-2-foo"}}'

# ══════════════════════════════════════════════
echo ""; echo "== T5b: team_name present but dir does NOT exist → DENY =="
# ══════════════════════════════════════════════

T5B_HOME="$TMPDIR/t5b"
mkdir -p "$T5B_HOME/.claude/teams"
# No team dirs at all
TEST_HOME="$T5B_HOME" TEST_CONFIG_DIR=""
run_test "T5b ammo-champion + team_name=ammo-round-99-nonexistent → DENY" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"champion-1","team_name":"ammo-round-99-nonexistent"}}' \
    "does not exist"

# Also: a stale team dir exists but the requested team_name differs
T5B2_HOME="$TMPDIR/t5b2"
mkdir -p "$T5B2_HOME/.claude"
make_team_dir "$T5B2_HOME/.claude" "ammo-round-1-old"
TEST_HOME="$T5B2_HOME" TEST_CONFIG_DIR=""
run_test "T5b ammo-champion asks for round-2 team that wasn't created → DENY" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"champion-1","team_name":"ammo-round-2-new"}}' \
    "does not exist"

# ══════════════════════════════════════════════
echo ""; echo "== T6: Out-of-scope subagent types → ALLOW =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/t6" TEST_CONFIG_DIR=""
run_test "T6 ammo-researcher no team_name → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-researcher","name":"baseline-research"}}'

run_test "T6 general-purpose no team_name → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"general-purpose","name":"helper"}}'

run_test "T6 ammo-delegate no team_name → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-delegate","name":"dl"}}'

# ══════════════════════════════════════════════
echo ""; echo "== T7: Inside subagent (agent_type set) → ALLOW =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/t7" TEST_CONFIG_DIR=""
# Even if subagent_type is gated and team_name missing, we MUST allow because
# the hook is running inside a subagent, not on the orchestrator thread.
run_test "T7 agent_type=ammo-delegate top-level → ALLOW (not orchestrator)" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c"},"agent_type":"ammo-delegate"}'

run_test "T7 agent_type=ammo-researcher top-level → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-impl-champion","name":"i"},"agent_type":"ammo-researcher"}'

# ══════════════════════════════════════════════
echo ""; echo "== T9: Missing tool_input.subagent_type → ALLOW (fail-open) =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/t9" TEST_CONFIG_DIR=""
run_test "T9 no tool_input.subagent_type → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"name":"c"}}'

run_test "T9 empty tool_input → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{}}'

run_test "T9 no tool_input at all → ALLOW" 0 allow \
    '{"tool_name":"Agent"}'

# ══════════════════════════════════════════════
echo ""; echo "== T10: team_name with dots → sanitized dir lookup (dots → dashes) =="
# ══════════════════════════════════════════════

T10_HOME="$TMPDIR/t10"
mkdir -p "$T10_HOME/.claude"
# Filesystem dir uses dashes; team_name from orchestrator contains dots
make_team_dir "$T10_HOME/.claude" "ammo-round-1-qwen3-5-4b"
TEST_HOME="$T10_HOME" TEST_CONFIG_DIR=""
run_test "T10 team_name 'ammo-round-1-qwen3.5-4b' → sanitized match → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c","team_name":"ammo-round-1-qwen3.5-4b"}}'

# And when NEITHER the raw nor sanitized form exists:
T10B_HOME="$TMPDIR/t10b"
mkdir -p "$T10B_HOME/.claude/teams"
TEST_HOME="$T10B_HOME" TEST_CONFIG_DIR=""
run_test "T10b team_name with dots + no matching dir (raw or sanitized) → DENY" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c","team_name":"ammo-round-9-zzz.9"}}'

# ══════════════════════════════════════════════
echo ""; echo "== T11: CLAUDE_CONFIG_DIR overrides HOME/.claude =="
# ══════════════════════════════════════════════

T11_HOME="$TMPDIR/t11home"
T11_CFG="$TMPDIR/t11cfg"
mkdir -p "$T11_HOME" "$T11_CFG"
# Team dir lives under CLAUDE_CONFIG_DIR, NOT under HOME/.claude
make_team_dir "$T11_CFG" "ammo-round-1-config-override"
TEST_HOME="$T11_HOME" TEST_CONFIG_DIR="$T11_CFG"
run_test "T11 CLAUDE_CONFIG_DIR override → team found there → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c","team_name":"ammo-round-1-config-override"}}'

# Negative: same config dir, different team requested
TEST_HOME="$T11_HOME" TEST_CONFIG_DIR="$T11_CFG"
run_test "T11 CLAUDE_CONFIG_DIR override, unknown team → DENY" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c","team_name":"ammo-round-2-does-not-exist"}}'

# ══════════════════════════════════════════════
echo ""; echo "== T12: jq unavailable → ALLOW (fail-open) =="
# ══════════════════════════════════════════════

# Build a minbin dir that symlinks every PATH tool EXCEPT jq. The hook's
# `command -v jq` must return nothing and the hook must exit 0 silently.
MIN_PATH="$TMPDIR/minbin-nojq"
mkdir -p "$MIN_PATH"
for t in bash sh cat echo env sed grep tr dirname basename mktemp stat head tail awk printf test which rm ls mkdir touch cut date; do
    real=$(command -v "$t" 2>/dev/null || true)
    [ -n "$real" ] && ln -sf "$real" "$MIN_PATH/$t"
done
TEST_HOME="$TMPDIR/t12" TEST_CONFIG_DIR="" TEST_PATH_OVERRIDE="$MIN_PATH"
run_test "T12 jq unavailable → ALLOW (fail-open)" 0 no_output \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c"}}'
unset TEST_PATH_OVERRIDE

# ══════════════════════════════════════════════
echo ""; echo "== Fail-open edge cases =="
# ══════════════════════════════════════════════

TEST_HOME="$TMPDIR/fedge" TEST_CONFIG_DIR=""
run_test "Empty JSON {} → ALLOW (no subagent_type)" 0 allow '{}'
run_test "Non-Agent tool_name → ALLOW" 0 allow \
    '{"tool_name":"Bash","tool_input":{"command":"ls"}}'
run_test "tool_input is null → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":null}'

# ══════════════════════════════════════════════
echo ""; echo "== Round-transition scenario: two teams, both match their own calls =="
# ══════════════════════════════════════════════

T_RT_HOME="$TMPDIR/rt"
mkdir -p "$T_RT_HOME/.claude"
make_team_dir "$T_RT_HOME/.claude" "ammo-round-1-rtmodel"
make_team_dir "$T_RT_HOME/.claude" "ammo-round-2-rtmodel"
TEST_HOME="$T_RT_HOME" TEST_CONFIG_DIR=""
run_test "Round 1 team exists, spawn points at round 1 → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c","team_name":"ammo-round-1-rtmodel"}}'
run_test "Round 2 team exists, spawn points at round 2 → ALLOW" 0 allow \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-impl-champion","name":"i","team_name":"ammo-round-2-rtmodel"}}'
run_test "Both exist, spawn asks for round 3 → DENY" 0 deny \
    '{"tool_name":"Agent","tool_input":{"subagent_type":"ammo-champion","name":"c","team_name":"ammo-round-3-rtmodel"}}' \
    "does not exist"

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
