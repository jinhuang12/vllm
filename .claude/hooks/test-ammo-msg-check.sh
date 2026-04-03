#!/bin/bash
# Test harness for ammo-msg-check.sh (PreToolUse hook)
# Run: bash .claude/hooks/test-ammo-msg-check.sh
#
# Tests both detection strategies:
#   Strategy 1: inbox read:false — for in-process agents
#   Strategy 2: transcript counting — for tmux agents (read always true)

set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/ammo-msg-check.sh"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR" /tmp/hook-stderr /tmp/hook-stdout; }
trap cleanup EXIT

make_team_config() {
    local team_dir="$1" team_name="$2"; shift 2
    mkdir -p "$team_dir"
    local members="[" first=true
    for member in "$@"; do
        local name="${member%%:*}"
        local atype="${member#*:}"
        [ "$first" = "true" ] && first=false || members="$members,"
        members="$members{\"name\":\"$name\",\"agentType\":\"$atype\",\"agentId\":\"$name@$team_name\"}"
    done
    echo "{\"name\":\"$team_name\",\"leadSessionId\":\"lead-session\",\"members\":${members}]}" > "$team_dir/config.json"
}

# make_inbox: entries are "from|text|summary|read" (read defaults to true)
make_inbox() {
    local inbox_file="$1"; shift
    mkdir -p "$(dirname "$inbox_file")"
    local n=0 arr="[]"
    for entry in "$@"; do
        local from="${entry%%|*}"
        local rest="${entry#*|}"
        local text="${rest%%|*}"
        local rest2="${rest#*|}"
        local summary="${rest2%%|*}"
        local read_val="${rest2#*|}"
        [ "$read_val" = "$rest2" ] && read_val="true"
        n=$((n + 1))
        arr=$(echo "$arr" | jq --arg f "$from" --arg t "$text" \
            --arg ts "2026-04-03T18:$(printf '%02d' $n):00.000Z" \
            --arg s "$summary" --argjson r "$read_val" \
            '. + [{from: $f, text: $t, timestamp: $ts, read: $r, summary: $s}]')
    done
    echo "$arr" > "$inbox_file"
}

# make_transcript: creates JSONL with N teammate-message occurrences
make_transcript() {
    local file="$1" team="$2" agent="$3" count="${4:-0}"
    echo '{"type":"permission-mode","sessionId":"test-session"}' > "$file"
    echo "{\"teamName\":\"$team\",\"agentName\":\"$agent\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hello\"}}" >> "$file"
    local i=0
    while [ "$i" -lt "$count" ]; do
        echo "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"<teammate-message teammate_id=\\\"sender-$i\\\">msg $i</teammate-message>\"}}" >> "$file"
        i=$((i + 1))
    done
}

run_test() {
    local test_name="$1" expected_exit="$2" json_input="$3" check_stderr="${4:-}" actual_exit=0
    TOTAL=$((TOTAL + 1))
    echo "$json_input" | env HOME="$TMPDIR" bash "$HOOK" > /tmp/hook-stdout 2>/tmp/hook-stderr || actual_exit=$?
    local pass=true
    [ "$actual_exit" -ne "$expected_exit" ] && pass=false
    [ -n "$check_stderr" ] && ! grep -qF "$check_stderr" /tmp/hook-stderr 2>/dev/null && pass=false
    if [ "$pass" = "true" ]; then
        echo "  PASS [$TOTAL]: $test_name (exit=$actual_exit)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $test_name (expected=$expected_exit, got=$actual_exit)"
        [ -n "$check_stderr" ] && echo "        expected stderr: $check_stderr"
        echo "        stderr: $(head -5 /tmp/hook-stderr 2>/dev/null || echo '(none)')"
        FAIL=$((FAIL + 1))
    fi
}

TEAM_DIR="$TMPDIR/.claude/teams/test-team"
make_team_config "$TEAM_DIR" "test-team" "verifier-1:general-purpose" "champion-1:ammo-champion" "impl-1:ammo-impl-champion"

TRANSCRIPT="$TMPDIR/transcript.jsonl"

# ════════════════════════════════════════════
echo "== Tool filtering: Read/Grep/Glob skipped =="
run_test "Read tool → skip" 0 '{"tool_name":"Read","agent_type":"champion-1","session_id":"s1"}'
run_test "Grep tool → skip" 0 '{"tool_name":"Grep","agent_type":"champion-1","session_id":"s1"}'
run_test "Glob tool → skip" 0 '{"tool_name":"Glob","agent_type":"champion-1","session_id":"s1"}'

# ════════════════════════════════════════════
echo ""; echo "== Lead agent (no agent_type) =="
run_test "No agent_type → skip" 0 '{"tool_name":"Bash","session_id":"s1"}'

# ════════════════════════════════════════════
echo ""; echo "== Not a champion agent =="
run_test "general-purpose agent → skip" 0 '{"tool_name":"Bash","agent_type":"verifier-1","session_id":"s1"}'

# ════════════════════════════════════════════
echo ""; echo "== No inbox file =="
rm -rf "$TEAM_DIR/inboxes" 2>/dev/null || true
run_test "No inbox file → skip" 0 '{"tool_name":"Bash","agent_type":"champion-1","session_id":"s1"}'

# ════════════════════════════════════════════
echo ""; echo "== Strategy 1: inbox read:false (in-process mode) =="

make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn prompt|spawn|true"
run_test "All read:true → skip" 0 '{"tool_name":"Bash","agent_type":"champion-1","session_id":"s1"}'

make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn prompt|spawn|true" \
    "mon-1|DA-MONITOR: [CRITICAL] error|CRITICAL: ULP error|false"
run_test "1 unread from monitor → BLOCK (strategy 1)" 2 \
    '{"tool_name":"Bash","agent_type":"champion-1","session_id":"s1"}' \
    "CRITICAL: ULP error"

make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn prompt|spawn|true" \
    "champion-1|self update|self task|false"
run_test "Only self unread → skip" 0 '{"tool_name":"Bash","agent_type":"champion-1","session_id":"s1"}'

make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|warning|warn 1|false" \
    "mon-1|critical|CRITICAL: bad|false" \
    "team-lead|fix|fix needed|false"
run_test "3 unread from 2 senders → BLOCK (strategy 1)" 2 \
    '{"tool_name":"Bash","agent_type":"champion-1","session_id":"s1"}' \
    "3 unread teammate message"

# ════════════════════════════════════════════
echo ""; echo "== Strategy 2: transcript counting (tmux mode) =="
# Simulate tmux: all read:true, but transcript has fewer messages than inbox

make_transcript "$TRANSCRIPT" "test-team" "champion-1" 1
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|CRITICAL warning|CRITICAL: baseline issue|true"
run_test "2 inbox, 1 in transcript → BLOCK (strategy 2)" 2 \
    "{\"tool_name\":\"Bash\",\"agent_type\":\"champion-1\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" \
    "CRITICAL: baseline issue"

make_transcript "$TRANSCRIPT" "test-team" "champion-1" 2
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|delivered warning|warn|true"
run_test "2 inbox = 2 in transcript → skip" 0 \
    "{\"tool_name\":\"Bash\",\"agent_type\":\"champion-1\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}"

# Tmux: undelivered but only from self → allow
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 1
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "champion-1|self msg|self task|true"
run_test "Tmux: only self undelivered → skip" 0 \
    "{\"tool_name\":\"Bash\",\"agent_type\":\"champion-1\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}"

# Tmux: batched messages (multiple tags per JSONL line)
BATCH_T="$TMPDIR/batch.jsonl"
echo '{"type":"permission-mode","sessionId":"s1"}' > "$BATCH_T"
echo "{\"teamName\":\"test-team\",\"agentName\":\"champion-1\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"<teammate-message teammate_id=\\\"team-lead\\\">spawn</teammate-message>\"}}" >> "$BATCH_T"
echo "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"<teammate-message teammate_id=\\\"mon-1\\\">w1</teammate-message>\\n\\n<teammate-message teammate_id=\\\"team-lead\\\">g1</teammate-message>\"}}" >> "$BATCH_T"

make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" "mon-1|w1|warn|true" "team-lead|g1|guide|true"
run_test "Tmux: 3 inbox = 3 occurrences (batched) → skip" 0 \
    "{\"tool_name\":\"Bash\",\"agent_type\":\"champion-1\",\"session_id\":\"s1\",\"transcript_path\":\"$BATCH_T\"}"

make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" "mon-1|w1|warn|true" "team-lead|g1|guide|true" \
    "mon-1|DA-MONITOR: [CRITICAL] new|CRITICAL: new issue|true"
run_test "Tmux: 4 inbox vs 3 occurrences → BLOCK (strategy 2)" 2 \
    "{\"tool_name\":\"Bash\",\"agent_type\":\"champion-1\",\"session_id\":\"s1\",\"transcript_path\":\"$BATCH_T\"}" \
    "CRITICAL: new issue"

# ════════════════════════════════════════════
echo ""; echo "== ammo-impl-champion also gated =="
make_inbox "$TEAM_DIR/inboxes/impl-1.json" \
    "team-lead|spawn|spawn|true" \
    "team-lead|Fix correctness|CRITICAL: fix needed|false"
run_test "impl-champion with unread → BLOCK" 2 \
    '{"tool_name":"Bash","agent_type":"impl-1","session_id":"s1"}' \
    "CRITICAL: fix needed"

# ════════════════════════════════════════════
echo ""; echo "== Repeated blocking =="
run_test "Second call → still BLOCK" 2 \
    '{"tool_name":"Write","agent_type":"impl-1","session_id":"s1"}'
run_test "Third call (Agent) → still BLOCK" 2 \
    '{"tool_name":"Agent","agent_type":"impl-1","session_id":"s1"}'

# ════════════════════════════════════════════
echo ""; echo "== Fail-open =="
run_test "Empty JSON → exit 0" 0 '{}'
run_test "No agent_type → exit 0" 0 '{"tool_name":"Bash","session_id":"s1"}'
run_test "Unknown agent → exit 0" 0 '{"tool_name":"Bash","agent_type":"ghost","session_id":"s1"}'
run_test "Missing transcript → exit 0 (strategy 2 skipped)" 0 \
    '{"tool_name":"Bash","agent_type":"champion-1","session_id":"s1","transcript_path":"/nonexistent.jsonl"}'

# ════════════════════════════════════════════
echo ""; echo "== All unread from self (both strategies) =="
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "champion-1|update 1|u1|false" \
    "champion-1|update 2|u2|false"
run_test "Strategy 1: all self unread → skip" 0 \
    '{"tool_name":"Bash","agent_type":"champion-1","session_id":"s1"}'

make_transcript "$TRANSCRIPT" "test-team" "champion-1" 0
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "champion-1|update 1|u1|true" \
    "champion-1|update 2|u2|true"
run_test "Strategy 2: all self undelivered → skip" 0 \
    "{\"tool_name\":\"Bash\",\"agent_type\":\"champion-1\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}"

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
