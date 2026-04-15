#!/bin/bash
# Test harness for ammo-msg-check.sh (PreToolUse hook)
# Run: bash .claude/hooks/test-ammo-msg-check.sh
#
# Tests timestamp-based delivery detection for tmux team member agents.

set -euo pipefail

HOOK="$(cd "$(dirname "$0")" && pwd)/ammo-msg-check.sh"
PASS=0
FAIL=0
TOTAL=0

TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR" /tmp/hook-stderr /tmp/hook-stdout; rm -f /tmp/ammo-msg-gate-*.ts 2>/dev/null || true; }
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
        local rest3="${rest2#*|}"
        local ts_or_read="${rest3%%|*}"
        local read_val="${rest3#*|}"
        # If 5 fields: from|text|summary|timestamp|read
        # If 4 fields: from|text|summary|read (auto-gen ts)
        if [ "$read_val" != "$rest3" ]; then
            # 5 fields — explicit timestamp
            local ts="$ts_or_read"
        else
            # 4 fields — auto-gen timestamp
            read_val="$ts_or_read"
            n=$((n + 1))
            local ts="2026-04-03T18:$(printf '%02d' $n):00.000Z"
        fi
        arr=$(echo "$arr" | jq --arg f "$from" --arg t "$text" \
            --arg ts "$ts" \
            --arg s "$summary" --argjson r "$read_val" \
            '. + [{from: $f, text: $t, timestamp: $ts, read: $r, summary: $s}]')
    done
    echo "$arr" > "$inbox_file"
}

# Delivery timestamps use 18:10+i, so inbox timestamps 18:01-18:09 are "before delivery"
# and 18:20+ are "after delivery"
make_transcript() {
    local file="$1" team="$2" agent="$3" count="${4:-0}"
    echo '{"type":"permission-mode","sessionId":"test-session"}' > "$file"
    echo "{\"teamName\":\"$team\",\"agentName\":\"$agent\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"hello\"},\"sessionId\":\"test-session\"}" >> "$file"
    local i=0
    while [ "$i" -lt "$count" ]; do
        echo "{\"type\":\"user\",\"timestamp\":\"2026-04-03T18:$(printf '%02d' $((10+i))):00.000Z\",\"message\":{\"role\":\"user\",\"content\":\"<teammate-message teammate_id=\\\"sender-$i\\\">msg $i</teammate-message>\"}}" >> "$file"
        i=$((i + 1))
    done
}

make_lead_transcript() {
    local file="$1"
    echo '{"type":"user","message":{"role":"user","content":"hello"},"sessionId":"lead-session"}' > "$file"
}

run_test() {
    local test_name="$1" expected_exit="$2" json_input="$3" check_output="${4:-}" actual_exit=0
    rm -f /tmp/ammo-msg-gate-*.ts 2>/dev/null || true
    TOTAL=$((TOTAL + 1))
    echo "$json_input" | env HOME="$TMPDIR" bash "$HOOK" > /tmp/hook-stdout 2>/tmp/hook-stderr || actual_exit=$?
    local pass=true
    [ "$actual_exit" -ne "$expected_exit" ] && pass=false
    if [ -n "$check_output" ]; then
        if ! grep -qF "$check_output" /tmp/hook-stdout 2>/dev/null && ! grep -qF "$check_output" /tmp/hook-stderr 2>/dev/null; then
            pass=false
        fi
        # If expecting "deny", also verify stdout is valid JSON
        if [ "$check_output" = "deny" ] && [ -s /tmp/hook-stdout ]; then
            if ! jq . /tmp/hook-stdout >/dev/null 2>&1; then
                echo "  WARN: stdout is not valid JSON!"
                pass=false
            fi
        fi
    fi
    if [ "$pass" = "true" ]; then
        echo "  PASS [$TOTAL]: $test_name (exit=$actual_exit)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL [$TOTAL]: $test_name (expected=$expected_exit, got=$actual_exit)"
        [ -n "$check_output" ] && echo "        expected output: $check_output"
        echo "        stdout: $(head -3 /tmp/hook-stdout 2>/dev/null || echo '(none)')"
        echo "        stderr: $(head -3 /tmp/hook-stderr 2>/dev/null || echo '(none)')"
        FAIL=$((FAIL + 1))
    fi
}

TEAM_DIR="$TMPDIR/.claude/teams/test-team"
make_team_config "$TEAM_DIR" "test-team" "verifier-1:general-purpose" "champion-1:ammo-champion" "impl-1:ammo-impl-champion"

TRANSCRIPT="$TMPDIR/transcript.jsonl"

# ════════════════════════════════════════════
echo "== All tools checked =="
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 0
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|alert|alert|true"
run_test "Read tool → DENY" 0 \
    "{\"tool_name\":\"Read\",\"transcript_path\":\"$TRANSCRIPT\"}" "deny"
run_test "Grep tool → DENY" 0 \
    "{\"tool_name\":\"Grep\",\"transcript_path\":\"$TRANSCRIPT\"}" "deny"
run_test "Glob tool → DENY" 0 \
    "{\"tool_name\":\"Glob\",\"transcript_path\":\"$TRANSCRIPT\"}" "deny"

# ════════════════════════════════════════════
echo ""; echo "== Lead agent (no agentName in transcript) =="
run_test "No transcript path → skip" 0 '{"tool_name":"Bash","session_id":"s1"}'

LEAD_TRANSCRIPT="$TMPDIR/lead_transcript.jsonl"
make_lead_transcript "$LEAD_TRANSCRIPT"
run_test "Lead transcript → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$LEAD_TRANSCRIPT\"}"

# ════════════════════════════════════════════
echo ""; echo "== Identity from transcript =="
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 0
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|CRITICAL alert|CRITICAL: issue|true"
run_test "Champion identity → DENY (undelivered)" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" \
    "deny"

make_transcript "$TRANSCRIPT" "test-team" "verifier-1" 0
run_test "Non-champion identity → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}"

# ════════════════════════════════════════════
echo ""; echo "== No inbox file =="
rm -rf "$TEAM_DIR/inboxes" 2>/dev/null || true
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 0
run_test "No inbox file → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}"

# ════════════════════════════════════════════
echo ""; echo "== Timestamp-based delivery detection =="

# All delivered: 2 inbox msgs (ts 18:01, 18:02) < 2 delivery lines (ts 18:10, 18:11)
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 2
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|delivered warning|warn|true"
run_test "2 inbox (ts < delivery) = all delivered → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}"

# 1 undelivered: 1 delivery (ts 18:10), inbox msg 2 at 18:20 (after delivery)
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 1
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|CRITICAL warning|CRITICAL: baseline issue|2026-04-03T18:20:00.000Z|true"
run_test "inbox ts after delivery → DENY" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" \
    "deny"

# Only self undelivered → allow
# Inbox: team-lead at 18:01 (delivered via ts 18:10), champion-1 at 18:20 (after delivery, but self)
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 1
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "champion-1|self msg|self task|2026-04-03T18:20:00.000Z|true"
run_test "Only self undelivered → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}"

# Batched messages (multiple tags per JSONL line)
BATCH_T="$TMPDIR/batch.jsonl"
echo '{"type":"permission-mode","sessionId":"s1"}' > "$BATCH_T"
echo "{\"teamName\":\"test-team\",\"agentName\":\"champion-1\",\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"<teammate-message teammate_id=\\\"team-lead\\\">spawn</teammate-message>\"},\"timestamp\":\"2026-04-03T18:10:00.000Z\"}" >> "$BATCH_T"
echo "{\"type\":\"user\",\"timestamp\":\"2026-04-03T18:15:00.000Z\",\"message\":{\"role\":\"user\",\"content\":\"<teammate-message teammate_id=\\\"mon-1\\\">w1</teammate-message>\\n\\n<teammate-message teammate_id=\\\"team-lead\\\">g1</teammate-message>\"}}" >> "$BATCH_T"

# Inbox: 3 msgs all with ts before batch delivery at 18:15
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" "mon-1|w1|warn|true" "team-lead|g1|guide|true"
run_test "3 inbox (ts < batch delivery) → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$BATCH_T\"}"

# 4th msg with ts after batch delivery
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" "mon-1|w1|warn|true" "team-lead|g1|guide|true" \
    "mon-1|DA-MONITOR: [CRITICAL] new|CRITICAL: new issue|2026-04-03T18:20:00.000Z|true"
run_test "4th inbox msg ts after batch delivery → DENY" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$BATCH_T\"}" \
    "deny"

# ════════════════════════════════════════════
echo ""; echo "== impl-champion also gated =="
make_transcript "$TRANSCRIPT" "test-team" "impl-champion-op001" 0
make_inbox "$TEAM_DIR/inboxes/impl-champion-op001.json" \
    "team-lead|spawn|spawn|true" \
    "team-lead|Fix correctness|CRITICAL: fix needed|true"
run_test "impl-champion with undelivered → DENY" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" \
    "deny"

# ════════════════════════════════════════════
echo ""; echo "== Cooldown behavior =="
# First call denies, second within 5s is cooled down
make_transcript "$TRANSCRIPT" "test-team" "impl-champion-op001" 0
make_inbox "$TEAM_DIR/inboxes/impl-champion-op001.json" \
    "team-lead|spawn|spawn|true" \
    "team-lead|Fix correctness|CRITICAL: fix needed|true"
rm -f /tmp/ammo-msg-gate-*.ts 2>/dev/null || true
TOTAL=$((TOTAL + 1))
# First call — should deny
R1=$(echo "{\"tool_name\":\"Write\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" | \
    env HOME="$TMPDIR" bash "$HOOK" 2>/dev/null) || true
# Second call — should be cooled down (no output)
R2=$(echo "{\"tool_name\":\"Agent\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" | \
    env HOME="$TMPDIR" bash "$HOOK" 2>/dev/null) || true
if echo "$R1" | grep -q "deny" && [ -z "$R2" ]; then
    echo "  PASS [$TOTAL]: 1st call DENY, 2nd call cooldown (no output)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: Cooldown: 1st=$(echo "$R1" | grep -c deny || echo 0) 2nd='$R2'"
    FAIL=$((FAIL + 1))
fi

# ════════════════════════════════════════════
echo ""; echo "== Fail-open =="
run_test "Empty JSON → exit 0" 0 '{}'
run_test "No transcript → exit 0" 0 '{"tool_name":"Bash","session_id":"s1"}'
run_test "Missing transcript file → exit 0" 0 \
    '{"tool_name":"Bash","session_id":"s1","transcript_path":"/nonexistent.jsonl"}'

# ════════════════════════════════════════════
echo ""; echo "== All undelivered from self =="
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 0
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "champion-1|update 1|u1|true" \
    "champion-1|update 2|u2|true"
run_test "All self undelivered → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}"

# ════════════════════════════════════════════
echo ""; echo "== Edge: empty transcript file =="
EMPTY_T="$TMPDIR/empty.jsonl"
: > "$EMPTY_T"
run_test "Empty transcript (0 bytes) → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$EMPTY_T\"}"

# ════════════════════════════════════════════
echo ""; echo "== Edge: transcript with no agentName in first 5 lines =="
NO_AGENT_T="$TMPDIR/no_agent.jsonl"
for i in 1 2 3 4 5; do
    echo '{"type":"permission-mode","sessionId":"test-session"}' >> "$NO_AGENT_T"
done
run_test "5 permission-mode lines, no agentName → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$NO_AGENT_T\"}"

# ════════════════════════════════════════════
echo ""; echo "== Edge: agentName on line 5 (boundary) =="
BOUNDARY_T="$TMPDIR/boundary.jsonl"
for i in 1 2 3 4; do echo '{"type":"system","subtype":"compact"}' >> "$BOUNDARY_T"; done
echo '{"teamName":"test-team","agentName":"champion-1","type":"user"}' >> "$BOUNDARY_T"
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|alert|alert|true"
run_test "agentName on line 5 → resolves + DENY" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$BOUNDARY_T\"}" \
    "deny"

# ════════════════════════════════════════════
echo ""; echo "== Edge: agentName on line 6 (outside window) =="
OUTSIDE_T="$TMPDIR/outside.jsonl"
for i in 1 2 3 4 5; do echo '{"type":"system","subtype":"compact"}' >> "$OUTSIDE_T"; done
echo '{"teamName":"test-team","agentName":"champion-1","type":"user"}' >> "$OUTSIDE_T"
run_test "agentName on line 6 → skip (outside 5-line window)" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$OUTSIDE_T\"}"

# ════════════════════════════════════════════
echo ""; echo "== Edge: special characters in summaries =="
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 0
# Put quotes and backslashes in the SUMMARY field (which reaches JSON output)
SPECIAL_INBOX="$TEAM_DIR/inboxes/champion-1.json"
cat > "$SPECIAL_INBOX" << 'INBOX_EOF'
[{"from":"team-lead","text":"spawn","timestamp":"2026-04-03T18:01:00.000Z","read":true,"summary":"spawn"},
 {"from":"mon-1","text":"found issue","timestamp":"2026-04-03T18:02:00.000Z","read":true,"summary":"\"CRITICAL\" path\\to\\file & <tag>"}]
INBOX_EOF
run_test "Summaries with quotes/backslash/html → DENY (valid JSON)" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" \
    "deny"

# ════════════════════════════════════════════
echo ""; echo "== Edge: empty inbox ([]) =="
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 0
echo '[]' > "$TEAM_DIR/inboxes/champion-1.json"
run_test "Empty inbox [] → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}"

# ════════════════════════════════════════════
echo ""; echo "== Edge: various tool types =="
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 0
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|alert|alert|true"
run_test "SendMessage tool → checked + DENY" 0 \
    "{\"tool_name\":\"SendMessage\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" \
    "deny"
run_test "Edit tool → checked + DENY" 0 \
    "{\"tool_name\":\"Edit\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" \
    "deny"
run_test "Write tool → checked + DENY" 0 \
    "{\"tool_name\":\"Write\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" \
    "deny"
run_test "Agent tool → checked + DENY" 0 \
    "{\"tool_name\":\"Agent\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" \
    "deny"

# ════════════════════════════════════════════
echo ""; echo "== Edge: tool result containing teammate-message tag (false positive) =="
# Simulates reading a file that contains <teammate-message teammate_id= as text
FALSEP_T="$TMPDIR/false_positive.jsonl"
echo '{"type":"permission-mode","sessionId":"s1"}' > "$FALSEP_T"
echo '{"teamName":"test-team","agentName":"champion-1","type":"user","message":{"role":"user","content":"<teammate-message teammate_id=\"team-lead\">spawn</teammate-message>"},"timestamp":"2026-04-03T18:10:00.000Z"}' >> "$FALSEP_T"
# Tool result from Read that contains the tag as file content (array format — NOT string)
echo '{"type":"user","timestamp":"2026-04-03T18:15:00.000Z","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"grep -o '"'"'<teammate-message teammate_id='"'"' file\n<teammate-message teammate_id= in docs"}]}}' >> "$FALSEP_T"
# Inbox: 1 delivered (ts 18:01 < delivery 18:10), 1 new after 18:15 tool_result
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|alert|alert|2026-04-03T18:20:00.000Z|true"
run_test "Tool result with tag text → NOT counted as delivery → DENY" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$FALSEP_T\"}" \
    "deny"

echo ""; echo "== Edge: multiple undelivered messages (JSON newline handling) =="
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 0
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|first warning|warn 1|true" \
    "mon-2|second warning|warn 2|true"
run_test "2 undelivered from different senders → DENY (valid JSON)" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" \
    "deny"

echo ""; echo "== Edge: corrupt transcript line (jq resilience) =="
CORRUPT_T="$TMPDIR/corrupt.jsonl"
echo '{"type":"permission-mode","sessionId":"s1"}' > "$CORRUPT_T"
echo '{"teamName":"test-team","agentName":"champion-1","type":"user","message":{"role":"user","content":"<teammate-message teammate_id=\"team-lead\">spawn</teammate-message>"},"timestamp":"2026-04-03T18:10:00.000Z"}' >> "$CORRUPT_T"
# Corrupt line: not valid JSON
printf '%20000s\n' '' >> "$CORRUPT_T"
# Delivery AFTER the corrupt line — this is the last delivery (ts 18:15)
echo '{"type":"user","timestamp":"2026-04-03T18:15:00.000Z","message":{"role":"user","content":"<teammate-message teammate_id=\"mon-1\">post-corruption msg</teammate-message>"}}' >> "$CORRUPT_T"
# Inbox: both msgs have ts before 18:15 → all delivered
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|post-corruption|alert|true"
run_test "Corrupt line in transcript → deliveries on both sides counted → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$CORRUPT_T\"}"

echo ""; echo "== Edge: multiple undelivered from same sender =="
# 1 delivery at 18:10; inbox has 1 delivered (ts 18:01) + 3 after delivery (ts 18:20+)
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 1
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|warn 1|w1|2026-04-03T18:20:00.000Z|true" \
    "mon-1|warn 2|w2|2026-04-03T18:21:00.000Z|true" \
    "mon-1|warn 3|w3|2026-04-03T18:22:00.000Z|true"
run_test "3 undelivered from same sender → DENY" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}" \
    "deny"

# ════════════════════════════════════════════
echo ""; echo "== P0: In-process subagent skip =="
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 0
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|alert|alert|true"
run_test "agent_type=ammo-delegate → skip" 0 \
    "{\"tool_name\":\"Bash\",\"transcript_path\":\"$TRANSCRIPT\",\"agent_type\":\"ammo-delegate\"}"
run_test "agent_type=general-purpose → skip" 0 \
    "{\"tool_name\":\"Bash\",\"transcript_path\":\"$TRANSCRIPT\",\"agent_type\":\"general-purpose\"}"

# ════════════════════════════════════════════
echo ""; echo "== Docker path: CLAUDE_CONFIG_DIR differs from HOME =="

# Setup: team config + inbox at CLAUDE_CONFIG_DIR, empty HOME
DOCKER_CFG="$TMPDIR/docker-claude-config"
DOCKER_TEAM_DIR="$DOCKER_CFG/teams/test-team"
make_team_config "$DOCKER_TEAM_DIR" "test-team" "champion-1:ammo-champion"
make_inbox "$DOCKER_TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|CRITICAL alert|CRITICAL: baseline issue|true"

DOCKER_FAKE_HOME=$(mktemp -d)  # empty, no .claude/teams/

DOCKER_TRANSCRIPT="$TMPDIR/docker-transcript.jsonl"
make_transcript "$DOCKER_TRANSCRIPT" "test-team" "champion-1" 0

rm -f /tmp/ammo-msg-gate-*.ts 2>/dev/null || true
TOTAL=$((TOTAL + 1))
actual_exit=0
echo "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$DOCKER_TRANSCRIPT\"}" | \
    env HOME="$DOCKER_FAKE_HOME" CLAUDE_CONFIG_DIR="$DOCKER_CFG" \
    bash "$HOOK" > /tmp/hook-stdout 2>/tmp/hook-stderr || actual_exit=$?
if grep -qF "deny" /tmp/hook-stdout 2>/dev/null; then
    echo "  PASS [$TOTAL]: Docker path — inbox found via CLAUDE_CONFIG_DIR → DENY"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: Docker path — inbox found via CLAUDE_CONFIG_DIR → DENY (exit=$actual_exit, no deny in output)"
    echo "        stdout: $(head -3 /tmp/hook-stdout 2>/dev/null || echo '(none)')"
    echo "        stderr: $(head -3 /tmp/hook-stderr 2>/dev/null || echo '(none)')"
    FAIL=$((FAIL + 1))
fi
rm -rf "$DOCKER_FAKE_HOME"

echo ""; echo "== Verifier edge-case probe 2: CLAUDE_CONFIG_DIR set, but no teams there — fail-open =="

PROBE2_CFG=$(mktemp -d)  # exists but has no teams/ subdir
PROBE2_HOME=$(mktemp -d)  # also empty
PROBE2_TRANSCRIPT="$TMPDIR/probe2-transcript.jsonl"
make_transcript "$PROBE2_TRANSCRIPT" "test-team" "champion-1" 0

rm -f /tmp/ammo-msg-gate-*.ts 2>/dev/null || true
TOTAL=$((TOTAL + 1))
actual_exit=0
echo "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$PROBE2_TRANSCRIPT\"}" | \
    env HOME="$PROBE2_HOME" CLAUDE_CONFIG_DIR="$PROBE2_CFG" \
    bash "$HOOK" > /tmp/hook-stdout 2>/tmp/hook-stderr || actual_exit=$?
if [ "$actual_exit" -eq 0 ] && ! grep -qF "deny" /tmp/hook-stdout 2>/dev/null; then
    echo "  PASS [$TOTAL]: CLAUDE_CONFIG_DIR set, no teams — fail-open (exit=0, no deny)"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: CLAUDE_CONFIG_DIR set, no teams — fail-open (expected exit=0 no deny, got exit=$actual_exit)"
    echo "        stdout: $(head -3 /tmp/hook-stdout 2>/dev/null || echo '(none)')"
    FAIL=$((FAIL + 1))
fi
rm -rf "$PROBE2_CFG" "$PROBE2_HOME"

echo ""; echo "== Verifier edge-case probe: CLAUDE_CONFIG_DIR == HOME/.claude =="

PROBE_HOME=$(mktemp -d)
mkdir -p "$PROBE_HOME/.claude/teams/test-team/inboxes"
make_team_config "$PROBE_HOME/.claude/teams/test-team" "test-team" "champion-1:ammo-champion"
make_inbox "$PROBE_HOME/.claude/teams/test-team/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|alert|alert|true"
PROBE_TRANSCRIPT="$TMPDIR/probe-transcript.jsonl"
make_transcript "$PROBE_TRANSCRIPT" "test-team" "champion-1" 0

rm -f /tmp/ammo-msg-gate-*.ts 2>/dev/null || true
TOTAL=$((TOTAL + 1))
actual_exit=0
echo "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$PROBE_TRANSCRIPT\"}" | \
    env HOME="$PROBE_HOME" CLAUDE_CONFIG_DIR="$PROBE_HOME/.claude" \
    bash "$HOOK" > /tmp/hook-stdout 2>/tmp/hook-stderr || actual_exit=$?
if grep -qF "deny" /tmp/hook-stdout 2>/dev/null; then
    echo "  PASS [$TOTAL]: CLAUDE_CONFIG_DIR == HOME/.claude — inbox found, DENY"
    PASS=$((PASS + 1))
else
    echo "  FAIL [$TOTAL]: CLAUDE_CONFIG_DIR == HOME/.claude — inbox found, DENY (exit=$actual_exit, no deny)"
    echo "        stdout: $(head -3 /tmp/hook-stdout 2>/dev/null || echo '(none)')"
    echo "        stderr: $(head -3 /tmp/hook-stderr 2>/dev/null || echo '(none)')"
    FAIL=$((FAIL + 1))
fi
rm -rf "$PROBE_HOME"

# ════════════════════════════════════════════
echo ""; echo "== NEW: Self-message only undelivered (timestamp) =="
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 1
# Inbox: 1 delivered (ts 18:01 < delivery 18:10), 1 self-msg after delivery
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "champion-1|self note|self|2026-04-03T18:20:00.000Z|true"
run_test "Self-msg ts after delivery, non-self all before → skip" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}"

# ════════════════════════════════════════════
echo ""; echo "== NEW: No deliveries in transcript =="
# Fresh transcript with agentName but zero teammate-message tags
FRESH_T="$TMPDIR/fresh.jsonl"
echo '{"type":"permission-mode","sessionId":"s1"}' > "$FRESH_T"
echo '{"teamName":"test-team","agentName":"champion-1","type":"user","message":{"role":"user","content":"hello"},"timestamp":"2026-04-03T18:00:00.000Z"}' >> "$FRESH_T"
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|initial dispatch|true"
run_test "No deliveries + inbox has msgs → DENY" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$FRESH_T\"}" \
    "deny"

# ════════════════════════════════════════════
echo ""; echo "== NEW: Exact timestamp match (inbox ts == delivery ts) =="
# Delivery at 18:10, inbox msg with EXACT same timestamp → treated as delivered
make_transcript "$TRANSCRIPT" "test-team" "champion-1" 1
make_inbox "$TEAM_DIR/inboxes/champion-1.json" \
    "team-lead|spawn|spawn|true" \
    "mon-1|alert|alert|2026-04-03T18:10:00.000Z|true"
run_test "inbox ts == delivery ts → ALLOW (treated as delivered)" 0 \
    "{\"tool_name\":\"Bash\",\"session_id\":\"s1\",\"transcript_path\":\"$TRANSCRIPT\"}"

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed out of $TOTAL tests"
echo "================================"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
