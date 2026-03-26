# Design Spec: AMMO Transcript Monitor Agent

**Date**: 2026-03-25
**Status**: Draft (Rev 2 — addresses DA review round 1)
**Source**: 5 adversarial debates on agent architecture + empirical transcript analysis
**Relationship**: Additive proposal. The restructuring design spec (`2026-03-25-agent-restructuring-design.md`) deletes ammo-delegate and replaces it with "ad-hoc subagents + DA stop hooks." This design proposes an **additional** agent type not in that spec. It can be adopted independently or alongside the restructuring.

---

## 1. Overview

The ammo-transcript-monitor is a new agent type that passively reads a champion's `.jsonl` session transcript and interjects via SendMessage only when it detects methodology errors. It occupies the "during work" verification slot (Layer 0) that no existing mechanism covers.

**Key properties:**
- Zero coordination overhead on the champion (no checkpoints, no status reports)
- Unmediated access to champion actions, tool outputs, and thinking blocks
- Near-real-time detection (polling every ~5 seconds, detection latency 5-15 sec)
- Cost: $30-50/session (~2,160 polls over 3 hours, full content extraction — see Section 6)

## 2. Relationship to Rejected Approach 3

The consensus findings (`consensus_findings.md`, Dispute 3) rejected a "monitoring agent" approach. This section addresses each objection.

### Objection 1: "No real-time observation mechanism"

> "Monitors have their own separate context window and do NOT observe the champion's tool calls or reasoning."

**Addressed.** The monitor reads the `.jsonl` transcript file that Claude Code automatically generates for every session. This IS an observation mechanism — every tool call, thinking block, and result is recorded. The monitor reads the file via `Bash("python3 filter.py ...")`, not through shared context.

### Objection 2: "Message delivery latency — champion running a 20-minute benchmark blocks all monitor messages"

**Largely resolved.** SendMessage to a champion mid-Bash-command is queued until the command completes. With ~5 second polling, the detection latency is negligible; the dominant delay is message delivery (queued until champion's current tool call completes). Worst-case detection-to-delivery latency is the duration of the champion's longest Bash command (typically <10 min for benchmarks).

**Mitigation**: 5-second polling makes detection near-instantaneous. Even with message delivery delays, this is far earlier than post-hoc mechanisms (30-120 min for validator, end-of-session for stop hooks). The monitor catches errors within seconds of them occurring.

### Objection 3: "Degenerates to checkpoint messaging"

**Addressed.** Checkpoint messaging requires the champion to compose and send status updates — the champion controls what the monitor sees. Transcript reading is the inverse: the champion writes nothing to the monitor, and the monitor reads everything the champion does (tool calls, thinking, results). The champion cannot curate, filter, or selectively present data.

| Property | Checkpoint Messaging (rejected) | Transcript Reading (this design) |
|----------|--------------------------------|----------------------------------|
| Champion awareness | Must pause and report | Zero awareness |
| Information source | Champion-composed messages | Raw transcript (unmediated) |
| Champion can filter | Yes | No — transcript is automatic |
| Coordination overhead | High (round-trips) | Zero |

### Objection 4: "Anchoring — monitor that 'watched throughout' has more anchoring exposure than current validator"

**Acknowledged as a real risk, structurally mitigated but not eliminated.** The monitor reads the champion's thinking blocks, creating anchoring exposure. This design mitigates via:

1. **Process focus, not domain focus**: The monitor flags PROCESS errors (wrong methodology, missing tests, framing bias). It does NOT second-guess domain decisions (which kernel approach). This reduces the surface area where anchoring matters.
2. **Sonnet reviewing Opus**: The monitor (Sonnet) is structurally less likely to override Opus domain reasoning. Its value is in catching procedural shortcuts.
3. **Counter-anchoring protocol**: Explicit instruction to ask "Would I flag this if I hadn't read the justification?"
4. **This is Layer 0, not the only layer**: The structurally independent layers (validator at Layer 1, stop hook at Layer 3) remain unchanged. Anchoring in the monitor does NOT compromise those layers.

**Honest limitation**: The counter-anchoring protocol is behavioral instructions to a Sonnet agent, not a structural guarantee. The consensus findings valued structural guarantees. This design accepts reduced independence at Layer 0 in exchange for earlier detection, relying on Layers 1-3 for structural independence.

---

## 3. The Filter Script

### 3.1 Design Rationale

Raw transcripts are 80% noise (`progress` records) and heavily nested JSON. A Python filter script runs via Bash **outside the agent's context window**, extracting human-readable content and reporting it back. Only the digest enters the monitor's context.

Compression measured on 10 real AMMO transcripts: 11-138x (median 30x) when keeping FULL content of relevant fields. The prior debate measured higher compression (56-454x) with aggressive 200-char truncation. This design keeps full content for more thorough adversarial review, at a modest cost increase.

### 3.2 Full Filter Script (`transcript_filter.py`)

```python
#!/usr/bin/env python3
"""Filter Claude Code session transcripts for DA monitoring.

Reads a .jsonl transcript, drops noise records (progress, file-history-snapshot,
queue-operation), and extracts human-readable content from assistant/user/system
records. Outputs a conversation-log-style digest with timestamps.

Usage:
    python3 transcript_filter.py <path> [--start-line N] [--max-content-len N]
                                        [--state-file PATH]

Outputs digest to stdout.
Writes LAST_LINE_PROCESSED to --state-file (default: /tmp/monitor_state_{basename}.txt).
Also prints LAST_LINE_PROCESSED as the final stdout line for convenience.
"""

import json
import sys
import os
from datetime import datetime

NOISE_TYPES = {"progress", "file-history-snapshot", "queue-operation"}
MAX_CONTENT_LEN = 50000  # Truncate individual content blocks >50KB
WRITE_EDIT_PREVIEW_LEN = 500  # Show first N chars of Write/Edit content


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Filter Claude Code .jsonl transcripts")
    p.add_argument("path", help="Path to .jsonl transcript file")
    p.add_argument("--start-line", type=int, default=0,
                   help="Skip lines before this (0-indexed). Use LAST_LINE_PROCESSED from previous run.")
    p.add_argument("--max-content-len", type=int, default=MAX_CONTENT_LEN,
                   help="Max chars per content block before truncation")
    p.add_argument("--state-file", type=str, default=None,
                   help="Path to write LAST_LINE_PROCESSED (default: /tmp/monitor_state_{basename}.txt)")
    return p.parse_args()


def safe_truncate(text, max_len):
    """Truncate text with [TRUNCATED] marker."""
    if not text or len(text) <= max_len:
        return text
    return text[:max_len] + f"\n[TRUNCATED at {max_len} chars, total {len(text)}]"


def format_timestamp(ts):
    """Format ISO timestamp to readable form."""
    if not ts:
        return "??:??:??"
    try:
        ts_clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts_clean)
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return ts[:19] if len(ts) >= 19 else ts


def safe_get_message(record):
    """Safely extract message dict from record, handling string messages."""
    msg = record.get("message")
    if isinstance(msg, dict):
        return msg
    return {}


def extract_tool_use(block, max_len):
    """Extract readable summary from a tool_use block."""
    name = block.get("name", "unknown")
    inp = block.get("input", {})
    if not isinstance(inp, dict):
        inp = {}

    if name == "Bash":
        desc = inp.get("description", "")
        cmd = inp.get("command", "")
        bg = " [BACKGROUND]" if inp.get("run_in_background") else ""
        timeout = f" [timeout={inp['timeout']}ms]" if inp.get("timeout") else ""
        lines = [f"  BASH{bg}{timeout}: {desc}"]
        lines.append(f"    $ {safe_truncate(cmd, max_len)}")
        return "\n".join(lines)

    elif name == "Read":
        path = inp.get("file_path", "?")
        offset = f" (offset={inp['offset']})" if inp.get("offset") else ""
        limit = f" (limit={inp['limit']})" if inp.get("limit") else ""
        return f"  READ: {path}{offset}{limit}"

    elif name == "Write":
        path = inp.get("file_path", "?")
        content = inp.get("content", "")
        preview = safe_truncate(content, WRITE_EDIT_PREVIEW_LEN)
        return f"  WRITE: {path} ({len(content)} chars)\n    Preview: {preview}"

    elif name == "Edit":
        path = inp.get("file_path", "?")
        old = inp.get("old_string", "")
        new = inp.get("new_string", "")
        return (f"  EDIT: {path}\n"
                f"    old ({len(old)} chars): {safe_truncate(old, WRITE_EDIT_PREVIEW_LEN)}\n"
                f"    new ({len(new)} chars): {safe_truncate(new, WRITE_EDIT_PREVIEW_LEN)}")

    elif name == "Agent":
        desc = inp.get("description", "")
        agent_type = inp.get("subagent_type", "general")
        prompt = inp.get("prompt", "")
        agent_name = inp.get("name", "unnamed")
        model = inp.get("model", "default")
        bg = " [BACKGROUND]" if inp.get("run_in_background") else ""
        iso = f" [isolation={inp['isolation']}]" if inp.get("isolation") else ""
        team = f" [team={inp['team_name']}]" if inp.get("team_name") else ""
        lines = [f"  AGENT{bg}{iso}{team}: {desc} (type={agent_type}, name={agent_name}, model={model})"]
        lines.append(f"    Prompt: {safe_truncate(prompt, max_len)}")
        return "\n".join(lines)

    elif name == "SendMessage":
        to = inp.get("to", inp.get("recipient", "?"))
        summary = inp.get("summary", "")
        msg = inp.get("message", inp.get("content", ""))
        if isinstance(msg, dict):
            msg = json.dumps(msg, indent=2)
        lines = [f"  SENDMESSAGE -> {to}: {summary}"]
        lines.append(f"    {safe_truncate(str(msg), max_len)}")
        return "\n".join(lines)

    elif name in ("Grep", "Glob"):
        pattern = inp.get("pattern", "")
        path = inp.get("path", ".")
        return f"  {name.upper()}: pattern='{pattern}' path={path}"

    elif name == "TaskCreate":
        subject = inp.get("subject", "")
        return f"  TASK_CREATE: {subject}"

    elif name == "TaskUpdate":
        tid = inp.get("taskId", "?")
        status = inp.get("status", "")
        return f"  TASK_UPDATE: #{tid} -> {status}"

    elif name == "Skill":
        skill = inp.get("skillName", inp.get("name", "?"))
        return f"  SKILL: {skill}"

    elif name in ("TeamCreate", "TeamDelete"):
        team = inp.get("team_name", inp.get("name", "?"))
        return f"  {name.upper()}: {team}"

    elif name in ("EnterWorktree", "ExitWorktree"):
        wt = inp.get("name", "?")
        return f"  {name.upper()}: {wt}"

    else:
        inp_str = json.dumps(inp)
        return f"  {name.upper()}: {safe_truncate(inp_str, min(500, max_len))}"


def extract_tool_result(record, max_len):
    """Extract readable summary from a tool result user record."""
    tur = record.get("toolUseResult")
    if tur is None:
        return None

    if isinstance(tur, str):
        return f"  RESULT [ERROR]: {safe_truncate(tur, max_len)}"

    if not isinstance(tur, dict):
        return f"  RESULT: {safe_truncate(str(tur), 200)}"

    # Agent spawn result
    if tur.get("status") in ("teammate_spawned", "async_launched", "completed"):
        status = tur["status"]
        agent_id = tur.get("agentId", tur.get("agent_id", "?"))
        name = tur.get("name", "")
        team = tur.get("team_name", "")

        if status == "completed":
            content = tur.get("content", [])
            text = ""
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text = c.get("text", "")
                        break
            elif isinstance(content, str):
                text = content
            dur_ms = tur.get("totalDurationMs", 0)
            tokens = tur.get("totalTokens", 0)
            tools = tur.get("totalToolUseCount", 0)
            return (f"  AGENT_RESULT [completed]: id={agent_id} "
                    f"dur={dur_ms/1000:.0f}s tokens={tokens} tools={tools}\n"
                    f"    {safe_truncate(text, max_len)}")

        return f"  AGENT_RESULT [{status}]: name={name} id={agent_id} team={team}"

    # SendMessage result
    if "recipients" in tur or tur.get("routing"):
        success = tur.get("success", False)
        msg = tur.get("message", "")
        return f"  MSG_RESULT: {'OK' if success else 'FAIL'} - {msg}"

    # Shutdown request/response
    if "request_id" in tur and "target" in tur:
        target = tur.get("target", "?")
        return f"  SHUTDOWN_SENT: target={target} id={tur.get('request_id', '?')}"

    # Team create/delete result
    if "team_name" in tur and "team_file_path" in tur:
        return f"  TEAM_RESULT: {tur.get('team_name')} -> {tur.get('message', '')}"

    # File read result - show path only, skip content
    if "file" in tur and isinstance(tur["file"], dict):
        fp = tur["file"].get("filePath", "?")
        content_len = len(tur["file"].get("content", ""))
        return f"  READ_RESULT: {fp} ({content_len} chars)"

    # Write/Edit result
    if "filePath" in tur and ("newString" in tur or "content" in tur):
        fp = tur["filePath"]
        if "newString" in tur:
            return f"  EDIT_RESULT: {fp} (modified={not tur.get('userModified', False)})"
        tp = tur.get("type", "write")
        return f"  WRITE_RESULT [{tp}]: {fp}"

    # Bash result (stdout/stderr)
    if "stdout" in tur or "stderr" in tur:
        stdout = tur.get("stdout", "")
        stderr = tur.get("stderr", "")
        interrupted = tur.get("interrupted", False)
        bg_id = tur.get("backgroundTaskId", "")

        parts = []
        if interrupted:
            parts.append("[INTERRUPTED]")
        if bg_id:
            parts.append(f"[bg={bg_id}]")

        prefix = " ".join(parts) + " " if parts else ""

        if stderr and not stdout:
            return f"  BASH_RESULT {prefix}[STDERR]:\n    {safe_truncate(stderr, max_len)}"
        elif stdout:
            return f"  BASH_RESULT {prefix}:\n    {safe_truncate(stdout, max_len)}"
        else:
            return f"  BASH_RESULT {prefix}: (no output)"

    # Grep/Glob result
    if "filenames" in tur:
        mode = tur.get("mode", "?")
        num = tur.get("numFiles", len(tur.get("filenames", [])))
        content = tur.get("content", "")
        if mode == "content" and content:
            return f"  SEARCH_RESULT [{mode}]: {num} files\n    {safe_truncate(content, max_len)}"
        else:
            files = tur.get("filenames", [])
            file_list = ", ".join(files[:10])
            if len(files) > 10:
                file_list += f"... (+{len(files)-10} more)"
            return f"  SEARCH_RESULT [{mode}]: {num} files: {file_list}"

    # Task result
    if "task" in tur or "tasks" in tur:
        if "tasks" in tur:
            tasks = tur["tasks"]
            summaries = [f"#{t.get('id','?')}: {t.get('subject','?')} [{t.get('status','?')}]" for t in tasks[:5]]
            return f"  TASK_LIST: {'; '.join(summaries)}"
        else:
            t = tur["task"]
            return f"  TASK: #{t.get('id','?')}: {t.get('subject','?')}"

    if "taskId" in tur:
        change = tur.get("statusChange", {})
        if change:
            return f"  TASK_RESULT: #{tur['taskId']} {change.get('from','')} -> {change.get('to','')}"
        return f"  TASK_RESULT: #{tur['taskId']} success={tur.get('success')}"

    # Plan result
    if "plan" in tur:
        return f"  PLAN: {safe_truncate(str(tur.get('plan', '')), 300)}"

    # Skill result
    if "commandName" in tur:
        return f"  SKILL_RESULT: {tur['commandName']} success={tur.get('success')}"

    # Fallback
    keys = sorted(tur.keys())
    return f"  RESULT: keys={keys}, preview={safe_truncate(json.dumps(tur), 500)}"


def process_assistant_record(record, max_len):
    """Extract content from an assistant record."""
    lines = []
    content = safe_get_message(record).get("content", [])

    if not isinstance(content, list):
        return lines

    for block in content:
        if not isinstance(block, dict):
            continue

        btype = block.get("type")

        if btype == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                lines.append(f"  THINKING ({len(thinking)} chars):")
                lines.append(f"    {safe_truncate(thinking, max_len)}")

        elif btype == "text":
            text = block.get("text", "")
            if text:
                lines.append(f"  TEXT ({len(text)} chars):")
                lines.append(f"    {safe_truncate(text, max_len)}")

        elif btype == "tool_use":
            lines.append(extract_tool_use(block, max_len))

        elif btype == "tool_result":
            result_content = block.get("content", "")
            lines.append(f"  TOOL_RESULT: {safe_truncate(str(result_content), max_len)}")

    return lines


def process_user_record(record, max_len):
    """Extract content from a user record."""
    lines = []

    tool_result = extract_tool_result(record, max_len)
    if tool_result:
        lines.append(tool_result)
        return lines

    content = safe_get_message(record).get("content", [])
    if isinstance(content, str):
        lines.append(f"  MESSAGE: {safe_truncate(content, max_len)}")
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "")
                    lines.append(f"  MESSAGE: {safe_truncate(text, max_len)}")
                elif btype == "tool_result":
                    rc = block.get("content", "")
                    is_error = block.get("is_error", False)
                    tag = "ERROR" if is_error else "RESULT"
                    lines.append(f"  {tag}: {safe_truncate(str(rc), max_len)}")
            elif isinstance(block, str):
                lines.append(f"  MESSAGE: {safe_truncate(block, max_len)}")

    return lines


def process_system_record(record, max_len):
    """Extract content from a system record."""
    content = safe_get_message(record).get("content", [])
    lines = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                lines.append(f"  SYSTEM ({len(text)} chars): {safe_truncate(text, 300)}")
    elif isinstance(content, str):
        lines.append(f"  SYSTEM ({len(content)} chars): {safe_truncate(content, 300)}")
    return lines


def main():
    args = parse_args()
    path = args.path
    start_line = args.start_line
    max_len = args.max_content_len

    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    # State file for LAST_LINE_PROCESSED (avoids stdout truncation issues, M4)
    state_file = args.state_file
    if not state_file:
        basename = os.path.basename(path).replace(".jsonl", "")
        state_file = f"/tmp/monitor_state_{basename}.txt"

    line_num = 0
    last_successful = start_line  # C1 fix: only advance past successfully parsed lines
    processed = 0
    skipped_noise = 0
    skipped_parse = 0

    agent_name = None
    team_name = None
    session_id = None

    try:
        with open(path, "r") as f:
            for raw_line in f:
                current_line = line_num
                line_num += 1

                if current_line < start_line:
                    continue

                raw_line = raw_line.strip()
                if not raw_line:
                    last_successful = line_num  # blank lines are safe to skip past
                    continue

                if len(raw_line) > 5_000_000:
                    skipped_parse += 1
                    last_successful = line_num  # >5MB lines are deliberately skipped
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    # C1 fix: STOP at first parse error — likely the active write frontier.
                    # Do NOT advance last_successful past this line. Next poll retries it.
                    skipped_parse += 1
                    break

                rtype = record.get("type", "")

                if not agent_name and record.get("agentName"):
                    agent_name = record["agentName"]
                if not team_name and record.get("teamName"):
                    team_name = record["teamName"]
                if not session_id and record.get("sessionId"):
                    session_id = record["sessionId"]

                if rtype in NOISE_TYPES:
                    skipped_noise += 1
                    last_successful = line_num
                    continue

                ts = format_timestamp(record.get("timestamp"))

                # I1 fix: per-record error handling — one malformed record doesn't crash the run
                try:
                    if rtype == "assistant":
                        content_lines = process_assistant_record(record, max_len)
                        if content_lines:
                            print(f"[{ts}] ASSISTANT (line {current_line}):")
                            for cl in content_lines:
                                print(cl)
                            processed += 1

                    elif rtype == "user":
                        user_type = record.get("userType", "")
                        content_lines = process_user_record(record, max_len)
                        if content_lines:
                            label = "TEAMMATE" if user_type == "external" else "USER"
                            print(f"[{ts}] {label} (line {current_line}):")
                            for cl in content_lines:
                                print(cl)
                            processed += 1

                    elif rtype == "system":
                        content_lines = process_system_record(record, max_len)
                        if content_lines:
                            print(f"[{ts}] SYSTEM (line {current_line}):")
                            for cl in content_lines:
                                print(cl)
                            processed += 1

                except Exception as e:
                    print(f"[{ts}] PROCESS_ERROR (line {current_line}): {type(e).__name__}: {e}")
                    skipped_parse += 1

                last_successful = line_num

    except Exception as e:
        print(f"FATAL_ERROR: {type(e).__name__}: {e}", file=sys.stderr)

    # I1 fix: ALWAYS print summary + state, even after exceptions
    print(f"\n--- FILTER SUMMARY ---")
    if agent_name:
        print(f"AGENT: {agent_name}")
    if team_name:
        print(f"TEAM: {team_name}")
    if session_id:
        print(f"SESSION: {session_id}")
    print(f"LINES_READ: {line_num - start_line}")
    print(f"RECORDS_OUTPUT: {processed}")
    print(f"NOISE_SKIPPED: {skipped_noise}")
    print(f"PARSE_ERRORS: {skipped_parse}")
    print(f"LAST_LINE_PROCESSED: {last_successful}")

    # M4 fix: also write to state file so it's never lost to stdout truncation
    try:
        with open(state_file, "w") as sf:
            sf.write(str(last_successful))
    except Exception:
        pass  # best-effort


if __name__ == "__main__":
    main()
```

### 3.3 Edge Case Handling

| Edge Case | Handling |
|-----------|----------|
| Truncated final line (concurrent write) | `json.JSONDecodeError` triggers `break` — stops at write frontier. `last_successful` stays before the truncated line. Next poll retries it. **(C1 fix)** |
| Malformed record mid-file (genuine corruption) | `break` also stops here. If corruption is persistent, every poll stops at the same line. The monitor logs parse error count and can flag stale transcripts after repeated failures. |
| Single line > 5MB | Skipped deliberately, `last_successful` advances past it (not a write frontier). |
| Content block > 50KB | Truncated with `[TRUNCATED at N chars, total M]` marker |
| `toolUseResult` is string (error) | Treated as error text, truncated |
| `toolUseResult` is missing | Record still processed for `message.content` |
| `record["message"]` is string not dict | `safe_get_message()` returns `{}`, preventing `AttributeError` **(M3 fix)** |
| Per-record processing exception | Caught by inner try/except, logged as `PROCESS_ERROR`, does not crash the run **(I1 fix)** |
| Outer-loop exception (file I/O) | Caught by outer try/except. Summary + LAST_LINE_PROCESSED still printed **(I1 fix)** |
| stdout truncation hides LAST_LINE_PROCESSED | State also written to `/tmp/monitor_state_{basename}.txt` **(M4 fix)** |
| Read file content | Path shown, content skipped (`READ_RESULT: path (N chars)`) |
| Write/Edit content | Path + first 500 chars shown |
| Empty transcript | Script outputs summary with `RECORDS_OUTPUT: 0` |

### 3.4 Content Retention Decisions

The design keeps **full content** from thinking blocks, Bash commands, and tool results (up to 50KB per block). This is a deliberate tradeoff vs the filtered-advocate's 200-char truncation:

| Field | This Design | 200-char Design | Rationale |
|-------|------------|-----------------|-----------|
| Thinking blocks | FULL | 200 chars | Catches framing bias buried mid-reasoning |
| Bash commands | FULL | 300 chars | Long commands contain flags that reveal methodology issues |
| Tool results (Bash stdout) | FULL (50KB cap) | 200 chars | Benchmark output numbers needed for Amdahl cross-checks |
| Read file content | PATH ONLY | PATH ONLY | Avoid reading entire codebase into digest |
| Write/Edit content | 500 char preview | PATH ONLY | See what champion wrote without full content |
| SendMessage content | FULL | 150 chars | Delegation patterns and instructions are critical |
| Agent spawn prompts | FULL | 150 chars | Subagent instructions reveal champion strategy |

**Cost impact**: Full content increases per-poll digest size from ~8.4KB to ~30-80KB. This is more expensive than 200-char truncation but enables deeper adversarial review.

---

## 4. Agent Definition

### 4.1 Full Agent Definition (`ammo-transcript-monitor.md`)

```markdown
---
name: ammo-transcript-monitor
description: Periodic adversarial reviewer that monitors champion agents via session transcript logs. Detects methodology errors, framing biases, and procedural violations during champion work.
model: sonnet
---

# AMMO Transcript Monitor

You are an adversarial reviewer monitoring a champion's work via their session transcript. You observe the champion's ACTUAL actions, reasoning, and results — not curated summaries. Your job is to catch methodology errors, framing biases, and procedural violations EARLY, before the champion wastes hours on a flawed approach.

You apply general adversarial reasoning to everything the champion does. You are NOT a checklist robot — you think critically about whether the champion's approach will produce valid results.

## Setup (First Turn)

### 1. Write the Filter Script

Write the transcript filter script to `/tmp/transcript_filter_{your_monitor_name}.py`. The script is provided in your spawn prompt — write it exactly as given.

### 2. Discover the Champion's Transcript

Your spawn prompt provides the champion's `agent_name`. Discover their transcript by writing and running a discovery script:

```bash
python3 /tmp/transcript_discover_{your_monitor_name}.py "{champion_agent_name}"
```

Where the discovery script (also written on first turn) is:

```python
#!/usr/bin/env python3
import json, os, glob, sys

target_name = sys.argv[1]
files = sorted(glob.glob(os.path.expanduser(
    '~/.claude/projects/-home-jinhun-vllm/*.jsonl')),
    key=os.path.getmtime, reverse=True)[:50]

for f in files:
    with open(f) as fh:
        for i, line in enumerate(fh):
            if i > 20:  # agentName is in first ~10 lines
                break
            try:
                d = json.loads(line.strip())
                if d.get('agentName') == target_name:
                    print(f'FOUND:{f}')
                    sys.exit(0)
            except Exception:
                continue
print('NOT_FOUND')
```

If `NOT_FOUND`: the champion may not have started yet. Wait 2 minutes (`sleep 120`) and retry (up to 3 retries). If still not found after 3 retries, message the orchestrator: `DA-MONITOR: Cannot find champion transcript for agent_name={name}. Will keep retrying each poll cycle.`

Store the transcript path for subsequent polls.

### 3. Initialize Observation Log

Create your observation log at `{artifact_dir}/monitor_log_{champion_name}.md`:

```markdown
# Transcript Monitor Log: {champion_name}
## Poll History
(entries appended each cycle)
```

This log persists your observations across context compressions.

### 4. Initial State

Read the state file if it exists (`/tmp/monitor_state_{transcript_basename}.txt`). If not, set `last_line = 0`. The first poll reads the entire transcript.

## Polling Protocol

### Polling Interval: ~5 Seconds

The polling loop uses `sleep 5` between polls for near-real-time monitoring. Each poll runs the Python filter script (fast, <1s), analyzes the digest, and optionally logs/interjects. Over a 3-hour champion session, this produces ~2,000+ polls — the monitor is continuously active.

Your context window will grow rapidly. Rely on the observation log (`{artifact_dir}/monitor_log_{champion_name}.md`) to persist findings across context compressions. Keep per-poll analysis concise.

### Poll Execution

Each poll cycle:

1. **Run the filter**:
```bash
python3 /tmp/transcript_filter_{your_monitor_name}.py {transcript_path} --start-line {last_line}
```

2. **Read the output** — this is the champion's recent activity.

3. **Update state**: Extract `LAST_LINE_PROCESSED: N` from the output (or read `/tmp/monitor_state_{basename}.txt`). Set `last_line = N` for the next poll.

4. **Analyze the digest** for issues (see "What to Watch For" below).

5. **If issue found**: Send ONE message to the champion (see Interjection Protocol), respecting the 1-message-per-minute rate limit.

6. **If no issues**: Do nothing. Do NOT send "looks good" messages.

7. **Append to observation log**: Write a brief summary of this poll's findings (or "no issues") to `{artifact_dir}/monitor_log_{champion_name}.md`. This preserves state across context compressions.

8. **Re-read your observation log** if your context feels incomplete (earlier polls may have been compressed away). This recovers: which issues you already flagged, what the champion was doing in earlier polls, your running tally of messages sent.

9. **Wait**: `Bash(command="sleep 5")`, then repeat from step 1.

### When to Start Flagging

Skip the first poll result if it shows only startup activity (venv activation, initial file reads, reading debate artifacts). Start flagging issues only after you see the champion begin substantive work: writing proposals, running benchmarks, spawning subagents, implementing code.

### When to Stop Polling

Stop polling when ANY of:
- The champion's transcript shows a completion message (e.g., "VALIDATION_REQUEST", "Track complete", "Implementation infeasible")
- The orchestrator sends you a shutdown message
- The transcript stops growing for 5+ minutes AND re-discovery finds no new transcript (see Session Restart below)
- You have been running for 3 hours (safety time limit)

When stopping, send a final summary to the orchestrator:
```
DA-MONITOR SUMMARY: Monitored {champion_name} for {N} polls over {duration}.
Issues flagged: {count}. Categories: {list}.
Full log: {artifact_dir}/monitor_log_{champion_name}.md
```

### Session Restart Handling (I7)

If the transcript has no new lines for 5 minutes but you haven't seen a completion signal:

1. Re-run the discovery script to check for a **new** transcript from the same agent name
2. If a new transcript is found (different file path), switch to it and reset `last_line = 0`
3. If no new transcript, continue polling the old one for 1 more cycle, then stop with a stale-transcript warning to the orchestrator

## Interjection Protocol

### Message Format

```
DA-MONITOR: [{SEVERITY}] {one-sentence summary}.
Evidence: {specific text/command/thinking from transcript with line number}.
Recommended action: {what the champion should do differently}.
```

### Severity Levels

- **CRITICAL**: Stop immediately — wrong optimization target, production parity violation (`--enforce-eager`, `TORCH_COMPILE_DISABLE`), baseline reuse violation, working on wrong worktree/branch. These errors invalidate all subsequent work.
- **WARNING**: Investigate before continuing — single-BS testing, potential framing bias in thinking, missing CUDA graph capture, missing GPU pool reservation, dismissing high-f component without evidence.
- **INFO**: Note for later — minor methodology concern, unusual but possibly valid approach, missing but non-blocking artifact.

### Rate Limiting

- **Maximum 1 message per minute.** If multiple issues are found within a minute, batch them into a single message with the highest severity. Track the timestamp of your last sent message and skip sending if <60 seconds have passed.
- **Maximum 5 total messages per session.** After 5 messages, only send CRITICAL severity. This prevents the monitor from becoming noise.
- **Never send consecutive messages about the same issue.** If you flagged something and the champion hasn't addressed it yet, wait at least 2 minutes before re-flagging. They may be mid-work.

### Escalation Protocol (I6)

If you sent a CRITICAL message and the champion has not responded within 2 minutes, check the transcript for evidence of response:

- **Addressed**: The champion's subsequent actions changed in response (different methodology, corrective action taken) OR the champion's thinking/messages provide evidence-based justification for the current approach. Do NOT escalate.
- **Ignored**: The champion's subsequent actions show NO change — same methodology, same approach, no mention of the DA-MONITOR message. Escalate:
  ```
  SendMessage("team-lead", "DA-MONITOR ESCALATION: {champion_name} did not address
  CRITICAL finding from poll {N}: {summary}. Evidence of no response: {what the
  transcript shows they did instead}.")
  ```

An evidence-based dismissal counts as "addressed" even if you disagree. Your role is to flag process errors, not to override the champion's domain judgment.

### What NOT to Send

- "Looks good so far" — waste of champion attention
- Feedback on work-in-progress (half-written code, exploratory reads)
- Second-guessing the champion's strategic choices (which kernel, which approach)
- Flagging startup/setup activities (venv activation, reading bottleneck_analysis.md)
- Restating what the champion already knows (parroting their own thinking)

## What to Watch For

Apply general adversarial reasoning. These are patterns to be alert to, not a rigid checklist.

### Stage-Specific Focus (I5)

**Debate stage (Stage 3) — focus on:**
- Proposal methodology: evidence tiers, Amdahl consistency, baseline provenance
- Target selection: ignoring dominant f_decode component without justification
- Framing biases in thinking blocks
- Micro-experiment methodology: CUDA graph capture, multi-BS testing

**Implementation stage (Stages 4-5) — focus on:**
- Production parity: `--enforce-eager`, `TORCH_COMPILE_DISABLE` in benchmarks
- Worktree discipline: edits on correct branch, not main
- Validation integrity: not sharing test scripts with validator
- Gate completeness: all 5.1/5.2/5.3a/5.3b gates run before declaring success
- Stage 1 baseline reuse: not running own baseline measurements
- validation_results.md written before completion signal

**Both stages:**
- GPU pool reservation in all GPU commands
- Single batch size testing when multi-BS needed
- Premature conclusions without empirical backing

### Methodology Errors

- **Single batch size testing**: Champion runs benchmarks at only one BS when the target has multiple BS in target.json. Evidence: only one `--bs` or `-b` value in Bash commands.
- **Missing CUDA graph capture**: Benchmark scripts using raw `torch.cuda.Event` timing without graph capture.
- **`--enforce-eager` in benchmarks**: This disables torch.compile and CUDA graphs, producing invalid baselines. Also check for `TORCH_COMPILE_DISABLE=1` and `VLLM_TORCH_COMPILE_LEVEL=0`.
- **Missing GPU pool reservation**: GPU commands without `gpu_reservation.py reserve`. Evidence: `CUDA_VISIBLE_DEVICES` set directly without the pool script.
- **Running own baseline instead of using Stage 1**: Champion running `vllm bench latency` or similar for baseline comparison instead of reading from `{artifact_dir}/runs/baseline_bs*.json`.
- **Baseline provenance mismatch**: Micro-experiment baseline using different API than production (e.g., `torch.mm` instead of `F.linear`).

### Framing Biases

- **"Near-optimal" dismissal**: Thinking block characterizes 70-85% BW utilization as "near peak" or "already good" — 15-30% headroom is significant for BW-bound kernels.
- **Premature conclusion**: Thinking block reaches a conclusion before running a micro-experiment. Evidence: "this won't work because..." without a test.
- **Cherry-picking results**: Champion highlights favorable results while downplaying unfavorable ones.
- **Anchoring on debate plan**: During implementation, champion follows debate plan mechanically without verifying assumptions against actual ncu data.

### Procedural Violations

- **No validation_results.md before completion**: Champion signals completion without writing the required validation results file. [implementation only]
- **Source modification outside worktree**: Champion edits files in the main repo instead of the worktree. [implementation only]
- **Sharing test scripts with validator**: Champion sends test code to the validator. [implementation only]
- **Skipping gates**: Champion declares success without running all required gates. [implementation only]

### Scope/Target Issues

- **Low-impact target**: Champion targets a component with < 5% f_decode when higher-impact targets exist. [debate only]
- **Scope creep**: Champion modifies files outside the planned scope without justification. [implementation only]
- **Wrong branch/worktree**: Champion working on main instead of a worktree branch. [implementation only]

## Independence Considerations

You read the champion's thinking blocks. This creates anchoring risk — you may unconsciously adopt the champion's framing.

**Counter-anchoring protocol:**
1. Before accepting the champion's reasoning, ask: "Would I flag this if I hadn't read the justification?"
2. Focus on PROCESS errors (wrong methodology, missing tests, framing bias), not DOMAIN decisions (which kernel approach is best).
3. If the champion's thinking says "this is near-optimal at 73%", your job is to flag "73% leaves 27% headroom — where's the evidence that this headroom is unrecoverable?" — not to agree that 73% is fine because the champion explained it.
4. The champion is Opus; you are Sonnet. The champion is likely technically correct on domain questions. Your value is in catching procedural and methodological shortcuts that Opus may skip due to overconfidence.

**Limitation**: This protocol is behavioral instructions, not a structural guarantee. The structurally independent verification layers (validator at Layer 1, stop hook at Layer 3) remain the primary independence mechanisms.

## Communication

- Send interjections to the champion via `SendMessage("{champion_name}", "DA-MONITOR: ...")`
- Send status/summary to the orchestrator via `SendMessage("team-lead", "DA-MONITOR: ...")`
- Escalation protocol: see "Escalation Protocol" above

## References

Read these if needed for context on specific DA checks:
- `.claude/skills/ammo/references/debate-rules.md` — micro-experiment guidelines, evidence tiers
- `.claude/skills/ammo/references/validation-defaults.md` — gate definitions, thresholds
- `.claude/skills/ammo/references/gpu-pool.md` — GPU reservation pattern
- `.claude/skills/ammo/references/e2e-delta-math.md` — Amdahl's Law, E2E improvement math
```

---

## 5. Orchestrator Integration

### 5.1 When to Spawn

Spawn the monitor **alongside** the champion, at stage start:
- **Stage 3 (Debate)**: Spawn one monitor per champion after the team is created and champions are spawned.
- **Stage 4-5 (Implementation)**: Spawn one monitor per impl-champion after the implementation team is created.

### 5.2 Spawn Pattern

```python
# I2 fix: each monitor writes to /tmp/transcript_filter_{monitor_name}.py
# I3 fix: champion_name passed as argument to discovery script, not embedded in f-string
monitor_name = f"monitor-{champion_name}"

Agent(
    name=monitor_name,
    subagent_type="ammo-transcript-monitor",
    model="sonnet",
    team_name=existing_team_name,
    run_in_background=True,
    prompt=f"""
Monitor {champion_name} via their session transcript.

## Target Champion
- Agent name: {champion_name}
- Team name: {team_name}
- Stage: {"debate" if is_debate else "implementation"}
- Artifact dir: {artifact_dir}

## Your Identity
- Monitor name: {monitor_name}
- Write filter script to: /tmp/transcript_filter_{monitor_name}.py
- Write discovery script to: /tmp/transcript_discover_{monitor_name}.py

## Campaign Context
- Model: {model_id}
- Hardware: {hardware}
- Bottleneck analysis: {artifact_dir}/bottleneck_analysis.md
- Target config: {artifact_dir}/target.json

## Filter Script
Write this exact script to /tmp/transcript_filter_{monitor_name}.py on your first turn:

<filter_script>
{TRANSCRIPT_FILTER_SCRIPT_CONTENT}
</filter_script>

## Session Discovery
The champion's transcript is in ~/.claude/projects/-home-jinhun-vllm/.
Write the discovery script from your agent definition, then run:
  python3 /tmp/transcript_discover_{monitor_name}.py "{champion_name}"

Begin monitoring. First poll in 3 minutes (let the champion settle in).
"""
)
```

### 5.3 Session ID Discovery

The orchestrator does NOT need to know the champion's session_id. The monitor discovers it by:

1. Running the discovery script with the champion's agent name as `sys.argv[1]` (I3 fix: no f-string injection)
2. The script globs `~/.claude/projects/-home-jinhun-vllm/*.jsonl` sorted by mtime (newest first)
3. For each file, reads first 20 lines looking for `agentName` match
4. The `agentName` appears in a `user` type record near the start of the transcript (verified: first `user` record contains the agent's entrypoint message, typically within lines 0-5)

### 5.4 Failure Handling

- **Monitor errors out**: Champion continues unmonitored. The orchestrator logs the monitor failure but does NOT block the champion's work. DA stop hooks remain as the backstop.
- **Monitor can't find transcript**: Monitor retries with `sleep 120` between attempts, up to 3 times. If still not found, messages the orchestrator and retries each poll cycle.
- **Monitor sends false positive**: Champion can ignore it. The monitor's rate limiting (max 5 messages, max 1/minute) prevents spam.
- **Champion ignores CRITICAL finding**: Monitor escalates to orchestrator after 2 minutes of no response (see Escalation Protocol, I6 fix).
- **Champion session restarts**: Monitor detects stale transcript (no new lines for 5 minutes), re-runs discovery to find new transcript (I7 fix).

### 5.5 Lifecycle

```
Orchestrator spawns champion → Orchestrator spawns monitor (background)
                                    ↓
                              Monitor writes filter + discovery scripts
                              Monitor discovers transcript
                              Monitor creates observation log
                                    ↓
                              [~5 sec loop]
                              Poll → Filter → Analyze → Log → (maybe) Interject
                              sleep 5 → repeat
                                    ↓
                              Champion completes / stale / 3-hour time limit
                                    ↓
                              Monitor sends summary → Monitor stops
```

---

## 6. Cost Analysis

### 6.1 Per-Poll Cost (Full Content)

Each poll:
- **Bash command**: ~50 tokens
- **Filter output (incremental)**: 0-500 tokens (most 5s polls see 0-2 new transcript lines; bursts up to ~5,000 tokens when champion completes a long tool call)
- **Accumulated context**: grows with activity, subject to context compression
- **Output (reasoning + log write)**: ~100-500 tokens (minimal when no new activity)

### 6.2 Full Session Cost (~2,160 Polls over 3 Hours, Sonnet Pricing)

Sonnet pricing: $3/MTok input, $15/MTok output.

With 5-second polling, context accumulation is very different from 10-minute polling. Most polls see zero or minimal new content (champion is mid-tool-call), but the monitor still incurs input token costs for re-reading its accumulated context each turn.

**Key dynamics:**
- **Empty polls (~70% of polls)**: Filter returns "0 new lines." Input: accumulated context only. Output: ~100 tokens. These are cheap individually but add up over ~1,500 empty polls.
- **Active polls (~30% of polls)**: Filter returns new transcript content. Input: accumulated context + digest. Output: ~500 tokens (analysis + log write).
- **Context compression**: At 5s intervals, context hits compression thresholds frequently. After compression, earlier poll details are lost (but preserved in the observation log file). This caps effective per-turn input cost.
- **Prompt caching**: With 5s intervals, prior turns ARE prefix-cacheable within the ~5 min TTL. Cache hits are likely and significantly reduce costs.

**Conservative estimate (no caching, no compression):**

| Phase | Polls | Avg Input/Poll | Avg Output/Poll | Phase Cost |
|-------|-------|---------------|-----------------|------------|
| Setup | 1 | 8,000 | 3,000 | $0.07 |
| Early (first 30 min, ~360 polls) | 360 | 15,000 | 200 | $17.24 |
| Mid (30-90 min, ~720 polls) | 720 | 30,000 | 200 | $66.96 |
| Late (90-180 min, ~1,080 polls) | 1,080 | 50,000 | 200 | $165.24 |
| **TOTAL (uncached, no compression)** | **2,161** | | | **~$249** |

This uncached/uncompressed number is a **worst-case ceiling**. In practice, two mechanisms drastically reduce actual costs:

**Realistic estimate (with caching + compression):**

| Factor | Effect |
|--------|--------|
| Prompt caching (90% cache hit rate at 5s intervals) | ~90% reduction on cached input tokens (cached rate: $0.30/MTok vs $3/MTok) |
| Context compression (triggers every ~50-100 turns) | Caps effective context at ~50K tokens post-compression |
| Empty poll short-circuiting | ~70% of polls process no new content, minimal output |

With caching and compression, realistic cost: **$30-50/session**.

This is **10-15x more expensive** than the previous 540s polling design ($2.50-3.50). The tradeoff is detection latency dropping from ~10-20 minutes to ~5-15 seconds. The user has explicitly accepted this cost increase for near-real-time monitoring.

### 6.3 Comparison

| Mechanism | Cost/Session | Detection Timing | Independence |
|-----------|-------------|-----------------|--------------|
| Transcript monitor (this design, 5s) | $30-50 | Near-real-time (~5-15 sec lag) | Low (mitigated by protocol) |
| Transcript monitor (previous 540s) | $2.50-3.50 | Periodic (~10-20 min lag) | Low (mitigated by protocol) |
| Post-hoc DA agent | $2-5 | After session ends | High |
| DA stop hook (existing) | ~$0.50 | At session end only | High |
| Deleted ammo-delegate | $3-8 | SendMessage-based (variable) | Low |

The monitor is significantly more expensive than other mechanisms but provides near-real-time during-work detection that no other layer offers. This is a deliberate cost-for-latency tradeoff.

---

## 7. Relationship to Other Verification Layers

```
Layer 0: Transcript Monitor (periodic, during work)  ← THIS
  - Catches methodology errors during champion work
  - Sends corrections via SendMessage at turn boundaries
  - Sonnet model, passive observation, no champion coordination
  - NOT structurally independent (reads champion thinking) — compensated by process focus

Layer 1: Orchestrator-Spawned Validator (post-implementation)
  - Writes independent correctness tests and benchmarks
  - Runs all validation gates (5.1, 5.2, 5.3a, 5.3b)
  - Dual-reports to champion AND orchestrator
  - STRUCTURALLY INDEPENDENT (zero prior interaction with champion)

Layer 2: Champion Review (post-validation)
  - Cross-checks validator results against own measurements
  - Writes validation_results.md with evidence chain

Layer 3: DA Stop Hook (at session end)
  - Fires when champion tries to stop
  - Audits artifacts for completeness and consistency
  - Fresh Sonnet agent with zero prior context
  - STRUCTURALLY INDEPENDENT (separate session, no shared state)
```

The monitor does NOT replace any other layer. It adds earlier detection at the cost of reduced independence:
- Without monitor: errors caught at Layer 1 (30-120 min delay) or Layer 3 (end of session)
- With monitor: errors caught within ~5-15 seconds, champion can course-correct almost immediately
- Structural independence is preserved at Layers 1 and 3

---

## 8. Deployment Notes

### 8.1 File Locations

| Artifact | Path |
|----------|------|
| Agent definition | `.claude/agents/ammo-transcript-monitor.md` |
| Filter script (at runtime) | `/tmp/transcript_filter_{monitor_name}.py` (written by monitor on first turn) **(I2 fix: unique per monitor)** |
| Discovery script (at runtime) | `/tmp/transcript_discover_{monitor_name}.py` (written by monitor on first turn) |
| State file (line offset) | `/tmp/monitor_state_{transcript_basename}.txt` |
| Observation log | `{artifact_dir}/monitor_log_{champion_name}.md` **(I4 fix: persists across compressions)** |

### 8.2 SKILL.md Integration

Add to Stage 3 and Stage 4-5 agent tables:

```markdown
| ammo-transcript-monitor | Sonnet | Periodic DA reviewer via transcript reading | Per champion |
```

Add to orchestrator spawn logic in debate-protocol.md and parallel-tracks.md:
- After spawning each champion, spawn a transcript monitor in background
- Monitor name convention: `monitor-{champion_name}` (e.g., `monitor-champion-1`, `monitor-impl-champion-op007`)

### 8.3 Graceful Degradation

If the transcript monitor feature is not desired for a campaign, the orchestrator simply does not spawn monitors. No other agents reference the monitor — it is purely additive. The DA stop hooks and validator remain as the primary verification layers.

---

## 9. Review Findings Addressed

| ID | Severity | Issue | Fix |
|----|----------|-------|-----|
| C1 | Critical | `--start-line` bug: truncated lines permanently skipped | `break` on first parse error; track `last_successful` separately from `line_num` |
| C2 | Critical | No reconciliation with consensus rejection | Added Section 2 addressing all 4 objections explicitly |
| C3 | Critical | `sleep 900` exceeds Bash 600s timeout | Changed to `sleep 5` (5 sec); near-real-time monitoring |
| C4 | Critical | False dependency on restructuring spec | Reframed as additive proposal in header; removed "Depends on" claim |
| I1 | Important | Per-record crash risk; missing LAST_LINE_PROCESSED on exception | Inner try/except per record; outer try/except for file I/O; summary always prints |
| I2 | Important | `/tmp/transcript_filter.py` collision | Changed to `/tmp/transcript_filter_{monitor_name}.py` |
| I3 | Important | f-string injection in discovery | Discovery script uses `sys.argv[1]` instead of embedded f-string |
| I4 | Important | Context compression destroys tracking state | Monitor writes observation log to `{artifact_dir}/monitor_log_{champion_name}.md` |
| I5 | Important | Mixed stage patterns not differentiated | Added "Stage-Specific Focus" section mapping patterns to debate/implementation/both |
| I6 | Important | Vague escalation criteria | Defined "addressed" vs "ignored" explicitly with evidence criteria |
| I7 | Important | No session restart handling | Re-run discovery after 2 consecutive stale polls; switch to new transcript if found |
| M1 | Minor | Optimistic caching assumptions | Cost analysis includes both uncached ceiling ($249) and realistic cached estimate ($30-50); 5s interval makes caching more effective than 540s |
| M2 | Minor | "Real-time" labeling misleading | Changed to "periodic" throughout |
| M3 | Minor | `record.get("message", {})` assumes dict | Added `safe_get_message()` that returns `{}` for non-dict messages |
| M4 | Minor | LAST_LINE_PROCESSED lost to stdout truncation | Also written to state file at `/tmp/monitor_state_{basename}.txt` |
