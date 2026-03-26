# Design Spec: Background Transcript Watcher for AMMO Monitor

**Date**: 2026-03-26
**Status**: Draft — Rev 2 (addresses adversarial review round 1)
**Replaces**: The inline `transcript_filter.py` + poll-sleep-poll architecture in `2026-03-25-transcript-monitor-design.md`

---

## 1. Overview

The transcript monitor currently polls via repeated `Bash(command="python3 filter.py --start-line N")` calls every 5 seconds, producing ~2,160 Bash forks over a 3-hour session. Each fork starts a new Python process, re-imports libraries, opens/seeks the file, and exits. Most polls return zero new lines.

This design replaces that with a **single long-running background Python process** (`transcript_watcher.py`) that:
- Tails the champion's `.jsonl` transcript continuously (sub-second file checks)
- Auto-discovers and tails direct subagent transcripts when it detects Agent tool spawns
- Filters noise records and extracts human-readable content (reusing existing filter logic)
- Writes batched digests to a file every ~10 seconds
- Runs as a pre-existing script in `.claude/skills/ammo/scripts/` (no more writing scripts to `/tmp/` on first turn)

The monitor agent starts the watcher once via `Bash(run_in_background=True)` and then periodically `Read`s the digest file — no Bash overhead per poll.

### Key Properties

- **1 Bash call** (to start watcher) + **~1,080 sleep Bash calls** + **~1,080 Read calls** vs ~2,160 Bash-running-Python calls (old). Total tool calls are similar; the savings come from eliminating ~2,160 Python process startups and replacing Bash-with-output calls with lightweight Read calls.
- **~1,080 Read calls** over 3 hours (every 10s) — note: Read with high line offsets is O(n) scan, not O(1) seek, but the digest file is small (~5,000 lines max) so this is negligible
- **Zero Python startup overhead** per poll — single persistent process
- **Subagent coverage** — direct children auto-discovered and tailed
- **No /tmp script writing** — watcher lives in the repo at `.claude/skills/ammo/scripts/transcript_watcher.py`
- **Detection latency**: 0.5s (file check interval) + up to 10s (batch interval) = 0.5-10.5s

### Research Findings (Claude Code Capabilities)

Before settling on this design, we investigated Claude Code's native mechanisms:

| Mechanism | Verdict | Why |
|-----------|---------|-----|
| `FileChanged` hook | Not usable | Fires only within the same session; cannot inject prompts into other agents |
| `CronCreate` / `/loop` | Too coarse | 1-minute minimum granularity |
| Channels (research preview) | Not production-ready | Requires claude.ai login, Team/Enterprise opt-in, custom MCP server |
| `run_in_background` + `Read` output | **Best option** | Agent can read background process output while it's running; single process, periodic Read |
| Read tool byte-offset | Not available | Offset is line-based (O(n) scan), not byte-based (O(1) seek) |

The background watcher approach is confirmed as the best practical solution given current Claude Code capabilities.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│  Monitor Agent (Sonnet)                               │
│                                                       │
│  Setup (one-time):                                    │
│    Bash(run_in_background=True):                     │
│      python3 .claude/skills/ammo/scripts/             │
│        transcript_watcher.py                          │
│        --champion-name {name}                         │
│        --transcript {path}                            │
│        --digest-file /tmp/monitor_digest_{name}.md    │
│        --batch-interval 10                            │
│        --projects-dir ~/.claude/projects/...          │
│                                                       │
│  Poll loop (every 10s):                               │
│    Read(/tmp/monitor_digest_{name}.md, offset=N)     │
│    → analyze new batches                              │
│    → interject if issue found                         │
│    → update observation log                           │
│    Bash(command="sleep 10")                           │
│    → repeat                                           │
└──────────┬────────────────────────┬──────────────────┘
           │ starts (once)          │ reads (every 10s)
           ▼                        ▼
┌────────────────────────┐  ┌───────────────────────────┐
│  Watcher Process       │  │  Digest File              │
│  (background, single   │  │  /tmp/monitor_digest_X.md │
│   Python process)      │──▶                            │
│                        │  │  --- BATCH T1 (L450-467) │
│  • tail-f champion     │  │  [T] ASSISTANT (line 450) │
│  • detect Agent spawns │  │    THINKING: ...           │
│  • glob-match subagent │  │    BASH: ncu --set full   │
│    transcripts         │  │  --- BATCH T2 (L467-489) │
│  • filter noise        │  │  [sub:val-1] [T] ASST ... │
│  • batch & flush       │  │  --- BATCH T3 (L489-489) │
│    every 10s           │  │  (no new content)          │
│  • write status file   │  └───────────────────────────┘
└────────────────────────┘
           │ discovers & tails
           ▼
┌──────────────────────────────────────────────────┐
│  .jsonl transcript files                          │
│  ~/.claude/projects/-home-jinhun-vllm/           │
│                                                   │
│  {champion_session}.jsonl  ← primary target       │
│  {subagent_1_session}.jsonl ← auto-discovered     │
│  {subagent_2_session}.jsonl ← auto-discovered     │
└──────────────────────────────────────────────────┘
```

---

## 3. Watcher Script: `transcript_watcher.py`

### 3.1 Location

`.claude/skills/ammo/scripts/transcript_watcher.py` — checked into the repo, not written to `/tmp/` at runtime.

### 3.2 CLI Interface

```
python3 transcript_watcher.py \
    --champion-name <name>          # For labeling in digest
    --transcript <path>             # Path to champion's .jsonl (or use --discover)
    --discover                      # Auto-discover transcript by champion name
    --digest-file <path>            # Where to write batched digests
    --batch-interval <seconds>      # How often to flush (default: 10)
    --projects-dir <path>           # Where .jsonl files live (for subagent discovery)
    --status-file <path>            # Optional: write liveness/stats (default: /tmp/watcher_status_{name}.json)
    --max-content-len <int>         # Truncation limit per content block (default: 50000)
    --max-idle-seconds <int>        # Self-terminate after N seconds with no new transcript lines (default: 300)
    --max-runtime-seconds <int>     # Self-terminate after N seconds total runtime (default: 10800 = 3 hours)
```

### 3.3 Internal Architecture

```python
class TranscriptTailer:
    """Tails a single .jsonl file, yields filtered records."""
    def __init__(self, path, label, start_line=0):
        self.fh = open(path, 'r')
        self.path = path
        self.label = label          # "" for champion, "sub:{name}" for subagents
        self.line_num = 0
        self.last_successful = 0
        self._seek_to_line(start_line)

    def poll(self) -> list[str]:
        """Read any new complete lines, filter, return formatted strings."""
        new_lines = []
        self.last_raw_records = []  # Raw JSON dicts for caller inspection (I1 fix)
        while True:
            line = self.fh.readline()
            if not line:
                break  # No more data — file frontier reached
            line = line.strip()
            self.line_num += 1
            if not line:
                self.last_successful = self.line_num
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                break  # Truncated line — active write frontier (C1 logic)
            self.last_raw_records.append(record)
            # Filter noise
            if record.get('type') in NOISE_TYPES:
                self.last_successful = self.line_num
                continue
            # Format record using existing filter logic
            formatted = format_record(record, self.line_num - 1, self.label)
            if formatted:
                new_lines.append(formatted)
            self.last_successful = self.line_num
        return new_lines


class TranscriptWatcher:
    """Manages champion tailer + discovered subagent tailers."""
    def __init__(self, args):
        self.champion_tailer = TranscriptTailer(args.transcript, label="")
        self.subagent_tailers = {}   # name -> TranscriptTailer
        self.pending_discoveries = {} # name -> first_seen_time
        self._discovery_last_attempt = {}  # name -> last_attempt_time (C2/I2 throttle)
        self.digest_file = open(args.digest_file, 'w')
        self.batch_buffer = []
        self.last_flush = time.time()
        self.batch_interval = args.batch_interval
        self.projects_dir = args.projects_dir
        self.champion_name = args.champion_name
        self.stats = {'polls': 0, 'batches': 0, 'records': 0,
                      'subagents_discovered': 0}

    def run(self):
        """Main loop — runs until termination condition met."""
        self.start_time = time.time()
        self.last_activity = time.time()  # Last time any tailer returned new content

        while True:
            # Check termination conditions (C1 fix)
            elapsed = time.time() - self.start_time
            idle = time.time() - self.last_activity
            if elapsed >= self.max_runtime_seconds:
                self._write_termination("max_runtime_exceeded", elapsed)
                break
            if idle >= self.max_idle_seconds:
                self._write_termination("idle_timeout", idle)
                break

            # Poll champion
            new_records = self.champion_tailer.poll()
            if new_records:
                self.last_activity = time.time()
            self.batch_buffer.extend(new_records)

            # Detect Agent spawns from raw records (I1 fix: parse JSON, not formatted text)
            for raw_record in self.champion_tailer.last_raw_records:
                self._check_for_agent_spawn_from_json(raw_record)

            # Attempt pending subagent discoveries (no expiry — retries until session ends)
            self._try_discover_subagents()

            # Poll all active subagent tailers
            for name, tailer in list(self.subagent_tailers.items()):
                sub_records = tailer.poll()
                if sub_records:
                    self.last_activity = time.time()
                self.batch_buffer.extend(sub_records)

            # Flush batch if interval elapsed
            if time.time() - self.last_flush >= self.batch_interval:
                self._flush_batch()

            self.stats['polls'] += 1
            time.sleep(0.5)  # Sub-second file check interval

    def _check_for_agent_spawn_from_json(self, record):
        """Parse raw JSON record for Agent tool_use blocks, queue discovery. (I1 fix)"""
        if record.get('type') != 'assistant':
            return
        msg = record.get('message', {})
        if not isinstance(msg, dict):
            return
        for block in msg.get('content', []):
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use' and block.get('name') == 'Agent':
                inp = block.get('input', {})
                agent_name = inp.get('name')
                if agent_name and agent_name not in self.subagent_tailers:
                    if agent_name not in self.pending_discoveries:
                        self.pending_discoveries[agent_name] = time.time()  # track first-seen time

    def _try_discover_subagents(self):
        """For each pending discovery, glob .jsonl files and check agentName.

        C2 fix: Never give up on discovery — subagent startup can take 10-30s.
        Pending discoveries are retried every poll (0.5s) until the subagent is
        found or the watcher terminates. Discovery is throttled to once per 2s
        per agent to avoid excessive glob I/O (I2 fix: stagger for multi-watcher).
        """
        now = time.time()
        for name, first_seen_time in list(self.pending_discoveries.items()):
            if name in self.subagent_tailers:
                del self.pending_discoveries[name]
                continue
            # Throttle: only attempt discovery every 2 seconds per agent
            last_attempt = self._discovery_last_attempt.get(name, 0)
            if now - last_attempt < 2.0:
                continue
            self._discovery_last_attempt[name] = now
            # Glob and match
            path = self._find_transcript_by_agent_name(name)
            if path:
                self.subagent_tailers[name] = TranscriptTailer(
                    path, label=f"sub:{name}")
                self.stats['subagents_discovered'] += 1
                del self.pending_discoveries[name]
            # If not found, stays in pending_discoveries — retried next eligible poll

    def _find_transcript_by_agent_name(self, agent_name):
        """Glob projects_dir/*.jsonl sorted by mtime, check first 20 lines."""
        files = sorted(
            glob.glob(os.path.join(self.projects_dir, '*.jsonl')),
            key=os.path.getmtime, reverse=True)[:50]
        for f in files:
            if f == self.champion_tailer.path:
                continue  # Skip champion's own transcript
            # Skip files already tracked
            if any(t.path == f for t in self.subagent_tailers.values()):
                continue
            with open(f) as fh:
                for i, line in enumerate(fh):
                    if i > 20:
                        break
                    try:
                        d = json.loads(line.strip())
                        if d.get('agentName') == agent_name:
                            return f
                    except:
                        continue
        return None

    def _flush_batch(self):
        """Write accumulated records to digest file."""
        now = datetime.now().isoformat(timespec='seconds')
        champion_line = self.champion_tailer.last_successful
        if self.batch_buffer:
            self.digest_file.write(
                f"--- BATCH {now} (champion line {champion_line}) ---\n")
            for line in self.batch_buffer:
                self.digest_file.write(line + '\n')
            self.stats['records'] += len(self.batch_buffer)
        else:
            self.digest_file.write(
                f"--- BATCH {now} (champion line {champion_line}) [empty] ---\n")
        self.digest_file.flush()
        self.batch_buffer = []
        self.last_flush = time.time()
        self.stats['batches'] += 1
        self._write_status()

    def _write_status(self):
        """Write liveness/stats to status file."""
        # JSON with: timestamp, polls, batches, records, subagents tracked, etc.
        ...
```

### 3.4 Record Formatting

The `format_record()` function reuses the extraction logic from the existing `transcript_filter.py` in the design spec (Section 3.2 of `2026-03-25-transcript-monitor-design.md`). Specifically:

- `process_assistant_record()` — thinking blocks, text, tool_use formatting
- `process_user_record()` — tool results, teammate messages
- `process_system_record()` — system prompts (truncated)
- `extract_tool_use()` — per-tool formatting (Bash, Read, Agent, SendMessage, etc.)
- `extract_tool_result()` — agent results, bash results, search results, etc.
- `safe_truncate()`, `format_timestamp()`, `safe_get_message()` — utilities

These functions are extracted as-is from the existing filter script. The only additions:
- A `label` prefix for subagent attribution (e.g., `[sub:validator-1]`)
- The wrapping `format_record()` function that dispatches by record type

### 3.5 Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Truncated final line (concurrent write) | `JSONDecodeError` → `break` (same C1 logic as existing filter). Next `poll()` retries from the same position since `fh` stays seeked there. |
| Champion file deleted/rotated | `readline()` returns `""` indefinitely. Watcher self-terminates after `--max-idle-seconds` (default 300s) with no new content. **(C1 fix)** |
| Subagent transcript not yet created | Queued in `pending_discoveries`, retried every 2s indefinitely until found or watcher terminates. No expiry — subagent startup can take 10-30s. **(C2 fix)** |
| Subagent transcript from different champion | `_find_transcript_by_agent_name()` skips files already tracked and the champion's own file. However, if two champions spawn agents with the same name, there's an ambiguity. Mitigated by checking mtime (newest first) and expecting the subagent to start after the spawn was detected. |
| Watcher process dies | Monitor agent detects stale status file (no update for >30s) and can restart the watcher. |
| Digest file grows very large | The monitor reads with `offset=last_line`, only processing new batches. The file itself grows unboundedly but is in `/tmp/` and is small (filtered content only, ~1-5MB over 3 hours). |
| `fh.readline()` after file truncation | If the `.jsonl` file is truncated (shouldn't happen in normal Claude Code operation), `readline()` returns `""`. The tailer treats this as "no new data" and continues polling. |
| Multiple batch flushes with no new content | Empty batches written as `--- BATCH ... [empty] ---`. Monitor agent can skip these cheaply. |

### 3.6 Subagent Spawn Detection

The watcher detects Agent spawns by parsing the **raw JSON record** directly, not the formatted output. The `TranscriptTailer` exposes `last_raw_records` (the parsed JSON dicts from the most recent `poll()` call), and `_check_for_agent_spawn_from_json()` inspects `assistant` records for `tool_use` blocks where `name == "Agent"`, extracting the spawned agent name from `input.name`.

This approach is robust against changes to `extract_tool_use()` formatting — it depends only on the Claude Code transcript schema, which is stable. **(I1 fix: no regex on formatted output)**

### 3.7 Concurrent Watchers

During overlapped debate rounds with 2-4 debate champions + 2-3 impl-champions, **4-8 simultaneous watchers is the expected case**, not an edge case. Each watcher's subagent discovery involves globbing up to 50 `.jsonl` files and reading the first 20 lines of each.

Mitigations **(I2 fix)**:
- Discovery attempts are throttled to once per 2 seconds per pending agent (not every 0.5s poll)
- The watcher adds a randomized initial delay of 0-2 seconds on startup (`time.sleep(random.uniform(0, 2))`) to stagger discovery bursts when multiple watchers start simultaneously
- `_find_transcript_by_agent_name()` skips already-tracked files, reducing the search space as subagents are discovered

At worst case (8 watchers × 3 pending agents each × 50 files × 20 lines), this is ~24,000 line reads per discovery round — spread over 2 seconds and staggered across watchers, this is well within I/O capacity.

### 3.8 Depth Limit

The watcher monitors the champion + direct children only (depth 1). It does NOT recursively discover subagents spawned by subagents. This is a deliberate scope limit:
- Champions typically spawn 1-3 subagents (validator, researcher, micro-experiment runner)
- Sub-subagents are rare and their work is summarized in the subagent's result
- Recursive discovery would require watching an unbounded number of files

---

## 4. Monitor Agent Definition Changes

### 4.1 Setup (First Turn)

**Before** (current):
1. Write filter script to `/tmp/transcript_filter_{name}.py` (~570 lines)
2. Write discovery script to `/tmp/transcript_discover_{name}.py` (~15 lines)
3. Run discovery script to find transcript
4. Initialize observation log
5. Read state file

**After** (new):
1. Run discovery script (still needed — the transcript path must be found before starting the watcher). But the discovery logic can be embedded in the watcher's `--discover` mode, or the orchestrator can pass the transcript path directly.
2. Start watcher: `Bash(command="python3 .claude/skills/ammo/scripts/transcript_watcher.py --champion-name {name} --transcript {path} --digest-file /tmp/monitor_digest_{name}.md --batch-interval 10 --projects-dir ~/.claude/projects/-home-jinhun-vllm/", run_in_background=True)`
3. Initialize observation log (same as before)

**Simplification**: The watcher can handle transcript discovery itself via a `--discover` flag:
```
python3 transcript_watcher.py \
    --champion-name {name} \
    --discover --projects-dir ~/.claude/projects/-home-jinhun-vllm/ \
    --digest-file /tmp/monitor_digest_{name}.md
```
The watcher globs for the champion's transcript, then starts tailing. This eliminates the separate discovery script entirely.

### 4.2 Polling Protocol

**Before**:
```
sleep 5 → Bash(python3 filter.py --start-line N) → analyze → repeat
```

**After**:
```
sleep 10 → Read(/tmp/monitor_digest_{name}.md, offset=last_line) → analyze → repeat
```

Key differences:
- `Read` instead of `Bash` — cheaper, no process fork
- 10s interval instead of 5s — the watcher batches at 10s, so polling faster than the batch interval is wasteful
- No `--start-line` tracking — the monitor just tracks its last-read line in the digest file
- State management is simpler — no separate state file needed

### 4.3 Liveness Monitoring

The monitor should periodically check the watcher's status file (`/tmp/watcher_status_{name}.json`):
```json
{
    "last_flush": "2026-03-26T14:32:15",
    "polls": 1234,
    "batches": 56,
    "records": 890,
    "subagents": ["validator-1", "researcher-1"],
    "champion_line": 4567,
    "uptime_seconds": 1800
}
```

If `last_flush` is more than 30 seconds stale, the watcher may have died. The monitor should restart it.

### 4.4 Stopping

When the monitor decides to stop (completion signal, stale transcript, 3-hour limit):
1. The watcher process is automatically cleaned up when the monitor's Claude Code session exits
2. Alternatively, the monitor can kill it explicitly via `Bash(command="kill $(cat /tmp/watcher_pid_{name}.txt)")`
3. The watcher writes its PID to `/tmp/watcher_pid_{name}.txt` on startup for this purpose

---

## 5. Orchestrator Integration Changes

### 5.1 Spawn Pattern Update

The orchestrator's spawn prompt for the monitor no longer includes the filter script content. The `projects_dir` is derived dynamically from the working directory **(M4 fix)**:

```python
import os
projects_dir = os.path.expanduser("~/.claude/projects/") + os.getcwd().replace("/", "-")

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

## Watcher Script
Start the background watcher on your first turn:
  Bash(run_in_background=True):
    python3 .claude/skills/ammo/scripts/transcript_watcher.py \
      --champion-name {champion_name} \
      --discover \
      --projects-dir {projects_dir} \
      --digest-file /tmp/monitor_digest_{monitor_name}.md \
      --batch-interval 10 \
      --max-idle-seconds 300 \
      --max-runtime-seconds 10800

Then poll the digest file every 10 seconds using Read.

## Campaign Context
- Model: {model_id}
- Hardware: {hardware}
- Bottleneck analysis: {artifact_dir}/bottleneck_analysis.md
- Target config: {artifact_dir}/target.json
"""
)
```

The spawn prompt is dramatically smaller — no more embedding 570 lines of Python.

### 5.2 File Locations

| Artifact | Path |
|----------|------|
| Watcher script (checked in) | `.claude/skills/ammo/scripts/transcript_watcher.py` |
| Agent definition | `.claude/agents/ammo-transcript-monitor.md` |
| Digest file (runtime) | `/tmp/monitor_digest_{monitor_name}.md` |
| Status file (runtime) | `/tmp/watcher_status_{monitor_name}.json` |
| PID file (runtime) | `/tmp/watcher_pid_{monitor_name}.txt` |
| Observation log (runtime) | `{artifact_dir}/monitor_log_{champion_name}.md` |

---

## 6. Cost Analysis

### 6.1 Comparison with Previous Design

The total tool call count is similar between old and new designs **(M1 fix: honest framing)**. The savings come from (a) eliminating ~2,160 Python process startups, (b) replacing Bash-running-Python calls with lightweight Read calls, and (c) a dramatically smaller spawn prompt.

| Metric | Previous (5s Bash polls) | New (background watcher + 10s Read) |
|--------|------------------------|--------------------------------------|
| Bash tool calls (Python) | ~2,160 (filter script) | 1 (start watcher) |
| Bash tool calls (sleep) | 0 (sleep was in filter loop) | ~1,080 |
| Read tool calls | 0 | ~1,080 (digest reads) |
| **Total tool calls** | **~2,160** | **~2,161** |
| Python process startups | ~2,160 | 1 |
| Spawn prompt size | ~8,000 tokens (includes filter script) | ~1,500 tokens |
| Per-poll input cost | High (Bash overhead + filter output in context) | Lower (Read output is just new batches) |
| Subagent coverage | None | Auto-discovered direct children |

### 6.2 Token Cost Estimate

The primary cost savings come from:
1. **Smaller spawn prompt** — no 570-line filter script embedded (~6,500 fewer tokens on every turn)
2. **Read vs Bash output** — Read returns just file content; Bash returns command + stdout + metadata
3. **Batched digests** — 10s batches often contain 0 records; the monitor processes `[empty]` markers cheaply
4. **No state management overhead** — no reading/writing separate state files

**Note (I5)**: Read with high line offsets performs an O(n) scan from the start of the file, not O(1) seek. However, the digest file is small (~5,000-6,000 lines over 3 hours of filtered content), making this negligible compared to scanning a 50,000+ line raw transcript.

Estimated cost with caching + compression: **$20-40/session** (vs $30-50 previously). The savings are moderate because the dominant cost is still the accumulated context in the monitor agent, not the tool call overhead.

---

## 7. Migration Path

This design replaces the filter-script-per-poll approach. The following files need coordinated updates:

1. **Create** `.claude/skills/ammo/scripts/transcript_watcher.py`
2. **Update** `.claude/agents/ammo-transcript-monitor.md` — see Section 7.1 below for specific changes **(I4 fix)**
3. **Update** `.claude/skills/ammo/docs/specs/2026-03-25-transcript-monitor-design.md` — reference the watcher script, update Section 3 (filter script) and Section 4 (agent definition)
4. **Update** orchestrator spawn logic in `debate-protocol.md` and `parallel-tracks.md` — remove `<filter_script>` embedding from spawn prompts, use new spawn pattern from Section 5.1 **(I3 fix)**
5. **Remove** the filter script from `/tmp/` writing instructions (it's no longer needed)
6. The existing filter script logic is preserved inside `transcript_watcher.py`, not deleted

### 7.1 Agent Definition Changes (I4 fix)

The following sections of `ammo-transcript-monitor.md` must be updated:

**"Setup (First Turn)"** — replace all 4 setup steps with:
1. Start background watcher via `Bash(run_in_background=True)` with the command from Section 4.1
2. Initialize observation log (unchanged)

Remove entirely:
- "Write the Filter Script" section
- "Discover the Champion's Transcript" section (watcher handles discovery via `--discover`)
- "Initial State" section (watcher tracks state internally)

**"Polling Protocol"** — replace:
- Title: "Polling Interval: ~5 Seconds" → "Polling Interval: ~10 Seconds"
- Remove `sleep 5` / `Bash(command="sleep 5")` instructions
- Replace poll execution steps 1-3 (Bash filter + state tracking) with single `Read` call on digest file
- Replace step 9 (`sleep 5`) with `sleep 10`
- Remove "Between polls" context window note (watcher handles filtering externally)

**"When to Stop Polling"** — update:
- Remove "transcript stops growing for 5+ minutes" (watcher self-terminates on idle)
- Add "watcher status file stale for >30 seconds — restart watcher"
- Keep: completion signal, orchestrator shutdown, 3-hour safety limit

---

## 8. Resolved Questions

1. **Digest file cleanup**: Don't rotate — let it grow. Rotation risks the monitor missing content between reads. The watcher also maintains a **metadata sidecar file** (`/tmp/monitor_digest_{name}.meta`) with the current batch count and last-written line number, allowing the monitor to detect "no new content" without reading the digest at all. **(M3 fix: adopted)**
2. **Watcher error reporting**: The watcher writes errors to BOTH the digest file (as `--- ERROR {timestamp}: {message} ---` batch markers) AND the status file (`"error": "..."` field). The digest approach ensures the monitor sees errors even if it doesn't check the status file. Fatal errors (e.g., transcript deleted) are followed by self-termination.
3. **Multiple monitors**: 4-8 simultaneous watchers is the **expected case** during overlapped debate rounds (see Section 3.7). Each watcher uses unique digest/status/PID file paths via `--digest-file`, and discovery is staggered with randomized startup delay. **(I2 fix: acknowledged as normal, not edge case)**

---

## 9. Review Findings Addressed

| ID | Severity | Issue | Fix |
|----|----------|-------|-----|
| C1 | Critical | Orphaned watcher has no termination condition | Added `--max-idle-seconds` (300) and `--max-runtime-seconds` (10800) CLI flags; watcher self-terminates |
| C2 | Critical | Subagent discovery window too short (5s) | Removed attempt limit; pending discoveries retry every 2s indefinitely until found or watcher terminates |
| I1 | Important | Subagent spawn detection fragile (regex on formatted output) | Parse raw JSON record for `tool_use` blocks with `name="Agent"`, extract `input.name` directly |
| I2 | Important | Multiple simultaneous watchers cause I/O contention | Acknowledged 4-8 watchers as expected case; added discovery throttle (2s/agent) and randomized startup delay (0-2s) |
| I3 | Important | `debate-protocol.md` spawn prompt inconsistency | Added to migration path (Section 7, step 4) |
| I4 | Important | Agent definition changes not described in spec | Added Section 7.1 with specific section-by-section changes |
| I5 | Important | Read tool O(n) scan at high offsets | Noted in cost analysis; digest file is small (~5K lines) so impact is negligible |
| M1 | Minor | Cost savings overstated (tool call count similar) | Corrected comparison table; honest framing of what saves cost vs what doesn't |
| M2 | Minor | PID file path collision risk | Low risk; unique naming via `--digest-file` flag covers this |
| M3 | Minor | Digest rotation has clear answer + optimization | Adopted: no rotation, added metadata sidecar file for cheap "new content?" check |
| M4 | Minor | `--projects-dir` hardcoded in spawn prompt | Derived dynamically from `os.getcwd()` in orchestrator |
