# Design Spec: Extended Transcript Filter with Subagent Support

**Date**: 2026-03-26
**Status**: Draft — Rev 4 (simplified: filter extension replaces background watcher)
**Extends**: The `transcript_filter.py` in `2026-03-25-transcript-monitor-design.md` Section 3.2

---

## 1. Overview

The transcript monitor polls via `Bash(command="python3 filter.py --start-line N")` every 5 seconds. This works but has two gaps:

1. **No subagent coverage**: Champions spawn subagents (validator, researcher, micro-experiment runner) whose work happens in separate `.jsonl` transcripts. The monitor only sees `teammate_spawned` status in the champion's transcript — it cannot observe what those subagents actually do.

2. **Filter script written to /tmp each session**: The monitor writes ~570 lines of Python to `/tmp/` on first turn, wasting tokens in the spawn prompt and setup time.

This design addresses both by:
- Moving the filter script to `.claude/skills/ammo/scripts/transcript_filter.py` (checked into the repo)
- Adding `--include-subagents` and `--projects-dir` flags that auto-discover and include direct child subagent transcripts in the filtered output
- Keeping the existing poll-per-Bash architecture (stateless, self-healing, no background processes)

### Key Properties

- **Stateless per-poll**: Each invocation opens, reads, closes. No persistent file handles, no orphan risk, no crash recovery logic.
- **Self-healing on truncated lines**: File re-opened each poll from `--start-line N`. Truncated lines at the write frontier naturally retry next poll.
- **Subagent coverage**: Direct children auto-discovered via Agent tool_use records in the champion's transcript.
- **No /tmp script writing**: Script lives in the repo. Spawn prompt shrinks from ~8,000 to ~1,500 tokens.
- **Minimal change surface**: Extends the existing, battle-tested filter script rather than replacing it.

### Research Findings (Claude Code Capabilities)

Before settling on this design, we investigated alternatives:

| Mechanism | Verdict | Why |
|-----------|---------|-----|
| Background watcher process | Over-engineered | Adds process management, status files, restart logic for ~76s savings over 3 hours. Introduces file position corruption bug on truncated lines that requires `fh.tell()`/`fh.seek()`. |
| `FileChanged` hook | Not usable | Fires only within the same session; cannot inject prompts into other agents |
| `CronCreate` / `/loop` | Too coarse | 1-minute minimum granularity |
| Channels (research preview) | Not production-ready | Requires claude.ai login, Team/Enterprise opt-in |
| Extended filter script (this design) | **Best option** | Adds subagent support to proven stateless approach. Zero new complexity classes. |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│  Monitor Agent (Sonnet)                               │
│                                                       │
│  Poll loop (every 5s):                                │
│    Bash: python3 .claude/skills/ammo/scripts/         │
│      transcript_filter.py {transcript_path}           │
│      --start-line {last_line}                         │
│      --include-subagents                              │
│      --projects-dir ~/.claude/projects/...            │
│    → analyze digest                                   │
│    → interject if issue found                         │
│    → update observation log                           │
│    Bash(command="sleep 5")                            │
│    → repeat                                           │
└──────────────────────────────────────────────────────┘
         │ each poll
         ▼
┌──────────────────────────────────────────────────┐
│  transcript_filter.py (single invocation)         │
│                                                   │
│  1. Read champion transcript from --start-line    │
│  2. Filter noise, format records (existing logic) │
│  3. If --include-subagents:                       │
│     a. Scan formatted records for Agent tool_use  │
│        with name=X                                │
│     b. Discover subagent transcripts by globbing  │
│        --projects-dir/*.jsonl, matching agentName │
│     c. Read subagent transcripts from their own   │
│        --start-line (tracked in state file)       │
│     d. Format subagent records with [sub:X] prefix│
│  4. Output combined digest to stdout              │
│  5. Write LAST_LINE_PROCESSED to state file       │
│     (champion + per-subagent offsets)              │
└──────────────────────────────────────────────────┘
         │ reads
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

## 3. Filter Script Changes

### 3.1 New Location

Move from: embedded in spawn prompt, written to `/tmp/transcript_filter_{name}.py` at runtime
Move to: `.claude/skills/ammo/scripts/transcript_filter.py` — checked into the repo

### 3.2 New CLI Arguments

```
python3 transcript_filter.py <path> \
    [--start-line N]              # Existing: skip lines before N
    [--max-content-len N]         # Existing: truncation limit
    [--state-file PATH]           # Existing: LAST_LINE_PROCESSED output
    [--include-subagents]         # NEW: discover and include subagent transcripts
    [--projects-dir PATH]         # NEW: where .jsonl files live (required with --include-subagents)
```

### 3.3 Subagent Discovery Logic

When `--include-subagents` is set, after processing the champion transcript:

```python
def discover_subagents(champion_path, projects_dir, raw_records, pending_names=None):
    """Find subagent transcripts by matching Agent tool_use names.

    Scans the champion's raw records for Agent tool_use blocks,
    extracts spawned agent names, merges with pending_names from
    the state file, then globs projects_dir for matching .jsonl
    transcripts.

    Args:
        champion_path: Path to champion's .jsonl (to skip in glob)
        projects_dir: Directory containing .jsonl transcript files
        raw_records: List of raw JSON record dicts from champion processing
        pending_names: Set of agent names from state file that haven't been
                       found yet (R3-C1 fix: prevents spawn window gap)

    Returns:
        (discovered, still_pending):
            discovered: dict {agent_name: transcript_path}
            still_pending: set of names not yet found (write back to state)
    """
    # Step 1: Extract spawned agent names from raw records
    spawned_names = set(pending_names or [])
    for record in raw_records:
        if record.get('type') != 'assistant':
            continue
        msg = record.get('message', {})
        if not isinstance(msg, dict):
            continue
        for block in msg.get('content', []):
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use' and block.get('name') == 'Agent':
                inp = block.get('input', {})
                agent_name = inp.get('name')
                if agent_name:
                    spawned_names.add(agent_name)

    if not spawned_names:
        return {}, set()

    # Step 2: Glob .jsonl files, match agentName in first 20 lines
    discovered = {}
    champion_real = os.path.realpath(champion_path)  # R3-I2: normalize for comparison
    files = sorted(
        glob.glob(os.path.join(projects_dir, '*.jsonl')),
        key=os.path.getmtime, reverse=True)[:50]

    for f in files:
        if os.path.realpath(f) == champion_real:  # R3-I2: realpath comparison
            continue
        if len(discovered) == len(spawned_names):
            break  # All found
        with open(f) as fh:
            for i, line in enumerate(fh):
                if i > 20:
                    break
                try:
                    d = json.loads(line.strip())
                    name = d.get('agentName')
                    if name in spawned_names and name not in discovered:
                        discovered[name] = f
                except Exception:
                    continue

    still_pending = spawned_names - set(discovered.keys())  # R3-C1: names not yet found
    return discovered, still_pending
```

### 3.4 Subagent State Tracking

The state file (already used for `LAST_LINE_PROCESSED`) is extended to track per-subagent line offsets:

**Before** (`/tmp/monitor_state_{basename}.txt`):
```
1234
```

**After** (`/tmp/monitor_state_{monitor_name}.txt`) **(R3-M2: keyed by monitor name, not transcript basename)**:
```json
{
    "pending_subagents": ["micro-exp-1"],
    "subagents": {
        "validator-1": {"path": "/home/.../.jsonl", "last_line": 567},
        "researcher-1": {"path": "/home/.../.jsonl", "last_line": 89}
    }
}
```

Note: `champion_line` is NOT stored in the state file **(R3-I1 fix)**. The monitor agent is the sole authority for the champion offset — it reads `LAST_LINE_PROCESSED` from stdout and passes it as `--start-line` on the next poll. The state file only tracks subagent state.

The state file format changes from a bare integer to JSON. For backwards compatibility, `main()` checks if the state file content is a bare integer (old format) and ignores it (the old format only stored the champion line, which is now tracked via `--start-line`).

**`pending_subagents`** (R3-C1 fix): Tracks agent names whose spawn was detected but whose transcript hasn't been found yet. On each poll, the filter merges `pending_subagents` from the state file with any new Agent spawn names from the current `raw_records`, then attempts discovery for all pending names. On successful discovery, the name moves from `pending_subagents` to `subagents`. This prevents the spawn window gap where a subagent is permanently lost if its transcript doesn't exist on the poll that first sees the spawn record.

### 3.5 Combined Output Format

Subagent records are interleaved with champion records, prefixed with `[sub:{name}]`:

```
[14:32:01] ASSISTANT (line 450):
  THINKING (2341 chars):
    Let me check the bandwidth utilization...
  BASH: ncu --set full --target-processes all ...
[14:32:03] USER (line 455):
  BASH_RESULT:
    Kernel: sm80_xmma_gemm... Memory BW: 423 GB/s ...
[sub:validator-1] [14:32:20] ASSISTANT (line 12):
  BASH: python3 test_correctness.py --kernel fused_moe ...
[sub:validator-1] [14:32:25] USER (line 15):
  BASH_RESULT:
    All tests passed (12/12)

--- FILTER SUMMARY ---
AGENT: champion-1
LINES_READ: 17
RECORDS_OUTPUT: 4
NOISE_SKIPPED: 12
PARSE_ERRORS: 0
LAST_LINE_PROCESSED: 467
SUBAGENTS_DISCOVERED: 2 (validator-1, researcher-1)
SUBAGENTS_PENDING: 1 (micro-exp-1)
SUBAGENT_LINES: validator-1=15, researcher-1=0
```

Champion records come first (in order), then subagent records (grouped by subagent, in order). This avoids complex interleaving and makes the output predictable.

### 3.6 Processing Flow

```python
def main():
    args = parse_args()

    # Read state file (handles both old integer and new JSON format)
    state = read_state(args.state_file)

    # R3-I1: --start-line always wins for champion offset. The monitor agent
    # is the source of truth (it reads LAST_LINE_PROCESSED from stdout).
    # State file only stores subagent offsets and pending_subagents.
    start_line = args.start_line

    # Process champion transcript (existing logic, unchanged)
    champion_records, raw_records, last_line = process_transcript(
        args.path, start_line, args.max_content_len)

    # Output champion records
    for line in champion_records:
        print(line)

    # Subagent processing (new)
    subagent_state = state.get('subagents', {})
    pending_subagents = set(state.get('pending_subagents', []))
    still_pending = set()

    if args.include_subagents and args.projects_dir:
        # R3-C1: Merge pending names from state + new names from raw_records
        discovered, still_pending = discover_subagents(
            args.path, args.projects_dir, raw_records,
            pending_names=pending_subagents)

        # Also include subagents from previous state (already discovered)
        for name, info in subagent_state.items():
            if name not in discovered and os.path.exists(info['path']):
                discovered[name] = info['path']

        new_subagent_state = {}
        for name, path in discovered.items():
            sub_start = subagent_state.get(name, {}).get('last_line', 0)
            sub_records, _, sub_last_line = process_transcript(
                path, sub_start, args.max_content_len, label=f"sub:{name}")

            for line in sub_records:
                print(line)

            new_subagent_state[name] = {'path': path, 'last_line': sub_last_line}

        subagent_state = new_subagent_state

    # Print summary
    print_summary(last_line, subagent_state, still_pending)

    # Write state file (JSON format)
    # R3-C1: pending_subagents persisted so next poll retries discovery
    # R3-I1: champion_line NOT stored — monitor tracks via --start-line
    write_state(args.state_file, {
        'pending_subagents': sorted(still_pending),
        'subagents': subagent_state
    })
```

### 3.7 `process_transcript()` Refactor

The existing `main()` function's transcript processing logic is extracted into a reusable `process_transcript()` function:

```python
def process_transcript(path, start_line, max_content_len, label=""):
    """Process a single .jsonl transcript file.

    Args:
        path: Path to .jsonl file
        start_line: Skip lines before this (0-indexed)
        max_content_len: Truncation limit per content block
        label: Prefix for output lines (e.g., "sub:validator-1")

    Returns:
        (formatted_lines, raw_records, last_line_processed)
    """
    # ... existing processing logic from main(), extracted unchanged
    # Only addition: prepend [label] to each output line if label is non-empty
```

This is a straightforward extract-method refactor. The existing line-by-line processing, noise filtering, record formatting, and edge case handling (truncated lines, malformed records, >5MB lines) remain unchanged.

### 3.8 Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Truncated final line (concurrent write) | Existing C1 logic: `json.JSONDecodeError` → `break`. File re-opened from `start_line` next poll — naturally retries. No file handle corruption risk (stateless). |
| Subagent transcript not yet created | `discover_subagents()` won't find it. Agent name persisted to `pending_subagents` in state file. Next poll merges pending names + new names, retries discovery. Name moves to `subagents` on success. **(R3-C1 fix)** |
| Subagent transcript from wrong champion | Matched by `agentName` in first 20 lines + mtime ordering (newest first). Ambiguity risk low — AMMO agent names include champion identity (e.g., `validator-champion-1`). |
| --include-subagents without --projects-dir | Subagent discovery skipped (no error). |
| Old state file format (bare integer) | Detected and converted to `{"champion_line": N, "subagents": {}}`. |
| Subagent previously discovered but transcript deleted | `os.path.exists()` check before processing. Silently skipped. |
| Many subagents (>3) | Depth-1 only: champions spawn 1-3 subagents typically. Even 5+ is fine — each is a sequential file read, not a persistent resource. |
| Discovery I/O with concurrent watchers | During overlapped debate rounds (4-8 monitors), each filter invocation globs up to 50 files × 20 lines. At 5s intervals, this is bursty but brief (~50ms per invocation). No persistent contention. |

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
1. Run discovery (still needed to find transcript path — or use orchestrator-provided path)
2. Initialize observation log (same as before)
3. Read state file if exists (same as before)

Remove entirely:
- "Write the Filter Script" section — script is pre-existing at `.claude/skills/ammo/scripts/transcript_filter.py`
- The discovery logic can remain a small inline snippet or be a `--discover` mode on the filter script itself

### 4.2 Polling Protocol

**Before**:
```
Bash: python3 /tmp/transcript_filter_{name}.py {path} --start-line {N}
sleep 5 → repeat
```

**After**:
```
Bash: python3 .claude/skills/ammo/scripts/transcript_filter.py {path} \
    --start-line {N} --include-subagents --projects-dir {projects_dir}
sleep 5 → repeat
```

Only change: script path (repo instead of /tmp) and two new flags. Everything else (sleep interval, state tracking, observation log, rate limiting, interjection protocol) remains unchanged.

### 4.3 No Other Changes

The following sections of `ammo-transcript-monitor.md` are **unchanged**:
- Polling interval (5 seconds)
- Rate limiting (1 message/minute, 5 total/session)
- Escalation protocol
- When to start/stop flagging
- Session restart handling
- What to watch for
- Independence considerations

---

## 5. Orchestrator Integration Changes

### 5.1 Spawn Pattern Update

The spawn prompt no longer includes the filter script. The `projects_dir` is derived dynamically:

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

## Filter Script
The transcript filter is pre-installed at:
  .claude/skills/ammo/scripts/transcript_filter.py

Run it each poll with:
  python3 .claude/skills/ammo/scripts/transcript_filter.py {{transcript_path}} \
    --start-line {{last_line}} \
    --include-subagents \
    --projects-dir {projects_dir}

## Campaign Context
- Model: {model_id}
- Hardware: {hardware}
- Bottleneck analysis: {artifact_dir}/bottleneck_analysis.md
- Target config: {artifact_dir}/target.json
"""
)
```

### 5.2 File Locations

| Artifact | Path |
|----------|------|
| Filter script (checked in) | `.claude/skills/ammo/scripts/transcript_filter.py` |
| Agent definition | `.claude/agents/ammo-transcript-monitor.md` |
| State file (runtime) | `/tmp/monitor_state_{monitor_name}.txt` **(R3-M2: keyed by monitor name)** |
| Observation log (runtime) | `{artifact_dir}/monitor_log_{champion_name}.md` |

### 5.3 Migration Path

1. **Create** `.claude/skills/ammo/scripts/transcript_filter.py` — extract from design spec Section 3.2 of `2026-03-25-transcript-monitor-design.md`, add subagent extensions from this spec
2. **Update** `.claude/agents/ammo-transcript-monitor.md` — remove "Write the Filter Script" setup step, update Bash command in poll execution to use repo path + new flags
3. **Update** orchestrator spawn logic in `debate-protocol.md` and `parallel-tracks.md` — remove `<filter_script>` embedding, use new spawn pattern from Section 5.1
4. **Update** `.claude/skills/ammo/docs/specs/2026-03-25-transcript-monitor-design.md` — reference the repo-hosted script, note subagent support

---

## 6. Subagent Depth Limit

The filter monitors the champion + direct children only (depth 1). It does NOT recursively discover subagents spawned by subagents:
- Champions typically spawn 1-3 subagents (validator, researcher, micro-experiment runner)
- Sub-subagents are rare and their work is summarized in the subagent's result record in the parent transcript
- Recursive discovery would be unbounded complexity for near-zero value

---

## 7. Review History

This design went through 2 rounds of adversarial review on a background watcher approach. The reviews identified a critical file-handle corruption bug (truncated lines permanently desync persistent file handles) and significant over-engineering. The simpler filter-extension approach was proposed by a reviewer and adopted as the design.

Key findings that shaped this design:
- Persistent file handles require `fh.tell()`/`fh.seek()` for truncated line recovery — the stateless per-poll approach avoids this entirely
- Background processes need status files, PID files, restart logic, signal handling, append-mode file management — all unnecessary with per-poll execution
- The background watcher's main advantage (eliminating Python startup overhead) saves ~76 seconds over 3 hours — negligible
- The real value is subagent coverage and no /tmp script writing — both achievable with the simpler approach
