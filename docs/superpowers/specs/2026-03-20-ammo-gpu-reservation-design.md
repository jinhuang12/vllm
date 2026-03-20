# AMMO GPU Reservation System

## Problem

During AMMO Stages 4-5, parallel implementation tracks and overlapped debate agents share the same physical GPUs. The current system has three gaps:

1. **Advisory-only GPU assignment**: Spawn prompts tell agents which GPU to use (`CUDA_VISIBLE_DEVICES={gpu_id}`), but nothing enforces it.
2. **Sweep-only locking**: The E2E sweep script (`run_vllm_bench_latency_sweep.py`) holds a per-process flock keyed by `CUDA_VISIBLE_DEVICES`, but kernel benchmarks, ncu profiling, and ad-hoc experiments bypass it entirely.
3. **No TP-aware partitioning**: The orchestrator assigns 1 GPU per track regardless of TP. With TP=4 on 4 GPUs, E2E sweeps consume ALL GPUs — but tracks don't know this and run overlapping experiments.

**Observed failure**: With TP=4, one track ran the sweep script (all GPUs) while another track ran ad-hoc kernel experiments on the same GPUs. Neither was aware of the contention.

## Design Overview

Two layers, each solving a different problem:

1. **Orchestrator: TP-aware GPU partitioning** — decides whether tracks get exclusive GPU partitions or share GPUs, and passes the right CVD values in spawn prompts.
2. **Hooks: automatic per-call reservation** — PreToolUse auto-reserves GPUs when a GPU command runs, PostToolUse auto-releases when it completes. Agents never call reserve/release manually. Zero forgotten releases.

```
Orchestrator (at track spawn):
  Computes: can_partition = (gpu_count >= tp * num_tracks)?
  YES → exclusive partitions (no contention possible)
  NO  → shared mode (E2E sweeps serialize via hooks)
  Passes CVD values in spawn prompts

Agent runs GPU command:
  CUDA_VISIBLE_DEVICES=2 python benchmark.py

PreToolUse hook (auto-reserve):
  Detects GPU command → extracts CVD=2
  Checks state.json → GPU 2 free? → writes reservation → allows command
  GPU 2 held? → HARD BLOCK (agent retries)

Command executes...

PostToolUse hook (auto-release):
  Detects matching GPU command → releases reservation from state.json
```

## Layer 1: TP-Aware GPU Partitioning (Orchestrator)

At Stage 4-5 track spawn, the orchestrator computes GPU assignments:

```
gpu_count = state.json -> gpu_resources.gpu_count
tp = state.json -> target.tp
num_tracks = len(debate.selected_winners)

can_partition = (gpu_count >= tp * num_tracks)
```

**Immutability**: The GPU assignment plan is computed once per round and recorded in `state.json.gpu_assignment`. If the plan already exists for the current round, the orchestrator uses the existing plan (does not recompute). If `target.tp` changes mid-campaign, the current round must complete or be abandoned before the new TP takes effect.

### Exclusive Mode (`can_partition = true`)

Each track gets TP GPUs exclusively. Fully parallel kernel AND E2E work. Zero contention.

| Example | GPUs | TP | Tracks | Assignment |
|---------|------|----|--------|------------|
| TP=1, 4 GPUs, 3 tracks | 4 | 1 | 3 | A:0, B:1, C:2 |
| TP=2, 4 GPUs, 2 tracks | 4 | 2 | 2 | A:0,1  B:2,3 |

Spawn prompt for each track:
```
GPU assignment (EXCLUSIVE — your GPUs are dedicated to this track):
  Kernel work:  CUDA_VISIBLE_DEVICES=0,1
  E2E sweep:    CUDA_VISIBLE_DEVICES=0,1
  (Same partition — no contention with other tracks)
```

### Shared Mode (`can_partition = false`)

Tracks share GPUs for E2E sweeps. Kernel benchmarks get 1 GPU each (parallel). E2E sweeps need all TP GPUs (serialized via hooks).

| Example | GPUs | TP | Tracks | Kernel Assignment | E2E Assignment |
|---------|------|----|--------|-------------------|----------------|
| TP=4, 4 GPUs, 2 tracks | 4 | 4 | 2 | A:0, B:1 | Both:0,1,2,3 |
| TP=2, 4 GPUs, 3 tracks | 4 | 2 | 3 | A:0, B:1, C:2 | Shared:0,1,2,3 |

Spawn prompt for each track:
```
GPU assignment (SHARED — E2E sweeps share GPUs with other tracks):
  Kernel work:  CUDA_VISIBLE_DEVICES=0
  E2E sweep:    CUDA_VISIBLE_DEVICES=0,1,2,3
  WARNING: E2E sweeps may block if another track's sweep is running.
  If a GPU command blocks, retry after a short wait (the hook will
  tell you which command is holding the GPU).
```

### Orchestrator Records GPU Plan in state.json

```json
{
  "gpu_assignment": {
    "mode": "exclusive|shared",
    "gpu_count": 4,
    "tp": 2,
    "num_tracks": 2,
    "tracks": {
      "op003": {"kernel_cvd": "0,1", "e2e_cvd": "0,1"},
      "op007": {"kernel_cvd": "2,3", "e2e_cvd": "2,3"}
    }
  }
}
```

In shared mode:
```json
{
  "gpu_assignment": {
    "mode": "shared",
    "gpu_count": 4,
    "tp": 4,
    "num_tracks": 2,
    "tracks": {
      "op003": {"kernel_cvd": "0", "e2e_cvd": "0,1,2,3"},
      "op007": {"kernel_cvd": "1", "e2e_cvd": "0,1,2,3"}
    }
  }
}
```

## Layer 2: Hook-Managed Per-Call Reservations

Every GPU operation in AMMO is a single Bash tool call — there are no persistent GPU servers or background processes holding GPUs across calls. This means reservations can be scoped to individual tool calls, managed entirely by hooks.

### State Model

**Location**: `/tmp/ammo_gpu_res/state.json`

Machine-scoped (not per-campaign). Multiple concurrent AMMO sessions are disambiguated by `session_id`.

**Filesystem requirement**: MUST be on a local filesystem (`flock` is unreliable on NFS). If `/tmp/` is NFS-mounted, use `/run/ammo_gpu_res/` instead. Init verifies via `stat -f`.

```json
{
  "gpus": {
    "0": {
      "command_hash": "a8f3b2c1",
      "session_id": "a1b2c3d4",
      "reserved_at": "2026-03-20T14:32:00Z",
      "lease_expires": "2026-03-20T16:32:00Z",
      "cvd_requested": "0,1,2,3",
      "command_snippet": "python run_vllm_bench_latency_sweep.py ..."
    },
    "1": null,
    "2": null,
    "3": null
  },
  "gpu_count": 4,
  "audit": []
}
```

**Key difference from the previous design**: No PID or holder name. Reservations are keyed by the command being executed, identified by a hash of the command string. PreToolUse writes it; PostToolUse removes it. Crash recovery uses lease expiry, not PID liveness.

**Atomicity**: `flock(LOCK_EX | LOCK_NB)` on `/tmp/ammo_gpu_res/.lock` with exponential backoff (5 attempts: 100ms, 200ms, 400ms, 800ms — ~1.5s total budget).

**Lease expiry**: Each reservation has a `lease_expires` timestamp. Default: 2 hours. If `now() > lease_expires`, the reservation is treated as stale and auto-reclaimed (crash recovery). The 2-hour window is generous — the longest AMMO command (multi-BS sweep) takes ~30 min. Lease expiry only matters when PostToolUse fails to fire (session crash, compaction).

**GPU discovery**: `gpu_count` set from `nvidia-smi -L | wc -l` on first init.

### PreToolUse Hook: Auto-Reserve

Extend `ammo-pretool-guard.sh` (or add a new hook script in the same PreToolUse matcher):

#### Detection Patterns (Conservative)

```bash
# Pattern 1: Python/pytest with GPU keywords
# Matched by: command contains python/pytest AND
#   (torch|cuda|triton|vllm OR script name contains benchmark|kernel|gpu)

# Pattern 2: Inline python with GPU imports
# Matched by: python -c AND (torch|cuda|triton)

# Pattern 3: GPU profiling tools
# Matched by: nsys|ncu at command start, or nvidia-smi with --query-compute

# NOT matched:
# - cmake --build (no GPU usage)
# - nvidia-smi (bare)
# - python -c without GPU keywords
# - Read-only commands (git, grep, cat, etc.)

# Known limitation: bash -c "..." wrappers bypass detection.
# Skill docs mandate CVD prefix regardless.

# Exemptions:
# - python -c "import vllm" / "import torch" (bare imports, worktree setup)
```

#### Reserve Flow

The hook uses a **one-shot block** model (like `ammo-stop-guard.sh`): it blocks once per session to make the agent aware of GPU reservation, then trusts the agent's judgment on subsequent commands.

Three CVD states drive different behavior:

| CVD in command | Behavior | Reservation? |
|----------------|----------|-------------|
| `CUDA_VISIBLE_DEVICES=0,1` (digit IDs) | Auto-reserve GPUs, block on contention | Yes |
| `CUDA_VISIBLE_DEVICES=""` (empty) | Agent explicitly signals "no GPU needed". Allow immediately. | No |
| No CVD at all | **One-shot block** (first time), then allow | No |

When a GPU pattern is detected:

1. Check if `/tmp/ammo_gpu_res/state.json` exists. If not, skip.
2. Extract `CUDA_VISIBLE_DEVICES=...` from command text.

**Case A — CVD with GPU IDs** (e.g., `CUDA_VISIBLE_DEVICES=0,1,2,3`):
3. Parse GPU IDs.
4. Acquire flock on `.lock`, read state.json.
5. For each requested GPU:
   - If free → OK
   - If held but lease expired → auto-reclaim (log to audit), treat as free
   - If held and lease active → **BLOCK**: `"GPU {id} held by command: {snippet} (since {time}). Retry shortly."`
6. If all requested GPUs free: write reservations (command_hash, session_id, lease_expires=now+2h [or 4h for nsys], cvd_requested, command_snippet=first 80 chars), release flock.
   - If flock acquisition fails after 5 retries: **BLOCK** with distinct message: `"GPU reservation lock timeout — system is busy. Retry in a few seconds."` (distinct from "GPU held by command" to clarify the failure is transient lock contention, not a held GPU).
7. Exit 0 (allow command).

**Case B — CVD="" (empty string)**:
3. Agent explicitly says this command doesn't need GPUs. `CUDA_VISIBLE_DEVICES=""` also disables CUDA at the driver level, preventing accidental GPU usage.
4. **Sanity check** (advisory): if the command contains known GPU-heavy indicators (`run_vllm_bench`, `benchmark_kernel`, `nsys profile`, `ncu`) AND CVD is empty, emit a stderr warning: `"CVD is empty but command looks GPU-intensive. Verify this is intentional."` This catches misuse of the sentinel on GPU-heavy commands. Does not block.
5. Skip reservation. Exit 0 (allow command).

**Case C — No CVD at all**:
3. Check one-shot flag file `/tmp/ammo_gpu_res/.warned_{session_hash}`.
4. If flag does NOT exist (first time):
   - Write flag file.
   - **BLOCK** (one-shot): `"AMMO GPU: Command matches GPU pattern but has no CUDA_VISIBLE_DEVICES prefix. Add CUDA_VISIBLE_DEVICES=X (from your spawn prompt) for GPU work, or CUDA_VISIBLE_DEVICES=\"\" if this command does not use GPUs."`
5. If flag exists (already warned):
   - Exit 0 (allow command). Agent was already made aware and chose to proceed without CVD.

**Command hash**: `sha256(command_string)[:16]`. 16 hex chars = 64 bits of hash space. Birthday collision probability at 1000 commands is ~2.7e-11, effectively zero across any AMMO campaign lifetime. Used by PostToolUse to match the reservation for cleanup.

**Session hash**: Derived from `$CLAUDE_SESSION_ID` only (not `$PPID` — PIDs are small integers that reuse frequently across sessions). If `$CLAUDE_SESSION_ID` is not set, the hook skips the one-shot mechanism and exits 0 (fail-open). Flag files (`/tmp/ammo_gpu_res/.warned_{session_hash}`) are cleaned up on `SessionStart` via the existing `ammo-postcompact.sh` hook to prevent stale flags from suppressing warnings in new sessions.

**Lease duration**: Default 2 hours. If the command contains `nsys` (nsys profiling on large models can take 45-75 min), set `lease_hours=4.0` instead.

### PostToolUse Hook: Auto-Release

New hook in `.claude/settings.local.json`:

```json
{
  "PostToolUse": [
    {
      "matcher": "Bash",
      "hooks": [
        {
          "type": "command",
          "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/ammo-gpu-release.sh"
        }
      ]
    }
  ]
}
```

#### Release Flow

**Implementation prerequisite**: Verify empirically that `PostToolUse` for Bash provides `tool_input.command` (the original command string) in the hook input JSON. The existing PreToolUse hook (`ammo-pretool-guard.sh` line 15) handles field name ambiguity between `tool_input.command` and `input.command` — the PostToolUse hook should use the same extraction pattern.

1. Check if `/tmp/ammo_gpu_res/state.json` exists. If not, skip.
2. Extract command from the PostToolUse input JSON (try `tool_input.command`, then `input.command`).
3. **Fallback if command extraction fails**: If command is empty/null, scan state.json for reservations with the current `$CLAUDE_SESSION_ID` and release the most recently reserved entry (heuristic). Log a warning about the fallback.
4. Check if command contains `CUDA_VISIBLE_DEVICES=` with digit IDs (same regex as PreToolUse Case A). Commands with `CVD=""` or no CVD had no reservation — skip.
5. Compute command_hash, acquire flock, read state.json.
6. For each GPU entry where `command_hash` matches: clear to null.
7. Write state.json, release flock.
8. Exit 0.

**Missed PostToolUse (crash recovery)**: If the session crashes mid-command, PostToolUse never fires. The reservation stays with its lease timestamp. The next PreToolUse for that GPU (from any agent) checks the lease — if expired, auto-reclaims. With a 2-hour lease, this means GPUs are unavailable for up to 2 hours after a crash. For faster recovery, the orchestrator can manually clear stale reservations via `gpu_status.py` + `gpu_force_clear.py` (see below).

### Sweep Script Retrofit

The sweep script no longer manages its own locking. The PreToolUse hook handles reservation when the sweep command is invoked. The PostToolUse hook releases when it completes.

**Removal**: Delete `_acquire_gpu_lock()`, `_check_gpu_idle()`, and the `/tmp/ammo_gpu_locks/` directory logic.

**Retained**: The existing `nvidia-smi` idle check is moved into the PreToolUse hook as an additional advisory layer (detects non-AMMO GPU processes).

**Transition safety**: On first run, the new system checks for existing `/tmp/ammo_gpu_locks/*.lock` files and warns if any are held.

### Diagnostic Scripts

Two scripts in `.claude/skills/ammo/scripts/` for orchestrator and human use:

#### `gpu_status.py`

```
python .claude/skills/ammo/scripts/gpu_status.py [--json]
```

Human-readable table or JSON of current reservation state.

#### `gpu_force_clear.py`

```
python .claude/skills/ammo/scripts/gpu_force_clear.py [--gpu-ids 0,1] [--all] --session-id <id>
python .claude/skills/ammo/scripts/gpu_force_clear.py --all --force-no-session  # emergency: clear ALL regardless of session
```

Orchestrator-only tool. Clears reservations for specific GPUs or all GPUs within a session. Used for crash recovery when the 2-hour lease is too long to wait. Writes to audit log. The `--force-no-session` flag is an emergency override for when the session ID was never recorded (crash before `session_id` MUST was fulfilled). Logs a prominent warning.

### Shared Module

`.claude/skills/ammo/scripts/gpu_reservation.py` — Internal library used by hooks and diagnostic scripts:

```python
# Public API
def read_state() -> dict: ...
def write_reservation(gpu_ids: list[int], command_hash: str, session_id: str,
                      cvd_requested: str, command_snippet: str,
                      lease_hours: float = 2.0) -> None: ...
def release_by_hash(command_hash: str) -> None: ...
def force_clear(gpu_ids: list[int] | None, session_id: str) -> None: ...
def check_and_reclaim_expired() -> list[int]: ...  # returns reclaimed GPU IDs

# Internal
def _acquire_flock() -> IO: ...
def _release_flock(fh: IO) -> None: ...
def _init_state() -> dict: ...
```

## Skill Documentation Updates

### SKILL.md

Add to Non-Negotiables (after item 4):

> **GPU isolation**: GPU commands MUST include a `CUDA_VISIBLE_DEVICES=X` prefix (from spawn prompt assignment) for GPU work, or `CUDA_VISIBLE_DEVICES=""` to explicitly signal no GPU is needed. The PreToolUse hook auto-reserves GPUs when CVD contains GPU IDs and blocks on contention. PostToolUse auto-releases. No manual reservation or release needed. *(Enforced by `ammo-pretool-guard.sh` PreToolUse + `ammo-gpu-release.sh` PostToolUse — one-shot block on first missing CVD, then trusts agent judgment)*

Add to Helper Scripts table:

| Script | Purpose |
|--------|---------|
| `scripts/gpu_status.py` | Print current GPU reservation state (orchestrator/human diagnostic) |
| `scripts/gpu_force_clear.py` | Force-clear stale GPU reservations after crashes (orchestrator-only) |

Update Hook Enforcement table:

| Hook Event | Script | Purpose |
|------------|--------|---------|
| **PreToolUse** (Bash) | `ammo-pretool-guard.sh` | Production parity reminders (existing) + GPU auto-reserve (new): blocks GPU commands without CVD or with GPU contention |
| **PostToolUse** (Bash) | `ammo-gpu-release.sh` | GPU auto-release: clears reservation after GPU command completes |

**session_id recording**: Upgrade "The lead SHOULD record the session ID" to "The lead MUST record the session ID in state.json at campaign start." If `session_id` is null when the orchestrator spawns tracks, the orchestrator MUST generate a UUID and record it before proceeding. The session_id is passed to agents in spawn prompts and used by hooks to tag reservations.

**Soft-reservation deletion**: Remove SKILL.md line 141 ("On multi-GPU systems (N >= 3), the last GPU is soft-reserved for debate micro-experiments..."). Debate agents are CPU-only.

### orchestration/parallel-tracks.md

Replace the GPU Assignment section with the TP-aware partitioning model described in Layer 1 above.

**Deletions**:
- Remove lines 36-41 (old GPU Assignment table with flock references)
- Remove lines 76, 113 (`GPU assignment: CUDA_VISIBLE_DEVICES={gpu_id}` in spawn templates)
- Remove lines 299-307 (GPU Allocation During Overlap — debate soft-reservation)

**Additions**:
- TP-Aware GPU Partitioning section (exclusive vs shared mode)
- Updated spawn prompt templates with kernel_cvd + e2e_cvd
- GPU Assignment Plan in state.json schema

### Agent .md files

Update `ammo-impl-champion` and `ammo-impl-validator`:

```
## GPU Usage

Your spawn prompt includes GPU assignments:
  Kernel work:  CUDA_VISIBLE_DEVICES=X
  E2E sweep:    CUDA_VISIBLE_DEVICES=Y

Prefix GPU commands with the appropriate CUDA_VISIBLE_DEVICES value:
  CUDA_VISIBLE_DEVICES=0 python benchmark_kernel.py
  CUDA_VISIBLE_DEVICES=0,1,2,3 python run_vllm_bench_latency_sweep.py ...

If a command does NOT need GPUs but matches GPU patterns (e.g., a torch
script that only does CPU tensor ops), prefix with CUDA_VISIBLE_DEVICES=""
to signal no GPU is needed and skip reservation.

The hooks auto-manage GPU reservations — no manual reserve/release needed.
If a command blocks ("GPU held by..."), wait briefly and retry.
If persistent blocking, report to orchestrator via SendMessage.
```

### Resume Protocol Update

Add step 3b:

> **3b.** Check GPU reservation state: run `gpu_status.py`. If stale reservations exist from the crashed session, clear them: `gpu_force_clear.py --all --session-id <crashed_session_id>`. Re-spawned agents will have their GPUs auto-reserved by hooks when they run commands.

## Implementation Scope

This is a **design spec** — the "Modified/New files" lists below are the implementation TODO.

### File Inventory

New files:
- `.claude/skills/ammo/scripts/gpu_reservation.py` — Shared module (state I/O, flock, lease logic)
- `.claude/skills/ammo/scripts/gpu_status.py` — Diagnostic: print reservation state
- `.claude/skills/ammo/scripts/gpu_force_clear.py` — Orchestrator crash recovery tool
- `.claude/hooks/ammo-gpu-release.sh` — PostToolUse hook for auto-release

Modified files:
- `.claude/hooks/ammo-pretool-guard.sh` — Add GPU auto-reserve (hard block on contention/missing CVD)
- `.claude/settings.local.json` — Add PostToolUse hook registration
- `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py` — Remove `_acquire_gpu_lock`, `_check_gpu_idle`, `/tmp/ammo_gpu_locks/` logic
- `.claude/skills/ammo/SKILL.md` — Add non-negotiable, hooks table, session_id MUST, remove debate soft-reservation
- `.claude/skills/ammo/orchestration/parallel-tracks.md` — Replace GPU assignment with TP-aware partitioning
- `.claude/skills/ammo/agents/ammo-impl-champion.md` — Simplified GPU usage instructions
- `.claude/skills/ammo/agents/ammo-impl-validator.md` — Simplified GPU usage instructions

### Edge Cases

**Exclusive mode — no contention**: When `gpu_count >= tp * num_tracks`, each track has dedicated GPUs. PreToolUse still auto-reserves (for non-AMMO process detection) but never blocks another track.

**Shared mode — E2E sweep serialization**: Two tracks running E2E sweeps on the same GPUs → second one blocks. Agent retries. First sweep finishes → PostToolUse releases → second retry succeeds.

**Crash without PostToolUse**: Lease expiry (2 hours) auto-reclaims. For faster recovery, orchestrator runs `gpu_force_clear.py`. Resume protocol step 3b handles this.

**Hook false positives**: Conservative patterns. Commands without GPU keywords pass through instantly. GPU commands without CVD are blocked (intentional — agents must use their assigned GPUs).

**Hook false negatives**: `bash -c` wrappers bypass detection. Skill docs mandate CVD prefix regardless.

**Multiple campaigns on same machine**: `session_id` disambiguates. `gpu_force_clear.py` requires `--session-id` to prevent cross-campaign clearing.

**Worktree smoke tests**: `python -c "import vllm"` exempted from GPU reservation (bare import, no compute).

**No AMMO campaign active**: Hook bails early if no `kernel_opt_artifacts/*/state.json` exists AND no `/tmp/ammo_gpu_res/state.json` exists.

**Transition from old locking**: New system warns if `/tmp/ammo_gpu_locks/*.lock` files exist.

**Debate agents**: CPU-only. MUST NOT use GPUs. No CVD in spawn prompt.

### What This Does NOT Solve

- **Non-AMMO GPU processes**: nvidia-smi advisory check (moved to PreToolUse) partially addresses this.
- **Network-level GPU scheduling**: Out of scope.
- **Automatic preemption**: E2E sweeps cannot preempt kernel benchmarks. In shared mode, the sweep blocks until the kernel benchmark command finishes naturally (seconds to minutes). This is acceptable — kernel benchmarks are short-lived.
