# AMMO GPU Reservation System

## Problem

During AMMO Stages 4-5, parallel implementation tracks and overlapped debate agents share the same physical GPUs. The current system has three gaps:

1. **Advisory-only GPU assignment**: Spawn prompts tell agents which GPU to use (`CUDA_VISIBLE_DEVICES={gpu_id}`), but nothing enforces it.
2. **Sweep-only locking**: The E2E sweep script (`run_vllm_bench_latency_sweep.py`) holds a per-process flock keyed by `CUDA_VISIBLE_DEVICES`, but kernel benchmarks, ncu profiling, and ad-hoc experiments bypass it entirely.
3. **No cross-agent visibility**: An agent running a kernel benchmark on GPU 0 has no way to know that another agent is about to launch an E2E sweep that needs all GPUs.

**Observed failure**: With TP=4, one track ran the sweep script (all GPUs) while another track ran ad-hoc kernel experiments. Neither was aware of the other, producing unreliable measurements for both.

## Design

### Architecture

```
+-------------------+     +-------------------+     +-------------------+
| impl-champion-003 |     | impl-validator-007|     | sweep script      |
+--------+----------+     +--------+----------+     +--------+----------+
         |                         |                          |
    gpu_reserve.py            gpu_reserve.py            gpu_reserve.py
    --gpu-count 1             --gpu-count 1             --gpu-count 4
         |                         |                          |
   prints CVD=2              prints CVD=3              prints CVD=0,1,2,3
   agent prefixes            agent prefixes            sweep uses internally
   every GPU command         every GPU command
         |                         |                          |
         v                         v                          v
+------------------------------------------------------------------------+
|                /tmp/ammo_gpu_res/state.json                            |
|  Atomic read-modify-write via brief flock (LOCK_NB + retry) on .lock  |
|  Per-GPU entries keyed by physical GPU ID                              |
+------------------------------------------------------------------------+
         ^                         ^                          ^
         |                         |                          |
    PreToolUse hook           PreToolUse hook           (integrated)
    checks CVD prefix         checks CVD prefix
    in command text           in command text
```

### State Model

**Location**: `/tmp/ammo_gpu_res/state.json`

Machine-scoped (not per-campaign) because GPU contention is a machine-level concern. Multiple concurrent AMMO sessions on the same machine are disambiguated by `session_id`.

**Filesystem requirement**: `/tmp/ammo_gpu_res/` MUST be on a local filesystem. `flock` is unreliable on NFS. If `/tmp/` is NFS-mounted (e.g., in containerized environments), use `/run/ammo_gpu_res/` instead (always ramfs/tmpfs on Linux). The init routine should verify local filesystem via `stat -f /tmp/ | grep -v nfs`.

```json
{
  "gpus": {
    "0": {
      "holder": "impl-champion-op003",
      "session_id": "a1b2c3d4",
      "reserved_at": "2026-03-20T14:32:00Z",
      "pid": 12345,
      "pid_starttime": 98765432,
      "purpose": "kernel_benchmark",
      "artifact_dir": "/home/jinhun/vllm/kernel_opt_artifacts/Nemotron..."
    },
    "1": null,
    "2": {
      "holder": "impl-validator-op007",
      "session_id": "a1b2c3d4",
      "reserved_at": "2026-03-20T14:35:00Z",
      "pid": 12346,
      "pid_starttime": 98765500,
      "purpose": "e2e_sweep",
      "artifact_dir": "/home/jinhun/vllm/kernel_opt_artifacts/Nemotron..."
    },
    "3": null
  },
  "gpu_count": 4,
  "audit": [
    {"timestamp": "2026-03-20T15:00:00Z", "action": "force_release", "holder": "impl-champion-op003", "released_by": "orchestrator", "session_id": "a1b2c3d4", "reason": "coordination_timeout"}
  ]
}
```

**Atomicity**: `flock(LOCK_EX | LOCK_NB)` on `/tmp/ammo_gpu_res/.lock` with exponential backoff (5 attempts: 100ms, 200ms, 400ms, 800ms — ~1.5s total budget). If the lock cannot be acquired after retries, exit with error (indicates hung or deadlocked process). The flock is held only for the JSON update (~milliseconds), not for the reservation duration. The larger budget handles filesystem stalls under multi-agent contention (4-6 agents reserving simultaneously at track start).

**Stale detection**: On every reserve attempt, for each GPU that shows as held:
1. Check `kill -0 <pid>`. If the PID is dead → stale, auto-reclaim with warning.
2. If PID is alive, read `/proc/<pid>/stat` field 22 (starttime in clock ticks since boot) and compare against the recorded `pid_starttime`. A mismatch means the OS recycled the PID → stale, auto-reclaim with warning.
3. If both match → reservation is live, skip.

This two-step check eliminates PID-reuse false positives.

**GPU discovery**: `gpu_count` is set on first init via `nvidia-smi -L | wc -l`.

### Scripts API

Three new scripts in `.claude/skills/ammo/scripts/`, plus a shared module:

#### `gpu_reserve.py`

```
python .claude/skills/ammo/scripts/gpu_reserve.py \
  --gpu-count 2 \
  --holder impl-champion-op003 \
  --session-id <session_uuid> \
  [--purpose kernel_benchmark|e2e_sweep|micro_experiment|ncu_profiling]
```

`--session-id` is **required** (not optional). The script validates it is non-empty.

**One reservation per holder**: If the holder already has a reservation, the script exits with an error. To change GPU count, release first then re-reserve. This prevents partial-release ambiguity.

Behavior:
1. Acquire flock (LOCK_NB + retry) on `/tmp/ammo_gpu_res/.lock`
2. Read `state.json` (create if missing, discover GPU count)
3. Check if holder already has a reservation → error if so
4. Run stale detection on all held GPUs
5. Find N free physical GPUs (lowest-numbered first)
6. If N free GPUs found: write holder info (including `pid_starttime` from `/proc/self/stat`), release flock, exit 0
7. If not enough free: print which GPUs are held by whom + since when, release flock, exit 1
8. On success, print **exactly one line** to stdout:
   ```
   CUDA_VISIBLE_DEVICES=3,4
   ```
   All other output (reservation confirmation, stale detection warnings, diagnostic info) goes to **stderr only**. This is critical: agents capture the output via `CVD=$(python gpu_reserve.py ...)`, and any extra stdout lines would corrupt the captured value.

**Why stdout, not an env file**: Claude Code's Bash tool creates a fresh shell for each invocation. `source env_file.sh` in one Bash call does NOT persist env vars to the next call. Instead, agents capture the output and prefix every GPU command with the printed `CUDA_VISIBLE_DEVICES=X` value. This is the only reliable way to propagate env vars across independent shell invocations.

**Agent usage pattern** (documented in skill docs and agent .md files):
```bash
# Step 1: Reserve (once per track)
CVD=$(python .claude/skills/ammo/scripts/gpu_reserve.py --gpu-count 1 --holder impl-champion-op003 --session-id abc123)

# Step 2: Use the captured value as a prefix on every GPU command
CUDA_VISIBLE_DEVICES=$CVD python benchmark_kernel.py
CUDA_VISIBLE_DEVICES=$CVD python -m pytest tests/test_kernel.py
CUDA_VISIBLE_DEVICES=$CVD nsys profile python run_model.py
```

Note: steps 1 and 2 can be in the same Bash tool call (chained with `&&`) or in separate calls as long as the agent remembers the `CVD` value and reuses it. Since Claude Code agents can read their own prior tool outputs, they can reference the value from the reserve call.

**Why gpu-count, not gpu-ids**: When `CUDA_VISIBLE_DEVICES=3,4`, PyTorch sees devices as `cuda:0` and `cuda:1`. Physical-to-logical ID mapping is confusing for agents. By requesting a count, agents never need to reason about physical IDs. The reservation system handles allocation and prints the correct `CUDA_VISIBLE_DEVICES` value.

**Allocation policy**: Lowest-numbered free physical GPUs. This is deterministic and simple. No affinity or NUMA awareness (not needed for current L40S/H100 configurations).

#### `gpu_release.py`

```
python .claude/skills/ammo/scripts/gpu_release.py --holder impl-champion-op003 [--force]
```

Behavior:
1. Acquire flock (LOCK_NB + retry) on `.lock`
2. Read `state.json`, clear all entries matching `holder`
3. Write updated `state.json`, release flock

`--force` mode: Release a reservation held by a DIFFERENT holder. Used by the orchestrator after a coordination timeout (see "Coordination Timeout" below). Requires passing the actual holder name to release. When `--force` is used, the script verifies that the `session_id` in the reservation entry matches the orchestrator's session (passed via `--session-id`) before releasing. This prevents cross-campaign force-releases — an orchestrator from campaign A cannot force-release campaign B's reservations. Logs a warning about the forced release.

#### `gpu_status.py`

```
python .claude/skills/ammo/scripts/gpu_status.py [--json] [--holder <name> --format cuda_visible]
```

Behavior:
- Default: human-readable table:
  ```
  GPU  Status    Holder                    Since               Purpose          Session
  0    HELD      impl-champion-op003       2026-03-20 14:32    kernel_benchmark a1b2c3d4
  1    FREE      -                         -                   -                -
  2    HELD      impl-validator-op007      2026-03-20 14:35    e2e_sweep        a1b2c3d4
  3    FREE      -                         -                   -                -
  ```
- `--json`: raw state.json content.
- `--holder X --format cuda_visible`: print ONLY the `CUDA_VISIBLE_DEVICES` value for that holder to stdout (useful for scripting if agent lost the original reserve output). Exits non-zero with stderr message if holder has no active reservation.

### Sweep Script Retrofit

Replace the existing `_acquire_gpu_lock()` in `run_vllm_bench_latency_sweep.py` with calls to the reservation system. The entire post-reservation body MUST be wrapped in `try/finally` to ensure release on any exit path (including `SystemExit`, which the sweep script raises in ~30 places).

**Before** (current):
```python
lock_handle = _acquire_gpu_lock(artifact_dir=artifact_dir, is_child=is_child)
# ... sweep body ... (no try/finally)
lock_handle.close()  # only reached on normal exit
```

**After**:
```python
from gpu_reservation import reserve, release

reserve(gpu_count=tp, holder=f"sweep:{agent_name}", session_id=session_id, purpose="e2e_sweep")
try:
    # ... entire sweep body ...
finally:
    release(holder=f"sweep:{agent_name}")
```

The sweep script imports the reservation functions directly from the shared module. The `flock`-based `_acquire_gpu_lock` function and its `/tmp/ammo_gpu_locks/` directory are removed.

**`_check_gpu_idle()` update**: Retained as an additional warning layer (catches non-AMMO GPU processes), but updated to filter out PIDs that are listed as reservation holders in `state.json`. This prevents spurious warnings when other tracks have valid reservations on different GPUs.

**Transition safety**: During rollout, the old `/tmp/ammo_gpu_locks/` flock system and the new JSON reservation system may coexist if worktrees have different versions of the sweep script. To handle this: on first run, the new system checks for the existence of `/tmp/ammo_gpu_locks/*.lock` files and warns if any are held. This is advisory only — full migration completes when all active worktrees are rebased to include the new sweep script.

### PreToolUse Hook Enhancement

Extend `ammo-pretool-guard.sh` with a GPU reservation check section.

#### Detection Patterns (Conservative)

Only match clear GPU indicators. The existing read-only command bailout at the top of the hook (grep, rg, cat, git, etc.) continues to apply.

```bash
# Pattern 1: Python/pytest scripts with GPU keywords in command
# e.g., "python benchmark_kernel.py", "pytest test_cuda.py"
# Matched by: command contains python/pytest AND (command contains torch|cuda|triton|vllm
#   OR script name contains benchmark|kernel|gpu)

# Pattern 2: Inline python with GPU imports
# e.g., "python -c 'import torch; ...'"
# Matched by: python -c AND (torch|cuda|triton)

# Pattern 3: GPU profiling tools
# e.g., "nsys profile ...", "ncu ...", "nvidia-smi --query-compute-apps"
# Matched by: command starts with nsys|ncu, or nvidia-smi with --query-compute

# NOT matched (false positive prevention):
# - cmake --build (builds don't use GPU; only subsequent tests do)
# - nvidia-smi (bare, for status checks)
# - python -c 'print("hello")' (no GPU keywords)
# - Any command already in the read-only bailout list

# Known limitation: Commands wrapped in `bash -c "..."` or executed via `.sh` scripts
# bypass pattern matching because the inner command is not visible to the hook.
# This is accepted as a tradeoff of conservative detection. The skill docs mandate
# reservation regardless of hook detection.
```

#### Check Flow

When a GPU pattern is detected:

1. Check if `/tmp/ammo_gpu_res/state.json` exists. If not, skip (no reservation system initialized).
2. Extract `CUDA_VISIBLE_DEVICES=X` from the command text (regex for `CUDA_VISIBLE_DEVICES=[\d,]+` in the command string).
3. If `CUDA_VISIBLE_DEVICES` found in command:
   - Read `state.json`, check if those physical GPU IDs are reserved by anyone.
   - If reserved: silent pass (agent is using reserved GPUs correctly).
   - If reserved by a different holder: warn about contention.
   - If not reserved: warn about missing reservation.
4. If no `CUDA_VISIBLE_DEVICES` in command:
   - Warn: `"AMMO GPU WARNING: GPU command detected without CUDA_VISIBLE_DEVICES prefix. Reserve GPUs first via gpu_reserve.py, then prefix your command with the assigned CUDA_VISIBLE_DEVICES=X."`

**Enforcement mode**: Warn only (exit 0 with stderr message). Does not block execution. The warning gives agents the signal to self-correct. *(Uses "Warned by" language in SKILL.md, not "Enforced by", to match the actual behavior.)*

**Worktree smoke test exemption**: Commands matching `python -c 'import vllm'` or `python -c "import torch"` (bare import checks without compute) are exempted from the GPU reservation warning. These run during worktree setup for environment verification and allocate only a minimal CUDA context without running compute that would perturb benchmarks.

### Coordination Timeout and Force-Release

When the orchestrator needs all GPUs for an E2E sweep but tracks hold reservations:

1. Orchestrator sends `"Release GPUs for E2E sweep"` to all track agents via SendMessage.
2. **Timeout**: If any track agent has not released within **10 minutes**, the orchestrator may force-release:
   ```bash
   python .claude/skills/ammo/scripts/gpu_release.py --holder <stuck_agent_name> --force
   ```
3. The force-release is logged (the release script writes a warning to stderr and to `state.json` audit trail).
4. The orchestrator notifies the affected agent: `"Your GPU reservation was force-released for E2E sweep. Re-reserve when the sweep completes."`

This mirrors the existing 90-minute timeout on overlapped debate — bounded escalation with logging.

### Skill Documentation Updates

#### SKILL.md

Add to Non-Negotiables (after item 4):

> **GPU reservation**: All agents MUST acquire a GPU reservation via `gpu_reserve.py` before any GPU work (kernel benchmarks, E2E sweeps, ncu profiling, experiments). Prefix every GPU command with the `CUDA_VISIBLE_DEVICES=X` value printed by the reserve script. Release via `gpu_release.py` when GPU work is complete. *(Warned by `ammo-pretool-guard.sh` — detects missing CUDA_VISIBLE_DEVICES prefix)*

Add to Helper Scripts table:

| Script | Purpose |
|--------|---------|
| `scripts/gpu_reserve.py` | Reserve N GPUs for an agent (fail-fast if unavailable) |
| `scripts/gpu_release.py` | Release GPUs held by an agent (supports `--force` for orchestrator) |
| `scripts/gpu_status.py` | Print current GPU reservation state |

Update Hook Enforcement table to note the GPU reservation check in `ammo-pretool-guard.sh`.

**session_id recording**: Upgrade the existing "The lead SHOULD record the session ID" to "The lead MUST record the session ID in state.json at campaign start." The orchestrator interpolates `{session_id}` in spawn prompts from `state.json["session_id"]`. If `session_id` is null when the orchestrator attempts to spawn tracks, the orchestrator MUST generate a UUID and record it before proceeding.

#### orchestration/parallel-tracks.md

Replace the GPU Assignment table:

| Operation | Reservation | Details |
|-----------|-------------|---------|
| Track kernel benchmarks | `--gpu-count 1` per track | Reserved at track start, released at track end |
| E2E sweep (TP=N) | `--gpu-count N` | All other reservations must be released first. Sweep auto-reserves via integrated API. |
| Debate micro-experiments | None (CPU-only) | Debate agents MUST NOT use GPUs. CPU-based analysis only (roofline calcs, ISA inspection, ncu --query-metrics static analysis). This aligns with SKILL.md "CPU-based analysis preferred" for debate. **Deletions required**: (1) Remove parallel-tracks.md lines 306-307 ("On systems with 3+ GPUs, the last GPU is soft-reserved for debate micro-experiments..."). (2) Remove SKILL.md line 141 ("On multi-GPU systems (N >= 3), the last GPU is soft-reserved for debate micro-experiments..."). (3) Remove parallel-tracks.md "GPU Allocation During Overlap" table rows referencing debate GPU access. All debate GPU references across all files must be replaced with CPU-only language. |

Add "GPU Reservation Protocol" section:

```
## GPU Reservation Protocol

Before any GPU work, every agent must:

1. Reserve and capture the assigned CUDA_VISIBLE_DEVICES:
   CVD=$(python .claude/skills/ammo/scripts/gpu_reserve.py \
     --gpu-count <N> --holder <agent-name> --session-id <session-id> \
     --purpose <kernel_benchmark|e2e_sweep|ncu_profiling>)

2. Prefix every GPU command with the assigned value:
   CUDA_VISIBLE_DEVICES=$CVD python benchmark.py
   CUDA_VISIBLE_DEVICES=$CVD nsys profile python model.py

3. Release when done:
   python .claude/skills/ammo/scripts/gpu_release.py --holder <agent-name>

IMPORTANT: Do NOT use `source` on env files — Claude Code Bash tool calls are
independent shell processes. Env vars set in one call do NOT persist to the next.
Always use the inline CUDA_VISIBLE_DEVICES=$CVD prefix pattern.

If reservation fails (GPUs held by another agent):
- The script prints who holds which GPUs and since when.
- Report the contention to the orchestrator via SendMessage.
- Do NOT proceed with GPU work without a reservation.
- Do NOT busy-poll retries. Wait for orchestrator coordination.

E2E sweeps need all TP GPUs. If tracks hold per-GPU reservations,
they must release before the sweep can acquire. The orchestrator
coordinates this via SendMessage:
  1. Orchestrator messages all track agents: "Release GPUs for E2E sweep"
  2. Track agents release via gpu_release.py, acknowledge via SendMessage
  3. Sweep acquires all GPUs (10-minute timeout on step 2, then force-release)
  4. After sweep completes: agents re-reserve their per-GPU allocations
```

#### Agent .md files

Update `ammo-impl-champion` and `ammo-impl-validator` agent definitions:

```
## GPU Reservation (MANDATORY)

Before any GPU command (kernel benchmark, ncu, pytest with CUDA, experiments):

1. Reserve (once at track start):
   CVD=$(python .claude/skills/ammo/scripts/gpu_reserve.py --gpu-count 1 \
     --holder <your-name> --session-id <session-id> --purpose <purpose>)

2. Prefix EVERY GPU command with the assigned value:
   CUDA_VISIBLE_DEVICES=$CVD python benchmark_kernel.py
   CUDA_VISIBLE_DEVICES=$CVD python -m pytest tests/test_kernel.py

3. Release (at track end):
   python .claude/skills/ammo/scripts/gpu_release.py --holder <your-name>

WARNING: Do NOT use `source` — env vars do not persist across Bash tool calls.
If reservation fails, report to orchestrator via SendMessage. Do NOT proceed.
```

#### Spawn Prompt Updates

Replace hardcoded `GPU assignment: CUDA_VISIBLE_DEVICES={gpu_id}` with:

```
GPU reservation: Reserve 1 GPU at track start:
  CVD=$(python .claude/skills/ammo/scripts/gpu_reserve.py --gpu-count 1 \
    --holder {agent_name} --session-id {session_id})
Then prefix every GPU command: CUDA_VISIBLE_DEVICES=$CVD <command>
Release at track end:
  python .claude/skills/ammo/scripts/gpu_release.py --holder {agent_name}
```

#### Resume Protocol Update

Add step 3b to SKILL.md Resume Protocol (after step 3, before step 4):

> **3b.** Check GPU reservation state: run `python .claude/skills/ammo/scripts/gpu_status.py`. Cross-reference against `parallel_tracks` in campaign `state.json`. Force-release any reservations from the crashed session (`gpu_release.py --holder <name> --force --session-id <crashed_session_id>`). Active tracks that are being re-spawned will re-reserve their GPUs as part of their startup sequence.

### Implementation Scope

This is a **design spec** — it describes changes to be made during implementation. The "Modified files" list below is the implementation TODO. The actual files on disk have NOT been updated yet. During implementation, each file modification should be verified against this spec.

### File Inventory

New files:
- `.claude/skills/ammo/scripts/gpu_reserve.py` — Reservation script (prints CVD to stdout)
- `.claude/skills/ammo/scripts/gpu_release.py` — Release script (supports `--force`)
- `.claude/skills/ammo/scripts/gpu_status.py` — Status display script
- `.claude/skills/ammo/scripts/gpu_reservation.py` — Shared module with public API:
  - `reserve(gpu_count: int, holder: str, session_id: str, purpose: str = "general") -> str` — returns `CUDA_VISIBLE_DEVICES` value; raises `ReservationError` on failure
  - `release(holder: str, force: bool = False, session_id: str | None = None) -> None` — releases all GPUs for holder; `force=True` requires `session_id`
  - `status() -> dict` — returns parsed state.json content
  - Internal: `_read_state()`, `_write_state()`, `_acquire_flock()`, `_check_stale()`

Modified files:
- `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py` — Replace `_acquire_gpu_lock`/`_check_gpu_idle` with reservation API calls, wrap in `try/finally`
- `.claude/hooks/ammo-pretool-guard.sh` — Add GPU reservation check (CVD prefix detection)
- `.claude/skills/ammo/SKILL.md` — Add non-negotiable ("Warned by"), helper scripts, hook docs, resume protocol step 3b
- `.claude/skills/ammo/orchestration/parallel-tracks.md` — Replace GPU assignment table, add protocol section, debate = CPU-only
- `.claude/skills/ammo/agents/ammo-impl-champion.md` — Add GPU reservation instructions (CVD prefix pattern)
- `.claude/skills/ammo/agents/ammo-impl-validator.md` — Add GPU reservation instructions (CVD prefix pattern)

### Edge Cases

**Multiple campaigns on same machine**: `session_id` in each reservation entry disambiguates. `gpu_status.py` shows which session holds each GPU. One campaign's agents cannot accidentally reclaim another campaign's GPUs (different holder names with different session_ids).

**Agent crash without release**: Two-step stale detection (PID liveness + starttime comparison) on next reserve attempt. Handles both dead PIDs and PID reuse by the OS.

**E2E sweep needs all GPUs but tracks hold some**: The orchestrator coordinates release via SendMessage before launching the sweep. 10-minute timeout, then force-release. Documented in GPU Reservation Protocol.

**Hook false positives**: Conservative patterns minimize these. A `python -c 'print("hello")'` won't trigger because it lacks GPU keywords. The warn-only mode means false positives are annoying but not blocking.

**Hook false negatives**: Commands wrapped in `bash -c "..."` or executed via `.sh` wrapper scripts bypass pattern detection. This is a known limitation of conservative detection. The skill docs mandate reservation regardless — the hook is a safety net, not the sole enforcement.

**No AMMO campaign active**: The pretool hook has two independent early-exit checks: (1) the existing campaign check — bail if no `state.json` exists in any `kernel_opt_artifacts/` subdirectory; (2) the new reservation check — bail if `/tmp/ammo_gpu_res/state.json` does not exist. Both must pass for the GPU reservation warning to fire. These are different files: campaign state is in the artifact dir, reservation state is in `/tmp/`.

**Worktree agents**: `/tmp/ammo_gpu_res/` is universally accessible regardless of worktree path. The scripts use absolute paths.

**Worktree smoke tests**: `python -c "import vllm"` / `python -c "import torch"` during worktree setup are exempted from GPU reservation warnings (bare import checks, no compute).

**Campaign resume after crash**: Resume protocol step 3b explicitly reconciles GPU reservation state — force-releases stale reservations from the crashed session before re-spawning agents.

**flock contention under high parallelism**: Uses `LOCK_NB` + exponential backoff (5 attempts: 100ms, 200ms, 400ms, 800ms — ~1.5s total budget) instead of blocking `LOCK_EX`. If the lock cannot be acquired after retries, exits with error rather than hanging indefinitely.

**Transition from old locking system**: The new system checks for existing `/tmp/ammo_gpu_locks/*.lock` files and warns on first run. Old worktrees with the pre-modification sweep script will use the old flock; new worktrees use JSON reservations. Full migration completes when all worktrees are rebased.

### What This Does NOT Solve

- **Non-AMMO GPU processes**: Other users or system processes on the same machine. The retained `_check_gpu_idle()` nvidia-smi advisory check partially addresses this (updated to filter out known AMMO reservation holders).
- **Network-level GPU scheduling**: Multiple machines are out of scope.
- **Priority/preemption**: No mechanism for high-priority work (E2E sweep) to preempt lower-priority work (kernel benchmark) automatically. The orchestrator coordinates this via SendMessage + timeout + force-release.
