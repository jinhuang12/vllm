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
         v                         v                          v
+------------------------------------------------------------------------+
|                /tmp/ammo_gpu_res/state.json                            |
|  Atomic read-modify-write via brief flock on .lock                     |
|  Per-GPU entries keyed by physical GPU ID                              |
+------------------------------------------------------------------------+
         ^                         ^                          ^
         |                         |                          |
    PreToolUse hook           PreToolUse hook           (integrated)
    reads state.json          reads state.json
    warns if no reservation   warns if contention
```

### State Model

**Location**: `/tmp/ammo_gpu_res/state.json`

Machine-scoped (not per-campaign) because GPU contention is a machine-level concern. Multiple concurrent AMMO sessions on the same machine are disambiguated by `session_id`.

```json
{
  "gpus": {
    "0": {
      "holder": "impl-champion-op003",
      "session_id": "a1b2c3d4",
      "reserved_at": "2026-03-20T14:32:00Z",
      "pid": 12345,
      "purpose": "kernel_benchmark",
      "artifact_dir": "/home/jinhun/vllm/kernel_opt_artifacts/Nemotron..."
    },
    "1": null,
    "2": {
      "holder": "impl-validator-op007",
      "session_id": "a1b2c3d4",
      "reserved_at": "2026-03-20T14:35:00Z",
      "pid": 12346,
      "purpose": "e2e_sweep",
      "artifact_dir": "/home/jinhun/vllm/kernel_opt_artifacts/Nemotron..."
    },
    "3": null
  },
  "gpu_count": 4
}
```

**Atomicity**: Brief `flock(LOCK_EX)` on `/tmp/ammo_gpu_res/.lock` during every read-modify-write. The flock is held only for the JSON update (~milliseconds), not for the reservation duration.

**Stale detection**: On every reserve attempt, for each GPU that shows as held, check `kill -0 <pid>`. If the PID is dead, treat as stale, auto-reclaim with a warning to stderr.

**GPU discovery**: `gpu_count` is set on first init via `nvidia-smi -L | wc -l`.

### Scripts API

Three new scripts in `.claude/skills/ammo/scripts/`:

#### `gpu_reserve.py`

```
python .claude/skills/ammo/scripts/gpu_reserve.py \
  --gpu-count 2 \
  --holder impl-champion-op003 \
  --session-id <session_uuid> \
  [--purpose kernel_benchmark|e2e_sweep|micro_experiment|ncu_profiling]
```

Behavior:
1. Acquire brief flock on `/tmp/ammo_gpu_res/.lock`
2. Read `state.json` (create if missing, discover GPU count)
3. Run stale detection on all held GPUs
4. Find N free physical GPUs (lowest-numbered first)
5. If N free GPUs found: write holder info for each, release flock, exit 0
6. If not enough free: print which GPUs are held by whom + since when, release flock, exit 1
7. On success: write env file to `/tmp/ammo_gpu_res/env_{holder}.sh`:
   ```bash
   export CUDA_VISIBLE_DEVICES=3,4
   export AMMO_GPU_HOLDER=impl-champion-op003
   ```

**Why gpu-count, not gpu-ids**: When `CUDA_VISIBLE_DEVICES=3,4`, PyTorch sees devices as `cuda:0` and `cuda:1`. Physical-to-logical ID mapping is confusing for agents. By requesting a count, agents never need to reason about physical IDs. The reservation system handles allocation and writes the correct `CUDA_VISIBLE_DEVICES` to the env file.

**Allocation policy**: Lowest-numbered free physical GPUs. This is deterministic and simple. No affinity or NUMA awareness (not needed for current L40S/H100 configurations).

#### `gpu_release.py`

```
python .claude/skills/ammo/scripts/gpu_release.py --holder impl-champion-op003
```

Behavior:
1. Acquire brief flock on `.lock`
2. Read `state.json`, clear all entries matching `holder`
3. Write updated `state.json`, release flock
4. Remove `/tmp/ammo_gpu_res/env_{holder}.sh`

#### `gpu_status.py`

```
python .claude/skills/ammo/scripts/gpu_status.py [--json]
```

Behavior:
- Human-readable table by default:
  ```
  GPU  Status    Holder                    Since               Purpose
  0    HELD      impl-champion-op003       2026-03-20 14:32    kernel_benchmark
  1    FREE      -                         -                   -
  2    HELD      impl-validator-op007      2026-03-20 14:35    e2e_sweep
  3    FREE      -                         -                   -
  ```
- `--json` outputs the raw state.json content.

### Sweep Script Retrofit

Replace the existing `_acquire_gpu_lock()` and `_check_gpu_idle()` in `run_vllm_bench_latency_sweep.py` with calls to the reservation system.

**Before** (current):
```python
lock_handle = _acquire_gpu_lock(artifact_dir=artifact_dir, is_child=is_child)
```

**After**:
```python
# At sweep start:
_gpu_reserve(gpu_count=tp, holder=f"sweep:{agent_name}", session_id=session_id, purpose="e2e_sweep")

# At sweep end (or in finally/exception handler):
_gpu_release(holder=f"sweep:{agent_name}")
```

The sweep script imports the reservation functions directly (they'll be refactored into a shared module) rather than shelling out to the scripts. The `flock`-based `_acquire_gpu_lock` function and its `/tmp/ammo_gpu_locks/` directory are removed.

The `_check_gpu_idle()` advisory nvidia-smi check is retained as an additional warning layer (it catches non-AMMO GPU processes that the reservation system can't track).

### PreToolUse Hook Enhancement

Extend `ammo-pretool-guard.sh` with a GPU reservation check section.

#### Detection Patterns (Conservative)

Only match clear GPU indicators. The existing read-only command bailout at the top of the hook (grep, rg, cat, git, etc.) continues to apply.

```bash
# Pattern 1: Python/pytest scripts with GPU keywords in command
# e.g., "python benchmark_kernel.py", "pytest test_cuda.py"
# Matched by: command starts with python/pytest AND (command contains torch|cuda|triton|vllm OR script name contains benchmark|kernel|gpu)

# Pattern 2: Inline python with GPU imports
# e.g., "python -c 'import torch; ...'"
# Matched by: python -c AND (torch|cuda|triton)

# Pattern 3: GPU profiling tools
# e.g., "nsys profile ...", "ncu ...", "nvidia-smi --query-compute-apps"
# Matched by: command starts with nsys|ncu, or nvidia-smi with --query-compute

# Pattern 4: CUDA compilation
# e.g., "cmake --build ... --target install"
# Matched by: cmake --build (building CUDA extensions needs GPU for testing)

# NOT matched (false positive prevention):
# - nvidia-smi (bare, for status checks)
# - python -c 'print("hello")' (no GPU keywords)
# - Any command already in the read-only bailout list
```

#### Check Flow

When a GPU pattern is detected:

1. Check if `/tmp/ammo_gpu_res/state.json` exists. If not, skip (no reservation system initialized).
2. Read `AMMO_GPU_HOLDER` from the environment. If not set:
   - Warn: `"AMMO GPU WARNING: GPU command detected but AMMO_GPU_HOLDER is not set. Reserve GPUs first: python .claude/skills/ammo/scripts/gpu_reserve.py --gpu-count N --holder <name> --session-id <id>, then source the env file."`
3. If `AMMO_GPU_HOLDER` is set, read `state.json` and check if any GPU is reserved with that holder.
   - If no matching reservation found: warn about missing reservation.
   - If reservation found: silent pass (agent is properly reserved).
4. Additionally, if `CUDA_VISIBLE_DEVICES` is set in the command or env, check if those specific physical GPUs are reserved by the current holder. Warn if they're held by someone else.

**Enforcement mode**: Warn only (exit 0 with stderr message). Does not block execution. The warning gives agents the signal to self-correct.

### Skill Documentation Updates

#### SKILL.md

Add to Non-Negotiables (after item 4):

> **GPU reservation**: All agents MUST acquire a GPU reservation via `gpu_reserve.py` before any GPU work (kernel benchmarks, E2E sweeps, ncu profiling, experiments). Source the generated env file to set `CUDA_VISIBLE_DEVICES` and `AMMO_GPU_HOLDER`. Release via `gpu_release.py` when GPU work is complete. The PreToolUse hook warns on GPU commands without a reservation. *(Enforced by `ammo-pretool-guard.sh`)*

Add to Helper Scripts table:

| Script | Purpose |
|--------|---------|
| `scripts/gpu_reserve.py` | Reserve N GPUs for an agent (fail-fast if unavailable) |
| `scripts/gpu_release.py` | Release GPUs held by an agent |
| `scripts/gpu_status.py` | Print current GPU reservation state |

Update Hook Enforcement table to note the GPU reservation check in `ammo-pretool-guard.sh`.

#### orchestration/parallel-tracks.md

Replace the GPU Assignment table:

| Operation | Reservation | Details |
|-----------|-------------|---------|
| Track kernel benchmarks | `--gpu-count 1` per track | Reserved at track start, released at track end |
| E2E sweep (TP=N) | `--gpu-count N` | All other reservations must be released first. Sweep auto-reserves via integrated API. |
| Debate micro-experiments | `--gpu-count 1` (optional) | CPU-based analysis preferred. On 3+ GPU systems, may reserve last GPU. |

Add "GPU Reservation Protocol" section:

```
## GPU Reservation Protocol

Before any GPU work, every agent must:

1. Reserve:
   python .claude/skills/ammo/scripts/gpu_reserve.py \
     --gpu-count <N> --holder <agent-name> --session-id <session-id> \
     --purpose <kernel_benchmark|e2e_sweep|ncu_profiling>

2. Source the env file:
   source /tmp/ammo_gpu_res/env_<agent-name>.sh

3. Do GPU work (benchmarks, profiling, experiments).

4. Release:
   python .claude/skills/ammo/scripts/gpu_release.py --holder <agent-name>

If reservation fails (GPUs held by another agent):
- The script prints who holds which GPUs and since when.
- Wait for the other agent to release, then retry.
- Do NOT proceed with GPU work without a reservation.

E2E sweeps need all TP GPUs. If tracks hold per-GPU reservations,
they must release before the sweep can acquire. The orchestrator
coordinates this via SendMessage:
  1. Orchestrator messages all track agents: "Release GPUs for E2E sweep"
  2. Track agents release via gpu_release.py
  3. Sweep acquires all GPUs
  4. After sweep: agents re-reserve their per-GPU allocations
```

#### Agent .md files

Update `ammo-impl-champion` and `ammo-impl-validator` agent definitions to include:

```
## GPU Reservation (MANDATORY)

Before any GPU command (kernel benchmark, ncu, pytest with CUDA, experiments):
1. python .claude/skills/ammo/scripts/gpu_reserve.py --gpu-count 1 --holder <your-name> --session-id <session-id> --purpose <purpose>
2. source /tmp/ammo_gpu_res/env_<your-name>.sh
3. [GPU work]
4. python .claude/skills/ammo/scripts/gpu_release.py --holder <your-name>

If reservation fails, report to your champion/orchestrator via SendMessage — do NOT proceed without a reservation.
```

#### Spawn Prompt Updates

Replace hardcoded `GPU assignment: CUDA_VISIBLE_DEVICES={gpu_id}` with:

```
GPU reservation: Reserve 1 GPU via:
  python .claude/skills/ammo/scripts/gpu_reserve.py --gpu-count 1 --holder {agent_name} --session-id {session_id}
  source /tmp/ammo_gpu_res/env_{agent_name}.sh
Release when your track completes:
  python .claude/skills/ammo/scripts/gpu_release.py --holder {agent_name}
```

### File Inventory

New files:
- `.claude/skills/ammo/scripts/gpu_reserve.py` — Reservation script
- `.claude/skills/ammo/scripts/gpu_release.py` — Release script
- `.claude/skills/ammo/scripts/gpu_status.py` — Status display script
- `.claude/skills/ammo/scripts/gpu_reservation.py` — Shared module (state read/write, flock, stale detection)

Modified files:
- `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py` — Replace `_acquire_gpu_lock`/`_check_gpu_idle` with reservation API calls
- `.claude/hooks/ammo-pretool-guard.sh` — Add GPU reservation check section
- `.claude/skills/ammo/SKILL.md` — Add non-negotiable, helper scripts, hook docs
- `.claude/skills/ammo/orchestration/parallel-tracks.md` — Replace GPU assignment table, add protocol section
- `.claude/skills/ammo/agents/ammo-impl-champion.md` — Add GPU reservation instructions
- `.claude/skills/ammo/agents/ammo-impl-validator.md` — Add GPU reservation instructions

### Edge Cases

**Multiple campaigns on same machine**: `session_id` in each reservation entry disambiguates. `gpu_status.py` shows which session holds each GPU. One campaign's agents cannot accidentally reclaim another campaign's GPUs (different holder names with different session_ids).

**Agent crash without release**: PID-based stale detection on next reserve attempt. If the PID is dead, the reservation is auto-reclaimed with a warning.

**E2E sweep needs all GPUs but tracks hold some**: The orchestrator coordinates release via SendMessage before launching the sweep. This is documented in the GPU Reservation Protocol. The sweep script's reserve call fails fast if GPUs are still held, giving a clear error.

**Hook false positives**: Conservative patterns minimize these. A `python -c 'print("hello")'` won't trigger because it lacks GPU keywords. The warn-only mode means false positives are annoying but not blocking.

**No AMMO campaign active**: The pretool hook already bails out early if no `state.json` exists in any `kernel_opt_artifacts/` subdirectory. The GPU reservation check is only active when the reservation state file exists.

**Worktree agents**: `/tmp/ammo_gpu_res/` is universally accessible regardless of worktree path. The scripts use absolute paths.

### What This Does NOT Solve

- **Non-AMMO GPU processes**: Other users or system processes on the same machine. The retained `_check_gpu_idle()` nvidia-smi advisory check partially addresses this.
- **Network-level GPU scheduling**: Multiple machines are out of scope.
- **Priority/preemption**: No mechanism for high-priority work (E2E sweep) to preempt lower-priority work (kernel benchmark). This requires orchestrator coordination via SendMessage.
