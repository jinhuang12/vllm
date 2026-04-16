# GPU Pool Reservation

All GPU commands must use the pool reservation pattern. This ensures GPU isolation across concurrent agents.

## Reservation Pattern

```bash
CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve \
  --num-gpus N --session-id {op_id} --no-auto-release) && \
  CUDA_VISIBLE_DEVICES=$CVD <command>
```

- `--session-id {op_id}`: Use your agent's op_id (e.g., `op001`). Prevents cross-agent eviction.
- `--no-auto-release`: Prevents your reservation from silently clearing another agent's GPUs that share a session_id.

GPUs auto-release when the command completes. The PostToolUse hook detects the reservation pattern and releases by session ID. A SubagentStop hook also releases any GPUs still held by `$CLAUDE_SESSION_ID` (or `$CLAUDE_SESSION_ID:$AGENT_ID`) when you return to the parent. Lease expiry (**15 min default**) handles crashes.

Pass `--lease-hours 2` for long sweeps or nsys captures that run >10 min. The script does NOT auto-extend leases for nsys.

**Session-id form rules (important — the PostToolUse hook regex is strict):**
- Use `--session-id foo` (space form) OR `--session-id=foo` (equals form). Both work.
- Valid id chars: `[A-Za-z0-9_.\-]`. Anything else (spaces, `)`, `&`, `;`, quotes) terminates the id.
- Always release explicitly before returning if you minted an ad-hoc id:
  `python .claude/skills/ammo/scripts/gpu_reservation.py release-session --session-id {op_id} || true`

## Contention Handling

If the pool is exhausted, the reserve command **fails immediately** with a `ReservationError`. Retry with backoff until a GPU frees up:

```bash
# Retry loop — keep trying until a GPU is available
MAX_RETRIES=60
for i in $(seq 1 $MAX_RETRIES); do
  CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve \
    --num-gpus N --session-id {op_id} --no-auto-release 2>/dev/null) && break
  echo "GPU pool exhausted, retry $i/$MAX_RETRIES (sleeping 30s)..." >&2
  sleep 30
done
if [ -z "$CVD" ]; then
  echo "ERROR: Could not acquire GPU after $MAX_RETRIES retries" >&2
  exit 1
fi
CUDA_VISIBLE_DEVICES=$CVD <command>
```

For CPU-only commands (file reads, roofline math, ISA inspection), no reservation is needed.

## Process Isolation Rules

**Never kill processes on GPUs you don't own.** When multiple agents share a GPU pool:

- Only terminate processes YOU started. Do not `kill`, `pkill`, or `killall` processes belonging to other agents.
- If `nvidia-smi` shows a process on "your" GPU that you didn't start, it belongs to another agent whose reservation overlaps. Wait for them to finish — do not kill it.
- If you suspect a zombie/leaked process, report it to the orchestrator rather than killing it yourself.

## GPU Count by Task Type

| Task | `--num-gpus` | Notes |
|------|-------------|-------|
| Kernel benchmarks | 1 | Single GPU sufficient |
| Micro-experiments (debate) | 1 | Keep brief to minimize contention |
| Static analysis (`ncu --query-metrics`) | 1 | No kernel execution |
| nsys single-kernel traces | 1 | Existing binary only |
| E2E sweeps | `{tp}` | Match tensor parallelism from target.json |

For TP > 1, the pool allocates contiguous GPU blocks. If no contiguous block is available, the command fails — retry after other agents release.

For sweep or nsys runs that legitimately exceed 15 min, add `--lease-hours 2` (or higher) to the reserve call so the lease doesn't expire mid-run.

## Diagnostics (Orchestrator Only)

- `scripts/gpu_status.py` — print current reservation state
- `scripts/gpu_force_clear.py` — clear stale reservations after crashes
