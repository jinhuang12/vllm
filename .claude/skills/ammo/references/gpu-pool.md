# GPU Pool Reservation

All GPU commands must use the pool reservation pattern. This ensures GPU isolation across concurrent agents.

## Reservation Pattern

```bash
CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus N) && CUDA_VISIBLE_DEVICES=$CVD <command>
```

GPUs auto-release when the command completes. The PostToolUse hook detects the reservation pattern and releases by session ID. Lease expiry (2h default, 4h for nsys) handles crashes.

## Contention Handling

If the pool is exhausted, the reserve command fails. Wait briefly and retry. For CPU-only commands (file reads, roofline math, ISA inspection), no reservation is needed.

## GPU Count by Task Type

| Task | `--num-gpus` | Notes |
|------|-------------|-------|
| Kernel benchmarks | 1 | Single GPU sufficient |
| Micro-experiments (debate) | 1 | Keep brief to minimize contention |
| Static analysis (`ncu --query-metrics`) | 1 | No kernel execution |
| nsys single-kernel traces | 1 | Existing binary only |
| E2E sweeps | `{tp}` | Match tensor parallelism from target.json |

For TP > 1, the pool allocates contiguous GPU blocks. If no contiguous block is available, the command fails — retry after other agents release.

## Diagnostics (Orchestrator Only)

- `scripts/gpu_status.py` — print current reservation state
- `scripts/gpu_force_clear.py` — clear stale reservations after crashes
