---
name: ammo
description: Profile and optimize GPU kernels for vLLM inference on NVIDIA GPUs. Use when targeting specific (model, hardware, dtype, TP) deployments to improve latency. Triggers on requests to speed up any vLLM kernel.
---

# AMMO - Automated Model Micro-Optimizer

Profile and optimize **GPU kernels** for **vLLM inference** that beat the **production-parity baseline** (CUDA graphs / torch.compile), without regressing correctness.

## Invocation

User provides: model_id, hardware, dtype, tp, component (or "auto").

Lead (you) scaffolds artifact directory, creates team, spawns agents, creates task graph.

```bash
python .claude/skills/ammo/scripts/new_target.py \
  --artifact-dir kernel_opt_artifacts/{component}_{model}_{hardware}_{dtype}_tp{tp} \
  --model-id <MODEL_ID> --hardware <HW> --dtype <DTYPE> --tp <TP>
```

### Team Setup

```
TeamCreate: ammo-{target}
Spawn researcher: Task(subagent_type="general-purpose", name="researcher", team_name=...)
  → Agent def: .claude/agents/ammo-researcher.md
Spawn implementer: Task(subagent_type="general-purpose", name="implementer", team_name=...)
  → Agent def: .claude/agents/ammo-implementer.md
```

## 5-Stage Workflow

```
Stage 1: Baseline Capture       → constraints.md
Stage 2: Bottleneck Mining       → bottleneck_analysis.md
Stage 3: Optimization Planning   → optimization_plan.md
Stage 4: Implementation          → kernel code + correctness tests
Stage 5: Validation              → validation_results.md
         → SHIP (done) or KILL → retry with next opportunity (up to max_attempts)
```

## Task Graph

| Task | Description | Owner | Depends |
|------|-------------|-------|---------|
| B1 | Scaffold artifact directory | lead | — |
| B2 | Capture baseline (env + nsys + source + timings) | researcher | B1 |
| B3 | Write constraints.md (invariants, baseline snapshot) | researcher | B2 |
| B4 | **GATE**: run `verify_phase1_baseline.py` | lead | B3 |
| B5 | Mine bottlenecks + rank opportunities | researcher | B4 |
| B6 | **GATE**: Stage 2 review (bottleneck quality) | lead | B5 |
| B7 | Select approach + write optimization_plan.md | researcher | B6 |
| B8 | **GATE**: Stage 3 plan review | lead | B7 |
| B9 | Implement optimization + write correctness tests | implementer | B8 |
| B10 | **GATE**: compilation check (`python -c "import ..."`) | lead | B9 |
| B11 | Review tests + correctness + kernel benchmarks | researcher | B10 |
| B12 | E2E benchmarks + kill criteria eval + write results | researcher | B11 |
| B13 | **GATE**: run `verify_validation_gates.py` | lead | B12 |
| B14 | Route decision (SHIP → close, KILL → iterate) | lead | B13 |

B11→B12 enforces GPU sequencing (no concurrent benchmarks).
B14 is **always pre-created at startup**, blocked by B13.

### Iteration Loop

B14 handles routing after B13 completes:

**IF SHIP:**
- Verify all kill criteria passed
- Update state.json: status → "shipped"
- Shut down team

**IF KILL:**
1. Record attempt in state.json `opportunity_attempts[]`
2. Check attempt count vs `max_attempts`
3. If exhausted → status: "exhausted", shut down
4. Select next opportunity from bottleneck_analysis.md
5. Update state.json: `current_opportunity_id`, stage → "3_planning_iteration"
6. Create tasks:

| Task | Description | Owner | Depends |
|------|-------------|-------|---------|
| B15 | Write updated optimization_plan.md | researcher | B14 |
| B16 | **GATE**: iteration plan review | lead | B15 |
| B9' | Implement optimization | implementer | B16 |
| B10' | **GATE**: compilation check | lead | B9' |
| B11' | Correctness + kernel benchmarks | researcher | B10' |
| B12' | E2E benchmarks + kill eval | researcher | B11' |
| B13' | **GATE**: run `verify_validation_gates.py` | lead | B12' |
| B14' | Route decision (attempt N+1) | lead | B13' |

Repeats until SHIP, all attempts exhausted (`max_attempts`), or BLOCKED.

## Non-Negotiables

1. **Production parity**: CUDA graphs + torch.compile in ALL measurements. FORBIDDEN: `TORCH_COMPILE_DISABLE=1`, `--enforce-eager`, `VLLM_TORCH_COMPILE_LEVEL=0`.
2. **vLLM baseline**: Compare against production kernel, NOT naive PyTorch loops.
3. **Numerical correctness**: `torch.allclose()` is mandatory in every correctness test.
4. **GPU sequencing**: E2E benchmarks (B12) blocked by kernel benchmarks (B11). Use `scripts/run_vllm_bench_latency_sweep.py` for all E2E measurements (holds system-wide GPU lock).
5. **Full-model E2E**: Do not skip E2E because "weights aren't available" — download them. Only skip if user explicitly waives.
6. **E2E delta math**: `E2E_improvement ≈ f × kernel_speedup`, where `f` = component share of total latency. If `f` is small, large kernel wins yield small E2E gains — this is expected, not a bug.

## State Management

`state.json` in artifact directory tracks stage, status, opportunity, attempts:

```json
{
  "target": { "model_id": "...", "hardware": "...", "dtype": "...", "tp": 1, "ep": 1, "component": "auto" },
  "stage": "1_baseline",
  "status": "in_progress",
  "current_opportunity_id": null,
  "max_attempts": 3,
  "opportunity_attempts": [],
  "route_decision": {},
  "verification_run": { "stage1": null, "validation": null },
  "last_update": "2026-02-23",
  "summary": "Initialized.",
  "team": { "name": "ammo-{target}", "members": ["lead", "researcher", "implementer"] },
  "gpu_resources": { "gpu_count": 1, "gpu_model": "...", "memory_total_gib": 0, "cuda_visible_devices": "0" }
}
```

### Status Values

- `"in_progress"` — stage is actively being worked
- `"shipped"` — SHIP decision, workflow done
- `"iterating"` — KILL decision, creating next attempt
- `"exhausted"` — all `max_attempts` consumed, workflow done
- `"blocked"` — unresolvable blocker, needs human

`"completed"` is NOT a valid workflow-level status. It is only valid for individual task status.

`TaskList` tracks task progress, ownership, and dependencies.

## Communication Patterns

- **Blocker escalation** (agent → lead): `SendMessage` with "BLOCKER [{severity}]: {description}". Save details to `{artifact_dir}/blockers/{stage}_{date}.md`.
- **Critical stop** (lead → all): Broadcast to halt all work immediately.
- **GPU contention alert**: Detected concurrent GPU processes during benchmark — stop and re-run.
- **Shutdown**: Clean team termination via `SendMessage(type="shutdown_request")`.
- **KILL→Pivot** (lead → researcher): Opportunity killed, selecting next from ranked list.

## Resume Protocol

After interruption or compaction:
1. Read team config: `~/.claude/teams/ammo-*/config.json`
2. Run `TaskList` to see task progress
3. Read `state.json` from artifact directory
4. Check artifact files for current stage deliverables
5. Message idle teammates to resume work
6. You are the LEAD — manage tasks and gates, do not implement directly.

## Helper Scripts

Run, don't modify:
- `scripts/new_target.py` — Scaffold artifact directory
- `scripts/collect_env.py` — Capture environment for reproducibility
- `scripts/verify_phase1_baseline.py` — Stage 1→2 gate verification
- `scripts/verify_validation_gates.py` — Stage 5 gate verification
- `scripts/run_vllm_bench_latency_sweep.py` — Batch E2E benchmark runner (GPU-locked)
- `scripts/generate_validation_report.py` — Structured reporting

## References

| Topic | File |
|-------|------|
| Nsys profiling | `references/nsys-profiling-guide.md` |
| Validation gates | `references/validation-defaults.md` |
| CUDA graph safety | `references/cudagraph-safety.md` |
| E2E latency | `references/e2e-latency-guide.md` |
