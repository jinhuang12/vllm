# Nsys Probe Script & Profiling Fallback Guidance

**Date**: 2026-03-23
**Status**: Draft
**Scope**: AMMO skill profiling pipeline (Stage 1-2)

## Problem Statement

In the GLM-5-FP8 B200 TP=8 campaign (session `190b247b`), the ammo-researcher spent ~2 hours attempting nsys profiling and only succeeded for BS=1. BS=8 and BS=32 profiling data was never captured, leaving the debate with single-batch-size grounding. The bottleneck analysis was incomplete, and a MoE GEMV optimization that regressed at BS=8 was not caught during debate.

### Root Cause Analysis

The researcher subagent's JSONL transcript reveals:

1. **The researcher read `nsys-profiling-guide.md` but only §3.1B** (two-step delimited capture). It **never consulted §3.9** (scaling limits, pre-profiling probe, `kernels_per_step` formula).

2. **The `ammo-researcher.md` agent prompt references the guide generally but steers the agent to §3.1B and §3.3 specifically.** The "Profiling Strategy Selection" section cites "§3.1B and §3.3" by name. Section 3.9 — which contains the probe, the formula, and the overhead model — is never referenced directly. Additionally, the sweep command example in `ammo-researcher.md` omits `--nsys-output-len`, so agents copying the example get no OL control.

3. **The sweep script's `--nsys-profile` flag has no pre-profiling check.** It launches directly into full nsys capture without estimating whether it will succeed within a reasonable time budget.

4. **When nsys failed (RPC timeout at BS=8), the researcher had no structured fallback path.** It jumped to `--enforce-eager` (a parity violation caught by the stop hook), then spent another hour on ad-hoc manual nsys attempts before eventually succeeding with mitigations it discovered through trial and error (`output_len=1`, `--cudagraph-capture-sizes 1`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200`).

5. **The mitigations the researcher eventually found are exactly what §3.9's formula would have recommended.** For GLM-5-FP8: `kernels_per_step ≈ 78 layers × 15 ≈ 1,170`. With TP=8: effective events per step ≈ 1,170 × 8 = 9,360. Safe `nsys_OL = floor(20000 / 9360) = 2`. The formula would have told the researcher upfront: "OL=2 is the maximum safe value, and even that is marginal."

### What Exists Today

The `nsys-profiling-guide.md` §3.9 already contains:
- The `nsys_OL = min(32, floor(20000 / kernels_per_step))` formula
- A pre-profiling probe example (nsys at OL=2 to count kernels)
- An overhead model table (events → estimated wall time)
- Escape hatches (reduce OL, use `--cuda-graph-trace=graph`, two-pass approach)

But this guidance is:
- **Buried** in §3.9 of a 610-line document
- **Not directly referenced** by the `ammo-researcher.md` agent prompt (only the guide file is referenced, not §3.9)
- **Not automated** — the probe is a manual multi-step process
- **Not integrated** into the sweep script workflow

## Solution

Three deliverables that work together:

### 1. Standalone `scripts/nsys_probe.py` (~200 LOC)

A lightweight script that estimates nsys profiling cost before the real capture.

**Inputs:**
- `--artifact-dir` (required) — points to the artifact directory containing `target.json`
- `--probe-bs` (optional) — which batch size to probe with (default: smallest from `workload.batch_sizes`)
- `--prewarm-timeout-s` (optional, default: 1800) — pre-warm step timeout (must accommodate torch.compile on cold cache, which can take 10-15 min for large models)
- `--probe-timeout-s` (optional, default: 600) — nsys probe step timeout

**Flow:**

```
Step 1: Pre-warm (no nsys)
  - Runs `vllm bench latency` with OL=2, warmup=1, iters=1
  - Includes --cudagraph-capture-sizes [probe_bs] to minimize graph surface
  - Populates torch.compile + Triton autotuning caches on disk
  - Validates the model can load and run (catches --trust-remote-code, tokenizer issues)
  - Timeout: --prewarm-timeout-s (default 1800s — cold torch.compile for large
    MoE models can take 10-15 min; CUDA graph capture is in-memory only and
    takes ~12-15s for large models)
  - Reads bench.extra_args from target.json and passes them through

Step 2: Nsys probe (plain full-run capture, matching §3.9 pattern)
  - Runs nsys profile wrapping a short bench run:
    nsys profile \
      --trace=cuda \
      --sample=none \
      --cuda-graph-trace=node \
      --trace-fork-before-exec=true \
      -o {artifact_dir}/nsys/probe \
      vllm bench latency \
        --model {model_id} \
        --tensor-parallel-size {tp} \
        --max-model-len {max_model_len} \
        --cudagraph-capture-sizes {probe_bs} \
        --batch-size {probe_bs} \
        --input-len 64 \
        --output-len 2 \
        --num-iters-warmup 3 \
        --num-iters 1 \
        {extra_args from target.json}
  - Uses plain full-run capture (no --capture-range) — matches the
    existing §3.9 probe pattern exactly
  - Restricts --cudagraph-capture-sizes to [probe_bs] only to minimize
    CUDA graph memory pressure
  - Step 1 pre-warm ensures torch.compile/Triton caches are warm, so
    Step 2 skips recompilation (~18s instead of ~840s)
  - Timeout: --probe-timeout-s (default 600s)

Step 3: Parse kernel count
  - Runs `nsys stats --report cuda_gpu_kern_sum` on the probe output
  - Sums total kernel instances across all unique kernel names
  - Estimates kernels_per_decode_step using the §3.9 rough-division method:
    - Trace contains warmup (3 iters × ~3 steps each = ~9 steps) +
      benchmark (1 iter × 3 steps = 3 steps) ≈ 12 total steps
    - kernels_per_decode_step ≈ total_instances / 12
  - Cross-validates against heuristic: N_layers × 13-20 for MoE/hybrid,
    N_layers × 10-15 for standard transformers
  - If measured and heuristic differ by > 2x, warns about possible
    transient overhead inflation and uses the lower value

Step 4: Compute per-bucket AND total sweep estimates
  For each batch_size in workload.batch_sizes:
    safe_nsys_OL = min(32, floor(20000 / (kernels_per_decode_step * TP)))
    per_bucket_events = kernels_per_decode_step * TP * safe_nsys_OL
    per_bucket_time = lookup from overhead model table
    risk_level = GREEN (<5 min) | YELLOW (5-15 min) | RED (>15 min)

  Total sweep estimate (what actually determines if the sweep will succeed):
    total_events = kernels_per_decode_step * TP * safe_nsys_OL * num_buckets
    total_time = lookup from overhead model table using total_events
    total_risk = GREEN | YELLOW | RED

Step 5: Output
  - Prints color-coded table to stdout (per-bucket + total sweep row)
  - Writes {artifact_dir}/nsys/probe_results.json
```

**Output format (`probe_results.json`):**

```json
{
  "probe_time": "2026-03-23T12:00:00Z",
  "model_id": "zai-org/GLM-5-FP8",
  "tp": 8,
  "probe_bs": 1,
  "kernels_per_decode_step": 1170,
  "estimation_method": "total_instances / 12 (§3.9 rough division)",
  "heuristic_estimate": "78 layers × 15 = 1170",
  "per_bucket": {
    "1": {
      "safe_nsys_OL": 2,
      "estimated_events": 18720,
      "estimated_time_min": 4.5,
      "risk_level": "YELLOW"
    },
    "8": {
      "safe_nsys_OL": 2,
      "estimated_events": 18720,
      "estimated_time_min": 4.5,
      "risk_level": "YELLOW"
    },
    "32": {
      "safe_nsys_OL": 2,
      "estimated_events": 18720,
      "estimated_time_min": 4.5,
      "risk_level": "YELLOW"
    }
  },
  "total_sweep": {
    "num_buckets": 3,
    "total_events": 56160,
    "estimated_time_min": 13.5,
    "risk_level": "YELLOW"
  },
  "suggested_sweep_args": {
    "--nsys-output-len": 2,
    "--nsys-num-iters": 1,
    "--nsys-timeout-s": 600
  }
}
```

**Console output example:**

```
=== nsys probe results ===
Model: zai-org/GLM-5-FP8 (TP=8)
Kernels/decode step: 1,170 (measured) vs 1,170 (heuristic: 78 layers × 15)

| Batch Size | Safe nsys_OL | Est. Events | Est. Time | Risk   |
|---:|---:|---:|---:|---|
| 1  | 2 | 18,720 | ~4.5 min  | YELLOW |
| 8  | 2 | 18,720 | ~4.5 min  | YELLOW |
| 32 | 2 | 18,720 | ~4.5 min  | YELLOW |
| TOTAL SWEEP (3 buckets) | — | 56,160 | ~13.5 min | YELLOW |

Suggested sweep args:
  --nsys-output-len 2 --nsys-num-iters 1 --nsys-timeout-s 600

WARNING: All batch sizes require nsys_OL <= 2 due to high kernel count × TP.
Decode profiling at OL=2 captures kernel identity and count accurately but
NOT duration scaling with KV length. For duration analysis, use targeted ncu
on specific kernels (see nsys-profiling-guide.md §4).
```

**Failure modes:**

| Failure | Exit Code | Message |
|---|---|---|
| Pre-warm fails or times out (model can't load, torch.compile exceeds timeout) | 2 | Error details + suggestions (e.g., "add --trust-remote-code to bench.extra_args in target.json", or "increase --prewarm-timeout-s for cold-cache torch.compile") |
| Probe nsys hangs (timeout) | 3 | "Model too heavy for --cuda-graph-trace=node at OL=2. Try: (1) reduce --cudagraph-capture-sizes further, (2) use torch.profiler for kernel ID, (3) use --cuda-graph-trace=graph (loses per-kernel detail, see §3.6)" |
| nsys not installed | 1 | "nsys not found on PATH" |
| Probe succeeds but all BS are RED | 0 | Normal output with warnings |

### 2. Guidance Changes

#### `ammo-researcher.md` — Two additions

**Addition 1** (under "E2E Baseline & Profiling Execution", before the sweep script example, AND update the sweep command example to include `--nsys-output-len`):

```markdown
**Pre-profiling probe (REQUIRED for TP > 1 or models > 10B params)**:

Before running `--nsys-profile`, estimate profiling cost:

    python .claude/skills/ammo/scripts/nsys_probe.py --artifact-dir {artifact_dir}

This takes ~5-15 minutes and outputs per-BS risk estimates with suggested
`--nsys-output-len`, `--nsys-num-iters`, and `--nsys-timeout-s` values.
See `references/nsys-profiling-guide.md` §3.9-3.10 for the theory.

For small TP=1 models (< 10B params), the probe is optional — nsys
profiling at default settings rarely has issues.
```

Updated sweep command example (replaces the existing example):
```bash
# Combined E2E baseline + nsys profiling (default for Stage 1):
python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --nsys-profile \
  --nsys-output-len {probe_suggested_OL}
```

**Addition 2** (new section after "Steady-State vs Transient Classification"):

```markdown
## When nsys Profiling Fails

If nsys `--cuda-graph-trace=node` fails or hangs for a batch size, follow
this escalation hierarchy:

1. **Reduce `--nsys-output-len`** to the probe's suggested value (or lower)
2. **Restrict `--cudagraph-capture-sizes`** to `[target_bs]` only
3. **Skip nsys for that BS** — document "BS=N profiling unavailable" in
   bottleneck_analysis.md
4. **NEVER fall back to `--enforce-eager`** for profiling

If a batch size has no profiling data, flag it explicitly:

> WARNING: No nsys profiling data for BS={N}. Debate proposals targeting
> this batch size lack empirical grounding for kernel-level claims.

If the probe itself times out at OL=2, the model may be too heavy for
`--cuda-graph-trace=node` entirely. In that case:
- Use `torch.profiler` for lightweight kernel identification
- Or use `--cuda-graph-trace=graph` (loses per-kernel detail inside CUDA
  graphs — see nsys-profiling-guide.md §3.6 for caveats)
- Document all methodology caveats prominently in bottleneck_analysis.md
```

#### `nsys-profiling-guide.md` — New section §3.10

After §3.9 "Scaling limits of --cuda-graph-trace=node":

```markdown
### 3.10 Profiling Decision Tree

Before attempting nsys profiling on models with TP > 1 or > 10B params,
run the probe script to estimate cost:

    python scripts/nsys_probe.py --artifact-dir {artifact_dir}

Decision tree based on probe results:

    Probe succeeds?
    ├── YES: All BS green/yellow?
    │   ├── YES → Run --nsys-profile with suggested --nsys-output-len
    │   └── NO (some BS red) →
    │       ├── Use suggested --nsys-output-len (auto-reduces for expensive BS)
    │       └── OR skip nsys for red BS, document the gap
    └── NO: Probe timed out at OL=2?
        ├── Restrict --cudagraph-capture-sizes to [1] only, retry probe
        ├── If still fails → use torch.profiler for kernel identification
        └── Document all caveats in bottleneck_analysis.md

For small TP=1 models (< 10B params), the probe is optional — proceed
directly to --nsys-profile with default settings.

If nsys profiling fails AFTER the probe passed (unexpected), run the probe
as a diagnostic to compare expected vs actual behavior.
```

#### `SKILL.md` — One-line addition

In the "Profiling strategy selection" paragraph under Stages 1-2, the existing text reads:

> For TP > 1 or models > 10B params, the lead should instruct the researcher to use two-step delimited capture. The researcher handles this automatically when using the sweep script with `--nsys-profile` [...]

Insert after "The researcher handles this automatically when using the sweep script with `--nsys-profile`":

> The lead should also instruct the researcher to run `scripts/nsys_probe.py` first to estimate profiling cost and determine safe `--nsys-output-len` values. See `references/nsys-profiling-guide.md` §3.10.

### 3. Trivial Fixes (out of scope for this spec, but noted)

These are simple fixes identified during the GLM-5-FP8 session analysis that should be done separately:

- **`--trust-remote-code` propagation**: The sweep script reads `bench.extra_args` from `target.json`. If the model requires `--trust-remote-code`, it must be in `extra_args`. The `new_target.py` scaffold script should auto-detect this from `tokenizer_config.json` when possible, or at minimum add a comment in the generated `target.json` noting the requirement.
- **`VLLM_RPC_TIMEOUT` documentation**: The `nsys-profiling-guide.md` should mention that nsys instrumentation overhead can cause `VLLM_RPC_TIMEOUT` (default 10s) to fire before `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (default 300s), and that increasing `VLLM_RPC_TIMEOUT` (e.g., `VLLM_RPC_TIMEOUT=600000`) may be needed for heavily-instrumented runs.

## Design Decisions

### Why a standalone script (not integrated into the sweep)?

1. **Agent control**: The probe outputs recommendations; the agent decides. This matches the user's preference for "warn and suggest, agent decides."
2. **Separation of concerns**: The sweep script is already 1,689 lines. The probe is a distinct pre-flight check.
3. **Reusable diagnostic**: The probe can be run independently when nsys fails unexpectedly, not just as a precursor to the sweep.
4. **No unnecessary overhead**: Small TP=1 models skip the probe entirely.

### Why plain full-run capture for the probe (not --capture-range)?

The probe uses a plain `nsys profile` wrapping a short bench run — the exact same pattern as the existing §3.9 example. This is simpler and avoids diverging from the known-working §3.9 approach. The `--capture-range=cudaProfilerApi` + `--profile` path (used by the sweep script's nsys mode) calls `cudaProfilerStart/Stop` via different code paths than the sweep's direct `torch.cuda.cudart().cudaProfilerStart()` — using it in the probe would create a subtle methodology difference.

The downside of full-run capture is that the trace includes warmup and one-time costs. The §3.9 rough-division method (total instances / ~12) accounts for this empirically. The pre-warm step ensures torch.compile and Triton caches are already warm, so the trace has minimal JIT overhead and the division-by-12 estimate is reliable.

### Why OL=2 for the probe?

- Keeps the probe fast (~2-5 min nsys time) even on the largest models
- Gives 3 decode steps in the profiled iteration (including warmup steps), which is sufficient for the rough-division kernel count method
- OL=1 would also work but gives fewer decode steps, making the division less stable

### Why not enforce a hard time budget?

The user specified "no hard budget, but warn at thresholds." The probe's risk levels (green/yellow/red) provide the warning; the agent or orchestrator makes the call on whether to proceed or skip.

### Why require the probe only for TP > 1 or > 10B?

- Small TP=1 models (< 10B) have low `kernels_per_step` (~200-400), yielding `nsys_OL = 32` (capped). Nsys profiling at OL=32 takes minutes and rarely fails.
- Large TP>1 models have high `kernels_per_step × TP`, making nsys profiling risky. These are exactly the models where the GLM-5-FP8 failures occurred.
- The probe itself costs ~5-15 minutes (model load + pre-warm + nsys at OL=2). For small models where profiling takes ~5 minutes, this triples the time for zero benefit.

### Why the §3.9 rough-division method for kernel counting?

The probe could attempt a more sophisticated method (e.g., isolating decode-step kernels by instance count patterns, or using NVTX ranges). However:
- The §3.9 rough-division method (total instances / ~12) is already validated in the guide
- It's simpler to implement and doesn't require NVTX instrumentation
- The cross-validation against the heuristic (N_layers × 13-20) catches gross errors
- The result feeds into a floor() operation and risk-level thresholds, not precision-sensitive calculations — a 20% error in kernel count still produces the correct `nsys_OL` recommendation

## Testing

The probe script should be tested against:

1. **Small model (TP=1, < 10B)**: Verify probe completes in < 5 min, all BS green, suggested OL = 32.
2. **Large MoE model (TP=8, > 100B)**: Verify probe identifies high kernel count, suggests OL=2-4, flags some BS as yellow/red. Total sweep row shows combined risk.
3. **Probe timeout scenario**: Verify graceful exit with code 3 and helpful message when nsys hangs at OL=2.
4. **Model load failure**: Verify graceful exit with code 2 when pre-warm fails (e.g., missing `--trust-remote-code`) or times out (cold-cache torch.compile exceeds timeout).
5. **Heuristic cross-validation**: Verify that measured kernel count and heuristic agree within 2x for known model architectures.

## Files Changed

| File | Change Type | Description |
|---|---|---|
| `scripts/nsys_probe.py` | New | Standalone probe script (~200 LOC) |
| `agents/ammo-researcher.md` | Edit | Add mandatory probe step, fallback hierarchy, update sweep command example to include `--nsys-output-len` |
| `references/nsys-profiling-guide.md` | Edit | Add §3.10 decision tree |
| `SKILL.md` | Edit | Add probe instruction in orchestrator's profiling strategy selection |
