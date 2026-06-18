# Nsight Systems Profiling Guide

Stage 2 has one default profiling workflow:

1. Run a clean E2E baseline sweep with no profiler flags.
2. Run a short Nsight Systems node capture with the architecture-appropriate CUDA trace backend.
3. Mine the nsys report for kernel ranking, launch chains, per-rank/per-device skew, CUDA memcpy/NVLink activity, and timing shares.
4. Run targeted Nsight Compute only for physical-ceiling claims such as occupancy, achieved bandwidth counters, or SM utilization.

Do not run a profiling probe. Do not use profiler-contaminated latency as the official E2E baseline.

## Required Stage 1 Commands

Clean baseline, used for all speedup math:

```bash
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --target-json {artifact_dir}/target.json \
  --round {N} \
  --slot baseline \
  --labels baseline \
  --capture-golden-refs
```

Bounded selected-step nsys capture, used for bottleneck attribution on Hopper/Ampere and any platform where default CUDA tracing is stable:

```bash
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --target-json {artifact_dir}/target.json \
  --round {N} \
  --slot profiling \
  --labels baseline \
  --nsys-profile \
  --nsys-mode node \
  --nsys-capture-output-steps 2,50%,100% \
  --nsys-num-iters 1 \
  --nsys-timeout-s 1800
```

`--nsys-capture-output-steps 2,50%,100%` resolves against each workload bucket's `output_len`, then profiles short shape-equivalent windows that land on a genuine full-batch decode step (not a chunked-prefill chunk). For target step `k`, the sweep captures decode depth `input_len + k` (invariant of the window) by running `input_len + k - w_eff`, `output_len = w_eff`, and vLLM's CUDA profiler captures only the final worker step. `w_eff` is an effective capture window that the script floors child-wide to clear chunked prefill: `w_eff = max(--nsys-capture-window-output-len, max over (bucket, step) of ceil((input_len + step - requested_window) * batch_size / 16384) + 6)`, where `requested_window = --nsys-capture-window-output-len`. The `ceil(...)` term is the number of chunked-prefill worker-steps (chunk = 16384 tokens) the capture must arm past, evaluated at the shifted prompt length `input_len + step - requested_window` (longest at the deepest requested step). `--nsys-capture-window-output-len` (default 2) is therefore only a LOWER BOUND — the script auto-raises it as needed (e.g. 2 → 11 for an 8192-token prompt at batch 8, where the deepest step 512 gives `ceil((8192+512-2)*8/16384)+6 = 5+6 = 11`) and you cannot force the effective window below the floor. You normally do not set it. `--nsys-output-len` remains a horizon override for percentage resolution.

For Blackwell (B200/B300) runs, always use `--nsys-trace cuda-sw` to avoid hardware-tracing stalls under CUDA graph replay:

```bash
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --target-json {artifact_dir}/target.json \
  --round {N} \
  --slot profiling \
  --labels baseline \
  --nsys-profile \
  --nsys-mode node \
  --nsys-trace cuda-sw \
  --nsys-capture-output-steps 2,50%,100% \
  --nsys-num-iters 1 \
  --nsys-timeout-s 1800
```

This produces one `.nsys-rep` per selected depth and batch bucket. The trace
sidecar records both the synthetic capture shape and source shape:
`source_input_len`, `source_output_len`, `capture_output_step`,
`capture_window_output_len`, and `capture_target_output_len`.

Use the bounded fallback `--nsys-output-len 2 --nsys-num-iters 1` only if
selected-step capture fails; document that depth coverage was omitted.

Outputs:

```text
{artifact_dir}/rounds/{N}/sweeps/baseline/e2e_latency_results.json
{artifact_dir}/rounds/{N}/profiling/nsys/*.nsys-rep
```

`--slot baseline` plus any profiler flag is invalid. The sweep script enforces this guard so official timing cannot be contaminated by profiler overhead.

## Trace Backend Matrix

| Hardware family | Stage 2 trace backend | Notes |
|---|---|---|
| Blackwell B200/B300 (SM100/SM120) | `--nsys-trace cuda-sw` | Always use software tracing. Hardware tracing stalls under CUDA graph replay on Blackwell regardless of model type or TP size. |
| Hopper H100/H200 | default `--nsys-trace cuda` | Use normal CUDA tracing unless timeout/hang failure appears. |
| Ampere A100 | default `--nsys-trace cuda` | Use normal CUDA tracing unless timeout/hang failure appears. |
| Unknown NVIDIA target | default `--nsys-trace cuda` first | Switch to `cuda-sw` only after logs match the Blackwell-style replay/collective timeout failure. |

On Blackwell, the hardware event system for CUDA tracing interacts poorly with CUDA graph replay, causing long warmup/profiling stalls followed by RPC or NCCL watchdog timeouts. This affects all workloads (not just MoE or TP>1). `cuda-sw` still captures CUDA API/software activity and graph node attribution while avoiding that hardware-tracing path.

Keep `--nsys-mode node` for the default Stage 2 path because node mode is the ranking source for CUDA graph workloads.

`--nsys-capture-output-steps` accepts comma-separated integers and percentages.
Percentages resolve against `--nsys-output-len` when supplied, otherwise against
each bucket's workload `output_len`. Duplicate resolved steps are removed in
stable order. A step shallower than the effective capture window is still
captured — the script shifts `input_len` (`il_eff = input_len + step - w_eff`)
so the trace lands at the requested decode depth. The only step that is dropped
is one where `il_eff < 1` (the context is too short to host a steady-state
decode); that bucket/step logs a loud WARNING and the remaining steps continue.
To recover a dropped step, use a larger `input_len` or a shallower step —
lowering `--nsys-capture-window-output-len` will not help, because it is only a
lower bound that the script auto-raises to clear chunked prefill.

If the command fails:

- Check the supervisor log under `rounds/{N}/sweeps/profiling/logs/`.
- Confirm the hardware family. On Blackwell (B200/B300), confirm the command used `--nsys-trace cuda-sw`; on Hopper/Ampere, start from default `cuda`.
- Keep `--nsys-capture-output-steps 2,50%,100%` and `--nsys-num-iters 1` for the default capture. Use `--nsys-output-len 2 --nsys-num-iters 1` only when selected-step capture itself fails.
- Increase `--nsys-timeout-s` only if logs show useful forward progress.
- Reduce the profiled bucket set only as a last resort, and document the omitted buckets.

## Stage 2 Mining

Mine the existing nsys reports. Do not re-run the sweep just to analyze traces.

Minimum analysis:

- Export nsys reports to SQLite or stats tables.
- Rank kernels by total GPU time and count.
- Group kernel chains in timestamp order, not architecture order.
- Map top kernels to source paths or generated backends when possible.
- For TP > 1, compare all rank/device reports and report per-rank skew.
- Separate compute kernels from communication kernels and memcpy/P2P traffic.
- Compute `f_decode`, `decode_share_of_e2e`, and `f_e2e` using measured Stage 1 data.

Useful commands:

```bash
nsys stats --force-export=true --report cuda_gpu_kern_sum \
  {artifact_dir}/rounds/{N}/profiling/nsys/baseline_profile*.nsys-rep

nsys stats --force-export=true --report cuda_gpu_trace \
  {artifact_dir}/rounds/{N}/profiling/nsys/baseline_profile*.nsys-rep

nsys export --type sqlite --force-overwrite=true \
  --output {artifact_dir}/rounds/{N}/profiling/nsys/baseline.sqlite \
  {artifact_dir}/rounds/{N}/profiling/nsys/baseline_profile*.nsys-rep
```

Report approximate trace timings honestly. A value such as `~74 us` is valid when it comes from the nsys trace. It is not valid to infer kernel duration or ordering from source code or architecture diagrams.

## Optional Graph Diagnostics

`--nsys-mode graph` is optional diagnostic enrichment. Use it only when node-mode results leave a specific open question about graph structure, launch grouping, CUDA graph replay, or metadata available only in graph view.

Graph mode is not the ranking source for Stage 2. If node and graph mode disagree, use node-mode timing for bottleneck ranking and explain why graph mode was captured.

Graph diagnostic command:

```bash
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --target-json {artifact_dir}/target.json \
  --round {N} \
  --slot profiling \
  --labels baseline \
  --nsys-profile \
  --nsys-mode graph \
  --nsys-output-len 2 \
  --nsys-num-iters 1 \
  --nsys-timeout-s 1800
```

Add `--nsys-trace cuda-sw` to graph diagnostics on Blackwell (B200/B300) — same reasoning as node mode above.

## Targeted NCU

Nsight Compute is required before making any physical-ceiling claim:

- Occupancy or achieved occupancy.
- SM utilization or tensor-core utilization.
- Achieved memory bandwidth from hardware counters.
- Register pressure, shared-memory pressure, or stall-reason claims.
- "Kernel X can improve by at most Y%" where Y comes from a hardware ceiling.

Keep NCU narrow. Profile only the top kernels identified by nsys, with representative bucket sizes. Store results under:

```text
{artifact_dir}/rounds/{N}/profiling/ncu/
```

Stage 2 may rank bottlenecks from nsys without NCU. It may not claim a physical ceiling without NCU or an explicitly cited hardware spec and math.

## Valid Stage 2 Evidence

Acceptable:

- `rounds/{N}/profiling/nsys/*.nsys-rep`
- nsys stats/export tables derived from those reports
- sweep JSON from `rounds/{N}/sweeps/baseline/`
- targeted NCU CSV/report for hardware-counter claims
- source-code mapping used only to explain what a measured kernel is

Not acceptable:

- Profiler-run latency as official E2E timing.
- Architecture-inferred kernel chains without trace timestamps.
- Occupancy/bandwidth-counter claims without targeted NCU.
- Graph-mode timing as the primary bottleneck ranking source.

## Report Checklist

`rounds/{N}/mining/bottleneck_analysis.md` must include:

- Exact Stage 1 baseline and profiling commands.
- Artifact paths for every trace used.
- Top kernels/components by measured time.
- Per-rank/per-device comparison for TP/DP runs.
- `f_decode`, `decode_share_of_e2e`, and `f_e2e` tables.
- Technology Landscape entries for the top components.
- Any NCU-backed physical-ceiling claims with paths to the raw NCU artifacts.
