---
name: vllm-fusion-opportunity-miner
description: Mine and rank GPU kernel fusion and tuning opportunities in vLLM inference using production-parity profiling (CUDA graphs + torch.compile where applicable). Use when Codex needs to (1) profile vLLM end-to-end (prefill and/or decode) with Nsight Systems, (2) extract kernel-time rankings and repeated kernel chains, (3) map hot kernels/chains back to vLLM code paths, (4) produce a prioritized, copy/paste patch plan for fusions/tuning with clear feasibility/risk scoring, and (5) generate reproducible artifacts for benchmarking and correctness validation before implementation.
---

# vLLM Fusion Opportunity Miner

Profile vLLM prefill/decode in production-parity mode and turn profiler evidence into a ranked, copy/paste patch plan for kernel fusions/tuning.

When executing `scripts/*`, run them via this skill directory (for example: `<skill_dir>/scripts/nsys_mine.py`).

## Two modes

- **Fast path (default)**: capture → mine → pick top 1–3 → draft patch plan.
- **Stage-gated investigation (recommended for larger work)**: keep a structured artifact directory (`state.json`, baseline snapshot, analysis outputs, patch plan) using `references/output-templates.md`.

## Quick start (fast path)

1) Capture a trace (store artifacts next to the exact command):

```bash
skill_dir="<path-to-vllm-fusion-opportunity-miner>"
mkdir -p artifacts/2026-01-02
"${skill_dir}/scripts/run_nsys_profile.sh" \
  --out-dir artifacts/2026-01-02 \
  --name decode \
  -- <YOUR_VLLM_BENCHMARK_COMMAND>
```

2) Export sqlite:

```bash
nsys export --type sqlite -o artifacts/2026-01-02/decode_sqlite artifacts/2026-01-02/decode.nsys-rep
```

3) Mine hot kernels + repeated chains + patch-plan scaffold:

```bash
"${skill_dir}/scripts/nsys_mine.py" \
  --sqlite artifacts/2026-01-02/decode_sqlite.sqlite \
  --out-dir artifacts/2026-01-02
```

Outputs:
- `artifacts/2026-01-02/nsys_mining.md` (human-readable ranking + patch-plan checklist)
- `artifacts/2026-01-02/nsys_mining.json` (machine-readable stats)

## Stage-gated investigation (artifacts + templates)

- Create an artifact directory that encodes model+hw+tp+date as a `run_id`.
- Write/run logs and progress to `state.json` and keep baseline/patch artifacts in the same directory.
- Use the copy/paste templates in `references/output-templates.md`:
  - `baseline_snapshot.md`
  - `fusion_opportunities.md`
  - `patch_plan.md`

## Workflow decision tree

- If the goal is **decode latency**: use a decode-heavy workload (steady-state generation) and profile after warmup.
- If the goal is **prefill throughput**: use a prefill-heavy workload (long prompts) and profile after warmup.
- If you need **code-path mapping**: enable NVTX ranges if available (vLLM/PyTorch), and always include `--trace=nvtx` in Nsight Systems.

## Step 1: Ensure production parity (before profiling)

- Use the same: model, dtype/quantization, attention backend, tensor/pipeline parallel settings, and batching as production.
- Enable compilation/capture if production uses it:
  - `--compilation-config` (`-O`) to control `torch.compile` behavior
  - `--cudagraph-capture-sizes` / `--max-cudagraph-capture-size` to control CUDA graph capture sizes
- Warm up first, then profile a steady-state window (avoid including initialization, weight load, graph capture setup).
- Record the exact command + env in `validation_results.md` (repo convention) and/or alongside artifacts.
- Read `references/production-parity.md` when parity details are unclear.

## Step 2: Capture with Nsight Systems

Prefer using the wrapper script so the command is saved next to the trace:

```bash
"${skill_dir}/scripts/run_nsys_profile.sh" \
  --out-dir <artifact_dir> \
  --name <prefill|decode> \
  -- <command...>
```

Tips:
- Keep traces short and targeted (seconds, not minutes) to reduce noise and analysis time.
- Include `cuda,nvtx` tracing; add more via `NSYS_ARGS` when needed.
- If CUDA graphs are enabled, consider `NSYS_ENABLE_CUDA_GRAPH_TRACE=1` (script will add node tracing if supported).
- Read `references/nsys-playbook.md` for delimited capture, multi-GPU tips, and report interpretation.

## Step 3: Export sqlite and mine rankings/chains

- Export sqlite:

```bash
nsys export --type sqlite -o <artifact_dir>/<name>_sqlite <artifact_dir>/<name>.nsys-rep
```

- Mine (sqlite → markdown + json):

```bash
"${skill_dir}/scripts/nsys_mine.py" \
  --sqlite <artifact_dir>/<name>_sqlite.sqlite \
  --out-dir <artifact_dir>
```

If the chain list is empty, tune thresholds:
- Decrease `--min-chain-count`
- Increase `--small-kernel-us` (the “small kernel” heuristic threshold)

Alternative (CSV exports → CSV analysis, good for spreadsheets / fallback):

```bash
"${skill_dir}/scripts/nsys_export_reports.sh" \
  <artifact_dir>/<name>.nsys-rep \
  <artifact_dir>/<name>_reports

"${skill_dir}/scripts/mine_kernel_chains.py" \
  --gpu-trace-csv <artifact_dir>/<name>_reports/cuda_gpu_trace.csv \
  --kern-sum-csv  <artifact_dir>/<name>_reports/cuda_gpu_kern_sum.csv \
  --out-dir       <artifact_dir>/analysis/<name>
```

Outputs:
- `<artifact_dir>/analysis/<name>/kernel_ranking.csv`
- `<artifact_dir>/analysis/<name>/kernel_chains.csv`

## Step 4: Map hotspots back to vLLM code paths

- Use `nsys_mining.md` to pick a kernel or a repeated chain.
- Search the repo for likely entry points:
  - `rg -n "<kernel substring>" csrc vllm`
  - If names look C++-mangled, try `c++filt` on the symbol name and search again.
- When the mapping is unclear, add NVTX ranges around suspected regions and re-profile.

## Step 5: Produce a prioritized patch plan (with feasibility/risk)

- Start from the “Copy/paste patch plan” section in `nsys_mining.md`.
- Score each item using `references/fusion_patch_plan_rubric.md`.
- Use `references/fusion-heuristics.md` for fusion pattern heuristics and stop-ship criteria.
- For each proposed fusion/tuning change, define:
  - acceptance criteria (latency/throughput)
  - correctness plan (tests + golden outputs)
  - benchmark commands (baseline vs candidate)
- If the best candidate is MoE-layer fusion, hand implementation off to `moe-monokernel-optimizer` and reuse your artifacts/evidence.

## Resources

- `scripts/run_nsys_profile.sh`: capture Nsight Systems traces with command logging.
- `scripts/nsys_mine.py`: extract top kernels and repeated kernel chains, emit `nsys_mining.md` + `nsys_mining.json`.
- `scripts/nsys_export_reports.sh`: export a small set of Nsight Systems CSV reports.
- `scripts/mine_kernel_chains.py`: mine kernel rankings/chains from `nsys stats` CSV reports.
- `references/fusion_patch_plan_rubric.md`: feasibility/risk scoring and patch-plan checklist.
- `references/nsys-playbook.md`: Nsight Systems capture + attribution guidance.
- `references/production-parity.md`: CUDA graphs + `torch.compile` parity checklist.
- `references/fusion-heuristics.md`: fusion heuristics + feasibility gates + stop-ship list.
- `references/output-templates.md`: `state.json` + baseline/opportunity/patch-plan templates.
