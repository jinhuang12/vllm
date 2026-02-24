---
name: vllm-fusion-opportunity-miner
description: Mine and rank GPU kernel fusion opportunities in vLLM inference using production-parity profiling (CUDA graphs + torch.compile). Use when Claude needs to (1) profile vLLM end-to-end (prefill/decode) with Nsight Systems, (2) extract kernel-time rankings and repeated kernel chains from sqlite exports, (3) map hot kernels/chains back to vLLM code paths, (4) produce a prioritized patch plan for fusions/tuning with feasibility/risk scoring, or (5) generate reproducible artifacts for GPU kernel optimization benchmarking.
---

# vLLM Fusion Opportunity Miner

Profile vLLM prefill/decode in production-parity mode and turn profiler evidence into a ranked patch plan for kernel fusions/tuning.

Run scripts via this skill directory: `<skill_dir>/scripts/<script>`.

## Two modes

- **Fast path**: capture → mine → pick top 1–3 → draft patch plan
- **Stage-gated** (larger work): structured artifact directory with `state.json`, baseline snapshot, analysis outputs — use `references/output-templates.md`

## Quick start (fast path)

```bash
skill_dir="<path-to-skill>"

# 1) Capture trace
mkdir -p artifacts/$(date +%Y-%m-%d)
"${skill_dir}/scripts/run_nsys_profile.sh" \
  --out-dir artifacts/$(date +%Y-%m-%d) \
  --name decode \
  -- <YOUR_VLLM_BENCHMARK_COMMAND>

# 2) Export sqlite
nsys export --type sqlite -o artifacts/$(date +%Y-%m-%d)/decode_sqlite \
  artifacts/$(date +%Y-%m-%d)/decode.nsys-rep

# 3) Mine hot kernels + chains
"${skill_dir}/scripts/nsys_mine.py" \
  --sqlite artifacts/$(date +%Y-%m-%d)/decode_sqlite.sqlite \
  --out-dir artifacts/$(date +%Y-%m-%d)
```

Outputs: `nsys_mining.md` (human-readable), `nsys_mining.json` (machine-readable).

## Workflow

### Step 1: Ensure production parity

Before profiling, match production exactly:
- Model, dtype/quantization, attention backend, TP/PP settings, batching
- Enable `torch.compile` / CUDA graphs if production uses them
- Warm up first, profile steady-state only (skip init/graph capture)
- Record exact command + env

See `references/production-parity.md` for details.

### Step 2: Capture with Nsight Systems

```bash
"${skill_dir}/scripts/run_nsys_profile.sh" \
  --out-dir <artifact_dir> --name <prefill|decode> \
  -- <command...>
```

- Keep traces short (seconds, not minutes)
- Include `cuda,nvtx` tracing; add more via `NSYS_ARGS`
- For CUDA graphs: `NSYS_ENABLE_CUDA_GRAPH_TRACE=1`

See `references/nsys-playbook.md` for advanced capture options.

### Step 3: Export sqlite and mine

```bash
nsys export --type sqlite -o <dir>/<n>_sqlite <dir>/<n>.nsys-rep

"${skill_dir}/scripts/nsys_mine.py" \
  --sqlite <dir>/<n>_sqlite.sqlite --out-dir <dir>
```

Empty chains? Tune `--min-chain-count` (lower) or `--small-kernel-us` (higher).

Alternative CSV workflow (good for spreadsheets):
```bash
"${skill_dir}/scripts/nsys_export_reports.sh" <n>.nsys-rep <n>_reports
"${skill_dir}/scripts/mine_kernel_chains.py" \
  --gpu-trace-csv <n>_reports/cuda_gpu_trace.csv \
  --kern-sum-csv <n>_reports/cuda_gpu_kern_sum.csv \
  --out-dir analysis/<n>
```

### Step 4: Map hotspots to code

- Use `nsys_mining.md` to pick a kernel/chain
- Search repo: `rg -n "<kernel substring>" csrc vllm`
- For mangled names: `c++filt <symbol>` then search
- Add NVTX ranges if mapping unclear, re-profile

### Step 5: Produce patch plan

- Start from "Copy/paste patch plan" in `nsys_mining.md`
- Score using `references/fusion_patch_plan_rubric.md`
- Apply heuristics from `references/fusion-heuristics.md`
- Define: acceptance criteria, correctness plan, benchmark commands

## Resources

| Resource | Purpose |
|----------|---------|
| `scripts/run_nsys_profile.sh` | Capture traces with command logging |
| `scripts/nsys_mine.py` | Extract top kernels/chains → md + json |
| `scripts/nsys_export_reports.sh` | Export CSV reports from nsys-rep |
| `scripts/mine_kernel_chains.py` | Mine from CSV (fallback) |
| `references/fusion_patch_plan_rubric.md` | Feasibility/risk scoring |
| `references/nsys-playbook.md` | Nsight Systems guidance |
| `references/production-parity.md` | CUDA graphs + compile parity |
| `references/fusion-heuristics.md` | Fusion patterns + stop-ship list |
| `references/output-templates.md` | Templates for structured artifacts |
