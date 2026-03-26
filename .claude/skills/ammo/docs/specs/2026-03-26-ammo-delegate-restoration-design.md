# AMMO Delegate Restoration Design

## Problem

The agent restructuring (2026-03-25) removed `ammo-delegate.md` and replaced it with generic "Subagents" sections in champion definitions that tell champions to spawn ad-hoc Sonnet subagents via `Agent()`. These generic subagents have zero knowledge of:

- 17 AMMO reference files (nsys-profiling-guide.md, debate-rules.md, gpu-pool.md, etc.)
- 10+ helper scripts (gpu_reservation.py, nsys_probe.py, run_vllm_bench_latency_sweep.py, etc.)
- GPU pool reservation pattern
- Production parity constraints
- Domain conventions (evidence tiers, baseline provenance, cache sensitivity testing)

Champions must manually include all relevant context in every subagent prompt, which is error-prone and leads to subagents that miss critical constraints.

## Solution

Re-create `.claude/agents/ammo-delegate.md` as a context-rich custom agent type. Champions spawn via `Agent(subagent_type="ammo-delegate")` instead of generic `Agent()`. The agent definition front-loads all domain context so delegates are immediately productive.

## Delegate Agent Definition

### Identity
- **Name**: `ammo-delegate`
- **Model**: sonnet (cost-effective for research/benchmarking tasks)
- **Interaction**: Fire-and-forget subagent (no team membership, no SendMessage)
- **Spawned by**: ammo-champion (debate phase) and ammo-impl-champion (implementation phase)

### Responsibilities
- Profiling data extraction (parsing nsys/ncu exports, extracting kernel timings)
- Dispatch path tracing (following kernel calls from Python through vLLM to CUDA)
- Roofline and bandwidth calculations
- Codebase research (finding prior art, checking existing kernel patterns)
- Micro-experiment script writing and execution
- ncu profiling runs and result parsing
- Shape and layout computation (M/N/K, tile sizes, SMEM budgets)
- Running test scripts and collecting output
- Reading and summarizing reference files

### Explicitly NOT responsible for
- DA auditing (transcript monitor's job)
- Decision-making (champion interprets results)
- Team coordination (fire-and-forget, results return to champion context)
- Implementation decisions (champion owns kernel design)

### Context included in agent definition

**Environment rules**:
- `.venv` activation requirement
- No package installation (NEVER `pip install` or `uv pip install`)
- CUDA build commands (`cmake --preset release && cmake --build --preset release --target install`)

**GPU pool reservation** (full pattern):
```
CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus N) && CUDA_VISIBLE_DEVICES=$CVD <command>
```

**Production parity constraints**:
- CUDA graphs + torch.compile required in ALL measurements
- FORBIDDEN: `--enforce-eager`, `TORCH_COMPILE_DISABLE=1`, `VLLM_TORCH_COMPILE_LEVEL=0`

**All references** (listed with one-line descriptions so delegate reads the relevant ones per task):
- `debate-rules.md` — micro-experiment guidelines, cache sensitivity, baseline provenance
- `gpu-pool.md` — GPU reservation pattern and contention handling
- `fusion-feasibility-heuristics.md` — H1-H5 heuristics for fusion candidates
- `gpu-configs.md` — SMEM budgets, cooperative launch limits, TMA availability
- `optimization-techniques.md` — Full technique catalog (T1-T14, U1-U6)
- `code-templates.md` — C++ kernel patterns, MMA templates, tile configs
- `e2e-delta-math.md` — E2E improvement math
- `cudagraph-safety.md` — CUDA graph capture checklist
- `nsys-profiling-guide.md` — nsys commands, report exports
- `impl-track-rules.md` — worktree build rules, verdict thresholds
- `validation-defaults.md` — tolerances, gate definitions
- `e2e-latency-guide.md` — E2E latency methodology
- `agent-responsiveness-guide.md` — (EXCLUDED: delegate is fire-and-forget, no messaging)
- `debate-scoring-rubric.md` — scoring criteria
- `crossover-probing.md` — crossover point analysis
- `validator-troubleshooting.md` — common validation issues
- `kernel-benchmark-template.py` — benchmark script template

**All scripts** (listed with usage):
- `gpu_reservation.py` — GPU pool reserve/release
- `nsys_probe.py` — estimate profiling cost, determine safe output-len
- `run_vllm_bench_latency_sweep.py` — batch E2E benchmark runner (GPU-locked)
- `verify_phase1_baseline.py` — Stage 1->2 gate verification
- `verify_validation_gates.py` — Stage 5 gate verification
- `generate_validation_report.py` — structured reporting
- `gpu_status.py` — GPU reservation diagnostics
- `gpu_force_clear.py` — clear stale reservations
- `collect_env.py` — environment capture
- `transcript_filter.py` — transcript filtering for monitors
- `new_target.py` — artifact directory scaffolding

### Stage-specific context from champion

The delegate definition is phase-agnostic. Champions provide stage-specific context in the spawn prompt:

**Debate phase** (champion provides):
- Artifact directory path
- Which bottleneck/candidate they're investigating
- Specific task (e.g., "parse nsys trace and extract top-10 kernels by time")
- Which references are most relevant for this task

**Implementation phase** (impl-champion provides):
- Artifact directory path and worktree path
- op_id and optimization description
- Specific task (e.g., "run ncu on baseline kernel and report occupancy/BW")
- Target batch sizes from target.json

Example spawn from a debate champion:
```
Agent(
  subagent_type="ammo-delegate",
  run_in_background=True,
  description="Extract nsys kernel timings",
  prompt="""
  Parse the nsys sqlite export at {artifact_dir}/runs/baseline_bs8/nsys_trace.sqlite
  and extract the top-15 kernels by total GPU time. For each kernel, report:
  - Kernel name, total time (us), call count, avg time per call
  - Whether it's a GEMM, attention, or elementwise kernel

  The nsys sqlite export format is documented in references/nsys-profiling-guide.md.
  Artifact directory: {artifact_dir}
  """
)
```

Example spawn from an impl-champion:
```
Agent(
  subagent_type="ammo-delegate",
  run_in_background=True,
  description="Profile baseline kernel with ncu",
  prompt="""
  Run ncu on the baseline silu_and_mul kernel for shape M=8, N=11008, K=1.
  Report: SM utilization, achieved DRAM BW, register count, occupancy.

  Use GPU pool reservation. Target batch sizes: [1, 8, 32].
  Artifact directory: {artifact_dir}
  Worktree: {worktree_path}
  """
)
```

## Changes Required

### 1. Create `.claude/agents/ammo-delegate.md`
New file with the full agent definition described above.

### 2. Update `.claude/agents/ammo-champion.md`
Replace the "Subagents" section (lines 131-153) to:
- Change "Sonnet subagents via `Agent()`" to "delegates via `Agent(subagent_type="ammo-delegate")`"
- Add guidance that champions should provide stage-specific context in spawn prompts
- Keep the "What to delegate" / "What to keep" lists (they're good)
- Update the "How to spawn" section with the new pattern
- Remove "Remind subagents of GPU pool reservation rules and the `.venv` activation requirement" (line 152) — the delegate definition bakes this in, so manual reminders are unnecessary boilerplate

### 3. Update `.claude/agents/ammo-impl-champion.md`
Same changes as ammo-champion.md in its "Subagents" section (lines 61-83).
- Also remove "Remind subagents of GPU pool reservation rules and the `.venv` activation requirement" (line 83) — same reason

### 4. Update `.claude/skills/ammo/README.md`
- Re-add ammo-delegate to the agent table and file tree
- Also add `ammo-resolver` to the agent table and file tree (currently missing despite existing on disk — zero-cost fix while editing)

### 5. Update `.claude/skills/ammo/SKILL.md`
No changes needed — SKILL.md doesn't reference the delegate directly. The orchestrator spawns champions and monitors; champions spawn delegates themselves.
