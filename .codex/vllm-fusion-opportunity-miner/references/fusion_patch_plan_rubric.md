# Fusion/tuning patch plan rubric

Use this as the scoring + checklist format when turning profiler evidence into an implementation plan.

## Scoring (suggested)

- **Impact (0–5)**: how much GPU time it could plausibly save.
  - 5: >10% end-to-end win or top-1 hotspot
  - 3: 3–10% end-to-end win or top-5 hotspot
  - 1: <3% end-to-end or uncertain
- **Feasibility (0–5)**: how likely it is to implement quickly and robustly.
  - 5: local kernel fusion/tuning with stable shapes and existing infra
  - 3: requires moderate refactor / new kernel path
  - 1: large graph-level changes or invasive re-architecture
- **Risk (0–5)**: likelihood of correctness or maintenance issues (higher = riskier).
  - 5: numerics-sensitive, many shapes/dtypes, backend-specific, tricky graph capture
  - 3: some constraints / limited coverage
  - 1: low-risk micro-optimization

Suggested priority: **(Impact + Feasibility) - Risk** (higher is better).

## Evidence checklist

- Capture settings match production parity (same model, dtype, attention backend, CUDA graph + `torch.compile` settings).
- Warmup performed; steady-state window profiled (avoid including initialization / weight load).
- Nsight Systems export saved (`.nsys-rep` + `.sqlite`) alongside the benchmark command.
- Hot kernels and/or repeated chains are linked to vLLM code paths (file+symbol).

## Patch plan template

For each opportunity, include:

- **What**: describe the chain/op to fuse or tune
- **Where**: vLLM code pointers (`vllm/` + `csrc/`)
- **Why**: profiler evidence (kernel time + counts + chain context)
- **How**: proposed fusion boundary / tuning knob / new kernel path
- **Constraints**: shapes, dtypes, backend, graph capture assumptions
- **Validation**: correctness tests + benchmark commands + acceptance criteria

