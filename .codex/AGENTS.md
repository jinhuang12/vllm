# Codex instructions for vLLM MoE monokernel work

This repository includes a Codex workflow bundle under:

- `.codex/prompts/moe-monokernel-optimizer.md` (custom slash command prompt)
- `.codex/moe-monokernel-optimizer/**` (supporting references, assets, orchestration)

## Default behavior

When asked to optimize MoE via a “monokernel”:

1. Follow the 5-phase workflow in the `moe-monokernel-optimizer` prompt:
   constraints → plan → implementation → validation → integration.
2. Keep work resumable: maintain `state.json` and write artifacts in the chosen `ARTIFACT_DIR`.
3. Prefer correctness + reproducibility over speculative optimizations.

## Repo hygiene

- Avoid drive-by refactors. Touch only files needed for the kernel + integration.
- Prefer generating a patch file (`git diff > ARTIFACT_DIR/monokernel.patch`) over making commits unless explicitly requested.
- If you make substantial edits, run *some* verification (compile + a minimal runtime smoke test).

## Non-interactive constraints (codex exec / CI)

- Never start interactive programs (editors/pagers).
- Assume `stdin` may be closed; commands must not prompt.
- Write logs and intermediate notes into the artifact directory.

Recommended env guards for automation:

- `GIT_TERMINAL_PROMPT=0`
- `GIT_EDITOR=true`
- `PAGER=cat`

## Helpful repo navigation commands

- Find model MoE implementation:
  - `rg -n "MoE|MixtureOfExperts|FusedMoE|fused_moe" vllm/model_executor/models`
- Find kernel dispatch sites:
  - `rg -n "fused_moe|experts|router" vllm csrc`

## Profiling tools available

You may use (when requested):

- `nsys` for timelines (kernel launch gaps, overlap)
- `ncu` for kernel-level metrics (occupancy, TC utilization, memory bottlenecks)
- `torch.profiler` for end-to-end attribution

Always include the exact commands and key findings in `validation_results.md`.
