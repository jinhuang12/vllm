# Implementation Track Rules

Agent-facing rules for Stages 4-5 parallel implementation tracks. Both the impl-champion and impl-validator must follow these rules.

## Worktree Build Rules

| Change Type | Required Action | Time |
|-------------|----------------|------|
| **Pure Python** (model code, Triton kernels, configs) | Edit, test, commit. **No rebuild.** | Immediate |
| **C++ kernel** (csrc/ changes) | `cmake --preset release && cmake --build --preset release --target install` | ~5-55s (ccache) |

Only the champion compiles. The kernel validation sub-agent never runs cmake — it only executes against committed, compiled code.

## Source Modification Rules

- Only the champion modifies source files (`csrc/`, `vllm/`, etc.).
- The kernel validation sub-agent reads files and writes to `{artifact_dir}/tracks/{op_id}/validator_tests/` only.
- The sub-agent runs sequentially before the champion's E2E sweep — no GPU coordination needed.

## Independent Validation Principle

The kernel validation sub-agent writes its OWN correctness tests and benchmarks from the **optimization plan and debate summary** — not from the champion's scripts or implementation. This is the structural guarantee against reward hacking (cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation).

The sub-agent derives tests from what the optimization SHOULD do (the plan) rather than what it DOES do (the implementation).

## Champion-Owned Validation

The champion owns all Stage 5 validation, with a sub-agent for kernel-level gates:

```
Kernel-Level (Sub-Agent):
  Gate 5.1a: Independent kernel correctness tests
  Gate 5.2: Kernel speedup benchmark under production parity

E2E-Level (Champion):
  Gate 5.1b: Sweep --verify-correctness
  Gate 5.3a: Sweep --nsys-profile (kernel proof)
  Gate 5.3b: Sweep E2E latency (per-BS verdicts)
  Writes final validation_results.md with evidence chain
```

## Handling Validation Failures

When the kernel validation sub-agent returns a gate failure:

1. Sub-agent returns failure details to champion (via Agent tool return value)
2. Champion diagnoses root cause
3. Champion fixes implementation, recompiles if needed
4. Champion completes Self-Validation Gate checklist (root cause reasoning, smoke test, fix-attempt counter)
5. Champion commits and spawns a fresh kernel validation sub-agent for re-validation

Each sub-agent invocation is independent — the champion cannot "fix" by influencing the test methodology.

## GATING_REQUIRED Workflow

> **This is the canonical definition.** Other files reference this section.

When per-BS verdicts show mixed results (some PASS + some REGRESSED), the track enters GATING_REQUIRED:

1. Sweep reports per-BS verdict table showing mixed results
2. Champion evaluates gating feasibility (is the dispatch site compatible with a gating mechanism?)
3. If feasible: champion spawns sub-agent for crossover probing benchmarks
4. Sub-agent runs kernel sweep + E2E confirmation per `crossover-probing.md`, returns probe results
5. Champion implements gating mechanism per `code-templates.md` dispatch decision tree
6. Champion registers env var in `vllm/envs.py`: `VLLM_{OP_NAME}=0`
7. Champion commits gated implementation
8. Champion spawns sub-agent for re-validation of gated kernel (5.1a + 5.2)
9. Champion re-runs sweep on gated code (5.1b + 5.3a + 5.3b) — all BS must be PASS or NOISE
10. If both kernel re-validation and sweep pass: verdict = `GATED_PASS`. If either fails: verdict = `FAIL`.

One gating attempt per track — no nested gating. The validator runs benchmarks and reports results but does NOT implement gating code.

## Stage 1 Baseline Reuse

All E2E comparisons use Stage 1 baseline numbers. Never run a new baseline during implementation.

Baseline data locations:
- Per-BS E2E latency: `{artifact_dir}/runs/baseline_bs{N}.json`
- Summary table: `{artifact_dir}/constraints.md` — "Baseline E2E latency" section
- Kernel breakdown: `{artifact_dir}/constraints.md` — "Baseline Truth Snapshot" section

## Track Constraints

These constraints apply to both champion and validator:

1. **All batch sizes.** Test every batch size in target.json. No exceptions. No cherry-picking.
2. **Production parity.** CUDA graphs + torch.compile in ALL measurements. NEVER use `--enforce-eager`, `TORCH_COMPILE_DISABLE=1`, or `VLLM_TORCH_COMPILE_LEVEL=0`.
3. **vLLM baseline.** Compare against vLLM's production kernel, NOT naive PyTorch.

## References

- `validation-defaults.md` — verdict thresholds (noise_tolerance_pct, catastrophic_regression_pct) and per-BS classification logic
- `crossover-probing.md` — crossover probing protocol for GATING_REQUIRED tracks
- `code-templates.md` — dispatch patterns and kernel templates
- `gpu-pool.md` — GPU reservation pattern
- `cudagraph-safety.md` — CUDA graph capture checklist
