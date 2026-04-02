# Implementation Track Rules

Agent-facing rules for Stages 4-5 parallel implementation tracks. Both the impl-champion and impl-validator must follow these rules.

## Worktree Build Rules

| Change Type | Required Action | Time |
|-------------|----------------|------|
| **Pure Python** (model code, Triton kernels, configs) | Edit, test, commit. **No rebuild.** | Immediate |
| **C++ kernel** (csrc/ changes) | `cmake --preset release && cmake --build --preset release --target install` | ~5-55s (ccache) |

Only the champion compiles. The validator never runs cmake. The validator only executes against committed, compiled code.

## Source Modification Rules

- Only the champion modifies source files (`csrc/`, `vllm/`, etc.).
- The validator reads files and writes to `{artifact_dir}/tracks/{op_id}/validator_prep/` only.
- Champion has GPU priority. The validator coordinates via SendMessage before GPU-intensive work (ncu profiling, E2E sweeps). The champion signals when it needs the GPU.

## Independent Validation Principle

When validating, the validator writes its OWN correctness tests and benchmarks from the **optimization plan and debate summary** — not from the champion's scripts or implementation. This is the structural guarantee against reward hacking (cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation).

The validator can know everything about the codebase from its support work and still write unbiased validation tests, as long as tests are derived from what the optimization SHOULD do (the plan) rather than what it DOES do (the implementation).

## Two Layers of Verification

Each track undergoes two layers of verification:

```
Layer 1: Independent Validator (Sonnet)
  Writes OWN synthetic correctness tests (Gate 5.1a only)
  Reports raw correctness results — no interpretation

Layer 2: Champion Review
  Evaluates E2E results against min_e2e_improvement_pct threshold
  Cross-checks Gate 5.2 numbers against own smoke-test
  Writes final validation_results.md with evidence chain
```

## Handling Validation Failures

When the validator reports a gate failure:

1. Validator reports failure details to champion (SendMessage with full error context)
2. Champion diagnoses root cause
3. Champion fixes implementation, recompiles if needed, commits
4. Champion messages validator directly for re-validation with new commit SHA
5. Validator re-runs Gate 5.1a from scratch with fresh independent tests

The validator writes new tests each re-validation cycle — the champion cannot "fix" by influencing the test methodology.

## GATING_REQUIRED Workflow

> **This is the canonical definition.** Other files reference this section.

When per-BS verdicts show mixed results (some PASS + some REGRESSED), the track enters GATING_REQUIRED:

1. Validator reports per-BS verdict table to champion
2. Champion evaluates gating feasibility (is the dispatch site compatible with a gating mechanism?)
3. If feasible: champion requests validator to run crossover probing benchmarks
4. Validator runs kernel sweep + E2E confirmation per `crossover-probing.md`, reports probe results
5. Champion implements gating mechanism per `code-templates.md` dispatch decision tree
6. Champion registers env var in `vllm/envs.py`: `VLLM_{OP_NAME}=0`
7. Champion commits gated implementation
8. Champion requests validator to re-validate gated version
9. Validator re-validates at all BS — all must be PASS or NOISE
10. If re-validation passes: verdict = `GATED_PASS`. If fails: verdict = `FAIL`.

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
