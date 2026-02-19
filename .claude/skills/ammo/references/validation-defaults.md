# Validation Defaults and Reporting Standard (AMMO Stage 5)

Use this as the default policy for `{artifact_dir}/validation_results.md`.

This document defines:
- gate order
- default pass/fail criteria
- minimum statistical rigor
- required evidence and reporting format
- parallel top-3 validation ownership model

## Gate order (strict)

Do not reorder (both local and acceptance layers):

1. Correctness
2. Kernel-time (production parity)
3. E2E (production parity)

If a gate fails, set `status=needs_investigation`, run investigation flow, and do not promote candidate.

## Validation ownership model (parallel default)

For each candidate:

1. Subagent local gates
- subagent runs full gate sequence under parity controls.
- subagent must emit evidence manifest and artifacts.

2. Orchestrator acceptance gates
- orchestrator independently reruns the same gate sequence.
- orchestrator pass is required for promotion.

Subagent local pass without orchestrator acceptance pass is non-promotable.

## Production parity requirements

All baseline vs candidate comparisons must keep parity knobs identical unless the test explicitly studies that knob:

- CUDA graphs mode
- compile/runtime mode (`torch.compile` or adapter equivalent)
- TP/EP and sharding topology
- bucket mix and sequence lengths
- model/dtype/quant format
- warmup and measured iteration counts

Any parity mismatch invalidates the comparison until re-run.

## A/B validity precheck (mandatory before 5.2 and 5.3)

Promotion-grade kernel/E2E A/B requires a feature-differentiated candidate path.

Minimum proof:
- baseline and candidate command provenance is recorded,
- candidate includes at least one intentional dispatch/config/code-path delta vs baseline,
- activation evidence for candidate path is present (log or profiler signature), when applicable.
- parity signature and environment comparability proof are present.

Invalid A/B conditions:
- baseline and candidate commands/env/flags are effectively identical,
- candidate activation evidence is missing when the claim depends on path activation.

If invalid A/B is detected:
- set kernel-time and E2E gates to `blocked`,
- do not treat results as promotion evidence,
- return to Stage 4 to implement or wire a differentiated candidate path.

## Irrefutable evidence minimum (promotion-grade)

Each candidate must provide:
- exact baseline and candidate command strings,
- normalized env/flag diff proving path differentiation,
- activation proof (required signatures in profiler/log),
- reproducible artifact paths for correctness/kernel/e2e,
- pass/fail outcomes per gate with rationale.

Orchestrator must verify the same evidence independently before acceptance.

## 5.1 Correctness gate defaults

Use model-specific tolerances when available. If missing, start here and justify:

- FP32: `atol=1e-3`, `rtol=1e-3`
- BF16/FP16: `atol=1e-2`, `rtol=1e-2`
- FP8/int quantized: use model-specific test tolerances; do not rely on generic values

Mandatory checks:
- no NaN/Inf outputs
- expected shape and dtype parity
- deterministic behavior where baseline requires determinism
- edge-case coverage for changed semantics (routing/reduction/masking/quant boundaries)

Correctness pass condition:
- no unwaived semantic regression across validated buckets.

## 5.2 Kernel-time gate defaults

Measure GPU kernel-time under production parity for same validated buckets.

Default pass criteria:
- per-bucket non-regression vs baseline in envelope
- per-bucket non-regression vs incumbent in envelope
- optimized path activation is proven (expected dispatch/log/profiler signature)
- A/B validity precheck passed

Required decomposition:
- total delta
- win-source kernel/path delta
- new-overhead kernel/path delta
- residual explanation

If a deliberate tradeoff regresses one bucket:
- document explicitly,
- narrow envelope or dispatch policy,
- and reflect risk in ROI decision.

## 5.3 E2E gate defaults

Run E2E with identical parity knobs and realistic workload buckets.

Default target policy:
- primary buckets: positive aggregate improvement required
- worst bucket: no silent regression; regressions must be explicit and justified by envelope restriction

Use upper-bound math from `references/core/e2e-delta-math.md` to assess plausibility.
If expected gain is physically bounded below target, re-scope before further tuning.
If A/B validity precheck fails, E2E gate is `blocked` (not `pass`/`fail`).

## Statistical rigor defaults

Minimum recommendation:
- warmup iterations: >= 3
- measured iterations: >= 20 (or justify lower)
- report central tendency and spread (mean + p50/p95 if available)
- run enough repetitions to identify unstable results

If variance is high:
- report it explicitly,
- avoid hard ship decisions from noisy runs,
- collect additional evidence.

## Required reporting checklist

`validation_results.md` must include:

1. Repro details
- exact commands
- env vars and runtime flags affecting dispatch/graphs/compile

2. Environment
- GPU + driver + CUDA
- framework version/commit
- model + dtype/quant
- TP/EP topology

3. Correctness evidence
- tolerances and rationale
- max/mean error summary
- edge-case outcomes

4. Kernel-time evidence
- per-bucket baseline vs candidate vs incumbent
- decomposition and activation proof
- profiler artifact paths (`nsys`/`ncu`/logs)
 - explicit A/B validity statement (differentiated candidate path confirmed)

5. E2E evidence
- per-bucket baseline vs candidate deltas
- aggregate summary on primary buckets
- variance notes and run counts

6. Gate decisions
- explicit pass/fail per gate
- explicit local-vs-acceptance outcomes per candidate
- final recommendation input for Stage 6

7. Parallel summary (when applicable)
- candidate states (`pass|fail|blocked|not_run`)
- winner set and stack order
- regression cut-point if stacking stopped

## Waiver policy

A waiver must state:
- which gate/check is waived
- why waiver is necessary
- risk and rollback implications
- approving authority (user or operator)

Without explicit waiver, failed gates block promotion.

## Machine-readable summary block (mandatory)

`validation_results.md` must include this marker and JSON block:

~~~~markdown
<!-- AMMO_VALIDATION_SUMMARY_V1 -->
```json
{
  "candidates": [
    {
      "candidate_id": "OP-001",
      "local": {"correctness": "pass", "kernel": "pass", "e2e": "pass"},
      "acceptance": {"correctness": "pass", "kernel": "pass", "e2e": "pass"},
      "a_b_valid": true,
      "activation_proof": true,
      "acceptance_run_context": "clean-env-acceptance-run",
      "evidence_paths": {
        "correctness": ["path/to/correctness.json"],
        "kernel": ["path/to/kernel.nsys"],
        "e2e": ["path/to/e2e.json"]
      }
    }
  ]
}
```
~~~~

For autonomous runs, this block is validated by:
- `scripts/check_validation_results.py`

## Convergence guard (mandatory for autonomy)

- Track repeated failures per `(candidate, gate, command_fingerprint)`.
- If the same signature occurs twice consecutively, mark candidate `blocked`.
- Record this in `artifact_bundle.json.validation_convergence.failure_signatures`.
