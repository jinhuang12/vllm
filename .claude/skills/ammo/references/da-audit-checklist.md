# Devil's Advocate Audit Checklist

This checklist is used by the DA (devil's advocate) embedded in the implementer's Stop hook and by the orchestrator during integration review.

## Validation Completeness

- [ ] `validation_results.md` contains Gate 5.1, 5.2, and 5.3 results with actual numeric measurements (not placeholders or TODOs)
- [ ] All kill criteria have definitive PASS/FAIL verdicts

## Baseline Citation

- [ ] `validation_results.md` cites "Stage 1 (not re-run)" or "Stage 1 baseline"
- [ ] Baseline numbers in `validation_results.md` match `{artifact_dir}/runs/` JSON files

## Production Parity

- [ ] No `TORCH_COMPILE_DISABLE=1`, `--enforce-eager`, or `VLLM_TORCH_COMPILE_LEVEL=0` in benchmark commands

## Amdahl's Law Sanity

- [ ] Actual E2E improvement is within 1.5x of expected (`f * (1 - 1/s)`)
- [ ] If violated: flag cross-track contamination or measurement error

## Cross-Track Awareness

- [ ] If other tracks have C++ changes and this track is Python-only, `.so` contamination risk is noted

## Kernel-to-E2E Coherence

- [ ] If kernel speedup > 1.1x but E2E improvement < 1%, investigate whether the optimized code path is executing during E2E

## E2E Output Paths

- [ ] E2E results are in structured sweep output paths (`{artifact_dir}/e2e_latency/json/` or `{artifact_dir}/tracks/{op_id}/`), NOT in ad-hoc paths like `/tmp/`. If results are in `/tmp/`, the implementer used raw `vllm bench latency` instead of the sweep script -- this is a gate failure.

## Scope Adherence

- [ ] Compare implemented scope against `debate/summary.md` winner specification. If any components from the plan were omitted, the implementer MUST have flagged this in `validation_results.md` with explicit rationale. Undisclosed descoping is a gate failure.
