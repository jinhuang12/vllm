# Validation Investigation Playbook

Use when Stage 5 gates fail.

## Hard limits

Per failure mode:
- max 3 hypothesis cycles
- max 2 deep kernel-analysis cycles

If still blocked: create blocker report and escalate.

## Common prerequisites

- Reproduce with same parity knobs and workload buckets.
- Confirm baseline vs candidate comparability.
- In parallel runs, isolate the failing candidate and keep unaffected candidates running.
- Collect minimum artifacts:
  - `artifact_bundle.json`
  - `constraints.md`
  - `optimization_plan.md`
  - `implementation_notes.md`
  - `validation_results.md`
  - relevant logs/traces

## 5.1 Correctness failure

Questions:
- Where does first divergence occur?
- Is divergence deterministic?
- Does it depend on shape/bucket/runtime mode?

Actions:
- add targeted assertions for shape/dtype/dispatch path
- isolate first bad stage
- apply minimal fix with explicit success criteria

## 5.2 Local performance regression

Questions:
- Which kernels/paths regressed?
- Was incumbent path accidentally disabled?
- Is overhead launch-bound or kernel-bound?

Actions:
- compare baseline vs incumbent vs candidate decomposition
- run focused kernel analysis on top regression kernels
- adjust candidate boundary/implementation based on evidence

## 5.3 E2E below significance

Questions:
- Is optimized path active in real workload?
- Is measured local gain too small relative to hotspot share?
- Did time shift to non-target components?

Actions:
- verify activation and parity
- compute expected upper bound from hotspot share
- narrow envelope, re-scope candidate, or stop

## Blocker handoff

If blocked after limits:
1. write `{artifact_dir}/investigation/blocker.md`
2. for parallel runs, also write `{artifact_dir}/investigation/blocker.<candidate_id>.md`
3. include repro, attempted fixes, rejected hypotheses, and evidence links
4. update `artifact_bundle.json` status
