# AMMO Multi-Agent Playbook (Stage 4+ Default)

Use this playbook after Stage 3 has selected top candidates.

## Objective

Run the top 3 feasible hypotheses in parallel with one subagent per hypothesis, isolated by git worktree, while the main agent acts as strict orchestrator.

## Preconditions

- `constraints.md` and `optimization_plan.md` are evidence-complete.
- Top candidates and differentiation contracts are explicit.
- `artifact_bundle.json` execution mode is `parallel_default`.

## Worktree protocol

1. Resolve baseline commit for all candidates.
2. Create one worktree per candidate:
   - `{artifact_dir}/parallel/worktrees/OP-001`
   - `{artifact_dir}/parallel/worktrees/OP-002`
   - `{artifact_dir}/parallel/worktrees/OP-003`
3. Create one branch per worktree:
   - `ammo/<run_id>/op-001`
   - `ammo/<run_id>/op-002`
   - `ammo/<run_id>/op-003`
4. Record worktree paths and branch names in bundle `parallel_candidates[]`.

## Subagent launch contract

For each candidate, pass a fixed contract:

- Candidate ID, ranked priority, expected profiler signature, kill criteria.
- Allowed file ownership scope and out-of-scope paths.
- Exact baseline and candidate command/env/flags contracts.
- Required local gate order:
  1. correctness
  2. kernel-time
  3. e2e
- Required outputs:
  - patch/diff summary
  - activation proof
  - gate evidence links
  - `subagent_evidence_manifest` JSON

## Orchestrator loop (strict auto-interrupt)

Repeat until all candidates are terminal (`pass|fail|blocked|aborted`):

1. Poll each subagent status.
2. Verify latest checkpoint against contract.
3. Interrupt immediately on:
   - out-of-scope edits,
   - missing differentiation evidence,
   - parity mismatch vs baseline contract,
   - unsupported claims without artifacts,
   - repeated stalls or invalid command usage.
4. Send corrective instructions and require explicit acknowledgement.
5. Log every intervention in `{artifact_dir}/parallel/orchestrator_log.md`.
6. Track repeated failure signatures in `artifact_bundle.json.validation_convergence.failure_signatures`.
7. If the same `(candidate, gate, command_fingerprint)` failure repeats twice consecutively, force candidate `blocked`.

## Triple-check acceptance policy

Never accept subagent pass claims directly.

For each candidate:
1. Validate evidence manifest completeness (`scripts/check_parallel_evidence.py`).
2. Re-run orchestrator acceptance gates independently.
3. Confirm no A/B invalidity and activation proof consistency.
4. Update machine-readable Stage 5 summary block in `validation_results.md`.

Only candidates with local-pass and acceptance-pass can be promoted.

## Multi-winner promotion policy

When more than one candidate passes:

1. Sort by rank + ROI tie-break.
2. Integrate sequentially (stack) in that order.
3. Re-run full acceptance gates after each addition.
4. Keep only non-regressing stack state.
5. Stop stacking at first regression and record decision rationale.

## Bounded-exhaustion loop

When `autonomy_mode=bounded_exhaustion`:

1. Complete one wave of top-3 parallel candidates.
2. Re-run Stage 2 mining and refresh ranked backlog.
3. Launch next wave from remaining feasible candidates.
4. Update `exhaustion_state` each wave:
   - `cycle`
   - `newly_mined_count`
   - `eligible_remaining_count`
   - `terminate_reason`
5. End only when terminate reason is not `pending`; no fixed time/cost cap is required.
6. Before final stop, run:
   - `scripts/check_parallel_evidence.py`
   - `scripts/check_validation_results.py`
   - `scripts/check_autonomy_completion.py`

## Required artifacts

- `{artifact_dir}/parallel/orchestrator_log.md`
- `{artifact_dir}/parallel/worktrees/*`
- `{artifact_dir}/parallel/evidence/OP-xxx.manifest.json`
- `{artifact_dir}/parallel/validation_records/*.json`
- updated `implementation_notes.md`
- updated `validation_results.md`
- updated `maintenance_decision.md`
