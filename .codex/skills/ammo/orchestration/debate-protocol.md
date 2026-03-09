# Stage 3: Adversarial Debate Protocol

Champions independently propose optimization candidates from grounded Stage 2 data, then debate them adversarially. Stage 2 provides measured facts and physical ceilings only; champions generate feasibility estimates from their own micro-experiments.

## Agent Structure

- Only the lead may spawn champion agents.
- Spawn exactly 3 `ammo-champion` roles.
- Store active agent IDs in `state.json.debate.agent_ids` when helpful.
- Every champion reads `bottleneck_analysis.md` independently and writes proposal artifacts.

## Debate Is Mandatory

Every run includes:

1. Phase 0 proposal generation
2. At least 1 full debate round

Default rounds: 2. Maximum: 4.

Convergence shortcut: if every champion independently converges on the same candidate in Phase 0 and all proposals include valid micro-experiment evidence, the lead may stop after 1 round.

## Phase 0: Independent Proposals

Each champion writes `{artifact_dir}/debate/proposals/{champion_id}_proposal.md` with these sections:

| Section | Contents |
|---|---|
| Candidate specification | Kernel/component, technique, and target envelope |
| Grounded data | Measured timings, component share `f`, BW utilization, constraints |
| Micro-experiment result | Mandatory empirical evidence |
| Integrated-path proof | Mandatory real vLLM dispatch/layer evidence |
| Feasibility math | Kernel speedup derived from the micro-experiment |
| Expected E2E impact | Explicit direct claim or proxy bound |
| Kill criteria | Thresholds that define failure |
| Kernel code scope | Files, language, and approximate code surface |

## Proposal Eligibility Gate (Lead)

Reject any proposal that fails the custom kernel mandate.

- Pass: new or substantially modified CUDA, Triton, or CUTLASS kernel work
- Reject: config-only changes, flag flips, parameter tuning, or missing kernel scope
- Reject: proxy-only candidates without integrated-path proof

Non-compliant proposals do not advance to Round 1.

## Round Structure

### Phase A: Evidence

Each champion writes `{artifact_dir}/debate/round_{N}/{op_id}_argument.md` with claim, evidence, feasibility math, and expected E2E impact.

### Phase B: Critique

Use round-robin critique assignment and write `{artifact_dir}/debate/round_{N}/{op_id}_critique_{target_id}.md`.

Critiques must identify:

1. Weaknesses in feasibility math
2. Overlooked CUDA graph, precision, or dispatch risks
3. Incorrect hardware or baseline assumptions

### Phase C: Rebuttal

Each champion writes `{artifact_dir}/debate/round_{N}/{op_id}_rebuttal.md` with counter-evidence, explicit concessions, or concrete mitigations.

## Conflicting Experimental Data

If two champions report contradictory measurements for the same shape with more than 1.5x disagreement, the lead must resolve the discrepancy before advancing the disputed claim.

Resolution options:

1. rerun a standardized CUDA-graphed micro-benchmark
2. compare exact methodology and privilege the production-parity result
3. cap feasibility at 5/10 if still unresolved

Do not send a disputed claim into Stage 4.

## Codex Orchestration Pattern

1. Broadcast phase instructions with `send_input`.
2. `wait` for all active champions.
3. Verify every required artifact exists before advancing.
4. Score candidates with `references/debate-scoring-rubric.md`.
5. Write winner selection to `{artifact_dir}/debate/summary.md`.

If a champion misses a required artifact deadline:

1. re-prompt once
2. respawn the same role once if still incomplete
3. mark the run `BLOCKED` if 3 complete lanes still cannot be produced

## Micro-Experiment Guardrails

### Allowed

- roofline calculations
- ISA inspection (`cuobjdump`, `ncu --query*`)
- tiny prototypes under 100 lines and 2 minutes
- single-kernel traces
- static memory-layout analysis
- CUDA-graphed kernel benchmarks

### Forbidden

- full-model benchmarks
- vLLM source modifications
- model downloads
- experiments over 2 minutes
- kernel timing claims without CUDA graph capture

### Cache-Sensitivity Requirement

Bandwidth-bound candidates must report warmed and cold-cache timings. If the ratio exceeds 1.5x, use the cold-cache result for E2E projections and flag the cache dependence.

## Winner Selection

After the final round, the lead:

1. scores every candidate
2. selects 2-3 winners
3. records rationale, known risks, and unresolved mitigations in `debate/summary.md`

## Teardown

After winner selection:

1. send final completion instructions to champions
2. close champion agents
3. clear `state.json.debate.agent_ids`
