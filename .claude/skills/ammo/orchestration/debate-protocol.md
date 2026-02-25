# Stage 3: Adversarial Debate Protocol

Champions independently propose optimization candidates from grounded Stage 2 data, then debate them adversarially. The main session acts as moderator. Stage 2 provides ONLY measured facts and physical ceilings — champions generate candidates and feasibility estimates themselves.

## Team Structure

- **Team name**: `ammo-debate-{component}-{model_short}-{hardware}`
  - Example: `ammo-debate-attention-llama70b-h100`
- **Champions**: 2-4 `ammo-champion` agents. Each reads the grounded bottleneck_analysis.md independently.
- Each champion is spawned with:
  - `name="champion-{N}"` (e.g., `champion-1`, `champion-2`)
  - `team_name` set to the team name above

## Debate is Always Mandatory

There is no fast-track exception. Every run must go through at least Phase 0 (proposals) + 1 debate round.

**Convergence Shortcut**: If ALL champions independently propose the same candidate in Phase 0 AND all proposals cite micro-experiment evidence, the lead may reduce to 1 debate round instead of the normal 2.

## Phase 0: Independent Proposals

All champions execute **in parallel**. Each champion reads the grounded bottleneck_analysis.md (which contains ONLY measured facts and physical ceilings) and independently proposes 1-2 optimization candidates.

Each champion writes:

```
{artifact_dir}/debate/proposals/{champion_id}_proposal.md
```

Required sections in the proposal file:

| Section | Contents |
|---------|----------|
| **Candidate Specification** | What kernel/component to optimize and the proposed technique |
| **Grounded Data** | Cite measured timings, component share `f`, bandwidth utilization from bottleneck_analysis.md |
| **Micro-Experiment Result** | At least one empirical data point: roofline calc, ncu query, or tiny prototype (MANDATORY) |
| **Feasibility Math** | Expected kernel speedup derived from the micro-experiment, NOT from Stage 2 |
| **Expected E2E Impact** | `f × kernel_speedup` where both factors have provenance |
| **Kill Criteria** | What threshold defines failure for this candidate |

**CRITICAL**: `kernel_speedup` estimates MUST come from the champion's own micro-experiment. Stage 2 does not provide speedup estimates.

## Round Structure

Minimum **2 rounds** (1 if convergence shortcut applies), maximum **4 rounds**. Each round has three sequential phases.

### Phase A: Evidence Presentation

All champions execute **in parallel**.

Each champion writes:

```
{artifact_dir}/debate/round_{N}/{op_id}_argument.md
```

Required sections in the argument file:

| Section | Contents |
|---------|----------|
| **Claim** | One-sentence optimization thesis |
| **Evidence** | Profiling data, calculations, or references supporting the claim |
| **Feasibility Math** | FLOPs/bytes analysis, register pressure, occupancy estimates |
| **Expected E2E Impact** | Projected latency or throughput improvement with derivation |

Champions **may** run micro-experiments during this phase (see Micro-Experiment Guidelines below).

### Phase B: Critique

Round-robin assignment:

- Champion for OP-001 critiques OP-002
- Champion for OP-002 critiques OP-003
- ...
- Champion for OP-N critiques OP-001

Each champion writes:

```
{artifact_dir}/debate/round_{N}/{op_id}_critique_{target_id}.md
```

The critique **must** identify:

1. Weaknesses in the target's feasibility math
2. Overlooked risks (numerical stability, edge cases, memory pressure)
3. Incorrect assumptions about hardware behavior or kernel characteristics

### Phase C: Rebuttal

All champions execute **in parallel**.

Each champion responds to the critique they received and writes:

```
{artifact_dir}/debate/round_{N}/{op_id}_rebuttal.md
```

The rebuttal **must** do one or more of:

- Provide counter-evidence disproving the critique
- Concede valid points explicitly
- Propose concrete mitigation for acknowledged weaknesses

## Communication Flow

The main session moderates the debate using `SendMessage`:

1. **Broadcast** phase-start message to all champions (includes round number and phase identifier).
2. Each champion **messages main** upon phase completion with a short status line.
3. Main **waits** for all champions to report before advancing to the next phase.
4. After each complete round, main evaluates convergence criteria.

## Convergence Criteria

Stop early (before round 4) if **either** condition is met:

1. **Clear winners**: The top 2-3 candidates have no unaddressed critiques remaining, and all other candidates have conceded material weaknesses.
2. **Stagnation**: Round N+1 arguments substantially repeat round N with no new evidence or counter-arguments introduced.

## Winner Selection

After the final round:

1. Main session reads **all** debate artifacts across all rounds.
2. Scores each candidate per `references/debate-scoring-rubric.md`.
3. Selects **2-3 winners** to advance to Stage 4 parallel tracks.
4. Writes the decision to:

```
{artifact_dir}/debate/summary.md
```

The summary includes: per-candidate scores, rationale for selection, and any conceded weaknesses that Stage 4 implementation must address.

## Micro-Experiment Guidelines

### Allowed

| Experiment Type | Constraint |
|----------------|------------|
| Roofline calculations | Pure arithmetic, no GPU required |
| ISA inspection | `cuobjdump`, `ncu --query-metrics`, static analysis only |
| Tiny kernel prototypes | <100 lines of code, <2 min wall-clock execution |
| nsys single-kernel traces | One kernel invocation, existing binary only |
| Memory layout analysis | Static analysis of tensor shapes and strides |

### Forbidden

| Experiment Type | Reason |
|----------------|--------|
| Full-model benchmarks | Too slow, belongs in Stage 5 |
| vLLM source modifications | Belongs in Stage 4 |
| Model weight downloads | Too slow, too large |
| Any experiment >2 min | Blocks debate progress |

## Teardown

After winner selection:

1. Send `shutdown_request` to all champion agents.
2. Execute `TeamDelete` to clean up the debate team.

## Artifact Structure

```
{artifact_dir}/debate/
  summary.md
  proposals/
    champion-1_proposal.md
    champion-2_proposal.md
    champion-3_proposal.md
  round_1/
    champion-1_argument.md
    champion-1_critique_champion-2.md
    champion-1_rebuttal.md
    champion-2_argument.md
    champion-2_critique_champion-3.md
    champion-2_rebuttal.md
    ...
  round_2/
    ...
  micro_experiments/
    champion-1_roofline.py
    champion-2_ncu_query.txt
    ...
```
