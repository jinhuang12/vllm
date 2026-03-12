# Stage 3: Adversarial Debate Protocol

Champions independently propose optimization candidates from grounded Stage 2 data, then debate them adversarially. The main session acts as moderator. Stage 2 provides ONLY measured facts and physical ceilings — champions generate candidates and feasibility estimates themselves.

## Team Structure

- **Team name**: `ammo-debate-{component}-{model_short}-{hardware}`
  - Example: `ammo-debate-attention-llama70b-h100`
- **Champions**: 2-4 `ammo-champion` agents. Each reads the grounded bottleneck_analysis.md independently.
- Each champion is spawned with:
  - `name="champion-{N}"` (e.g., `champion-1`, `champion-2`)
  - `team_name` set to the team name above

## Delegation

When `state.json` has `debate.delegation.enabled: true`, each champion is assigned 1-N Sonnet-model delegate agents for research and micro-experiments. Delegates are pre-spawned as teammates in the same team — champions direct them via SendMessage.

### Team Composition with Delegates

```
Team: ammo-debate-{component}-{model_short}-{hardware}
├── champion-1 (Opus)
│   └── delegate-1a (Sonnet)
├── champion-2 (Opus)
│   └── delegate-2a (Sonnet)
└── champion-3 (Opus)
    └── delegate-3a (Sonnet)
```

Naming convention: `delegate-{champion_number}{letter}` (e.g., `delegate-1a`, `delegate-1b` for champion-1's first and second delegates).

The mapping is stored in `state.json` at `debate.delegation.champion_delegate_mapping`:
```json
{
  "champion-1": ["delegate-1a"],
  "champion-2": ["delegate-2a"],
  "champion-3": ["delegate-3a"]
}
```

### Communication Flow

1. **Lead broadcasts phase-start to champions only**: Include routing clarity — "Champions: advance to Phase 0. Delegates: await champion instructions."
2. **Champion → Delegate via SendMessage**: Champion assigns specific research tasks with structured output requirements.
3. **Delegate → Champion via SendMessage**: Delegate reports results with structured data (kernel name, f-values, methodology, measurements).
4. **Champion writes debate artifacts**: Only champions write to `debate/proposals/` and `debate/round_{N}/`. Delegates write to `debate/delegate_work/` only.

### Phase Scope for Delegates

| Phase | Delegate Role |
|-------|--------------|
| **Phase 0 (Proposals)** | Active — profiling data extraction, codebase research, roofline calcs, ISA inspection |
| **Phase A (Evidence)** | Idle unless champion assigns specific research |
| **Phase B (Critique)** | Idle unless champion assigns specific research |
| **Phase C (Rebuttal)** | Optionally active — gather counter-evidence for critiques received |

### Delegate Constraints

- **No GPU kernel benchmarks**: Roofline calculations, codebase research, and `ncu --query-metrics` (static) only. No kernel benchmarks that require GPU allocation — this avoids GPU contention with the async pipeline.
- **No vLLM source modifications**: Research and analysis only.
- **15-minute task timeout**: Champions should time-box delegate tasks. If a delegate exceeds the timeout, the champion proceeds with partial data.

### Delegate Artifacts

Delegates write results to:
```
{artifact_dir}/debate/delegate_work/
  delegate-1a_bottleneck_top3.md
  delegate-1a_roofline_attention.py
  delegate-2a_kernel_source_trace.md
  ...
```

Champions cite these in proposals: `[Source: delegate-1a, {path}]`

### Timeout Handling

If a delegate does not respond within the phase deadline:
1. Champion proceeds with its own analysis or partial delegate data
2. Champion notes in proposal: "Delegate research incomplete; used own analysis for [section]"
3. This is NOT a debate failure — champions must be self-sufficient

### Teardown

On TeamDelete (after winner selection), ALL agents are shut down — both champions and delegates. No special delegate cleanup needed.

### Without Delegation

If `debate.delegation.enabled` is `false` (default), the debate runs with champions only — identical to the current protocol. No delegates are spawned, no `delegate_work/` directory is created.

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
| **Kernel Code Scope** | Specific kernel files to create/modify, language (CUDA/Triton/CUTLASS), estimated LOC — demonstrates this is custom kernel work |
| **Micro-Experiment Cache Audit** | For BW-bound kernels (AI < breakeven): (1) Were both warm-cache and cold-cache times reported? (2) If warm/cold > 1.5x, was cold-cache speedup used for E2E? For fusion proposals: was the fused kernel tested under production L2 conditions (data footprint matching pipeline working set)? |

**CRITICAL**: `kernel_speedup` estimates MUST come from the champion's own micro-experiment. Stage 2 does not provide speedup estimates.

### Proposal Eligibility Gate (Lead)

After Phase 0 submissions, the lead checks each proposal against the **Custom Kernel Mandate** before any debate begins:

- **Pass**: Proposal involves writing new or substantially modifying existing CUDA/Triton/CUTLASS kernel code.
- **Reject**: Config-only changes, flag-flipping, parameter tuning, or no kernel code scope described.

**Rejection action**: Message the champion to revise with a kernel-based proposal. If no compliant revision is submitted, the candidate is eliminated.

**Non-compliant proposals MUST NOT advance to Round 1.**

### Diversity Check (Lead)

After the eligibility gate, the lead reviews proposal diversity:

1. **f-value source check**: For each proposal, check whether the champion used `f_decode` (from the per-decode-step breakdown) or `f_total` (from the full trace). If the target kernel isn't in the decode breakdown, note this — the champion may be targeting prefill latency intentionally, or may have used a misleading f-value.

2. **Component diversity**: If all proposals target the same component, the lead should consider whether the debate will produce useful differentiation. If not, the lead may ask one champion to explore the next-highest-`f_decode` component as an alternative. The goal is at least 2 distinct target components among the eventual winners to reduce portfolio risk.

## Round Structure

Normal minimum: **2 rounds**. With convergence shortcut: **1 round** (see above). Maximum: **4 rounds**. Each round has three sequential phases.

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
| Kernel-level benchmarks | MUST use CUDA graph capture for both baseline and candidate kernels. Raw CUDA event timing without graph capture is INVALID for kernel speedup claims. See validation-defaults.md Kernel-Level Benchmark Requirements. |

### Forbidden

| Experiment Type | Reason |
|----------------|--------|
| Full-model benchmarks | Too slow, belongs in Stage 5 |
| vLLM source modifications | Belongs in Stage 4 |
| Model weight downloads | Too slow, too large |
| Any experiment >2 min | Blocks debate progress |
| Kernel benchmarks without CUDA graph capture | Inflates/deflates results due to launch overhead asymmetry (see OP-001 postmortem) |

### Cache-Sensitivity Requirements (BW-Bound Kernels)

For kernels identified as bandwidth-bound (AI < breakeven threshold), micro-experiments MUST report:
1. Loop-warmed time (100+ iterations on same tensors)
2. Cold-cache time (single iteration after L2 flush or fresh random tensors)

If the warm/cold ratio exceeds 1.5x, the speedup is cache-dependent. Use the cold-cache result for E2E projections and flag this in the proposal's feasibility math.

### Fusion-Specific Cache Testing

For proposals that fuse multiple kernels into one, the above requirements are necessary but not sufficient:

1. **Pipeline working set check**: Estimate total per-iteration working set (num_layers x per_layer_state). If this exceeds 2x the GPU's L2 cache, isolated benchmarks on small tensors overstate the fused kernel's benefit.
2. **L2-busting methodology**: Test the fused kernel with chained distinct data totaling > 2.5x L2 cache size, forcing DRAM streaming. This simulates production L2 competition.
3. **Report both**: Report speedup under (a) isolated warm-cache and (b) L2-busted cold conditions. If (a)/(b) > 1.5x, the E2E estimate MUST use the cold-cache speedup.

## Teardown

After winner selection:

1. Send `shutdown_request` to all champion agents.
2. Execute `TeamDelete` to clean up the debate team.

## Artifact Structure

### Round 1 (or single-pass campaigns)

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

### Campaign Round 2+ (scoped paths)

For campaign rounds beyond the first, debate artifacts must use campaign-round-scoped paths to avoid overwriting previous rounds' evidence:

```
{artifact_dir}/debate/campaign_round_{N}/
  summary.md
  proposals/
    champion-1_proposal.md
    ...
  round_1/          (debate round within this campaign round)
    ...
  round_2/
    ...
  micro_experiments/
    ...
```

The debate gate hook enforces this: round 2+ debates that use the legacy `debate/` path are blocked.

Note: "round_1" inside the campaign round directory refers to **debate rounds** (argument/critique/rebuttal cycles). The outer "campaign_round_{N}" refers to **campaign rounds** (profiling cycles).

## Re-validation After Re-profiling

When a debate completes while implementation is ongoing, winners are placed in `campaign.pending_queue`. After re-profiling (triggered by a shipped candidate), each queued winner must be re-validated before entering implementation:

1. Check if the target kernel still appears in the updated `bottleneck_analysis.md`.
2. Recalculate expected E2E impact using new f-values from the updated profiling data.
3. If `new_f × kernel_speedup < 1%` E2E improvement: discard (not worth implementing).
4. If still viable: candidate proceeds to implementation in the next available slot.

This is a feasibility recheck, NOT a full re-debate. The debate evidence (micro-experiments, kernel analysis) remains valid — only the component share `f` has changed due to the shifted bottleneck landscape.
