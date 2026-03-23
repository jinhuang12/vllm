# Stage 3: Adversarial Debate Protocol

Champions independently propose optimization candidates from grounded Stage 2 data, then debate them adversarially. The main session acts as moderator. Stage 2 provides ONLY measured facts and physical ceilings — champions generate candidates and feasibility estimates themselves.

## Team Structure

- **Team name**: `ammo-round-{round_id}-{model_short}-{hardware}` -- this is the **round team**, created once per round and reused for both debate (Stage 3) and implementation (Stages 4-5). During overlapped debate (round 2+ only), debate champions for round N+1 are spawned into the EXISTING round team from round N, alongside implementation agents. The debate uses a distinct `round_id` in its artifact paths (`debate/campaign_round_{N+1}/`) to avoid collisions.
  - Example: `ammo-round-1-llama70b-h100`
- **Champions**: 2-4 `ammo-champion` agents. Each reads the grounded bottleneck_analysis.md independently.
- Each champion is spawned with:
  - `name="champion-{N}"` (e.g., `champion-1`, `champion-2`)
  - `team_name` set to the round team name above

## Delegation

When `state.json` has `debate.delegation.enabled: true`, each champion is assigned 1-N Sonnet-model delegate agents for research and micro-experiments. Delegates are pre-spawned as teammates in the same team — champions direct them via SendMessage.

### Team Composition with Delegates

```
Round Team: ammo-round-{round_id}-{model_short}-{hardware}
|
| ... debate champions for round N (Stage 3) ...
| +-- champion-1 (Opus)          [Stage 3 debate -- shut down after selection]
| |   +-- delegate-1a (Sonnet)   [Stage 3 debate -- shut down after selection]
| +-- champion-2 (Opus)          [Stage 3 debate -- shut down after selection]
| |   +-- delegate-2a (Sonnet)   [Stage 3 debate -- shut down after selection]
| +-- champion-3 (Opus)          [Stage 3 debate -- shut down after selection]
|     +-- delegate-3a (Sonnet)   [Stage 3 debate -- shut down after selection]
|
| ... after debate: shutdown round N champions ...
|
| ... implementation agents for round N (Stages 4-5) ...
| +-- impl-champion-{op_id_1} (Opus)     [Stages 4-5]
| +-- impl-validator-{op_id_1} (Sonnet)  [Stages 4-5]
| +-- impl-champion-{op_id_2} (Opus)     [Stages 4-5]
| +-- impl-validator-{op_id_2} (Sonnet)  [Stages 4-5]
|
| ... OVERLAPPED: debate champions for round N+1 (if round 2+) ...
| +-- champion-r{N+1}-1 (Opus)    [next-round debate -- shut down after selection]
| +-- champion-r{N+1}-2 (Opus)    [next-round debate -- shut down after selection]
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

- **No GPU kernel benchmarks**: Roofline calculations, codebase research, and `ncu --query-metrics` (static) only. No kernel benchmarks that require GPU allocation — this avoids GPU contention with implementation tracks.
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

### Teardown After Debate Selection

After winner selection, shut down debate champions and delegates via `shutdown_request`. Do NOT call TeamDelete — the round team persists for Stages 4-5 implementation agents.

- Send `shutdown_request` to each champion agent (they are no longer needed).
- Delegates are shut down alongside their champions (or via explicit `shutdown_request`).
- The team itself (`ammo-round-{round_id}-{model_short}-{hardware}`) remains alive.
- TeamDelete happens later, after all implementation tracks complete (see `parallel-tracks.md`).

### Without Delegation

If `debate.delegation.enabled` is `false` (default), the debate runs with champions only — identical to the current protocol. No delegates are spawned, no `delegate_work/` directory is created.

## Debate is Always Mandatory

There is no fast-track exception. Every run must go through at least Phase 0 (proposals) + 1 debate round.

**No Convergence Shortcut**: Even if all champions converge on the same candidate, the minimum 2 debate rounds are mandatory. Convergence reduces critique diversity, making additional scrutiny MORE important, not less. (Rationale: c08b370fc campaign — convergence shortcut reduced scrutiny, contributing to 0% ship rate.)

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
| **Expected E2E Impact** | `f × kernel_speedup` where both factors have provenance. If optimization is batch-size-dependent, report per-BS projections and note gating candidacy |
| **Kill Criteria** | What threshold defines failure for this candidate. Include per-BS ranges when the optimization is expected to benefit only a subset of target batch sizes. Format: `beneficial_range: [BS_lo, BS_hi], improvement_threshold: 1.03, regression_tolerance: 0.5%` |
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

Normal minimum: **2 rounds** (no convergence shortcut). Maximum: **4 rounds**. Each round has three sequential phases.

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
4. **Hardware resource accounting**: SMEM budget, register usage, occupancy estimate, wave count at target batch sizes. If any resource exceeds the target GPU's limit, cite the specific limit and declare a hard constraint violation.

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

Flag proposals with kill criteria that imply per-BS differentiated impact (e.g., M<=32 kernel specialization, decode-only path). These are candidates for `GATED_PASS` and should be noted in `summary.md` so Stage 4-5 implementers are prepared for crossover probing.

### Post-Selection Evidence Gate

After winner selection but BEFORE shutting down debate champions, the lead opens a **15-minute evidence window**:

1. Broadcast to all champions: "Winners selected. 15-minute window for late findings that could override selection. Submit to `{artifact_dir}/debate/late_findings/`."
2. If any champion submits a late finding citing a hard constraint violation (SMEM exceeded, API incompatibility, optimization already deployed):
   - The lead evaluates the finding against available evidence
   - If the finding is substantiated: the winner is eliminated and the next-highest-scoring candidate advances
   - If the finding is unsubstantiated: noted but selection stands
3. After 15 minutes (or all champions respond "no findings"): proceed to shutdown.

This gate addresses the structural gap where Champion-1 in c08b370fc identified both fatal flaws in "Late Findings" after selection was finalized, with no mechanism for action.

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
| vLLM source modifications (except dispatch verification) | Belongs in Stage 4. **Exception**: Champions MAY run a minimal `import vllm` forward pass (no weight modifications, no training) to verify their kernel is invoked under torch.compile. This "dispatch verification" test uses existing model weights and takes <30 seconds. |
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

### Pipeline-Level Simulation (Fusion and BW-Bound Proposals)

For proposals that replace or modify kernels invoked per-layer (e.g., GEMM, normalization, activation):

1. **Multi-layer benchmark**: Run the candidate kernel N times sequentially (N = num_layers or min(N, 8)) with distinct input data per invocation, totaling > 2x L2 cache size.
2. **Launch overhead accounting**: For proposals replacing a fused/monolithic kernel with multiple separate kernels, explicitly measure and report the launch overhead (typically 2-5 us per launch under CUDA graphs). The pipeline speedup must account for this overhead.
3. **Report pipeline speedup, not just kernel speedup**: The proposal's E2E projection must use pipeline speedup (including launch overhead and multi-layer L2 effects), not raw kernel speedup from isolated benchmarks.

Proposals that report only isolated kernel speedup for per-layer optimizations have their feasibility score capped at 6/10.

## Micro-Experiment Artifact Requirements

Every kernel benchmark must produce:

1. A `.py` script in `debate/micro_experiments/` that is independently runnable
2. A `.log` file with: GPU device name on line 1 (`torch.cuda.get_device_name()`), kernel timing in microseconds, iteration count
3. For hardware utilization claims: an ncu CSV or nsys stats export

The `.log` file is the proof of execution. Missing log = Tier 1 (theoretical) regardless of script contents, and feasibility is capped at 3/10.

## NCU Triggers

ncu profiling is **optional by default** and **required only when one of these triggers fires**:

### Trigger 1: Triton Register Pressure Claims

**When**: Champion proposes a Triton kernel AND claims occupancy or register-count improvement as a primary benefit.

**Requirement**: Run `ncu --metrics l1tex__t_sector_hit_rate.pct,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active` on the proposed kernel prototype before Round 1 scoring.

**Rationale**: Register pressure in Triton's code generator cannot be estimated analytically — you must measure it.

### Trigger 2: Hardware Metric Claims Require Measurement

**When**: Any champion cites a specific hardware metric (achieved bandwidth GB/s, occupancy %, SMEM utilization, cache hit rate) as supporting evidence.

**Requirement**: The cited metric must come from an ncu or nsys measurement, not a roofline calculation. If purely theoretical, the champion must label it as an estimate and cannot cite it as achieved.

### Trigger 3: BW Assumption Divergence Escalation

**When**: A champion's roofline calculation assumes an achieved bandwidth that diverges from nsys-derived achieved BW (from Stage 2 bottleneck_analysis.md) by more than 2x.

**Requirement**: Before the proposal advances to scoring, require ncu verification of the actual achieved BW. The corrected BW value replaces the assumption in the Amdahl calculation.

**Governance**: ncu mandatory requirements are limited to these 3 triggers. No new mandatory trigger may be added without a documented failure mode from 3+ campaigns.

## Teardown (Post-Debate)

After winner selection:

1. Send `shutdown_request` to all champion agents (and delegates, if delegation was enabled).
2. Do NOT call TeamDelete. The round team persists for Stages 4-5 implementation.
3. TeamDelete is called only after all implementation tracks complete (see `parallel-tracks.md` and `SKILL.md` Stage 4-5 section).

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

## Overlapped Debate Considerations

When running as an overlapped debate during Stages 4-5:

1. **Debate champions MUST NOT message implementation agents.** The orchestrator does not provide implementation agent names. If a champion receives an unexpected message from an unknown agent, ignore it and notify the orchestrator.

2. **Artifact paths MUST use campaign-round scoping**: `debate/campaign_round_{N+1}/`. This is already enforced by the debate gate hook for round 2+, and remains enforced for overlapped debates.

3. **Micro-experiments are CPU-only during overlap**: Since GPUs are allocated to implementation tracks, debate champions are limited to roofline calculations, ISA inspection, and `ncu --query-metrics` (static analysis). The existing delegate constraint (no GPU kernel benchmarks) already covers this.

4. **Phase timing may be longer**: The orchestrator interleaves debate moderation with implementation monitoring. Debate phases may take longer to start because the orchestrator is busy gating implementation results. Champions should NOT assume a phase will start within any specific time window.

5. **The orchestrator decides debate timing**: Debate champions should wait for phase-start broadcasts from the orchestrator. Do not self-advance to the next phase.

