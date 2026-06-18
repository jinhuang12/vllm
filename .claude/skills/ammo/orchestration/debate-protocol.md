# Stage 3: Adversarial Debate Protocol

Champions independently propose optimization candidates from grounded Stage 2 data, then debate them adversarially. The main session acts as moderator. Stage 2 provides ONLY measured facts and physical ceilings — champions generate candidates and feasibility estimates themselves.

## Team Structure

- **Team name**: `ammo-round-{round_id}-{model_short}-{hardware}` -- this is the **round team**, created once per round and reused for both debate (Stage 3) and implementation (Stages 4-5).
  - Example: `ammo-round-1-llama70b-h100`
- **Champions**: 2-4 `ammo-champion` agents. Each reads the grounded bottleneck_analysis.md independently.
- Each champion is spawned with:
  - `name="champion-{N}"` (e.g., `champion-1`, `champion-2`)
  - `team_name` set to the round team name above
- **No debate-stage monitors**: Debate champions do NOT get transcript monitors. The adversarial debate structure (Phase B critique + Phase C rebuttal) provides quality control. Monitors are spawned only for impl-champions in Stages 4-5.

## Team Composition

```
Round Team: ammo-round-{round_id}-{model_short}-{hardware}
|
| ... debate champions for round N (Stage 3) ...
| +-- champion-1      [Stage 3 debate -- shut down after selection]
| +-- champion-2      [Stage 3 debate -- shut down after selection]
| +-- champion-3      [Stage 3 debate -- shut down after selection]
|
| ... after debate: shutdown round N champions ...
|
| ... implementation agents for round N (Stages 4-5) ...
| +-- impl-champion-{op_id_1}   [Stages 4-5]
| |   +-- impl-monitor-1        [Stages 4-5]
| +-- impl-champion-{op_id_2}   [Stages 4-5]
| |   +-- impl-monitor-2        [Stages 4-5]
```

## Debate is Always Mandatory

There is no fast-track exception. Every run must go through at least Phase 0 (proposals) + 1 full debate round (A/B/C).

**Conditional 2nd round**: After round 1 Phase C, champions self-declare open items (see § Open Items Declaration below). If ANY champion declares open items → round 2 is mandatory. If all champions declare `NONE` → proceed directly to scoring/selection. This replaces the former mandatory-2-round rule.

## Phase -1: Target Claim (NEW — runs BEFORE Phase 0)

### Why this phase exists

Without coordination, champions independently gravitate to the top bottleneck. Three champions all attacking the same component is *zero diversity*: when that component hits a wall in Stage 5, the entire round produces nothing. The campaign has no contingency.

The claim phase forces champions to coordinate **component** selection *before* developing proposals. Champions waterfall down the ranked-bottleneck list, ensuring at least one secondary target is covered as a hedge. The orchestrator reviews the resulting claim distribution holistically and redirects any champion whose target leaves the round under-diversified.

This is structural diversity insurance — independent of the per-proposal eligibility gates in Phase 0, which remain unchanged.

### The claim is COMPONENT-only — the mechanism comes in Phase 0

A champion claims a **component** (e.g., MoE GEMM, dense GEMM, attention). The optimization category (`kernel_replacement` / `kernel_fusion` / `dispatch_optimization`) is chosen in Phase 0, after the champion analyzes the existing profiling data for that component. The analysis of existing data sits *between* the two decisions:

```
claim COMPONENT  →  [orchestrator approves]  →  analyze existing profiling data  →  propose candidate + MECHANISM (Phase 0)
   (pre-data)                                      for THAT component                  (grounded in that data)
```

The category is a conclusion drawn from data — whether a fusable seam exists, whether a kernel can be replaced by a faster one, or whether host-side dispatch gaps dominate — so it is chosen once the champion has analyzed the component's slice of the existing profiling data. This uses the campaign's already-captured Stage 1-2 traces; it adds no new nsys/ncu capture. The total work is the same diligence champions already do, ordered so the mechanism follows from the existing evidence.

### Sequence

```
Stage 2 complete (mining done)
    │
    ▼
Orchestrator spawns N champions into round team (with claim-phase prompt)
    │
    ▼
CLAIM PHASE (component axis)
  1. Each champion reads bottleneck_analysis.md
  2. Each applies the component waterfall logic in the ammo-champion definition
  3. Each broadcasts a single COMPONENT claim line ("Claiming {component}") to the team
  4. Collisions handled by the orchestrator review (sharing a component is allowed;
     the LATER broadcaster yields ONLY if the review needs more component spread)
    │
    ▼
ORCHESTRATOR REVIEW
  - Read all claim broadcasts
  - Evaluate the holistic checks below (component axis)
  - Either: approve all → "Claims approved. Proceed to analyze + Phase 0."
  - Or: send per-champion redirects with reasoning
    │
    ▼
PER-COMPONENT ANALYSIS (champion, after approval)
  - Each champion analyzes the EXISTING profiling data for its assigned component (no new capture)
  - The mechanism/category follows from that data (fusion seam? replaceable kernel? dispatch gap?)
  - Scope: analyze only the assigned component — the orchestrator owns assignment
    │
    ▼
Phase 0 eligibility gates (unchanged) → Diversity Check (Lead): f-value source + per-technology exhausted check
```

### Champion responsibility

The full waterfall logic (steps, component fully-exhausted check, broadcast format, collision rules) lives in `.claude/agents/ammo-champion.md` § Target Claim Phase. The champion is the source of truth for the assignment math; the orchestrator only reviews the resulting claims. The two surfaces must agree: the claim is **component-only**, and the mechanism/category is selected in Phase 0 after the champion analyzes the existing profiling data for the component.

Each champion broadcasts exactly one claim message in the form:

```
Claiming {component}
```

`{component}` MUST match a component name from `bottleneck_analysis.md`. The claim does **NOT** include a category — the mechanism (`kernel_replacement` / `kernel_fusion` / `dispatch_optimization`) is chosen in the Phase 0 proposal, after the champion analyzes the existing profiling data for the component.

Collision resolution: sharing a component is **permitted** (two champions on the same component differentiate by mechanism in Phase 0). The later broadcaster does NOT automatically yield. The orchestrator's holistic review (below) decides whether the round needs more component spread; if so, it redirects the later broadcaster, who re-runs the waterfall to pick the next available component.

### Orchestrator Review

After all champions have broadcast, the orchestrator reads the complete set of **component** claims and evaluates them holistically. None of these checks duplicates the per-proposal eligibility gates — they're about *distribution* (which components), not *content* (which mechanism). The category is not known yet, so no check here keys on it.

| # | Check | Action if it fails |
|---|-------|--------------------|
| 1 | Claims collectively cover the highest-`f_e2e` components in proportion to opportunity (top component must have at least 1 claim) | Redirect a champion to the underserved top component |
| 2 | At least one claim targets a component *different from* the majority (anti-monoculture hedge) | If all claims target one component AND a viable secondary component exists: redirect the last broadcaster to the #2 component |
| 3 | No claim targets a **fully-exhausted** component. A component is fully-exhausted only when the `state.json.campaign.rounds[$IDX].exhausted_technologies` entries for it (matching `applies_to_component` + `applies_to_shape_bucket`) clearly close off **every** optimization avenue for it — across all catalog classes (the legacy trio `kernel_replacement` / `kernel_fusion` / `dispatch_optimization` plus any other cataloged class or novel descriptor that fits the component). Because the schema records `technology_class` + `failure_mode` (no category field) and one `technology_class` can serve more than one category, **default to viable when the closure is ambiguous** — a component with any plausibly-un-tried avenue stays in contention; the precise per-technology check runs at the Phase 0 Diversity Check (Lead), once the mechanism is known. | Send the champion the exhaustion entries showing every avenue is closed and ask for a different component |
| 4 | No claim targets a component whose `f_e2e` is below `campaign.config.min_e2e_improvement_pct` | Redirect to a viable component |

**Where the per-technology exhausted check lives now:** because the claim carries no category, the precise per-technology match runs at the **Phase 0 Diversity Check (Lead)** (§ Diversity Check (Lead) item 2), once the champion's analysis has set the mechanism and the proposal declares a concrete `## Category` + Technology Selection block. At claim time the orchestrator rejects a component only when *all* its categories are exhausted (Check 3); a champion who claims a still-viable component and then proposes the one exhausted technology for it is flagged at the Phase 0 Diversity Check. This matches the `exhausted_technologies[]` schema, which is keyed by `technology_class` + `applies_to_component` + `applies_to_shape_bucket` (never by category) — the concrete technology is known from the written proposal.

### Single-message orchestrator response

The orchestrator replies once. Either:

```
Claims approved. Proceed to analyze + Phase 0.
```

(The approval cues the next step: champions analyze the existing profiling data for their assigned component before writing the proposal.)

Or a redirect message addressed per-champion, e.g.:

```
champion-3: Your component ({component_X}) duplicates champion-1's and the round needs more spread (Check 2).
  Re-run the waterfall and pick a different component. Suggested: {component_Y} (next viable per the ranked `f_e2e` list).

champion-2: Approved.
champion-1: Approved.
```

Redirects are about *components*, never categories — the category isn't chosen until Phase 0. After redirects, the orchestrator waits for the redirected champion(s) to re-broadcast, then re-runs the review. Loop until all claims pass — bound to 3 cycles to prevent infinite back-and-forth; on the 3rd failure, the orchestrator picks an assignment manually and instructs the champion to comply.

### Why this replaces the old Diversity Check items

The previous Diversity Check Decision 2 ("Dominant Component Coverage (MANDATORY)") was a *post-hoc reject* — it ran after Phase 0 proposals existed and forced revision when coverage was missing. That wasted a champion's Phase 0 work. The claim phase guarantees coverage *before* anyone writes a proposal, so the post-hoc reject is no longer needed.

The previous Decision 3 ("Technology Diversity After EXHAUSTED Round (soft)") is now split by granularity, and stays a **soft** requirement. At claim time, Check 3 (above) + waterfall step 2 in the agent definition reject only *fully-exhausted components* (every category exhausted) — a component-level filter, since the category isn't known yet. The precise per-technology filter — "a champion re-attempting an `exhausted_technologies[]` entry must document what changed" — runs at the **Phase 0 Diversity Check (Lead) item 2**, after the champion's analysis of the existing data has set the mechanism. Together they keep a champion from silently repeating an exhausted technology, while still allowing a champion to claim a component that has *some* exhausted categories but a viable one remaining.

## Phase 0: Independent Proposals

All champions execute **in parallel**. Each champion reads the grounded bottleneck_analysis.md (which contains ONLY measured facts and physical ceilings) and independently proposes 1-2 optimization candidates.

Each champion writes:

```
{artifact_dir}/rounds/{CR}/debate/proposals/{champion_id}_proposal.md
```

Where `{CR}` is the current campaign round (`campaign.current_round` from `state.json`). All debate artifacts live under `rounds/{CR}/debate/` per the canonical layout in `references/artifact-layout.md`.

Champions write proposals per `references/debate-rules.md` (see Evidence Tiers for claim-evidence requirements, Micro-Experiment Rules for allowed/forbidden experiments, Baseline Provenance Rule for API matching, and Micro-Experiment Artifact Requirements for proof-of-execution).

### Proposal Eligibility Gate (Lead)

After Phase 0 submissions, the lead checks each proposal against the gates below before any debate begins.

**Gate 1 — Authored-Mechanism Mandate** (north star: *real engineering work, no flag-flipping*; full definition in `SKILL.md` NN#8):
- **Pass**: Proposal **authors mechanism logic OR host-side structure** — a custom/fused kernel (`custom_kernel` / `kernel_replacement` / `kernel_fusion` / `attention_kv_layout`), OR a load-time weight restructuring + library GEMM (`weight_layout_transform`), OR authored scheduling/dispatch/comm/graph-pass host-side code (`dispatch_optimization` / `execution_pipeline_restructuring` / `communication_strategy` / `compute_graph_pass`), OR a novel mechanism that authors logic/structure against a profiled bottleneck. *(This correctly passes a paused `dispatch_optimization` proposal — host-side, no kernel — which the old kernel-only wording wrongly excluded.)*
- **Reject**: A retuned scheduling/tuning value where the kernel/cubin body is byte-identical and a compiler, autotuner, library tactic-table, policy list, predicate flip, or env var emits the speedup — `num_warps` / `num_stages` / `BLOCK_SIZE_*` / `@autotune` tuples / tactic tables / `custom_ops` list edits / boolean predicate flips / env-var flips. *A constant in a `.py` file is still config, regardless of the measured win.* Also reject if no authored mechanism is described.

**Gate 2 — Technology Selection block:**
- **Pass**: Proposal includes a populated Technology Selection block per `references/technology-selection.md` § Required proposal fields.
- **Reject**: Block absent, incomplete, or fields contain unsupported assertions (e.g., "baseline technology: unknown" without evidence, "anti-regression check: not applicable" when the proposal is higher-abstraction than the baseline).

**Gate 3 — Precision Classification:**
- **Pass**: Proposal includes a `Precision Classification` field declaring `lossless` or `lossy`, citing the dtype boundary rule from `references/debate-scoring-rubric.md` § Lossy Classification Rule.
- **Reject**: Classification field absent or does not cite the dtype boundary rule.

**Gate 4 — Category Block:**
- **Pass**: Proposal includes a populated `## Category` block per `references/optimization-categories.md` § Verifying a Proposal's Category Block, with all five fields populated (`Selected`, `Slice targeted`, `Projection formula`, `Justification`, `Expected validation gates`). The declared `Selected` value MUST be a catalog class, a legacy alias, or a named novel descriptor (per `references/optimization-categories.md` § The Category Catalog) — `category` is a descriptor, not the eligibility decision (the gate is Gate 1 above). The declared `Projection formula` MUST match the formula for the **slice** that category attacks in `references/optimization-categories.md` § Per-Category Projection Formulas (nearest-analogue regime formula for a novel descriptor).
- **Reject**: Block absent, missing required fields, or `Projection formula` does not match the declared category's slice. Wrong-slice projection formula additionally caps E2E impact at 0/10 in scoring per `references/debate-scoring-rubric.md`. *(Note: an unrecognized `Selected` name is NOT a Gate-4 reject — a novel mechanism names a new descriptor; eligibility is decided by Gate 1.)*

> The per-technology exhausted check is **not** a hard eligibility gate — it is a soft check at the Diversity Check (Lead) below (item 2), because at claim time only the component is known and the mechanism isn't fixed until the proposal declares its `## Category` + Technology Selection block.

**Rejection action**: Message the champion to revise. If no compliant revision is submitted, the candidate is eliminated.

**Non-compliant proposals MUST NOT advance to Round 1.**

### Diversity Check (Lead)

After the eligibility gate, the lead reviews proposal diversity. Component-coverage concerns are now handled *before* proposals are written by the Phase -1 Target Claim phase. Two checks remain here: the f-value source check, and the per-technology exhausted check — the latter relocated from claim time, now that each proposal's mechanism is finally known.

1. **f-value source check**: For each proposal, check whether the champion used `f_e2e` (the correct Amdahl multiplier from the Top Components table — `f_decode × decode_busy × decode_share_of_e2e`), `f_decode` (the diagnostic ranking column), or `f_total` (full-trace, including warmup/prefill, generally wrong for decode-heavy workloads). If any workload-dilution red flag fires (`decode_busy < 0.85`, `prefill_share_of_e2e > 0.10`, or `input_len >= 512`) and the champion's projection plugs `f_decode` directly into Amdahl without the conversion to `f_e2e`, flag it — the scoring rubric applies a 2-point deduction (see `references/debate-scoring-rubric.md` § E2E impact potential). If the target kernel isn't in the decode breakdown, note this — the champion may be targeting prefill latency intentionally.

2. **Per-technology exhausted check (soft)**: For each proposal, now that its `## Category` / `technology_class` is declared, check whether the `(applies_to_component == proposal.component) × (technology_class == proposal.technology) × failure_mode` tuple is present in `state.json.campaign.rounds[$IDX].exhausted_technologies[]` for this shape bucket (treat `applies_to_component`/`applies_to_shape_bucket` = `null` as matching all). If it matches an exhausted entry (and `expires_after_reprofile` is not satisfied), send the champion the specific exhaustion entry and ask for a different mechanism/technology — or a documented justification of what changed vs. the prior attempt, per `references/technology-selection.md` § Technology diversity across rounds (the "document the loop" soft requirement). This is the natural home for the check: the mechanism is known here, whereas at claim time only component-level exhaustion (Phase -1 Check 3) can be evaluated. When `exhausted_technologies[]` is empty (e.g., round 1), this check is a no-op.

> **Replaced checks**: The previous "Dominant Component Coverage (MANDATORY)" check is subsumed by the Phase -1 Target Claim phase, which guarantees *component* coverage *before* proposals are written. The "Technology Diversity After EXHAUSTED Round (soft)" check is **relocated to item 2 above** (not removed, and still soft): per-technology (`component × technology_class × failure_mode`) exhaustion is checked HERE, once the mechanism is known; only the component-level remainder (skip a component when ALL its categories are exhausted) stays at Phase -1 Check 3. See § Phase -1: Target Claim above.

## Champion Spawn Context

Spawn 2-4 champions. A champion claims a *component* in Phase -1, analyzes the existing profiling data for that component, and self-selects `kernel_replacement` / `kernel_fusion` / `dispatch_optimization` in Phase 0 based on that analysis and the eligibility gates above. The spawn prompt assigns the component axis only; the mechanism is the champion's Phase 0 conclusion from the data.

The lead's spawn-prompt construction follows this order:

1. **Standard champion-orientation context** (artifact_dir, target component summary, paths to relevant artifacts). The "target component summary" lists the ranked `f_e2e` components for the *claim waterfall* and the paths to the existing profiling data; it states components only, not a category for any component.
2. **Workload Dilution summary** (decode_busy, decode_share_of_e2e, inter_kernel_share, prefill_share — copy directly from the researcher's table for the primary BS).
3. **Standard champion task body** (Claim Phase = claim a component; then analyze the existing profiling data for that component; then Phase 0 instructions, debate rules pointer, Technology Selection + Category block requirements).

## Round Structure

Normal minimum: **1 round**. Maximum: **5 rounds**. Each round has three sequential phases. A 2nd round triggers automatically if any champion declares open items after Phase C of round 1.

**Path template**: `{CR}` is `campaign.current_round` from `state.json` (1 for the first campaign round). `{N}` is the debate sub-round within that campaign round (1..5). Applies uniformly including R1 — every debate sub-round is scoped under `rounds/{CR}/debate/round_{N}/`.

### Phase A: Evidence Presentation

All champions execute **in parallel**.

Each champion writes:

```
{artifact_dir}/rounds/{CR}/debate/round_{N}/{op_id}_argument.md
```

Champions write arguments per `references/debate-rules.md` (see Evidence Tiers for required evidence levels). Champions **must** run micro-experiments during this phase (see `references/debate-rules.md` § Micro-Experiment Rules).

### Phase B: Critique

Round-robin assignment:

- Champion for OP-001 critiques OP-002
- Champion for OP-002 critiques OP-003
- ...
- Champion for OP-N critiques OP-001

Each champion writes:

```
{artifact_dir}/rounds/{CR}/debate/round_{N}/{op_id}_critique_{target_id}.md
```

Critiques must address: feasibility math weaknesses, overlooked risks, incorrect assumptions, and hardware resource accounting (SMEM budget, register usage, occupancy, wave count). See `references/debate-rules.md` for evidence tier requirements.

### Phase C: Rebuttal

All champions execute **in parallel**.

Each champion responds to the critique they received and writes:

```
{artifact_dir}/rounds/{CR}/debate/round_{N}/{op_id}_rebuttal.md
```

Rebuttals must provide counter-evidence, concede valid points explicitly, or propose concrete mitigation for acknowledged weaknesses. See `references/debate-rules.md` for evidence tier requirements.

### Open Items Declaration (end of Phase C)

At the end of their Phase C rebuttal file, each champion appends an Open Items Declaration. Champions emit ONLY the lines that apply — delete inapplicable categories:

If open items exist:
```markdown
## Open Items Declaration
- [UNADDRESSED_CRITIQUE] <description of critique not fully rebutted>
- [NEW_EVIDENCE] <new claim introduced in rebuttal that wasn't cross-examined>
```

If all critiques are satisfactorily addressed:
```markdown
## Open Items Declaration
- [NONE]
```

Categories:
- `UNADDRESSED_CRITIQUE`: A Phase B critique that the champion's rebuttal did not fully engage with
- `NEW_EVIDENCE`: New factual claim or evidence introduced in a rebuttal that other champions haven't had a chance to examine
- `NONE`: No open items remaining — champion considers all critiques satisfactorily addressed

**Decision logic** (orchestrator, after all Phase C rebuttals land):
- If ANY champion's declaration contains `[UNADDRESSED_CRITIQUE]` or `[NEW_EVIDENCE]` with a non-empty description → trigger round 2
- If ALL champions declare only `[NONE]` → proceed to scoring/selection

Champions are trusted. The scoring rubric naturally penalizes weak rebuttals — a champion who hides open items to avoid round 2 will score lower because unaddressed critiques are visible to the scorer.

## Communication Flow

The main session moderates the debate using `SendMessage`:

1. **Broadcast** phase-start message to all champions (includes round number and phase identifier).
2. Each champion **messages main** upon phase completion with a short status line.
3. Main **waits** for all champions to report before advancing to the next phase.
4. After each complete round, main evaluates convergence criteria.

## Convergence Criteria

For rounds 2+ (after open items trigger a continuation), stop early if **either** condition is met:

1. **Clear winners**: The top 3-4 candidates have no unaddressed critiques remaining, and all other candidates have conceded material weaknesses.
2. **Stagnation**: Round N+1 arguments substantially repeat round N with no new evidence or counter-arguments introduced.

## Winner Selection

After the final round:

1. Main session reads **all** debate artifacts across all rounds.
2. Scores each candidate per `references/debate-scoring-rubric.md`.
3. Selects **3-4 winners** to advance to Stage 4 parallel tracks.
4. Writes the decision to `state.json`:

```
state.json.campaign.rounds[N-1].debate.selected_candidates = [
  {op_id, track_assignment, score_breakdown,
   stage_4_validation_obligations, cited_evidence},
  ... one entry per winner (typically 2-4 per round)
]
```

This is the **authoritative cross-agent contract** — each Stage 4/5 implementation champion reads the entry matching its assigned `op_id` first. `op_id` values in this array must match entries in `selected_winners` (the list of chosen op_id strings). All downstream decisions (track_assignment, validation obligations, evidence citations) live in typed fields. No prose justification is part of the contract.

5. Runs the deterministic renderer to produce a human-readable summary:

```
python .claude/skills/ammo/scripts/render_debate_summary.py \
  --state {artifact_dir}/state.json \
  --out {artifact_dir}/rounds/{CR}/debate/summary.md
```

`summary.md` is a rendered view of `selected_candidates`; it contains numeric values, enum flags, and citation links only. If `summary.md` and `state.json` ever disagree, `state.json` wins — `summary.md` is regenerated from state, not hand-edited. No agent has a write path to `summary.md`.

Proposals with per-BS differentiated impact (e.g., `M<=32` kernel specialization, decode-only path) are candidates for `GATED_PASS`. These are encoded in `stage_4_validation_obligations` (e.g., `crossover_probe`), not in prose.

## Debate Rules Reference

Micro-experiment guidelines, artifact requirements, baseline provenance rules, and NCU triggers are defined in `references/debate-rules.md`. Champions must read this reference. The lead uses NCU Trigger 4 (Baseline BW Discrepancy) during the eligibility gate.

## Teardown (Post-Debate)

After winner selection:

1. Send `shutdown_request` to each champion: `SendMessage(to=<champion_id>, message={"type": "shutdown_request"})`. A champion approves only when its work is complete; if one replies `approve: false`, let it finish and re-send once it reports done.
2. Confirm each champion left the team roster (its `shutdown_approved` arrived) before spawning implementation agents. A champion still on the roster is still running.
3. Do NOT call TeamDelete. The round team persists for Stages 4-5 implementation. TeamDelete is called only after all implementation tracks complete (see `parallel-tracks.md` and `SKILL.md` Stage 4-5 section).

## Artifact Structure

All debate artifacts for campaign round `{CR}` live under `rounds/{CR}/debate/`. There is no separate first-round vs Nth-round shape — the round-scoped path applies uniformly from round 1 onward (see `references/artifact-layout.md`):

```
{artifact_dir}/rounds/{CR}/debate/
  summary.md
  proposals/
    champion-1_proposal.md
    champion-2_proposal.md
    champion-3_proposal.md
  round_1/                  (debate sub-round within campaign round CR)
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

The debate gate hook enforces the v2 layout: writes to legacy `debate/` (without a `rounds/{CR}/` prefix) are blocked.

Note: "round_1" inside `rounds/{CR}/debate/` refers to **debate sub-rounds** (argument/critique/rebuttal cycles). The outer `rounds/{CR}/` refers to **campaign rounds** (profiling cycles).

