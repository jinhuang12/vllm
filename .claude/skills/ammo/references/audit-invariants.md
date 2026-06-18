# Audit Invariants — Stage-Specific Checklists

Authoritative checklist the `ammo-auditor` agent reads at spawn time during **Phase 2** (Checklist Verification). Each row is a checkable invariant with a severity rating. BLOCKING items halt the campaign; HIGH items trigger investigation; LOW items are noted.

**Important**: The stage-completion reconstruction (Phase 1) runs BEFORE this checklist. This checklist is the floor of audit coverage, not the ceiling — it captures institutional memory of known failure patterns. Phase 1 catches novel issues (especially workflow/process violations); this checklist catches known artifact/math patterns.

Field references below use `state.json` JSONPath syntax where relevant. All paths have been validated against `.claude/schemas/state.schema.json` (schema version 4.0). When a row refers to an artifact field (not a state.json field), the path is relative to `{artifact_dir}`.

**How to use this file**: The auditor runs **Pre-Check**, then the section matching the stage being audited (`After Stage 1` / `After Stages 4-5` / `After Stage 6-7`), then **Holistic Cross-Reference**. Sections are additive, not alternatives.

**Precondition gating**: Before checking each row, verify its precondition holds. Rows marked with `[precondition: ...]` are skipped when that condition is not met. Skipped rows are noted in the verdict as `SKIPPED (precondition: <reason>)` — not silently omitted.

**Evolving this file**: New invariants get added when a post-mortem identifies a failure mode the auditor missed. The severity rating is the *default* — the auditor downgrades only with strong evidence the invariant doesn't apply.

---

## Pre-Check (runs before every stage-specific section)

Before attempting value-level invariants, verify the artifact is structurally valid. Failure here halts the audit — value-level checks against corrupted artifacts produce noise, not signal.

| # | Check | Invariant | Severity |
|---|-------|-----------|----------|
| 1 | Artifact existence | Required files for this stage exist on disk (not just referenced in `state.json`) | BLOCKING |
| 2 | Return code validity | All sweep legs have `returncode == 0` in their `e2e_latency_results.json`; any `returncode != 0` invalidates that leg's numbers | BLOCKING |
| 3 | Non-null values | Key numeric fields (`avg_s`, `avg_latency`, `aggregate.mean_latency`, `speedup`) are not null / None / NaN in the stage's result JSONs | BLOCKING |
| 4 | Schema completeness | Required sub-objects exist (e.g., `rounds[current_round-1].integration.e2e_latency_combined` for Stage 6 [precondition: integration ran]; `rounds[current_round-1].parallel_tracks.tracks` for Stage 4-5 [precondition: stage produced artifacts]) | BLOCKING |
| 5 | Architecture-claim grounding | Every architectural assertion in `constraints.md` (and `has_moe` / `architecture` fields declared in the constraints markdown header) is backed by the model's HuggingFace `config.json`. `has_moe == true` requires at least one of `num_experts`, `num_local_experts`, `num_experts_per_tok`, or `n_routed_experts` present and > 1 in the cached config (or nested `text_config`). `has_moe == false` requires all absent or <= 1. The auditor opens the actual config.json — kernel shapes alone cannot establish architecture type because fused MLPs produce shapes that mimic per-expert GEMMs | BLOCKING |

---

## After Stage 1 (Baseline Capture) — T_AUDIT_S1

| # | Check | Invariant | Severity |
|---|-------|-----------|----------|
| 1 | Latency plausibility | Every baseline `avg_latency` (seconds) ≥ `output_len × model_params × 2 / mem_bandwidth` floor AND ≤ 60s per iteration | BLOCKING |
| 2 | Workload consistency | All batch-size captures (`baseline_bs{N}.runner.json`) use identical `(input_len, output_len, num_launches)` that match `constraints.md` | BLOCKING |
| 3 | Param vs constraints match | Runner JSON params (`output_len`, `input_len`, `tp_size`) match `constraints.md` declared values exactly | BLOCKING |
| 4 | Not nsys_probe data | Baseline JSON `avg_latency` came from `--labels baseline` E2E sweep, not from an nsys_probe run (nsys_probe latencies are ~10-100x lower due to single-iteration profiling overhead) | BLOCKING |
| 5 | Baseline label provenance | Every `baseline_bs{N}.runner.json` has `label == "baseline"` AND `opt_env` is empty / null. `label == "opt"` or populated `opt_env` is a mislabelled opt sweep masquerading as baseline | BLOCKING |
| 6 | Not golden-refs capture | `output_len` must match track workload (typically 512). Reject any baseline with `output_len ≤ 32` — those are golden-refs captures, not E2E baselines | BLOCKING |
| 7 | Canonical baseline pointer | `state.json.campaign.rounds[current_round-1].profiling_baseline_path` resolves to exactly one baseline artifact; if multiple `e2e_latency*/` dirs exist, the pointer disambiguates | BLOCKING |
| 8 | Golden-refs spec-compliant | `golden_refs.json` has `num_questions ≥ 1319` AND `max_tokens ≥ 1024` unless an explicit override is documented | BLOCKING |
| 9 | Runner param capture | Every `baseline_bs{N}.runner.json` exists with full params recorded | HIGH |
| 10 | Variance sanity | `stddev / mean < 0.15` for each batch size (low noise) | HIGH |
| 11 | Latency scaling | Latency increases monotonically with batch size (BS=1 < BS=8 < BS=32) | HIGH |
| 12 | Outlier detection | Flag any per-iteration latency > `median × 1.10`; if present, verify trimmed mean matches reported mean within 2% | HIGH |
| 13 | Round-dir label consistency | For any `e2e_latency_round{N}_baseline/` dir, all `runner.json` files have `label == "baseline"` | HIGH |
| 14 | Profiler-free timing | The sweep script hard-errors if `--slot baseline` is combined with `--nsys-profile` or `--torch-profile`, making this invariant structurally enforced. Auditor check: verify `rounds/{N}/sweeps/baseline/` runner JSON does NOT contain `nsys_profile: true` or `torch_profile: true` in any `*_runner.json`. If present, the guard was bypassed and the baseline is contaminated (nsys: 10-30% process-wide overhead, torch: CUPTI subscription cost) | BLOCKING |
| 15 | Constraints claim grounding | Every non-trivial claim in `constraints.md` (kernel inventories, memory/compute-boundedness assertions, workload characterizations, decode-phase assumptions) is traceable to concrete evidence: model source code, profiling trace output, or `config.json`. Claims without cited evidence source are flagged. Architecture claims are covered by Pre-Check #5; this item covers the remaining prose | HIGH |

---

## After Stage 2 (Bottleneck Mining) — T_AUDIT_S2

Stage 2 invariants gate the Stage 2 → Stage 3 transition. They enforce that the v4.1 f_e2e framework outputs are present and self-consistent before champions are spawned.

**Schema-version gate**: this section is only enforced when `state.json.campaign.schema_version >= "4.1"` (split on `.`, compare as `(int, int)` tuples to avoid lexicographic bugs). Legacy campaigns (`<= "4.0"`) skip this section entirely — their `bottleneck_analysis.md` predates the dilution fields and would fail every check.

| # | Check | Invariant | Severity |
|---|-------|-----------|----------|
| 1 | Workload Dilution table present | `bottleneck_analysis.md` contains a `## Workload Dilution` section with one row per BS in `target.json:workload.batch_sizes` | BLOCKING |
| 2 | `decode_busy` present + bounded | Every BS row has a numeric `decode_busy` value in [0.20, 1.0] | BLOCKING |
| 3 | `decode_share_of_e2e` present + bounded | Every BS row has a numeric `decode_share_of_e2e` in [0.0, 1.0] | BLOCKING |
| 4 | `inter_kernel_share` present + bounded | Every BS row has a numeric `inter_kernel_share` in [0.0, 1.0] | BLOCKING |
| 5 | `f_e2e` column present in Top Components | `bottleneck_analysis.md` contains a `## Top Components` (or `## Top Components (by f_e2e)`) table with `f_e2e` as a column AND at least one row populated with a numeric `f_e2e` value | BLOCKING |
| 6 | Dilution identity cross-check | For every BS row, `\|decode_kernel_s / decode_wall_s − decode_busy\| ≤ 0.05`. Catches inconsistent population of the table's `decode_kernel_s` and `decode_busy` columns | HIGH |
| 7 | f_e2e budget invariant | `Σ f_e2e (kernel rows in Top Components) ≤ decode_busy × decode_share_of_e2e + 0.02` (small tolerance for measurement noise). Catches f_e2e values that exceed the total decode-kernel budget — usually indicates the conversion was skipped or applied wrong | HIGH |
| 8 | Inter-kernel slack identity | `\|inter_kernel_share − (1 − decode_busy) × decode_share_of_e2e\| ≤ 0.02` — derived field consistency check | HIGH |
| 9 | `prefill-active?` annotation present | Top Components rows include a `prefill-active?` column (Yes/No/n/a). Required so champions know which `f_e2e` values are lower bounds | HIGH |
| 10 | Bottleneck identification grounding | Every kernel or component identified as a "top bottleneck" in `bottleneck_analysis.md` must be verifiable against the raw nsys profiling trace. Cross-check: the component's claimed rank/time-share matches its actual time in the trace. Misidentified bottlenecks (e.g., confusing kernel aliases, attributing time to wrong op) waste entire champion tracks | HIGH |

---

## After Stages 4-5 (Parallel Tracks) — T_AUDIT_S45

| # | Check | Invariant | Severity |
|---|-------|-----------|----------|
| 1 | Kernel speedup vs e2e delta | Track's `kernel_speedup × component_share (f)` ≈ observed `e2e_speedup` (within 2x). Anchors Amdahl sanity: `rounds[current_round-1].parallel_tracks.tracks[op_id].kernel_speedup` and `.e2e_speedup` | BLOCKING |
| 2 | Lossy op GSM8K mandatory | Any op with `classification == "lossy"` MUST have a Gate 5.1b GSM8K result on disk. "N/A (Justified)" is not accepted — verified GSM8K delta or explicit user override file required | BLOCKING |
| 3 | Gate 5.1b mandatory (no skip) | Gate 5.1b (E2E GSM8K) runs even when Gate 5.1a (tensor bit-exact) passes — torch.compile / CUDA-graph restructuring can break E2E correctness for bit-exact kernels. `tracks[op_id].gate_5_1a == "PASS"` does not imply 5.1b is skippable | BLOCKING |
| 4 | Correctness baseline sanity | GSM8K baseline accuracy ≥ `max(10%, known_model_accuracy × 0.5)` — detects broken harness. From `correctness_verdict.json` | BLOCKING |
| 5 | Workload param match | Track profiling uses same `(input_len, output_len)` as Stage 1 baseline for this round | BLOCKING |
| 6 | Validation artifact on disk | Every integrated track has `tracks/{op_id}/validation_results.md` on disk; `rounds[current_round-1].parallel_tracks.tracks[op_id].validation_results_path` reference alone is insufficient | BLOCKING |
| 7 | Production vs micro gap | If `production_kernel_us / micro_warm_us > 1.5`, track must include a documented gap explanation in `validation_results.md` before shipping | HIGH |
| 8 | Headline uses verified only | Campaign headline uses `verified_cumulative_speedup`; `raw_including_unverified` is a footnote only. Top-line exceeding verified × 1.01 = fail | HIGH |
| 9 | PR-ready env-flag naming | For every track that authored a `VLLM_*` gating flag (read the new flag from the track's `vllm/envs.py` diff and from the GATED_PASS `env_var` line in `validation_results.md`), the flag name must communicate the optimization's mechanism to a vLLM maintainer who has never seen this campaign — because it ships verbatim in the PR diff and enable instructions, and renaming it post-SHIP means rewriting merged code. The auditor reads the flag and judges: does the name describe WHAT the optimization does (per `references/impl-track-rules.md` § Env Flag Naming (PR-Ready), e.g. `VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100`, `VLLM_MOE_TWO_STREAM`), or does it merely carry the internal `op_id` tracking handle (e.g. `VLLM_OP003`, `VLLM_OPT4` — the op_id with a `VLLM_` prefix, or any name that conveys nothing about the mechanism)? An op_id-derived or non-descriptive flag name is a confirmed leak of the internal handle into PR-facing code. Cite the exact flag name with `path:line:"quote"`. [precondition: track authored a `VLLM_*` flag — skip always-on/no-flag tracks] | BLOCKING |

---

## After Stage 6-7 (Integration + Post-SHIP) — T_AUDIT_S67

This consolidated section replaces the former separate "After Stage 6" and "After Stage 7" sections. T_AUDIT_S67 fires AFTER: SHIP + env promotion + golden-refs capture. The auditor dispatches two parallel delegate clusters.

### Cluster 1: Integration Invariants (pre- and post-merge)

| # | Check | Invariant | Severity |
|---|-------|-----------|----------|
| 1 | Integration baseline succeeded | Integration sweep's baseline leg has `returncode == 0` AND non-null `avg_s`. Null or crashed baseline → round cannot be shipped [precondition: round has SHIP decision] | BLOCKING |
| 2 | Integration opt succeeded | Integration sweep's opt leg has `returncode == 0` AND non-null `avg_s`. A crashed opt leg's numbers MUST NOT be substituted with baseline [precondition: round has SHIP decision] | BLOCKING |
| 3 | Opt_env completeness | `set(integration_opt_env) ⊇ set(env_flags(all_shipped_ops_through_round_N))` — every previously-shipped flag is present in the integration sweep's `opt_env` [precondition: round has SHIP decision] | BLOCKING |
| 4 | Combined-all excludes failed ops | If a combined/all sweep exists, `combined_all.opt_env.keys() ⊆ env_flags(shipped_optimizations)` — no FAILED op's flag appears in the sweep | BLOCKING |
| 5 | No dual-verdict override | If any `verdict_by_*` field in the integration leg reports FAIL, `final_decision ≠ SHIP` unless an explicit `override_justification` field is present | BLOCKING |
| 6 | E2E vs Round 1 baseline | Integrated `e2e_latency_combined[primary_bs].avg` ≤ Round 1 `baseline.e2e_latency[primary_bs].avg` (no overall regression) | BLOCKING |
| 7 | Baseline immutability | Round 1 `baseline.e2e_latency` must not be mutated after Stage 1 completes (source data for cumulative speedup computation) | BLOCKING |
| 8 | Merge conflict residue | No `<<<<<<<` / `=======` / `>>>>>>>` markers remain in integrated code | BLOCKING |
| 9 | `--fresh-cache` confirmed | Integration sweep included `--fresh-cache` flag (recorded in `runner.json.args` or sweep metadata). Skip this check when `integration.status == "single_pass"` — the short-circuit copies Stage 5 results which don't use `--fresh-cache` [precondition: round has SHIP decision] | HIGH |
| 10 | Fastpath activation proof | `fastpath_evidence.status == "confirmed"` with non-empty `require_patterns` for shipped op. `status: "unknown"` = no proof op activated [precondition: round has SHIP decision] | HIGH |
| 11 | Integration ≈ sum of parts | Multi-op integration `e2e_speedup` ≈ expected from individual ops' deltas (within 1.5x). Skip when `integration.status == "single_pass"` — single-op has no "sum of parts" to validate | HIGH |
| 12 | Selected-vs-shipped mapping | If `shipped[i] ∉ rounds[current_round-1].integration.selected_candidates`, require a `descoped_from` field + re-validation artifact | HIGH |
| 13 | Post-merge commit SHA | `rounds[current_round-1].integration.commit_sha` matches post-merge mainline HEAD (not a track-local worktree HEAD) | HIGH |

### Cluster 2: Post-SHIP Invariants (env promotion, baseline, continuity)

| # | Check | Invariant | Severity |
|---|-------|-----------|----------|
| 1 | New baseline vs Round 1 | Post-SHIP `e2e_latency_combined[primary_bs].avg` ≤ Round 1 `baseline.e2e_latency[primary_bs].avg` (must not regress) [precondition: round has SHIP decision] | BLOCKING |
| 2 | Integration source validity | `rounds[current_round-1].integration.e2e_latency_combined` populated from a sweep with `opt.returncode == 0`. Not a crashed run [precondition: round has SHIP decision] | BLOCKING |
| 3 | Env promotion to target.json | For every shipped op's env flag: `target.json:bench.baseline_env[flag] == "1"`. Missing promotion = next round's baseline is wrong [precondition: round has SHIP decision] | BLOCKING |
| 4 | shipped_optimizations complete | `set(state.json.campaign.shipped_optimizations) == ⋃ {round.shipped for all rounds with status ∈ {SHIPPED, completed}}` — no truncation [precondition: round has SHIP decision] | BLOCKING |
| 5 | e2e_latency_combined entries valid | Every `integration.e2e_latency_combined` entry must have `avg > 0` and `p50 > 0` [precondition: round has SHIP decision] | BLOCKING |
| 6 | Workload param stability | Integration sweep uses identical `(input_len, output_len, num_launches)` as Round 1 capture [precondition: round has SHIP decision] | BLOCKING |
| 7 | Drift magnitude | If post-SHIP latency differs > 5% from predicted (Round N-1 × shipped improvement), investigate | HIGH |
| 8 | Golden-refs captured | After env promotion, a golden-refs capture exists with `--labels baseline` and the promoted env. `golden_refs.json.metadata.baseline_env` matches current `target.json:bench.baseline_env` [precondition: round has SHIP decision] | HIGH |
| 9 | Exhausted-tech population | After EXHAUSTED status, `rounds[current_round-1].exhausted_technologies` is non-empty; next round targets a different bottleneck component or carries an `exhaustion_override` | HIGH |

---

## Holistic Cross-Reference (runs every audit)

Invariants that span rounds or cross-reference multiple fields. These run in addition to the stage-specific checks every time the auditor is invoked.

| # | Check | Invariant | Severity |
|---|-------|-----------|----------|
| 1 | Fixed-reference consistency | `campaign.cumulative_e2e_speedup` (computed at read time) matches `rounds[0].baseline.e2e_latency[primary_bs].avg / latest_integration.e2e_latency_combined[primary_bs].avg` | BLOCKING |
| 2 | Baseline file integrity | All `baseline_bs{N}.json` files across rounds have matching workload params (`input_len`, `output_len`, `num_launches`) | BLOCKING |
| 3 | Latency floor check | No latency value anywhere falls below the memory-bandwidth floor for decode (`output_len × model_params × 2 / mem_bandwidth`) | BLOCKING |
| 4 | Stage timestamp causality | Within each round's stage sub-objects (`baseline`, `bottleneck_mining`, `debate`, `parallel_tracks`, `integration`, `campaign_eval`), `started_at ≤ completed_at`; each subsequent stage's `started_at` ≥ the previous stage's `completed_at` | BLOCKING |
| 5 | Per-round cumulative populated | Every round with `status ∈ {SHIPPED, completed}` has a non-null `cumulative_speedup_after` field. Absence = BLOCKING | BLOCKING |
| 6 | Monotonic improvement | `cumulative_speedup_vs_round1` and per-round `cumulative_speedup_after` are non-decreasing across rounds (shipped ops don't silently regress) | HIGH |
| 7 | Round-over-round sanity | Each round's new baseline ≤ previous round's baseline at every BS | HIGH |
| 8 | Cumulative measured vs extrapolated | If both measured and extrapolated cumulative values exist (from different sources), divergence > 5% = flag | HIGH |
| 9 | Cross-session drift guard | If baseline and opt were measured in different sessions (different `session_id` in runner metadata), flag for same-session re-run | HIGH |

---

## Item Counts (for structural test)

The test suite (`tests/unit/test_ammo_auditor_docs.py`) enforces the following row counts so this file cannot silently drift:

| Section | Item count |
|---------|------------|
| Pre-Check | 5 |
| After Stage 1 | 15 |
| After Stage 2 (v4.1+ only) | 10 |
| After Stages 4-5 | 9 |
| After Stage 6-7 — Cluster 1 (Integration) | 13 |
| After Stage 6-7 — Cluster 2 (Post-SHIP) | 9 |
| Holistic Cross-Reference | 9 |
| **Total** | **70** |

If a new invariant is added, update this table and the corresponding test assertion.

---

## Field-Name Normalization Notes

This file uses canonical schema field names (schema v4.0). Historical aliases NOT used here:

- `cumulative_e2e_speedup` → computed at read time by the backend normalizer from `rounds[0].baseline.e2e_latency` vs latest `integration.e2e_latency_combined`; injected as `campaign.cumulative_e2e_speedup` in-memory (not stored in state.json)
- `cumulative_speedup_after` → per-round summary field (still stored)
- Free-form `stage_timestamps` dict → canonical is per-stage sub-object `started_at` / `completed_at` fields on each round (`baseline`, `bottleneck_mining`, `debate`, `parallel_tracks`, `integration`, `campaign_eval`)
- Bare-string `shipped_optimizations` → canonical is `[{op_id, round, classification}, ...]`

Any audit finding that cites these historical names must first be translated to the canonical name — otherwise the orchestrator's `state.json` update fails schema validation.
