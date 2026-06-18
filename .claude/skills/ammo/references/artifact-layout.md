# Artifact Layout (V2) — Path Resolution Reference

**Single source of truth for AMMO campaign artifact paths.** All agent docs, orchestration docs, and scripts reference this file. If another document contradicts a path here, this file wins.

Spec: `docs/superpowers/specs/2026-05-12-ammo-artifact-layout-design.md`

## Table of Contents

1. [Root Layout](#root-layout)
2. [Round Directory Tree](#round-directory-tree)
3. [Path Resolution Rules](#path-resolution-rules)
4. [Agent-Specific Output Paths](#agent-specific-output-paths)
5. [Authoritative vs Diagnostic Artifacts](#authoritative-vs-diagnostic-artifacts)
6. [Disambiguation Rules](#disambiguation-rules)
7. [Prohibited Patterns](#prohibited-patterns)
8. [Layout Detection (for scripts)](#layout-detection-for-scripts)

---

## Root Layout

```
kernel_opt_artifacts/{target}/
├── state.json              # Campaign state (orchestrator-only writes)
├── target.json             # Workload + bench config (mutated on SHIP)
├── REPORT.md               # Terminal deliverable (Stage 7b)
├── report_assets/          # Charts + generators for REPORT.md
│   ├── *.png
│   └── gen_*.py
├── rounds/                 # Round-scoped hierarchy (1-indexed)
│   └── {N}/
│       └── (see Round Directory Tree)
└── blockers/               # Escalation artifacts (cross-round)
    └── {stage}_{date}.md
```

`{target}` = `{model}_{hardware}_{dtype}_tp{tp}` (e.g., `deepseek-v4-flash_B200_fp8_tp1`).

### Root-Level Files

| File | Writer | Notes |
|------|--------|-------|
| `state.json` | Orchestrator only | Schema: `.claude/schemas/state.schema.json` |
| `target.json` | `new_target.py` + orchestrator (env promotion on SHIP) | Never written by sub-agents |
| `REPORT.md` | `ammo-report-writer` (terminal only) | Authoritative final deliverable |
| `report_assets/` | `ammo-report-writer` | 5 required PNGs + gen scripts |
| `blockers/{stage}_{date}.md` | Orchestrator on escalation | Cross-round (NOT round-scoped) |

> **Note:** `state.json` + path conventions are the sole sources of artifact metadata. The frontend lists files via `GET /api/campaigns/{id}/tree` and reads metrics from `state.json` (e.g., `rounds[N-1].bottleneck_mining`, `tracks[op_id].gate_5_2_metrics`). Per-artifact `.metrics.json` sidecars are no longer used.

---

## Round Directory Tree

Every round `N` (1-indexed) contains the full lifecycle for that optimization round.

```
rounds/{N}/
├── constraints.md                       # Baseline constraints for this round
│
├── profiling/
│   ├── nsys/                            # Stage 1 nsys node traces
│   │   └── baseline_bs{BS}.nsys-rep
│   └── ncu/                             # Targeted hardware-counter data
│   │   ├── sanity.csv                   # Top-3 kernel metrics (OL=8 driver)
│   │   └── sanity_results.md            # Researcher narrative
│
├── sweeps/
│   ├── baseline/                        # Stage 1 E2E baseline (AUTHORITATIVE)
│   │   ├── json/                        # golden_refs.json, baseline_bs{BS}.json
│   │   ├── logs/                        # Per-bucket + supervisor logs
│   │   ├── status/                      # Heartbeat files
│   │   └── e2e_latency_results.{json,md}
│   ├── opt/{op_id}/                     # Stage 5 per-track opt sweep (AUTHORITATIVE)
│   │   ├── json/                        # correctness_verdict.json, opt_outputs.json, opt_bs{BS}.json
│   │   ├── logs/
│   │   ├── status/
│   │   └── e2e_latency_results.{json,md}
│   ├── integration/                     # Stage 6 combined sweep (--fresh-cache)
│   │   └── (same sub-structure)
│   └── golden_capture/                  # Post-SHIP golden-refs for next round
│       └── json/golden_refs.json
│
├── mining/
│   └── bottleneck_analysis.md
│
├── debate/
│   ├── proposals/                       # {champion_id}_proposal.md
│   ├── round_{D}/                       # Debate rounds (D=1,2,...; NOT campaign rounds)
│   │   └── {op_id}_{argument|critique_{target}|rebuttal}.md
│   ├── micro_experiments/               # Champion feasibility scripts + logs
│   └── summary.md                       # DERIVED — never edit (regenerated from state.json)
│
├── tracks/
│   └── {op_id}/
│       ├── validation_results.md        # Champion's final verdict (AUTHORITATIVE)
│       ├── validator_tests/             # Impl-champion kernel correctness & speedup artifacts
│       │   ├── test_correctness.py
│       │   ├── bench_gate_5_2.py
│       │   ├── gate_5_1a_results.json
│       │   └── gate_5_2_results.json
│       ├── monitor_audits/              # Transcript monitor (Stages 4-5)
│       │   └── {monitor_id}_observations.md
│       └── _scratch/                    # Non-authoritative iteration artifacts
│           └── *.md, *.py, *.log
│
├── audits/
│   ├── stage_1.md                       # T_AUDIT_S1 verdict
│   ├── stage_45.md                      # T_AUDIT_S45 verdict
│   ├── stage_67.md                      # T_AUDIT_S67 verdict
│   └── stage_*_cycle_{C}.md             # Re-audit on BLOCKED (C=2,3)
│
└── _archive/                            # Superseded sweep runs (auto-managed)
    └── {slot}_{timestamp}/              # e.g., baseline_2026-05-05T181212Z/
```

---

## Path Resolution Rules

### Sweep Script (`run_vllm_bench_latency_sweep.py`)

Resolve sweep output via `--round` and `--slot`:

```bash
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir {artifact_dir} \
    --round {N} --slot {SLOT} \
    [--labels baseline|opt] [other flags...]
```

| `--slot` value | Resolves to | Used by |
|----------------|-------------|---------|
| `baseline` | `rounds/{N}/sweeps/baseline/` | Stage 1 clean E2E (profiling flags forbidden) |
| `profiling` | `rounds/{N}/sweeps/profiling/` | Stage 1 profiling traces (E2E here is contaminated, not authoritative) |
| `opt/{op_id}` | `rounds/{N}/sweeps/opt/{op_id}/` | Stage 5 per-track opt |
| `integration` | `rounds/{N}/sweeps/integration/` | Stage 6 combined sweep |
| `golden_capture` | `rounds/{N}/sweeps/golden_capture/` | Post-SHIP golden-refs |

**Resolution logic:**
1. `--round N --slot SLOT` → `out_root = {artifact_dir}/rounds/{N}/sweeps/{SLOT}/`
2. `--round N` only → fail (slot required)
3. Neither → read `state.json.campaign.current_round` + require `--slot`
4. `--out-name`: **removed**. Hard error with guidance pointing to `--round`/`--slot`.

**Archive behavior:** When the target `out_root` already exists and is non-empty, move existing contents to `rounds/{N}/_archive/{slot}_{ISO_timestamp}/`. Active slots NEVER carry timestamps in their names.

### Per-Bucket nsys Traces (sweep `--nsys-profile`)

When `--nsys-profile` is passed to the sweep script, traces are written to **`rounds/{N}/profiling/nsys/`** (NOT inside the sweep output dir). The sweep script extracts `--round` and writes traces to the sibling profiling dir.

Legacy campaigns may still contain `rounds/{N}/profiling/probe/` or `rounds/{N}/profiling/torch_profile/`. Readers may keep compatibility for those paths, but new campaigns do not scaffold or recommend them.

### Monitor Logs (Impl-Stage Only)

Impl-stage monitors MUST write to the **monitored entity's `monitor_audits/` subdirectory**:

| Monitor target | Output path |
|---------------|-------------|
| Impl-champion `{op_id}` | `rounds/{N}/tracks/{op_id}/monitor_audits/{monitor_id}_observations.md` |

Monitors receive `round_number` and `output_dir` in their dispatch prompt. They MUST NOT write to the campaign root. Debate champions do not have monitors.

### Champion Scratch Files

Intermediate artifacts (drafts, debug scripts, iteration logs) MUST go to `rounds/{N}/tracks/{op_id}/_scratch/`. The ONLY authoritative file at the track root is `validation_results.md`.

### Projection Accuracy (`check_projection_accuracy.py`)

Appends a `## Projection Accuracy` section to `rounds/{N}/tracks/{op_id}/validation_results.md` — NOT a campaign-root file. Takes `--round N --track-id {op_id}`.

---

## Agent-Specific Output Paths

### Orchestrator
| Writes to | When |
|-----------|------|
| `state.json` | Every stage transition, gate result, track update |
| `target.json` | SHIP env promotion |
| `rounds/{N}/debate/summary.md` (via script) | After debate winner selection |
| `blockers/{stage}_{date}.md` | On escalation |

### ammo-researcher
| Writes to | When |
|-----------|------|
| `rounds/{N}/sweeps/baseline/*` | T=1 (via sweep script) |
| `rounds/{N}/profiling/nsys/*.nsys-rep` | T=1 (via sweep `--nsys-profile`) |
| `rounds/{N}/constraints.md` | T=2 |
| `rounds/{N}/mining/bottleneck_analysis.md` | T=4 |
| `rounds/{N}/profiling/ncu/sanity.csv` | T=6 (targeted, only for physical-ceiling claims) |

### ammo-champion (debate)
| Writes to | When |
|-----------|------|
| `rounds/{N}/debate/proposals/{champion_id}_proposal.md` | Phase 0 |
| `rounds/{N}/debate/round_{D}/{op_id}_argument.md` | Phase A |
| `rounds/{N}/debate/round_{D}/{op_id}_critique_{target}.md` | Phase B |
| `rounds/{N}/debate/round_{D}/{op_id}_rebuttal.md` | Phase C |
| `rounds/{N}/debate/micro_experiments/{champion_id}_*.py` | Phase 0 (optional) |

### ammo-impl-champion
| Writes to | When |
|-----------|------|
| `rounds/{N}/tracks/{op_id}/validation_results.md` | After all gates |
| `rounds/{N}/tracks/{op_id}/validator_tests/*` | Kernel correctness & speedup tests |
| `rounds/{N}/tracks/{op_id}/_scratch/*` | During iteration |
| `rounds/{N}/sweeps/opt/{op_id}/*` | Via sweep script (Gates 5.1b/5.3) |

### ammo-transcript-monitor (impl-stage only)
| Writes to | When |
|-----------|------|
| `rounds/{N}/tracks/{op_id}/monitor_audits/{monitor_id}_observations.md` | Impl monitoring |

### ammo-auditor
| Writes to | When |
|-----------|------|
| `rounds/{N}/audits/stage_{1\|45\|67}.md` | After each audit gate |

### ammo-report-writer
| Writes to | When |
|-----------|------|
| `REPORT.md` | Terminal only |
| `report_assets/*.png` | Terminal only |
| `report_assets/gen_*.py` | Terminal only |

---

## Authoritative vs Diagnostic Artifacts

### Authoritative (consumed by downstream stages)

| Path | Consumer | Gate |
|------|----------|------|
| `state.json` | All agents, all stages | — |
| `rounds/{N}/sweeps/baseline/e2e_latency_results.json` | Debate, impl-champions, integration | T5 |
| `rounds/{N}/sweeps/baseline/json/golden_refs.json` | Stage 5.1b correctness | Gate 5.1b |
| `rounds/{N}/mining/bottleneck_analysis.md` | Debate champions, routing | T5 |
| `rounds/{N}/debate/summary.md` | Report writer (DERIVED) | — |
| `rounds/{N}/sweeps/opt/{op_id}/e2e_latency_results.json` | Orchestrator, integration | Gate 5.3b |
| `rounds/{N}/tracks/{op_id}/validation_results.md` | Orchestrator, auditor, report writer | T9 |
| `rounds/{N}/tracks/{op_id}/validator_tests/gate_5_1a_results.json` | Orchestrator state merge | Gate 5.1a |
| `rounds/{N}/tracks/{op_id}/validator_tests/gate_5_2_results.json` | Orchestrator state merge | Gate 5.2 |
| `rounds/{N}/sweeps/integration/e2e_latency_results.json` | Campaign eval, cumulative speedup | Stage 6 |
| `rounds/{N}/sweeps/golden_capture/json/golden_refs.json` | Next round's Stage 5.1b | Post-SHIP |
| `rounds/{N}/constraints.md` | Debate champions (current round) | — |

### Diagnostic (informational, not pipeline-consumed)

| Path | Purpose | Confusion Risk |
|------|---------|----------------|
| `rounds/{N}/profiling/ncu/sanity.csv` | HW counter sanity | MEDIUM — latency column is NOT baseline |
| `rounds/{N}/debate/micro_experiments/*` | Feasibility evidence | LOW |
| `rounds/{N}/tracks/{op_id}/_scratch/*` | Champion iteration | MEDIUM — drafts look like finals |
| `rounds/{N}/tracks/{op_id}/monitor_audits/*` | DA enforcement (impl-stage) | LOW |
| `rounds/{N}/_archive/*` | Superseded runs | LOW (quarantined) |
| `rounds/{N}/audits/*` | Audit trail | LOW |

---

## Disambiguation Rules

When multiple candidate files exist:

1. **"Which is the baseline E2E?"** → `rounds/{current_round}/sweeps/baseline/e2e_latency_results.json`. NEVER a profiling artifact. NEVER an `_archive/` dir.

2. **"Which nsys trace is the real baseline profile?"** → `rounds/{N}/profiling/nsys/baseline_bs{BS}.nsys-rep` or the matching `baseline_profile*.nsys-rep` companion path. Legacy probe directories are compatibility-only and are not Stage 2 ranking inputs.

3. **"Which is the authoritative track verdict?"** → `rounds/{N}/tracks/{op_id}/validation_results.md`. NEVER anything in `_scratch/`.

4. **"Which opt sweep is authoritative?"** → `rounds/{N}/sweeps/opt/{op_id}/e2e_latency_results.json`. Only one per track per round (superseded runs archived).

5. **"What are the golden refs for correctness?"** → For the CURRENT round's opt sweeps: `rounds/{N}/sweeps/baseline/json/golden_refs.json`. For round N+1 after SHIP: `rounds/{N}/sweeps/golden_capture/json/golden_refs.json`.

6. **"Is this file authoritative or derived?"** → `debate/summary.md` is DERIVED (regenerated from `state.json` by `render_debate_summary.py`). If it disagrees with `state.json.campaign.rounds[N-1].debate.selected_candidates`, state.json wins.

---

## Prohibited Patterns

The following are explicitly disallowed:

1. **Writing to campaign root** (except `state.json`, `target.json`, `REPORT.md`, `report_assets/`)
2. **Timestamped directory names in active slots** (only `_archive/` may carry timestamps)
3. **Ad-hoc `--out-name`** on sweep script (use `--round` + `--slot`)
4. **`monitor_log_*` at campaign root** — must be under `monitor_audits/`
5. **`validation_results_DRAFT.md`** or similar at track root — use `_scratch/`
6. **Nested `kernel_opt_artifacts/` paths** — monitors must receive absolute `output_dir`
7. **`e2e_latency_opt*`, `e2e_latency_combined/`** at campaign root — use semantic slots
8. **`investigation/`, `runs/`, `monitoring/`** directories at root — removed from scaffold
9. **Writing nsys traces into sweep output** — traces go to `profiling/nsys/`, not `sweeps/*/nsys/`
10. **`debate/campaign_round_{N}/` nesting** — replaced by top-level `rounds/{N}/debate/`
11. **`audits/stage_{N}_round_{M}.md` at campaign root** — replaced by `rounds/{M}/audits/stage_{N}.md`
12. **`bottleneck_analysis.md` at campaign root** — replaced by `rounds/{N}/mining/bottleneck_analysis.md`
13. **`constraints.md` at campaign root** — replaced by `rounds/{N}/constraints.md`

---

## Layout Detection (for scripts)

Scripts that need to support both v1 (legacy flat) and v2 (round-scoped) layouts use a single filesystem check:

```python
def _is_v2_layout(artifact_dir: Path) -> bool:
    return (artifact_dir / "rounds").is_dir()
```

This check is independent of `state.json["schema_version"]`. The schema version remains `"4.1"`; layout is detected by filesystem presence of `rounds/`.

When `_is_v2_layout(artifact_dir)` is `True`, scripts MUST emit v2 paths. When `False`, scripts MAY fall back to legacy paths (typically only relevant for old campaigns started before this spec).

---

## Hook Enforcement

A PostToolUse warn hook (`hooks/ammo-artifact-layout-warn.sh`) emits a non-blocking warning when files are created outside the allowed regex patterns. Allowed patterns mirror the tree above; the canonical regex list lives in the spec.

Op-id pattern: `[A-Za-z0-9_-]+` (supports `OP-001`, `op007`, etc.).
