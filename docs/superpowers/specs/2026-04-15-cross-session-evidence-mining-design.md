# Cross-Session Evidence Mining for AMMO Paper

**Date**: 2026-04-15
**Status**: Approved
**Purpose**: Mine all 18 archived AMMO eval runs to produce citation-backed evidence for the research paper (`ammo_transcript_monitor_paper_draft.md`)

---

## 1. Goals

Two evidence objectives:

1. **Generalizability** (preempt N=1 objection) — aggregate intervention taxonomy, anti-patterns, and campaign metrics across 18 runs spanning 4 models (Qwen3.5-4B, GLM-5, Qwen3.5-35B, Nemotron-3-Nano), 2 hardware targets (L40S, B200), and multiple skill versions.
2. **Coaching evidence** (support novel Section 4.5d claim) — specific instances where the monitor used actor data constructively to suggest fix paths. This is the paper's strongest differentiation from prior monitoring literature.

## 2. Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary goal | Generalizability (A) + Coaching (C) | Defensive + offensive for reviewers |
| Paper form | Both: aggregate section 6.9 + inline examples | Tables for breadth, examples for depth |
| Data extraction method | Agent-mined (B) with citation requirement | Needs qualitative reading of monitor logs; structured JSON alone misses coaching context |
| Evidence constraint | All findings require `{file_path, line_start, line_end, claim, quoted_content}` citations or are rejected | Academic rigor; verified via `verify_citations.py` |
| Agent deliverables | Evidence-only JSON (A) — no draft paper text | Paper voice consistency; writing is fast once evidence is solid |
| Parallelization | Per-run agents (Approach 1), 6 at a time | Coaching instances require within-run narrative context |
| Changelog | Each paper update gets a changelog entry matching existing format | Maintains traceability established in prior edits |

## 3. Architecture

### 3.1 Phase 1: Per-Run Mining (18 agents, 6 concurrent)

Each agent receives one run and reads:
- `changes_snapshot/campaign_artifacts/` (monitor logs, debate summaries, validation results, state.json)
- `transcript_grading.json`
- `scorecard.json`
- `artifacts_snapshot.json`

Produces a structured JSON report with 9 categories:

| # | Category | Description | Required citation |
|---|---|---|---|
| 1 | Intervention inventory | Every monitor intervention: category (methodology/evidence/reasoning/correctness/coaching), severity (CRITICAL/WARNING/INFO) | Monitor log file:line where intervention text appears |
| 2 | Coaching instances | Monitor used actor's own data to suggest fix paths constructively | Both: monitor log line (suggestion) + artifact the monitor referenced (actor's data source) |
| 3 | Intervention timing | When intervention arrived relative to failure state (early/mid/late; before/after actor noticed) | Monitor log line + actor transcript evidence of timing |
| 4 | Validator independence | Validator derived tests from spec, not actor code | Validation results file:line showing independent derivation |
| 5 | Enforcement instances | What enforcement type triggered (deny/block/warn/advisory) | Hook output or monitor log referencing the enforcement |
| 6 | Campaign metrics | Rounds, tracks shipped/failed, cumulative speedup, termination reason, technology classes | state.json or campaign report lines |
| 7 | Anti-patterns | Named anti-patterns with cross-session pattern matching | transcript_grading.json anti_pattern entries with evidence |
| 8 | Impact classification | HIGH/MEDIUM/LOW/NONE per intervention with counterfactual + false positive/negative flags | Monitor log + outcome evidence (validation_results, ship/fail decision) |
| 9 | Changelog entries | Draft changelog entry for each finding that makes it into the paper (What changed / Why / evidence source) | Same citations as the underlying finding |

### 3.2 Output Schema (per-run)

```json
{
  "run_idx": 4,
  "slug": "327b55e1e_12-round-campaign-...",
  "run_dir": "/path/to/run",
  "model": "Qwen3.5-4B",
  "hardware": "L40S",
  "dtype": "BF16",
  "skill_version": "latest|refactored|baseline-v1",

  "interventions": [
    {
      "id": "int-001",
      "category": "coaching",
      "severity": "WARNING",
      "summary": "Monitor used actor's per-shape BW data to redirect from tiled layout",
      "timing": "mid-session, before actor had finalized proposal",
      "impact": "HIGH",
      "counterfactual": "Champion would have submitted tiled layout proposal",
      "false_positive": false,
      "false_negative": false,
      "citations": [
        {
          "file_path": "monitor_log_champion-1.md",
          "line_start": 786,
          "line_end": 790,
          "claim": "Monitor redirected champion from tiled layout",
          "quoted_content": "Per-shape BW results are in champion1_pershape_bw.log..."
        }
      ]
    }
  ],

  "coaching_instances": [
    {
      "id": "coach-001",
      "intervention_ref": "int-001",
      "actor_data_source": {
        "file_path": "micro_experiments/champion1_pershape_bw.log",
        "line_start": 1,
        "line_end": 15,
        "claim": "Actor's own experiment showed 80-83% BW per shape",
        "quoted_content": "shape_0: 81.2% BW..."
      },
      "monitor_suggestion": {
        "file_path": "monitor_log_champion-1.md",
        "line_start": 786,
        "line_end": 790,
        "claim": "Monitor synthesized actor's data into redirect suggestion",
        "quoted_content": "Per-shape BW results show all shapes at 80-83%..."
      },
      "outcome": "Champion pivoted to in-proj concatenation, which shipped"
    }
  ],

  "validator_independence": [],
  "enforcement_instances": [],

  "campaign_metrics": {
    "rounds": 12,
    "tracks_total": 10,
    "tracks_shipped": 5,
    "tracks_failed": 5,
    "cumulative_speedup": 1.422,
    "termination_reason": "amdahl_threshold",
    "technology_classes": ["triton", "cuda_cpp"],
    "citations": []
  },

  "anti_patterns": [],
  "changelog_entries": []
}
```

### 3.3 Phase 2: Aggregation (1 agent)

After all 18 mining agents complete, one aggregation agent reads all 18 JSON reports and produces:

| Deliverable | Paper destination |
|---|---|
| Cross-session intervention taxonomy table (counts + shares by category, N=18) | Section 6.9 |
| Cross-session impact distribution (HIGH/MEDIUM/LOW/NONE aggregate) | Section 6.9 |
| Campaign metrics summary (rounds, tracks, speedups, termination reasons) | Section 6.9 |
| Anti-pattern frequency table (which patterns recur across sessions) | Section 6.9 |
| False positive/negative rate (aggregate across all classified interventions) | Section 6.9 + reference in 5.5 |
| Ranked coaching examples — top 3-5 by impact, diverse models/hardware | Section 4.5d (inline) or 6.6 (Case Study D) |
| Validator independence examples | Section 4.2 (inline) |
| Enforcement spectrum examples | Section 4.7 (inline) |
| Consolidated changelog entries | Paper changelog section |

### 3.4 Phase 3: Verification

Run `verify_citations.py` against every agent's output:
```bash
python .claude/skills/ammo/eval/scripts/verify_citations.py \
  <agent_output.json> --artifact-dir <run_artifact_dir> --strict
```

Any finding with a failed citation gets flagged for manual review or dropped. Only verified evidence goes into the paper.

## 4. Paper Placement

| Finding | Paper location | Form |
|---|---|---|
| Aggregate intervention taxonomy (N=18) | New Section 6.9 "Cross-Session Analysis" | Table + 1 paragraph |
| Aggregate impact distribution (N=18) | Section 6.9 | Table + 1 paragraph |
| Campaign metrics summary | Section 6.9 | Table |
| Anti-pattern frequency across sessions | Section 6.9 | Table + 1 paragraph |
| False positive/negative rates | Section 6.9 + reference in 5.5 | Inline stats |
| Top coaching case studies (1-2) | Section 4.5d (inline example) or Section 6.6 as Case Study D | 1-2 paragraphs each |
| Validator independence instances | Section 4.2 (inline) | 1-2 sentences with citation |
| Enforcement spectrum examples | Section 4.7 (inline) | 1-2 sentences with citation |
| Changelog entries | Appendix: Changelog | Standard format |

Section 6.9 target length: 1-1.5 pages (tight, not a second deep dive).

## 5. Run Inventory

Source: `/tmp/regrade_manifest.json`

| Idx | Model | Hardware | Rounds | DAG | Score | Slug prefix |
|---|---|---|---|---|---|---|
| 0 | Qwen3.5-4B | L40S | 17 | Y | 7.04 | 0258cbea2 |
| 1 | Qwen3.5-4B | L40S | 1 | - | 6.04 | 06f309c83 |
| 2 | Qwen3.5-4B | L40S | 1 | - | 6.34 | 06f309c83 |
| 3 | GLM-5 | B200 | 1 | Y | 4.21 | 2075d7d1e |
| 4 | Qwen3.5-4B | L40S | 12 | Y | 5.88 | 327b55e1e |
| 5 | Qwen3.5-35B | L40S | FP8 | Y | 2.85 | 3ed2cc190 |
| 6 | Qwen3.5-35B | L40S | FP8 | Y | 2.29 | 3ed2cc190 |
| 7 | Qwen3.5-4B | L40S | R7-R9 | Y | 4.12 | 6da5e40a0 |
| 8 | Qwen3.5-4B | L40S | 5 | Y | 3.81 | 6ee86c3e2 |
| 9 | Qwen3.5-4B | L40S | baseline | - | 4.25 | 85d8ded68 |
| 10 | Qwen3.5-4B | L40S | refactored | - | 3.79 | 99a0fbed6 |
| 11 | Nemotron-3-Nano | L40S | FP8 | Y | 5.17 | a779aa294 |
| 12 | Qwen3.5-4B | L40S | R7-R9 | Y | 3.56 | bcde2c0c2 |
| 13 | Qwen3.5-4B | L40S | R7-R9 | Y | 4.85 | bcde2c0c2 |
| 14 | Qwen3.5-4B | L40S | test | - | 2.89 | c08b370fc |
| 15 | Qwen3.5-4B | L40S | 4 | Y | 4.52 | d0803f604 |
| 16 | Qwen3.5-4B | L40S | 2 | - | 8.65 | faaac90ef |
| 17 | Qwen3.5-4B | L40S | 4 | Y | 5.38 | fef621ebc |

## 6. Constraints

- All agent findings MUST include structured citations: `{file_path, line_start, line_end, claim, quoted_content}`
- Findings without valid citations are rejected — not included in aggregation
- All citations verified post-hoc via `verify_citations.py` (>70% fuzzy match threshold)
- Agents must NOT draft paper text — evidence-only JSON output
- Agents must NOT modify any source files
- 6 agents max concurrent (12 agents peak avoided since no baseline comparison needed this time)
