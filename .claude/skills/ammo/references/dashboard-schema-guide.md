# AMMO Dashboard Schema Guide

> **Scope:** What the LIGHTGRID campaign dashboard **reads** from `state.json`.  
> This is a reference for dashboard developers — not a specification for the orchestrator.  
> The orchestrator produces `state.json`; the dashboard reads it with safe defaults for every field.

---

## Artifact Directory Convention

The dashboard locates `state.json` by globbing inside the session's git worktree:

```
kernel_opt_artifacts/{model_id}_{hardware}_{dtype}_tp{tp}/state.json
```

Fallback: `state.json` at the worktree root.

The same directory is used as the base for all artifact paths returned by  
`GET /api/campaigns/{session_id}/artifacts/{path}`.

---

## Root-Level Fields

The dashboard reads these top-level fields from `state.json`:

| Field | Type | Used for |
|-------|------|----------|
| `target` | object | L1 card: model, hardware, dtype, TP |
| `stage` | string | Pipeline progress position, breadcrumb |
| `parallel_tracks` | dict[op_id → TrackState] | L2 node status, L3 track table |
| `debate` | object | L2 debate column, L3 debate rationale |
| `integration` | object | L2 integration column, L3 integration section |
| `stage_timestamps` | dict[stage → {started_at, completed_at}] | (reserved, not rendered yet) |
| `campaign` | object | L1 speedup, current round, round history |
| `gpu_resources` | object | (not currently rendered) |
| `session_id` | string | Cross-reference only |
| `summary` | string | (not currently rendered) |

### `target`

```json
{
  "model_id": "Qwen/Qwen3-Coder-480B",
  "hardware": "A100",
  "dtype": "bf16",
  "tp": 4
}
```

All fields default to safe strings/numbers if missing.

### `stage`

String identifying the current pipeline stage. The dashboard uses this ordering:

| `stage` value | Column index | Display name |
|---------------|-------------|--------------|
| `1_baseline` | 0 | Baseline |
| `2_bottleneck_mining` | 1 | Mining |
| `3_debate` | 2 | Debate |
| `4_5_parallel_tracks` | 3 | Implement / 4 Validate |
| `6_integration` | 5 | Integrate |
| `7_campaign_eval`, `campaign_complete`, `campaign_exhausted` | 6 | (done) |

### `parallel_tracks`

Dict keyed by `op_id`. Each track:

```json
{
  "status": "IN_PROGRESS",       // "IN_PROGRESS" | "FAILED" | "SHIPPED"
  "verdict": "SPEEDUP",
  "classification": "memory_bound",
  "correctness": true,
  "kernel_speedup": 1.43,
  "e2e_speedup": 1.21,
  "fail_reason": null,
  "per_bs_verdict": {"bs1": "SPEEDUP", "bs4": "REGRESSION"},
  "validation_results_path": "attn_kernel/validation_results.md"
}
```

**Status normalisation** (dashboard applies this mapping):

| Raw value | Displayed as |
|-----------|-------------|
| `"IN_PROGRESS"` | `active` (cyan) |
| `"FAILED"` | `failed` (red + flickerRed animation) |
| op_id in `campaign.shipped_optimizations` | `shipped` (mint) |
| anything else | `active` |

**Graceful degradation:** missing `kernel_speedup`/`e2e_speedup` → shown as `—`.

### `debate`

```json
{
  "team_name": "debate-team-r3",
  "candidates": ["attn_flash", "kv_quant"],
  "rounds_completed": 2,
  "selected_winners": ["attn_flash"],
  "selection_rationale": "attn_flash achieves 1.43× kernel speedup...",
  "next_round_overlap": {}
}
```

- `candidates`: shown as individual debate nodes in L2 (col 2)
- `selected_winners`: highlighted green in L2 debate column
- `selection_rationale`: rendered as safe markdown in L3 Debate tab

### `integration`

```json
{
  "status": "completed",         // "pending" | "completed" | "failed"
  "passing_candidates": ["attn_flash"],
  "combined_patch_branch": "ammo/session-abc123/round-3-combined",
  "combined_e2e_result": {},
  "final_decision": {}
}
```

Shown in L2 integration column and L3 Integration section.

### `campaign`

```json
{
  "status": "active",
  "current_round": 3,
  "cumulative_e2e_speedup": 1.21,
  "shipped_optimizations": ["kv_quant"],
  "rounds": [
    {
      "round_id": 1,
      "selected_candidates": ["kv_quant"],
      "implementation_results": {
        "kv_quant": {"status": "PASSED", "e2e_speedup": 1.08, "reason": null}
      },
      "shipped": ["kv_quant"],
      "cumulative_speedup_after": 1.08,
      "profiling_baseline_path": "round_1/baseline_profile.csv",
      "top_bottleneck_share_pct": 62.3,
      "round_team_name": "round-1-team"
    }
  ]
}
```

---

## Current Round vs Past Rounds

### Current round (from root-level fields)

When `round_id == campaign.current_round`, the dashboard reads **root-level** fields for full detail:

- `parallel_tracks` → per-op status, speedup, correctness
- `debate` → candidates, winners, rationale
- `integration` → branch, passing candidates
- `stage` → which column is active

### Past rounds (from `campaign.rounds[]` summaries)

For completed rounds, only the `campaign.rounds[n]` summary is available:

| Field | Used for |
|-------|----------|
| `selected_candidates` | Node labels in L2 |
| `implementation_results[op_id].status` | `"PASSED"` → shipped, `"FAILED"` → failed |
| `shipped` | Mint colour in L2 |
| `cumulative_speedup_after` | Speedup bar chart in sidebar |

**Past rounds have no per-stage detail.** The dashboard renders compact summary nodes  
(no debate/validation breakdown) and gracefully shows `—` for missing speedup values.

---

## Graceful Degradation Rules

All models use `extra="ignore"` (Pydantic) so unknown future fields are silently dropped.

| Scenario | Dashboard behaviour |
|----------|-------------------|
| `parallel_tracks` missing | Empty track list; L2 shows "No tracks at this stage" |
| `debate.selection_rationale` missing | Debate tab hidden in L3 |
| `integration` missing/`status=="pending"` | Integration column shows ghost node |
| `campaign.rounds` empty | Single placeholder current round shown |
| `kernel_speedup`/`e2e_speedup` null | Shows `—` |
| Unknown `stage` value | Defaults to column 0 (Baseline) |
| `validation_results_path` null | Validation Report tab shows "File not found" |

---

## Artifact Path Conventions

Artifacts are served from `GET /api/campaigns/{session_id}/artifacts/{path}`.  
All paths are relative to the artifact directory (path traversal blocked by server).

### Per-operation artifacts (when `node = op_id`)

| Tab | Path pattern |
|-----|-------------|
| Validation Report | `{op_id}/validation_results.md` |
| Source Code | `{op_id}/kernel.py` |
| Profiling Data | `{op_id}/profiling_results.md` |
| Debate Rationale | `{op_id}/debate_rationale.md` |
| Correctness Report | `{op_id}/correctness_report.md` |
| Git Diff | `{op_id}/patch.diff` |

### Stage-level artifacts (when `node = stage-{N}`)

| Tab | Path |
|-----|------|
| Report | `REPORT.md` |
| Bottleneck | `bottleneck_report.md` |
| Debate Log | `debate_log.md` |

### Universal fallback

`state.json` is always available as the last tab.

### XSS safety

All fetched content is passed through `renderArtifactContent(content, mime)`:

- `text/markdown` → `DOMPurify.sanitize(marked.parse(content))`
- `text/x-python`, `application/json` → `hljs.highlight()` + HTML-escaped `<pre>`
- Other → HTML-escaped `<pre>`

Raw `x-html` is **never** used on unprocessed server strings.
