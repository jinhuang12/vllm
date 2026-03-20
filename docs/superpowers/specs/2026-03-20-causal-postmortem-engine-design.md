# Causal Post-Mortem Engine for AMMO Eval

**Date**: 2026-03-20
**Status**: Approved
**Location**: `.claude/skills/ammo/eval/causal/`

## Problem

The current AMMO eval subskill has three blind spots:

1. **E2E speedup at 40% weight rewards luck, not skill quality.** The same skill version produces wildly different E2E outcomes across runs due to stochastic debate selection, variable implementation success, and target-dependent optimization headroom.

2. **No causal attribution.** The eval treats the campaign as a black box: `skill_version → [?] → E2E speedup`. It cannot trace which agent decisions determined the outcome or distinguish skill quality from noise.

3. **The transcript grader is a defect detector, not a causal analyzer.** It catches hallucinated data and wasted retries but cannot trace data flow between agents, detect framing effects, identify ignored-correct-data, or attribute behavioral changes to specific skill edits.

## Solution

A **Causal Post-Mortem Engine** that runs alongside the existing eval pipeline (additive, not replacement). It builds an adaptive-depth causal DAG per campaign, traces data flow between agents, identifies trajectory-changing decisions, and diffs behavior across skill versions.

## Architecture

### Approach: Hybrid — Deterministic Core + Surgical LLM

The session JSONL is already structured (tool calls, file I/O, SendMessage are machine-parseable). We extract events deterministically and build the DAG automatically. LLM is reserved for the unstructured parts: assessing decision quality at anomaly nodes and generating narrative explanations.

### Pipeline

```
EXISTING EVAL (unchanged):
  parse_session_logs.py → parse_artifacts.py → score_campaign.py → archive → dashboard

NEW CAUSAL ENGINE (parallel):
  Step A: extract_events.py     Session data + JSONL → events.json
  Step B: build_dag.py          Events → causal_dag.json (coarse)
  Step C: score_nodes.py        DAG + snapshot → scored_dag.json + anomalies.json
  Step D: LLM Deep Dive         Anomaly transcript windows → deep_analysis.json
  Step E: generate_postmortem.py Merge → causal_dag.json (final) + narrative + viz
  Step F: diff_versions.py      Two DAGs → version_diff.json + regression_report.md
```

Steps A, B, C, E, F are pure Python — deterministic, reproducible. Step D is the only LLM call, reading ~20-50K tokens (flagged anomaly segments only). Step D can be skipped for fast/cheap runs.

### Integration Points

- `archive_run.py` stores `causal_dag.json` alongside existing `scorecard.json`
- `dashboard.html` gets a new "Causal Analysis" tab
- Existing 5-dimension scorecard is untouched

### Updated Archive Layout

```
runs/{target_slug}/run_1/
  scorecard.json
  report.md
  causal_dag.json          ← new
  postmortem_narrative.md  ← new
  causal_viz.html          ← new
  changes_snapshot/
    manifest.json
    patches/
    campaign_artifacts/
    skill.patch
```

## Component Design

### Step A: Event Extraction (`extract_events.py`)

**Input**: Session data JSON (output of existing `parse_session_logs.py`) + session JSONL file path
**Output**: `events.json` — structured event log

**Relationship to `parse_session_logs.py`**: Step A does NOT re-parse the JSONL from scratch. It accepts the Step 0 output (`session_data.json`) via `--session-data` and extends it with fine-grained events (`file_write`, `file_read`, `data_output`) that `parse_session_logs.py` does not extract. This ensures a single source of truth for agent IDs and stage timestamps. Stage assignments come from `session_data.json`'s pre-computed stage timestamps — Step A does not re-derive stage boundaries.

```python
@dataclass
class Event:
    timestamp: str           # ISO timestamp
    event_type: str          # "tool_call" | "file_write" | "file_read" | "send_message" | "agent_spawn" | "agent_complete" | "data_output"
    agent_id: str            # tool_use_id (UUID, authoritative identifier)
    agent_name: str | None   # Human-readable name (from input.name, may be null)
    agent_role: str          # "orchestrator" | "ammo-researcher" | "ammo-champion" | "ammo-impl-champion" | "ammo-impl-validator" | "ammo-delegate"
    stage: str               # AMMO stage: "1_baseline" | "2_mining" | "3_debate" | ... (from session_data stage timestamps)
    details: dict            # Event-type-specific payload
```

Event-type-specific details:

| Event Type | Details Fields |
|---|---|
| `tool_call` | `tool_name`, `args_summary`, `result_summary`, `success` |
| `file_write` | `path`, `size_bytes`, `content_hash` |
| `file_read` | `path` |
| `send_message` | `from_agent`, `to_agent`, `message_hash` |
| `agent_spawn` | `agent_name`, `agent_type`, `model`, `isolation` |
| `agent_complete` | `agent_name`, `total_tokens`, `duration_ms` |
| `data_output` | `datum_type`, `value`, `raw_text`, `source_file`, `source_line` |

#### Data Pattern Specification

The `data_output` event type requires heuristic extraction from agent text outputs. The following regex patterns define each datum type:

| Datum Type | Regex Pattern | Example Match | Tolerance for Citation Matching |
|---|---|---|---|
| `f_value` | `(\d{1,3}\.\d{1,2})\s*%\s*(of\s+(decode\|total\|latency))` | "29.8% of decode" | Relative: `\|a-b\|/max(a,b) < 0.005` (0.5%) |
| `kernel_timing` | `(\d+\.?\d*)\s*(µs\|us\|microsec)` | "2034 µs" | Relative: `\|a-b\|/max(a,b) < 0.02` (2%) |
| `speedup` | `(\d+\.\d+)\s*x\b` or `[+](\d+\.?\d*)\s*%` | "1.08x" or "+8.1%" | Relative: `\|a-b\|/max(a,b) < 0.01` (1%) |
| `memory_size` | `(\d+\.?\d*)\s*(MB\|GB\|KiB\|MiB\|GiB)` | "70.6 MB" | Relative: `\|a-b\|/max(a,b) < 0.01` (1%) |
| `bandwidth` | `(\d+\.?\d*)\s*(GB/s\|TB/s)` | "864 GB/s" | Relative: `\|a-b\|/max(a,b) < 0.02` (2%) |

All tolerances are relative: `|a - b| / max(|a|, |b|) < threshold`.

**Disambiguation**: When multiple upstream nodes produce data values matching within tolerance, prefer the node with the most recent `file_write` event chronologically (the freshest data source).

**Known limitation**: Heuristic extraction has false negatives — some data values in unstructured prose will be missed. The `data_claims_extracted` count is stored per-node so evaluators can audit the denominator.

### Step B: DAG Construction (`build_dag.py`)

**Input**: `events.json`
**Output**: `causal_dag.json` (coarse, stage-level)

**Node creation**: One node per AMMO stage per round, plus one per implementation track. Node boundaries determined by stage transitions in events (derived from `session_data.json` stage timestamps passed through Step A).

**Edge creation** — four types, all deterministic:

| Edge Type | Detection Method |
|---|---|
| `file_dependency` | Agent A has `file_write` event for path X, Agent B has `file_read` event for path X, B's read is after A's write |
| `message_dependency` | `send_message` events between agents |
| `data_citation` | Match `data_output` values in node A's outputs against `data_output` values in node B's inputs using per-datum-type tolerance (see Data Pattern Specification). When multiple upstream nodes match, prefer the most recent writer. |
| `decision_gate` | Structural — derived from AMMO stage ordering (hardcoded: debate → impl, mining → debate, etc.) |

**Coarse DAG** has 6-8 nodes per round: `baseline_capture`, `bottleneck_mining`, `debate`, `impl_{op_id}` (one per track), `integration`, `campaign_eval`.

### Step C: Node Scoring & Anomaly Detection (`score_nodes.py`)

**Input**: `causal_dag.json` + `events.json` + `artifacts_snapshot.json` (via `--snapshot`, for track outcome lookups)
**Output**: `scored_dag.json` + `anomalies.json`

**Per-node metrics** (all 0.0-1.0, computed deterministically):

| Metric | Computation | Anomaly Threshold |
|---|---|---|
| `data_grounding` | Count of `data_output` values in this node's outputs that match (via `data_citation` edges) at least one upstream `data_output` value, divided by `data_claims_extracted` for this node. Both numerator and denominator use the same heuristic extractor — see Known Limitation below. | < 0.5 |
| `output_influence` | Count of downstream nodes that consumed this node's outputs (via any edge type), divided by expected downstream consumers | < 0.3 |
| `prediction_accuracy` | For nodes with extractable predictions: `min(actual/predicted, predicted/actual)`, clamped to [0,1]. See Prediction Extraction below. Null if no prediction extractable — null does NOT trigger anomalies. | < 0.3 (only when non-null) |
| `retry_rate` | Count of failed tool calls / total tool calls at this node | > 0.3 |
| `transition_quality` | Average of this node's 4 non-null metrics minus the average of immediate downstream nodes' 4 non-null metrics. Positive = quality dropped after this node. | Δ > 0.3 |

**Known limitation (data_grounding)**: Both numerator and denominator are computed by the same heuristic extractor. Missed extractions (false negatives) inflate the metric artificially. The `data_claims_extracted` count is stored in node metrics so evaluators can audit: if a proposal has rich numeric content but `data_claims_extracted` is low, the extractor likely missed claims.

**Prediction Extraction**: For debate and proposal nodes, scan outputs for speedup predictions using patterns: `estimat(e|ed)\s+.*(\d+\.?\d*)(%|x)`, `expect(ed)?\s+.*(\d+\.?\d*)(%|x)`, `predict(ed)?\s+.*(\d+\.?\d*)(%|x)`, `~(\d+\.?\d*)(%|x)\s+(E2E|speedup|improvement)`. For range predictions (e.g., "3-5%"), use the midpoint. If no prediction is extractable, `prediction_accuracy` is null. Actual outcomes are read from `artifacts_snapshot.json` track results.

**Anomaly classification** (deterministic, based on metric combinations):

| Type | Condition | Meaning |
|---|---|---|
| `DATA_CORRUPTION` | `data_grounding < 0.5` AND any downstream node has `prediction_accuracy < 0.3` (non-null) | Bad data entered here and poisoned downstream |
| `QUALITY_DROP` | `transition_quality > 0.3` | Upstream was good, something went wrong at this node |
| `WASTED_EFFORT` | `output_influence < 0.3` AND `retry_rate < 0.1` | Work was done well but ignored downstream |
| `THRASHING` | `retry_rate > 0.3` | Agent struggled, burned tokens on retries |
| `LUCKY_WIN` | `prediction_accuracy` is non-null AND `< 0.3` AND node's track outcome is PASS (from `artifacts_snapshot.json`) | Succeeded for wrong reasons — not reproducible |

### Step D: LLM Deep Dive (Subagent)

**Input**: `anomalies.json` + relevant transcript segments
**Output**: `deep_analysis.json`

For each anomaly node:
1. Extract transcript window: ~2K tokens before + after the anomaly node's events from the session JSONL
2. Include: the node's input artifacts, the node's output artifacts, downstream consumer's output
3. Prompt the LLM subagent with anomaly type + evidence + transcript window
4. Also trigger adaptive DAG expansion: build agent-decision-level child nodes within the anomaly stage

LLM produces per-anomaly:

```json
{
  "node_id": "debate",
  "anomaly_type": "DATA_CORRUPTION",
  "root_cause": "Champion-3 cited f=22% for MoE dispatch but profiling shows 14.8%",
  "causal_chain": [
    "Champion-3 hallucinated f-value",
    "Inflated feasibility score in selection",
    "op003 selected over better candidate",
    "op003 failed correctness → round wasted"
  ],
  "counterfactual": "Without hallucinated f-value, op004 likely selected; targeted 6.2% bottleneck with sound roofline math",
  "skill_attribution": "agent_behavior",
  "actionable_fix": "Require champions to include exact source line from bottleneck_analysis.md for every f-value claim",
  "confidence": 0.85
}
```

`skill_attribution` is one of: `skill_guidance` (the skill text caused this), `agent_behavior` (agent deviated from skill), `input_data` (profiling data was ambiguous/wrong), `stochastic` (random variance, not attributable).

**Skippable**: When `--skip-deep-analysis` is passed, Step D is skipped entirely. The scored DAG and anomaly classifications are still produced (deterministic), just without LLM explanations.

#### Causal Analyzer Rubric (`causal_analyzer.md`)

The LLM subagent receives a rubric defining:

**Input format**: For each anomaly, the subagent receives:
- `node_id`, `anomaly_type`, `triggered_by` (which metric(s) tripped)
- `upstream_data`: the node's input artifacts (file contents or excerpts)
- `node_output`: the node's output artifacts
- `downstream_output`: the immediate downstream consumer's output
- `transcript_window`: ~2K tokens of agent conversation around the anomaly events
- `events_window`: structured events from the anomaly node

**Required output fields** per anomaly (all required, fail validation if missing):
- `node_id` (string): which node this analysis is for
- `anomaly_type` (string): echo back the classified type
- `root_cause` (string): 1-2 sentence explanation of what went wrong or right
- `causal_chain` (list of strings): ordered sequence of cause → effect steps
- `counterfactual` (string): what would likely have happened if the root cause were different
- `skill_attribution` (enum): `"skill_guidance"` | `"agent_behavior"` | `"input_data"` | `"stochastic"`
- `actionable_fix` (string): specific change to skill text, agent prompt, or process that would prevent recurrence. Must reference a concrete section/line if attributing to skill_guidance.
- `confidence` (float 0-1): how confident the analysis is, based on evidence quality

**Per-anomaly-type reasoning instructions**:
- `DATA_CORRUPTION`: Cross-reference every numeric claim in the node's output against `upstream_data`. Identify the exact value that diverges. Trace whether downstream nodes consumed the corrupt value.
- `QUALITY_DROP`: Compare the quality of upstream node's output with this node's output. Identify what information was available but not used, or what reasoning step went wrong.
- `WASTED_EFFORT`: Identify why downstream nodes did not consume this node's output. Was it inaccessible? Irrelevant? Contradicted by other sources?
- `THRASHING`: Identify the retry loop pattern. Was the root cause a skill instruction, a tooling issue, or an agent reasoning error?
- `LUCKY_WIN`: Identify what the agent predicted, what actually happened, and why the outcome was positive despite wrong reasoning. Assess reproducibility.

### Step E: Post-Mortem Generation (`generate_postmortem.py`)

**Input**: `scored_dag.json` + `deep_analysis.json` (optional) + `events.json`
**Output**: `causal_dag.json` (final, merged) + `postmortem_narrative.md` + `causal_viz.html`

1. Merge deep analysis into scored DAG nodes (attach `deep_analysis` field to anomaly nodes)
2. Identify critical path: longest chain of high-influence nodes from baseline to outcome
3. Identify trajectory-changing decisions: nodes where anomaly + high downstream impact
4. Generate `postmortem_narrative.md` from template:
   - Campaign Outcome (factual summary)
   - Critical Path (the decision chain that determined the result)
   - Trajectory-Changing Decisions (positive and negative, with attribution)
   - Anomalies (each with root cause, chain, impact, fix)
   - Skill Improvement Recommendations (aggregated actionable fixes)
   - Cross-Version Delta (if previous version DAG available)
5. Generate `causal_viz.html`: static HTML with embedded `causal_dag.json`, using dagre-d3 for force-directed layout. Click to expand nodes, hover for edge details, sidebar for node metrics/analysis.

### Step F: Version Diffing (`diff_versions.py`)

**Input**: Two `causal_dag.json` files via `--current-dag` and `--previous-dag` (paths to archived DAGs)
**Output**: `version_diff.json` + `regression_report.md` (optional, requires LLM)

**Automatic structural diff** (deterministic, runs on every eval):

- Per-stage metric deltas (e.g., `debate.data_grounding: +0.15`)
- New anomalies not present in previous version
- Resolved anomalies from previous version
- Edge count changes by type (e.g., `data_citation` edges: 12 → 8)
- Anomaly type distribution shift

**Deep investigation** (LLM, on-demand via `--deep` or auto-triggered when ≥2 new anomalies):

1. Load the **skill diff** between versions (already in archive from `snapshot_changes.py`)
2. Load regressed node transcript segments from both versions
3. LLM reads: skill diff + behavioral diff + transcript excerpts
4. LLM produces: `skill_change_impact` (narrative), `regression_chains` (skill change → behavior → outcome), `recommendation`

## DAG JSON Schema

```json
{
  "version": "1.0",
  "campaign": {
    "target": { "model_id": "...", "hardware": "...", "dtype": "...", "tp": 1 },
    "skill_commit": "abc123",
    "round": 1,
    "outcome": "campaign_exhausted"
  },
  "nodes": [
    {
      "id": "bottleneck_mining",
      "depth": "stage",
      "parent": null,
      "agent_id": "550e8400-e29b-41d4-a716-446655440000",
      "agent_name": "researcher-1",
      "agent_role": "ammo-researcher",
      "stage": "2_bottleneck_mining",
      "inputs": ["constraints.md", "nsys_profiles/"],
      "outputs": ["bottleneck_analysis.md"],
      "event_range": { "start_idx": 142, "end_idx": 387 },
      "data_claims_extracted": 12,
      "metrics": {
        "data_grounding": 0.85,
        "output_influence": 0.92,
        "prediction_accuracy": 0.71,
        "retry_rate": 0.05,
        "transition_quality": null
      },
      "anomaly": null,
      "deep_analysis": null,
      "children": []
    }
  ],
  "edges": [
    {
      "from": "bottleneck_mining",
      "to": "debate",
      "type": "file_dependency",
      "artifact": "bottleneck_analysis.md",
      "data_values_transferred": ["f_moe_dispatch=29.8%", "f_mamba_ssm=31.2%"]
    }
  ],
  "summary": {
    "total_nodes": 7,
    "expanded_nodes": 1,
    "anomalies": { "DATA_CORRUPTION": 1, "QUALITY_DROP": 0, "WASTED_EFFORT": 0, "THRASHING": 0, "LUCKY_WIN": 0 },
    "trajectory_changing_decisions": ["debate.champ3_propose"],
    "critical_path": ["baseline_capture", "bottleneck_mining", "debate", "impl_op003", "integration"],
    "actionable_fixes": ["Require exact source citation for f-value claims"]
  }
}
```

## Version Diff JSON Schema

```json
{
  "version": "1.0",
  "compared": {
    "current": { "version_id": "v1.3", "skill_commit": "def456" },
    "previous": { "version_id": "v1.2", "skill_commit": "abc123" }
  },
  "metric_deltas": {
    "bottleneck_mining": { "data_grounding": 0.15, "output_influence": -0.03 },
    "debate": { "data_grounding": -0.25, "prediction_accuracy": -0.18 }
  },
  "new_anomalies": [
    { "node": "debate", "type": "DATA_CORRUPTION", "not_in_previous": true }
  ],
  "resolved_anomalies": [
    { "node": "bottleneck_mining", "type": "THRASHING", "was_in_previous": true }
  ],
  "edge_changes": {
    "data_citation": { "previous": 12, "current": 8, "delta": -4 }
  },
  "anomaly_distribution_shift": {
    "previous": { "THRASHING": 1 },
    "current": { "DATA_CORRUPTION": 2 }
  },
  "deep_investigation": null
}
```

## EVAL-SKILL.md Integration

Add Steps 3b-3e to the existing eval pipeline (between Step 3 and Step 4):

```
Step 3b: Extract Events + Build Causal DAG
  python .claude/skills/ammo/eval/causal/extract_events.py \
    --session-id <SESSION_ID> \
    --session-data /tmp/ammo_eval_session_data.json \
    --output /tmp/ammo_eval_events.json

  python .claude/skills/ammo/eval/causal/build_dag.py \
    --events /tmp/ammo_eval_events.json \
    --artifact-dir <ARTIFACT_DIR> \
    --output /tmp/ammo_eval_causal_dag.json

Step 3c: Score Nodes + Detect Anomalies
  python .claude/skills/ammo/eval/causal/score_nodes.py \
    --dag /tmp/ammo_eval_causal_dag.json \
    --events /tmp/ammo_eval_events.json \
    --snapshot /tmp/ammo_eval_snapshot.json \
    --output /tmp/ammo_eval_scored_dag.json \
    --anomalies /tmp/ammo_eval_anomalies.json

Step 3d: LLM Deep Dive (optional, skip with --skip-deep-analysis)
  Spawn subagent with:
    Read the causal analyzer rubric at .claude/skills/ammo/eval/causal/agents/causal_analyzer.md,
    then analyze anomalies at /tmp/ammo_eval_anomalies.json
    with events at /tmp/ammo_eval_events.json
    and artifacts at <ARTIFACT_DIR>.
    Write deep_analysis.json to /tmp/ammo_eval_deep_analysis.json.

Step 3e: Generate Post-Mortem
  python .claude/skills/ammo/eval/causal/generate_postmortem.py \
    --scored-dag /tmp/ammo_eval_scored_dag.json \
    --deep-analysis /tmp/ammo_eval_deep_analysis.json \
    --events /tmp/ammo_eval_events.json \
    --output-dag /tmp/ammo_eval_causal_dag_final.json \
    --output-narrative /tmp/ammo_eval_postmortem.md \
    --output-viz /tmp/ammo_eval_causal_viz.html

Step 3f: Cross-Version Diff (if previous version exists in archive)
  python .claude/skills/ammo/eval/causal/diff_versions.py \
    --current-dag /tmp/ammo_eval_causal_dag_final.json \
    --previous-dag ~/.claude/ammo-eval/versions/<prev>/runs/<target>/run_1/causal_dag.json \
    --output /tmp/ammo_eval_version_diff.json \
    [--deep --skill-diff ~/.claude/ammo-eval/versions/<prev>/meta/skill.patch]
```

Step 5 (archive) stores the causal artifacts alongside existing scorecard.
Step 6 (dashboard) includes the causal viz in a new tab.

## File Structure

```
.claude/skills/ammo/eval/causal/
├── extract_events.py          # Step A: session_data + JSONL → events.json
├── build_dag.py               # Step B: events → causal DAG
├── score_nodes.py             # Step C: scoring + anomaly detection
├── generate_postmortem.py     # Step E: merge + narrative + viz
├── diff_versions.py           # Step F: cross-version comparison
├── agents/
│   └── causal_analyzer.md     # LLM deep-dive rubric (Step D)
├── templates/
│   ├── narrative_template.md  # Post-mortem report template
│   └── viz_template.html      # Interactive DAG visualization template
└── tests/
    ├── test_extract_events.py
    ├── test_build_dag.py
    ├── test_score_nodes.py
    └── fixtures/               # Sample JSONL + expected outputs
```

## Testing Strategy

- **Unit tests** for each deterministic script (A, B, C, E, F) using fixture JSONL data
- Fixtures derived from actual campaign session logs (Nemotron + Qwen campaigns)
- LLM deep dive (Step D) tested by checking output schema conformance, not content quality
- Integration test: run full pipeline on a real archived campaign, verify output schemas

## Edge Cases

- **Multi-round campaigns**: Each round gets its own DAG subtree. Cross-round edges connect via re-profiling nodes. The narrative summarizes per-round and cross-round patterns.
- **Overlapped debate**: Debate nodes for round N+1 appear alongside implementation nodes for round N. The temporal overlap means round N+1 debate starts before round N implementations complete. The debate's profiling inputs are from round N's `baseline_capture` (not round N's implementation outputs). A `decision_gate` edge from round N's `campaign_eval` to round N+1's `debate` captures the data dependency on the final selection decision, while the overlapped execution is represented by a temporal annotation on the edge.
- **Campaign with no anomalies**: All nodes green, narrative is short ("Clean campaign execution, no trajectory-changing anomalies detected"). Diff still runs.
- **Missing session JSONL**: If JSONL is unavailable, the causal engine cannot run. Fail gracefully with a message; existing eval proceeds normally.
- **Truncated session JSONL**: The JSONL may end abruptly (campaign interrupted). The parser treats all lines up to the first unparseable line as valid. Stage timestamps for in-progress stages will be absent; those stages are marked `status: "incomplete"` in the DAG.
- **GATED_PASS tracks**: The `prediction_accuracy` metric accounts for per-BS verdict complexity — a track that predicted "speedup" but achieved GATED_PASS (some BS regressed) gets partial accuracy credit.

## Non-Goals

- Replacing the existing 5-dimension scorecard (this is additive)
- Automated skill editing (recommendations are for humans to act on)
- Real-time monitoring during campaigns (this is post-mortem only)
- Scoring the orchestrator's behavior (focus is on subagent quality)
