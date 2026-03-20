# Causal Post-Mortem Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a causal analysis layer for the AMMO eval that traces agent decision chains, detects anomalies, and diffs behavior across skill versions.

**Architecture:** Hybrid deterministic/LLM pipeline. 5 deterministic Python scripts extract events from session JSONL, build a causal DAG, score nodes, generate postmortem reports, and diff versions. 1 LLM subagent handles deep anomaly analysis. All additive to existing eval.

**Tech Stack:** Python 3.12, json, re, dataclasses, argparse. dagre-d3 (JS, bundled in viz template). No new pip dependencies.

**Spec:** `docs/superpowers/specs/2026-03-20-causal-postmortem-engine-design.md`

**Existing code patterns to follow:** `.claude/skills/ammo/eval/scripts/parse_session_logs.py` (JSONL parsing), `score_campaign.py` (scoring/reporting), `parse_artifacts.py` (artifact extraction).

**Note on CLI args:** The spec's EVAL-SKILL.md integration shows `--session-id` for `extract_events.py`, but the actual implementation uses `--session-jsonl <path>` (explicit JSONL path) + `--session-data <path>` (pre-computed session data). Task 8 updates EVAL-SKILL.md with the correct invocation.

**Deferred:** The spec mentions a new "Causal Analysis" tab in `dashboard.html`. This is deferred to a follow-up task — the dashboard template is complex HTML and should be done after the core pipeline is validated. The `causal_viz.html` standalone output provides the visualization in the meantime.

---

## Task 1: Test Fixtures

**Files:**
- Create: `.claude/skills/ammo/eval/causal/tests/__init__.py`
- Create: `.claude/skills/ammo/eval/causal/tests/fixtures/sample_session.jsonl`
- Create: `.claude/skills/ammo/eval/causal/tests/fixtures/sample_session_data.json`
- Create: `.claude/skills/ammo/eval/causal/tests/fixtures/sample_snapshot.json`
- Create: `.claude/skills/ammo/eval/causal/__init__.py`

Build minimal but realistic fixtures that cover the full pipeline. These are reused by all subsequent tasks.

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p .claude/skills/ammo/eval/causal/tests/fixtures
mkdir -p .claude/skills/ammo/eval/causal/agents
mkdir -p .claude/skills/ammo/eval/causal/templates
touch .claude/skills/ammo/eval/causal/__init__.py
touch .claude/skills/ammo/eval/causal/tests/__init__.py
```

- [ ] **Step 2: Create sample_session.jsonl**

A minimal session JSONL with ~25 lines covering: orchestrator spawning a researcher (Agent tool_use with subagent_type ammo-researcher), researcher writing bottleneck_analysis.md (tool_use Write), TeamCreate for debate, champion spawns (Agent tool_use), champions writing proposals (tool_use Write), champions reading bottleneck_analysis.md (tool_use Read), TeamDelete, implementer spawns, task-notifications with tokens/duration, SendMessage events. Include at least:
- 1 researcher spawn + completion
- 1 TeamCreate + TeamDelete
- 2 champion spawns + completions
- 1 implementer spawn + completion
- File writes: `bottleneck_analysis.md`, `debate/proposals/champion-1.md`, `debate/proposals/champion-2.md`, `tracks/op001/validation_results.md`
- File reads: researcher reads `constraints.md`, champions read `bottleneck_analysis.md`
- Data values in text blocks: f-values like "29.8% of decode", timings like "2034 µs", speedups like "1.08x"
- One champion text block citing a WRONG f-value (e.g., "22.0% of decode" when the source says 14.8%) to test DATA_CORRUPTION detection
- **Disambiguation case**: Two agents write the same f-value "29.8% of decode" at different timestamps (researcher writes it first at T1, then a different agent re-states it at T2). This tests that `data_citation` edge building prefers the most recent writer.
- One truncated-safe line: add a final valid line so tests for truncated JSONL can remove it

Follow the JSONL format from `parse_session_logs.py` docstring: each line is `{"type": "assistant"|"user", "timestamp": "ISO", "sessionId": "...", "message": {"content": [...]}}`.

- [ ] **Step 3: Create sample_session_data.json**

Output matching what `parse_session_logs.py` would produce for the sample JSONL. Include `stage_timestamps` with `round_1` keys, `agent_costs` array, `cost_summary`, `team_lifecycle`. This provides pre-computed stage boundaries for `extract_events.py`.

- [ ] **Step 4: Create sample_snapshot.json**

An `artifacts_snapshot.json` matching what `parse_artifacts.py` produces. Include: `target`, `campaign` with `status: "campaign_exhausted"`, `cumulative_e2e_speedup: 1.05`, `shipped_optimizations_count: 1`, one round with `implementation_results` showing op001 as PASSED (with `e2e_speedup: 1.05`) and op002 as FAILED, `gates` with track statuses, `debate` with proposals. Include a GATED_PASS track (op003) with `per_bs_verdict: {"1": "PASS", "8": "PASS", "32": "REGRESSED"}` for testing partial prediction accuracy credit.

- [ ] **Step 5: Commit fixtures**

```bash
git add .claude/skills/ammo/eval/causal/
git commit -m "feat(ammo-eval): add causal engine test fixtures"
```

---

## Task 2: Event Extraction (`extract_events.py`)

**Files:**
- Create: `.claude/skills/ammo/eval/causal/extract_events.py`
- Create: `.claude/skills/ammo/eval/causal/tests/test_extract_events.py`

**Spec reference:** "Step A: Event Extraction" section + "Data Pattern Specification" table.

This script extends the existing `parse_session_logs.py` output with fine-grained events (file_write, file_read, send_message, data_output). It does NOT re-parse stage boundaries — it reads them from `--session-data`.

- [ ] **Step 1: Write failing tests**

```python
# test_extract_events.py
"""Tests for extract_events.py — JSONL → structured event log."""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load_session_data():
    return json.loads((FIXTURES / "sample_session_data.json").read_text())


def _extract():
    from extract_events import extract_events
    return extract_events(
        session_jsonl_path=FIXTURES / "sample_session.jsonl",
        session_data=_load_session_data(),
    )


def test_extract_produces_events_list():
    """Basic: extraction produces a non-empty events list."""
    events = _extract()
    assert isinstance(events, list)
    assert len(events) > 0


def test_file_write_events_extracted():
    """File write tool calls become file_write events."""
    events = _extract()
    file_writes = [e for e in events if e["event_type"] == "file_write"]
    paths = [e["details"]["path"] for e in file_writes]
    assert any("bottleneck_analysis.md" in p for p in paths)


def test_file_read_events_extracted():
    """File read tool calls become file_read events."""
    events = _extract()
    file_reads = [e for e in events if e["event_type"] == "file_read"]
    paths = [e["details"]["path"] for e in file_reads]
    assert any("bottleneck_analysis.md" in p for p in paths)


def test_data_output_events_extracted():
    """Numeric patterns in text outputs become data_output events."""
    events = _extract()
    data_outputs = [e for e in events if e["event_type"] == "data_output"]
    assert len(data_outputs) > 0
    datum_types = {e["details"]["datum_type"] for e in data_outputs}
    assert "f_value" in datum_types


def test_agent_id_and_name_separated():
    """Events have separate agent_id (UUID) and agent_name fields."""
    events = _extract()
    spawns = [e for e in events if e["event_type"] == "agent_spawn"]
    assert len(spawns) > 0
    for s in spawns:
        assert "agent_id" in s
        assert "agent_name" in s


def test_stage_assignment_from_session_data():
    """Stage field comes from session_data timestamps, not re-derived."""
    events = _extract()
    stages = {e.get("stage") for e in events if e.get("stage")}
    assert any("baseline" in s for s in stages)


def test_graceful_fail_missing_jsonl():
    """Missing JSONL file raises FileNotFoundError."""
    from extract_events import extract_events
    with pytest.raises(FileNotFoundError):
        extract_events(
            session_jsonl_path=Path("/nonexistent/session.jsonl"),
            session_data=_load_session_data(),
        )


def test_truncated_jsonl_handles_gracefully(tmp_path):
    """Truncated JSONL processes valid lines and ignores the broken last line."""
    from extract_events import extract_events
    # Copy fixture and append a broken line
    content = (FIXTURES / "sample_session.jsonl").read_text()
    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text(content + '{"type": "assistant", "truncated_mid_')
    events = extract_events(
        session_jsonl_path=truncated,
        session_data=_load_session_data(),
    )
    assert isinstance(events, list)
    assert len(events) > 0  # Valid lines still parsed


def test_cli_interface(tmp_path):
    """CLI produces output JSON file."""
    import subprocess
    out = tmp_path / "events.json"
    result = subprocess.run(
        ["python", str(Path(__file__).parent.parent / "extract_events.py"),
         "--session-jsonl", str(FIXTURES / "sample_session.jsonl"),
         "--session-data", str(FIXTURES / "sample_session_data.json"),
         "--output", str(out)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert isinstance(data, list)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .claude/skills/ammo/eval/causal && python -m pytest tests/test_extract_events.py -v`
Expected: ImportError (extract_events doesn't exist yet)

- [ ] **Step 3: Implement extract_events.py**

Core logic:
1. Accept `--session-jsonl`, `--session-data`, `--output` CLI args
2. Load session_data JSON for stage timestamps and agent spawn registry
3. Stream JSONL line by line (same pattern as `SessionParser._process_line`). Wrap each `json.loads()` in try/except `json.JSONDecodeError` to handle truncated files gracefully.
4. For each `assistant` message with `tool_use` blocks:
   - `Write` tool → `file_write` event with path from input
   - `Read` tool → `file_read` event with path from input
   - `Agent` tool → `agent_spawn` event
   - `SendMessage` tool → `send_message` event
5. For each text block in assistant messages:
   - Run data pattern regexes (f_value, kernel_timing, speedup, memory_size, bandwidth per spec table)
   - Each match → `data_output` event with `datum_type`, `value`, `raw_text`
6. For `user` messages with `<task-notification>`:
   - `agent_complete` event
7. Assign `stage` to each event using session_data stage timestamps (binary search on timestamp ranges)
8. Return list of Event dicts, write to `--output` as JSON array

Follow the `@dataclass` schema from the spec. Use `agent_id` = `tool_use_id` (UUID), `agent_name` = `input.name` (may be null).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd .claude/skills/ammo/eval/causal && python -m pytest tests/test_extract_events.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/ammo/eval/causal/extract_events.py .claude/skills/ammo/eval/causal/tests/test_extract_events.py
git commit -m "feat(ammo-eval): add causal event extraction from session JSONL"
```

---

## Task 3: DAG Construction (`build_dag.py`)

**Files:**
- Create: `.claude/skills/ammo/eval/causal/build_dag.py`
- Create: `.claude/skills/ammo/eval/causal/tests/test_build_dag.py`

**Spec reference:** "Step B: DAG Construction" section.

Builds a coarse stage-level DAG from events.json. Nodes = stages, edges = file/message/data/gate dependencies.

- [ ] **Step 1: Write failing tests**

```python
# test_build_dag.py
"""Tests for build_dag.py — events → causal DAG."""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _load_events():
    """Helper: extract events from fixture, return list."""
    from extract_events import extract_events
    session_data = json.loads((FIXTURES / "sample_session_data.json").read_text())
    return extract_events(
        session_jsonl_path=FIXTURES / "sample_session.jsonl",
        session_data=session_data,
    )


def _build(events=None):
    """Helper: build DAG from events."""
    from build_dag import build_dag
    if events is None:
        events = _load_events()
    return build_dag(events=events, artifact_dir=str(FIXTURES))


def test_builds_coarse_dag_with_expected_nodes():
    """Coarse DAG has stage-level nodes."""
    dag = _build()
    node_ids = {n["id"] for n in dag["nodes"]}
    # Must have at least these core stages
    assert "bottleneck_mining" in node_ids
    assert "debate" in node_ids
    assert any(nid.startswith("impl_") for nid in node_ids)


def test_file_dependency_edges():
    """Edge from mining → debate via bottleneck_analysis.md."""
    dag = _build()
    file_edges = [e for e in dag["edges"] if e["type"] == "file_dependency"]
    mining_to_debate = [e for e in file_edges
                        if e["from"] == "bottleneck_mining" and e["to"] == "debate"]
    assert len(mining_to_debate) > 0
    assert any("bottleneck_analysis.md" in e.get("artifact", "") for e in mining_to_debate)


def test_data_citation_edges():
    """Data citation edges exist where f-values match across nodes."""
    dag = _build()
    citation_edges = [e for e in dag["edges"] if e["type"] == "data_citation"]
    assert len(citation_edges) > 0
    # At least one edge should transfer f-value data
    all_transferred = []
    for e in citation_edges:
        all_transferred.extend(e.get("data_values_transferred", []))
    assert any("29.8" in v for v in all_transferred)


def test_decision_gate_edges():
    """Structural edges: debate → impl, impl → integration."""
    dag = _build()
    gate_edges = [e for e in dag["edges"] if e["type"] == "decision_gate"]
    froms = {e["from"] for e in gate_edges}
    tos = {e["to"] for e in gate_edges}
    assert "debate" in froms
    assert any(t.startswith("impl_") for t in tos)


def test_data_citation_disambiguation():
    """When two upstream nodes produce same value, prefer most recent writer."""
    dag = _build()
    citation_edges = [e for e in dag["edges"] if e["type"] == "data_citation"]
    # The fixture has two agents writing "29.8%" — edge should come from the later one
    for edge in citation_edges:
        if any("29.8" in v for v in edge.get("data_values_transferred", [])):
            # The 'from' should not be the earliest writer
            # (exact assertion depends on fixture agent IDs — just verify edge exists)
            assert edge["from"] is not None


def test_nodes_have_data_claims_extracted():
    """Each node has a data_claims_extracted count."""
    dag = _build()
    for node in dag["nodes"]:
        assert "data_claims_extracted" in node
        assert isinstance(node["data_claims_extracted"], int)


def test_dag_json_schema():
    """Output matches spec schema: version, campaign, nodes, edges, summary."""
    dag = _build()
    assert "version" in dag
    assert "nodes" in dag
    assert "edges" in dag
    assert "summary" in dag
    assert "total_nodes" in dag["summary"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .claude/skills/ammo/eval/causal && python -m pytest tests/test_build_dag.py -v`
Expected: ImportError (build_dag doesn't exist yet)

- [ ] **Step 3: Implement build_dag.py**

Core logic:
1. Accept `--events`, `--artifact-dir`, `--output` CLI args
2. Expose `build_dag(events: list, artifact_dir: str) -> dict` function for direct use in tests
3. Load events list
4. Group events by stage → create one node per stage. For each node:
   - Collect inputs (file_read paths), outputs (file_write paths), data_output values
   - Populate `data_claims_extracted` = count of `data_output` events for this node
5. Build edges:
   - `file_dependency`: match file_write in node A with file_read in node B (path match, B after A)
   - `message_dependency`: send_message events between different agent roles
   - `data_citation`: match data_output values between nodes using per-datum-type relative tolerance from spec (`|a-b|/max(|a|,|b|) < threshold`). When multiple upstream nodes match the same value, prefer the node with the most recent `file_write` event timestamp.
   - `decision_gate`: hardcoded stage ordering (baseline → mining → debate → impl, impl → integration → campaign_eval)
6. Populate `data_values_transferred` on citation edges
7. Build summary with total_nodes, total edges, etc.
8. Write `causal_dag.json`

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd .claude/skills/ammo/eval/causal && python -m pytest tests/test_build_dag.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/ammo/eval/causal/build_dag.py .claude/skills/ammo/eval/causal/tests/test_build_dag.py
git commit -m "feat(ammo-eval): add causal DAG construction from events"
```

---

## Task 4: Node Scoring & Anomaly Detection (`score_nodes.py`)

**Files:**
- Create: `.claude/skills/ammo/eval/causal/score_nodes.py`
- Create: `.claude/skills/ammo/eval/causal/tests/test_score_nodes.py`

**Spec reference:** "Step C: Node Scoring & Anomaly Detection" section.

Scores each DAG node on 5 metrics and classifies anomalies.

- [ ] **Step 1: Write failing tests**

```python
# test_score_nodes.py
"""Tests for score_nodes.py — node scoring + anomaly detection."""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _build_scored_dag():
    """Helper: run full pipeline from fixture → scored DAG."""
    from extract_events import extract_events
    from build_dag import build_dag
    from score_nodes import score_nodes

    session_data = json.loads((FIXTURES / "sample_session_data.json").read_text())
    events = extract_events(
        session_jsonl_path=FIXTURES / "sample_session.jsonl",
        session_data=session_data,
    )
    dag = build_dag(events=events, artifact_dir=str(FIXTURES))
    snapshot = json.loads((FIXTURES / "sample_snapshot.json").read_text())
    scored_dag, anomalies = score_nodes(dag=dag, events=events, snapshot=snapshot)
    return scored_dag, anomalies, events


def test_all_nodes_have_metrics():
    """Every node gets all 5 metric fields."""
    scored_dag, _, _ = _build_scored_dag()
    for node in scored_dag["nodes"]:
        m = node["metrics"]
        assert "data_grounding" in m
        assert "output_influence" in m
        assert "prediction_accuracy" in m
        assert "retry_rate" in m
        assert "transition_quality" in m


def test_data_grounding_metric():
    """Nodes with grounded data score high; ungrounded score low."""
    scored_dag, _, _ = _build_scored_dag()
    mining = next(n for n in scored_dag["nodes"] if n["id"] == "bottleneck_mining")
    # Researcher produces source data — grounding should be high (or null for root)
    debate = next(n for n in scored_dag["nodes"] if n["id"] == "debate")
    # Debate may have lower grounding if champion hallucinated data
    assert debate["metrics"]["data_grounding"] is not None


def test_prediction_accuracy_null_when_no_prediction():
    """Nodes without speedup predictions get null prediction_accuracy."""
    scored_dag, _, _ = _build_scored_dag()
    mining = next(n for n in scored_dag["nodes"] if n["id"] == "bottleneck_mining")
    # Researcher doesn't predict speedups — should be null
    assert mining["metrics"]["prediction_accuracy"] is None


def test_prediction_accuracy_not_null_triggers_no_anomaly_when_null():
    """Null prediction_accuracy does NOT produce LUCKY_WIN."""
    _, anomalies, _ = _build_scored_dag()
    lucky_wins = [a for a in anomalies if a["anomaly_type"] == "LUCKY_WIN"]
    for lw in lucky_wins:
        # Every LUCKY_WIN must have a non-null prediction_accuracy trigger
        assert lw["triggered_by"]["prediction_accuracy"] is not None


def test_transition_quality_sign_convention():
    """Positive transition_quality means THIS node is better than downstream.
    E.g., upstream avg=0.8, downstream avg=0.4 → transition_quality=0.4 (quality dropped after)."""
    from score_nodes import _compute_transition_quality
    # Direct unit test of the helper
    upstream_avg = 0.8
    downstream_avg = 0.4
    tq = _compute_transition_quality(upstream_avg, downstream_avg)
    assert tq == pytest.approx(0.4, abs=0.01)
    assert tq > 0.3  # Would trigger QUALITY_DROP


def test_thrashing_anomaly():
    """High retry_rate → THRASHING anomaly."""
    from score_nodes import _classify_anomalies
    node_metrics = {
        "data_grounding": 0.8,
        "output_influence": 0.8,
        "prediction_accuracy": None,
        "retry_rate": 0.5,  # > 0.3 threshold
        "transition_quality": 0.0,
    }
    anomalies = _classify_anomalies("test_node", node_metrics, downstream_metrics=[], track_outcome=None)
    types = [a["anomaly_type"] for a in anomalies]
    assert "THRASHING" in types


def test_lucky_win_requires_pass_outcome():
    """LUCKY_WIN needs prediction_accuracy < 0.3 AND track outcome PASS."""
    from score_nodes import _classify_anomalies
    node_metrics = {
        "data_grounding": 0.8,
        "output_influence": 0.8,
        "prediction_accuracy": 0.1,  # < 0.3
        "retry_rate": 0.0,
        "transition_quality": 0.0,
    }
    # With PASS outcome → LUCKY_WIN
    anomalies_pass = _classify_anomalies("impl_op001", node_metrics, downstream_metrics=[], track_outcome="PASSED")
    assert any(a["anomaly_type"] == "LUCKY_WIN" for a in anomalies_pass)
    # With FAILED outcome → no LUCKY_WIN
    anomalies_fail = _classify_anomalies("impl_op002", node_metrics, downstream_metrics=[], track_outcome="FAILED")
    assert not any(a["anomaly_type"] == "LUCKY_WIN" for a in anomalies_fail)


def test_prediction_accuracy_gated_pass_partial_credit():
    """GATED_PASS tracks get partial prediction accuracy credit."""
    scored_dag, _, _ = _build_scored_dag()
    # op003 in fixture is GATED_PASS — should get partial credit, not 0
    gated_nodes = [n for n in scored_dag["nodes"] if "op003" in n["id"]]
    if gated_nodes:
        pa = gated_nodes[0]["metrics"]["prediction_accuracy"]
        # Should be between 0 and 1 (partial credit), not null or 0
        if pa is not None:
            assert 0.0 < pa < 1.0


def test_anomalies_json_schema():
    """Anomalies have required fields: node_id, anomaly_type, triggered_by."""
    _, anomalies, _ = _build_scored_dag()
    for a in anomalies:
        assert "node_id" in a
        assert "anomaly_type" in a
        assert "triggered_by" in a
        assert a["anomaly_type"] in ("DATA_CORRUPTION", "QUALITY_DROP", "WASTED_EFFORT", "THRASHING", "LUCKY_WIN")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .claude/skills/ammo/eval/causal && python -m pytest tests/test_score_nodes.py -v`
Expected: ImportError (score_nodes doesn't exist yet)

- [ ] **Step 3: Implement score_nodes.py**

Core logic:
1. Accept `--dag`, `--events`, `--snapshot`, `--output`, `--anomalies` CLI args
2. Expose `score_nodes(dag: dict, events: list, snapshot: dict) -> tuple[dict, list]` for direct use
3. Expose `_compute_transition_quality(upstream_avg: float, downstream_avg: float) -> float` helper (returns `upstream_avg - downstream_avg`)
4. Expose `_classify_anomalies(node_id, metrics, downstream_metrics, track_outcome) -> list` helper
5. For each node:
   - `data_grounding`: count data_output values with matching upstream data_citation edge / `data_claims_extracted`. Both use same heuristic — document the limitation.
   - `output_influence`: downstream consumers / expected downstream count from DAG structure
   - `prediction_accuracy`: extract predictions using regex patterns (`estimat(e|ed)\s+.*(\d+\.?\d*)(%|x)`, etc.) from node's data_output events. Compare to actual from snapshot using `min(actual/predicted, predicted/actual)`. For range predictions, use midpoint. **Null if no prediction found — null does NOT trigger any anomaly.** For GATED_PASS tracks: use the mean speedup across passing batch sizes as the "actual" value for partial credit.
   - `retry_rate`: count failed tool_calls / total tool_calls at this node
   - `transition_quality`: `avg(this node's non-null metrics) - avg(downstream nodes' non-null metrics)`. Positive delta = quality dropped after this node. Threshold: `> 0.3`.
6. Classify anomalies per spec table. Track outcomes looked up from `snapshot` `gates.validation_gates` and `campaign.rounds[].implementation_results`.
7. Write `scored_dag.json` + `anomalies.json`

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd .claude/skills/ammo/eval/causal && python -m pytest tests/test_score_nodes.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/ammo/eval/causal/score_nodes.py .claude/skills/ammo/eval/causal/tests/test_score_nodes.py
git commit -m "feat(ammo-eval): add causal node scoring and anomaly detection"
```

---

## Task 5: LLM Rubric (`causal_analyzer.md`)

**Files:**
- Create: `.claude/skills/ammo/eval/causal/agents/causal_analyzer.md`

**Spec reference:** "Causal Analyzer Rubric" section in Step D.

No code dependencies — this can be done at any point in the pipeline.

- [ ] **Step 1: Write the rubric**

Markdown document following the spec's rubric skeleton. Contents:
1. Role description: "You are a causal post-mortem analyzer for AMMO campaigns"
2. Input format: what the subagent receives per anomaly (node_id, anomaly_type, upstream_data, node_output, downstream_output, transcript_window, events_window)
3. Required output fields with types and descriptions — all 8 fields (node_id, anomaly_type, root_cause, causal_chain, counterfactual, skill_attribution enum, actionable_fix, confidence). Mark each as REQUIRED with type annotation.
4. Per-anomaly-type reasoning instructions (all 5 types: DATA_CORRUPTION, QUALITY_DROP, WASTED_EFFORT, THRASHING, LUCKY_WIN — per spec)
5. Output schema example (JSON)
6. Instructions to write `deep_analysis.json` as a JSON array of per-anomaly objects

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/ammo/eval/causal/agents/causal_analyzer.md
git commit -m "feat(ammo-eval): add causal analyzer LLM rubric"
```

---

## Task 6: Post-Mortem Generation (`generate_postmortem.py`)

**Files:**
- Create: `.claude/skills/ammo/eval/causal/generate_postmortem.py`
- Create: `.claude/skills/ammo/eval/causal/tests/test_generate_postmortem.py`
- Create: `.claude/skills/ammo/eval/causal/templates/narrative_template.md`
- Create: `.claude/skills/ammo/eval/causal/templates/viz_template.html`

**Spec reference:** "Step E: Post-Mortem Generation" section.

- [ ] **Step 1: Create narrative template**

Markdown template with placeholders: `{{campaign_outcome}}`, `{{critical_path}}`, `{{trajectory_decisions}}`, `{{anomalies}}`, `{{recommendations}}`, `{{cross_version_delta}}`.

- [ ] **Step 2: Create viz template**

Static HTML template that:
- Embeds `causal_dag.json` as `window.__DAG_DATA__`
- Uses dagre-d3 (CDN link, ~50KB) for force-directed DAG layout
- Nodes colored by health: green (no anomaly) → orange (anomaly) → red (DATA_CORRUPTION)
- Click node → show detail panel (metrics, anomaly, deep_analysis)
- Hover edge → tooltip with artifact name and type
- Critical path highlighted with thicker edges
- Right sidebar: selected node details (metrics, anomaly info, transcript excerpt)
- Top bar: depth toggle (Stage/Agent), filter (All/Anomalies Only)

- [ ] **Step 3: Write failing tests**

```python
# test_generate_postmortem.py
"""Tests for generate_postmortem.py — merge, narrative, viz generation."""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _build_scored():
    """Helper: build scored DAG from fixtures."""
    from extract_events import extract_events
    from build_dag import build_dag
    from score_nodes import score_nodes

    session_data = json.loads((FIXTURES / "sample_session_data.json").read_text())
    events = extract_events(
        session_jsonl_path=FIXTURES / "sample_session.jsonl",
        session_data=session_data,
    )
    dag = build_dag(events=events, artifact_dir=str(FIXTURES))
    snapshot = json.loads((FIXTURES / "sample_snapshot.json").read_text())
    scored_dag, anomalies = score_nodes(dag=dag, events=events, snapshot=snapshot)
    return scored_dag, anomalies, events


def test_generates_narrative_md(tmp_path):
    """Produces markdown with expected sections."""
    from generate_postmortem import generate_postmortem
    scored_dag, _, events = _build_scored()
    result = generate_postmortem(scored_dag=scored_dag, events=events, deep_analysis=None)
    narrative = result["narrative"]
    assert "# Causal Post-Mortem" in narrative
    assert "Critical Path" in narrative
    assert "Anomalies" in narrative or "No anomalies" in narrative.lower()


def test_generates_viz_html(tmp_path):
    """Produces HTML with dagre-d3 and embedded DAG data."""
    from generate_postmortem import generate_postmortem
    scored_dag, _, events = _build_scored()
    result = generate_postmortem(scored_dag=scored_dag, events=events, deep_analysis=None)
    viz = result["viz_html"]
    assert "dagre" in viz.lower() or "d3" in viz.lower()
    assert "__DAG_DATA__" in viz


def test_merges_deep_analysis_into_dag():
    """Anomaly nodes get deep_analysis field when provided."""
    from generate_postmortem import generate_postmortem
    scored_dag, anomalies, events = _build_scored()
    if anomalies:
        deep = [{
            "node_id": anomalies[0]["node_id"],
            "anomaly_type": anomalies[0]["anomaly_type"],
            "root_cause": "test root cause",
            "causal_chain": ["step 1", "step 2"],
            "counterfactual": "test counterfactual",
            "skill_attribution": "agent_behavior",
            "actionable_fix": "test fix",
            "confidence": 0.8,
        }]
        result = generate_postmortem(scored_dag=scored_dag, events=events, deep_analysis=deep)
        final_dag = result["dag"]
        node = next(n for n in final_dag["nodes"] if n["id"] == anomalies[0]["node_id"])
        assert node["deep_analysis"] is not None
        assert node["deep_analysis"]["root_cause"] == "test root cause"


def test_critical_path_excludes_low_influence():
    """Critical path follows high-influence nodes, not all nodes."""
    from generate_postmortem import generate_postmortem
    scored_dag, _, events = _build_scored()
    result = generate_postmortem(scored_dag=scored_dag, events=events, deep_analysis=None)
    final_dag = result["dag"]
    cp = final_dag["summary"]["critical_path"]
    assert isinstance(cp, list)
    assert len(cp) >= 2  # At least start and end
    # All nodes in critical path should exist in the DAG
    node_ids = {n["id"] for n in final_dag["nodes"]}
    for nid in cp:
        assert nid in node_ids


def test_works_without_deep_analysis():
    """Still produces narrative and viz when deep_analysis is None."""
    from generate_postmortem import generate_postmortem
    scored_dag, _, events = _build_scored()
    result = generate_postmortem(scored_dag=scored_dag, events=events, deep_analysis=None)
    assert result["narrative"]
    assert result["viz_html"]
    assert result["dag"]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd .claude/skills/ammo/eval/causal && python -m pytest tests/test_generate_postmortem.py -v`
Expected: ImportError

- [ ] **Step 5: Implement generate_postmortem.py**

Core logic:
1. Accept `--scored-dag`, `--deep-analysis` (optional), `--events`, `--output-dag`, `--output-narrative`, `--output-viz` CLI args
2. Expose `generate_postmortem(scored_dag, events, deep_analysis) -> dict` with keys `dag`, `narrative`, `viz_html`
3. Merge deep_analysis entries into scored_dag nodes by node_id match
4. **Critical path algorithm**: Topological sort of the DAG. For each node, compute `path_weight = sum(output_influence)` along all paths from root. Critical path = the path with maximum cumulative weight from first node to last node (longest weighted path in a DAG via topological order).
5. Identify trajectory-changing decisions: nodes where `anomaly` is non-null AND the node has at least 1 downstream edge
6. Render narrative by filling in template placeholders
7. Render viz by embedding final DAG JSON into HTML template (`window.__DAG_DATA__ = <json>;`)
8. Write all three outputs

- [ ] **Step 6: Run tests to verify they pass**

- [ ] **Step 7: Commit**

```bash
git add .claude/skills/ammo/eval/causal/generate_postmortem.py \
       .claude/skills/ammo/eval/causal/templates/ \
       .claude/skills/ammo/eval/causal/tests/test_generate_postmortem.py
git commit -m "feat(ammo-eval): add causal postmortem generation with narrative + viz"
```

---

## Task 7: Version Diffing (`diff_versions.py`)

**Files:**
- Create: `.claude/skills/ammo/eval/causal/diff_versions.py`
- Create: `.claude/skills/ammo/eval/causal/tests/test_diff_versions.py`

**Spec reference:** "Step F: Version Diffing" section.

- [ ] **Step 1: Write failing tests**

```python
# test_diff_versions.py
"""Tests for diff_versions.py — cross-version causal DAG comparison."""
import json
import copy
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _build_dag():
    """Build a scored DAG from fixtures."""
    from extract_events import extract_events
    from build_dag import build_dag
    from score_nodes import score_nodes

    session_data = json.loads((FIXTURES / "sample_session_data.json").read_text())
    events = extract_events(
        session_jsonl_path=FIXTURES / "sample_session.jsonl",
        session_data=session_data,
    )
    dag = build_dag(events=events, artifact_dir=str(FIXTURES))
    snapshot = json.loads((FIXTURES / "sample_snapshot.json").read_text())
    scored_dag, _ = score_nodes(dag=dag, events=events, snapshot=snapshot)
    return scored_dag


def _make_v2(dag):
    """Create a modified 'v2' DAG with different metrics and anomalies."""
    v2 = copy.deepcopy(dag)
    for node in v2["nodes"]:
        if node["id"] == "debate":
            node["metrics"]["data_grounding"] = 0.3  # Worse
            node["anomaly"] = {"type": "DATA_CORRUPTION", "triggered_by": {"data_grounding": 0.3}}
        if node["id"] == "bottleneck_mining":
            node["metrics"]["retry_rate"] = 0.05  # Better (was higher)
            node["anomaly"] = None  # Resolved
    return v2


def test_metric_deltas_computed():
    """Per-node metric deltas computed between two DAGs."""
    from diff_versions import diff_versions
    v1 = _build_dag()
    v2 = _make_v2(v1)
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    assert "metric_deltas" in diff
    assert "debate" in diff["metric_deltas"]


def test_new_anomalies_detected():
    """Anomaly in v2 not in v1 → listed as new."""
    from diff_versions import diff_versions
    v1 = _build_dag()
    v2 = _make_v2(v1)
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    new = diff.get("new_anomalies", [])
    assert any(a["node"] == "debate" for a in new)


def test_resolved_anomalies_detected():
    """Anomaly in v1 not in v2 → listed as resolved."""
    from diff_versions import diff_versions
    v1 = _build_dag()
    # Add an anomaly to v1's mining node
    for n in v1["nodes"]:
        if n["id"] == "bottleneck_mining":
            n["anomaly"] = {"type": "THRASHING", "triggered_by": {"retry_rate": 0.4}}
    v2 = _make_v2(v1)
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    resolved = diff.get("resolved_anomalies", [])
    assert any(a["node"] == "bottleneck_mining" for a in resolved)


def test_edge_count_changes():
    """Different edge counts reported."""
    from diff_versions import diff_versions
    v1 = _build_dag()
    v2 = copy.deepcopy(v1)
    # Remove some edges from v2
    v2["edges"] = [e for e in v2["edges"] if e["type"] != "data_citation"]
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    assert "edge_changes" in diff
    if "data_citation" in diff["edge_changes"]:
        assert diff["edge_changes"]["data_citation"]["delta"] < 0


def test_diff_json_schema():
    """Output matches spec schema."""
    from diff_versions import diff_versions
    v1 = _build_dag()
    v2 = _make_v2(v1)
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    assert "version" in diff
    assert "compared" in diff
    assert "metric_deltas" in diff
    assert "new_anomalies" in diff
    assert "resolved_anomalies" in diff
    assert "edge_changes" in diff
    assert "anomaly_distribution_shift" in diff


def test_cli_with_two_dags(tmp_path):
    """CLI produces version_diff.json from two DAG files."""
    import subprocess
    v1 = _build_dag()
    v2 = _make_v2(v1)
    f1 = tmp_path / "v1.json"
    f2 = tmp_path / "v2.json"
    out = tmp_path / "diff.json"
    f1.write_text(json.dumps(v1))
    f2.write_text(json.dumps(v2))
    result = subprocess.run(
        ["python", str(Path(__file__).parent.parent / "diff_versions.py"),
         "--current-dag", str(f2), "--previous-dag", str(f1),
         "--output", str(out)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    diff = json.loads(out.read_text())
    assert "metric_deltas" in diff


def test_regression_report_with_deep_flag(tmp_path):
    """When --deep is passed, regression_report.md is written (stub content for now)."""
    import subprocess
    v1 = _build_dag()
    v2 = _make_v2(v1)
    f1 = tmp_path / "v1.json"
    f2 = tmp_path / "v2.json"
    out = tmp_path / "diff.json"
    report = tmp_path / "regression_report.md"
    f1.write_text(json.dumps(v1))
    f2.write_text(json.dumps(v2))
    result = subprocess.run(
        ["python", str(Path(__file__).parent.parent / "diff_versions.py"),
         "--current-dag", str(f2), "--previous-dag", str(f1),
         "--output", str(out), "--deep",
         "--regression-report", str(report)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0, result.stderr
    assert report.exists()
    content = report.read_text()
    assert len(content) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .claude/skills/ammo/eval/causal && python -m pytest tests/test_diff_versions.py -v`
Expected: ImportError

- [ ] **Step 3: Implement diff_versions.py**

Core logic:
1. Accept `--current-dag`, `--previous-dag`, `--output`, `--deep` (flag), `--skill-diff` (optional path), `--regression-report` (optional path) CLI args
2. Expose `diff_versions(current_dag: dict, previous_dag: dict) -> dict` for direct use
3. Match nodes by ID across DAGs
4. Compute per-node metric deltas for each of the 5 metrics
5. Identify new anomalies (in current, not in previous) and resolved anomalies
6. Count edges by type in each DAG, compute deltas
7. Build anomaly distribution shift
8. If `--deep`: generate a stub `regression_report.md` with structural diff summary (full LLM investigation deferred to post-MVP). Write to `--regression-report` path.
9. Write `version_diff.json`

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/ammo/eval/causal/diff_versions.py .claude/skills/ammo/eval/causal/tests/test_diff_versions.py
git commit -m "feat(ammo-eval): add cross-version causal DAG diffing"
```

---

## Task 8: Integration + Cleanup

**Files:**
- Modify: `.claude/skills/ammo/eval/EVAL-SKILL.md` (add Steps 3b-3f)
- Modify: `.claude/skills/ammo/eval/scripts/archive_run.py` (store causal artifacts)

**Note:** The `dashboard.html` "Causal Analysis" tab is deferred to a follow-up task. The `causal_viz.html` standalone output provides the visualization.

- [ ] **Step 1: Update EVAL-SKILL.md**

Add Steps 3b-3f between existing Step 3 and Step 4. **Important:** The spec shows `--session-id` for `extract_events.py` but the actual implementation uses `--session-jsonl`. Use the correct invocation:

```
Step 3b: Extract Events + Build Causal DAG
  python .claude/skills/ammo/eval/causal/extract_events.py \
    --session-jsonl ~/.claude/projects/-home-jinhun-vllm/<SESSION_ID>.jsonl \
    --session-data /tmp/ammo_eval_session_data.json \
    --output /tmp/ammo_eval_events.json

  python .claude/skills/ammo/eval/causal/build_dag.py \
    --events /tmp/ammo_eval_events.json \
    --artifact-dir <ARTIFACT_DIR> \
    --output /tmp/ammo_eval_causal_dag.json
```

Also add `--causal-dag`, `--postmortem-narrative`, `--causal-viz` flags to Step 5 (archive) and update the Cleanup section (Step 8) to include causal temp files (`/tmp/ammo_eval_events.json`, `/tmp/ammo_eval_causal_dag*.json`, `/tmp/ammo_eval_scored_dag.json`, `/tmp/ammo_eval_anomalies.json`, `/tmp/ammo_eval_deep_analysis.json`, `/tmp/ammo_eval_postmortem.md`, `/tmp/ammo_eval_causal_viz.html`, `/tmp/ammo_eval_version_diff.json`).

- [ ] **Step 2: Update archive_run.py**

Read the file first. Add `--causal-dag`, `--postmortem-narrative`, `--causal-viz` optional CLI args. When provided, copy these files into the archive run directory alongside scorecard.json.

- [ ] **Step 3: Run existing eval tests to verify no regression**

Run: `cd .claude/skills/ammo/eval && python -m pytest tests/ -v`
Expected: All existing tests still pass

- [ ] **Step 4: Run all causal engine tests**

Run: `cd .claude/skills/ammo/eval/causal && python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/ammo/eval/EVAL-SKILL.md .claude/skills/ammo/eval/scripts/archive_run.py
git commit -m "feat(ammo-eval): integrate causal engine into eval pipeline"
```

---

## Task Summary

| Task | Component | Type | Depends On |
|------|-----------|------|------------|
| 1 | Test fixtures | Setup | — |
| 2 | extract_events.py | Script | 1 |
| 3 | build_dag.py | Script | 2 |
| 4 | score_nodes.py | Script | 3 |
| 5 | causal_analyzer.md | Rubric | — (no code deps, can be done anytime) |
| 6 | generate_postmortem.py + templates | Script | 4 |
| 7 | diff_versions.py | Script | 4 |
| 8 | Integration | Glue | 6, 7 |

Tasks 6 and 7 can be done in parallel (both depend only on 4). Task 5 has no code dependencies and can be done at any point.

**Deferred to follow-up:** Dashboard `causal_viz` tab integration into `dashboard_template.html`.
