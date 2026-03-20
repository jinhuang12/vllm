# test_diff_versions.py
"""Tests for diff_versions.py — cross-version causal DAG comparison."""
import json
import copy
from pathlib import Path
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _build_dag():
    from extract_events import extract_events
    from build_dag import build_dag
    from score_nodes import score_nodes
    session_data = json.loads((FIXTURES / "sample_session_data.json").read_text())
    events = extract_events(session_jsonl_path=FIXTURES / "sample_session.jsonl", session_data=session_data)
    dag = build_dag(events=events, artifact_dir=str(FIXTURES))
    snapshot = json.loads((FIXTURES / "sample_snapshot.json").read_text())
    scored_dag, _ = score_nodes(dag=dag, events=events, snapshot=snapshot)
    return scored_dag


def _make_v2(dag):
    v2 = copy.deepcopy(dag)
    for node in v2["nodes"]:
        if node["id"] == "debate":
            node["metrics"]["data_grounding"] = 0.3
            node["anomaly"] = {"type": "DATA_CORRUPTION", "triggered_by": {"data_grounding": 0.3}}
        if node["id"] == "bottleneck_mining":
            node["metrics"]["retry_rate"] = 0.05
            node["anomaly"] = None
    return v2


def test_metric_deltas_computed():
    from diff_versions import diff_versions
    v1 = _build_dag()
    v2 = _make_v2(v1)
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    assert "metric_deltas" in diff
    assert "debate" in diff["metric_deltas"]


def test_new_anomalies_detected():
    from diff_versions import diff_versions
    v1 = _build_dag()
    v2 = _make_v2(v1)
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    new = diff.get("new_anomalies", [])
    assert any(a["node"] == "debate" for a in new)


def test_resolved_anomalies_detected():
    from diff_versions import diff_versions
    v1 = _build_dag()
    for n in v1["nodes"]:
        if n["id"] == "bottleneck_mining":
            n["anomaly"] = {"type": "THRASHING", "triggered_by": {"retry_rate": 0.4}}
    v2 = _make_v2(v1)
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    resolved = diff.get("resolved_anomalies", [])
    assert any(a["node"] == "bottleneck_mining" for a in resolved)


def test_edge_count_changes():
    from diff_versions import diff_versions
    v1 = _build_dag()
    v2 = copy.deepcopy(v1)
    v2["edges"] = [e for e in v2["edges"] if e["type"] != "data_citation"]
    diff = diff_versions(current_dag=v2, previous_dag=v1)
    assert "edge_changes" in diff
    if "data_citation" in diff["edge_changes"]:
        assert diff["edge_changes"]["data_citation"]["delta"] < 0


def test_diff_json_schema():
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
    assert len(report.read_text()) > 0
