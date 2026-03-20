#!/usr/bin/env python3
"""Cross-version comparison of two causal DAGs.

Detects metric regressions, new/resolved anomalies, and edge count changes
between a current scored DAG and a previous scored DAG.

Usage:
    python diff_versions.py \\
        --current-dag /path/to/current_dag.json \\
        --previous-dag /path/to/previous_dag.json \\
        --output /path/to/diff.json \\
        [--deep] \\
        [--skill-diff /path/to/skill_diff.json] \\
        [--regression-report /path/to/report.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Anomaly extraction helpers
# ---------------------------------------------------------------------------

def _extract_anomaly_info(anomaly_field: Any) -> Optional[Dict[str, Any]]:
    """Normalize the 'anomaly' field from a DAG node into a canonical dict.

    score_nodes.py sets anomaly to a list of type strings: ["THRASHING", ...]
    Tests may inject a single dict: {"type": "DATA_CORRUPTION", "triggered_by": {...}}
    This function normalizes both into Optional[{"type": str, "triggered_by": ...}].

    If the field is None, empty, or False-y: returns None (no anomaly).
    If the field is a non-empty list of strings: returns {"type": first_type}.
    If the field is a dict with "type": returns the dict as-is.
    """
    if anomaly_field is None:
        return None

    if isinstance(anomaly_field, dict):
        if anomaly_field.get("type"):
            return anomaly_field
        return None

    if isinstance(anomaly_field, list):
        # Filter out None/empty entries
        types = [t for t in anomaly_field if t]
        if not types:
            return None
        return {"type": types[0], "triggered_by": None}

    return None


def _has_anomaly(node: Dict[str, Any]) -> bool:
    """Return True if this node has an active anomaly."""
    return _extract_anomaly_info(node.get("anomaly")) is not None


# ---------------------------------------------------------------------------
# Core diff logic
# ---------------------------------------------------------------------------

def _compute_metric_deltas(
    current_node: Dict[str, Any],
    previous_node: Dict[str, Any],
) -> Dict[str, float]:
    """Compute per-metric deltas: current_value - previous_value.

    Skips metrics where either value is None.
    """
    _METRICS = [
        "data_grounding",
        "output_influence",
        "prediction_accuracy",
        "retry_rate",
        "transition_quality",
    ]
    deltas: Dict[str, float] = {}

    current_metrics = current_node.get("metrics", {}) or {}
    previous_metrics = previous_node.get("metrics", {}) or {}

    for metric in _METRICS:
        curr_val = current_metrics.get(metric)
        prev_val = previous_metrics.get(metric)
        if curr_val is not None and prev_val is not None:
            deltas[metric] = curr_val - prev_val

    return deltas


def _count_edges_by_type(edges: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count edges grouped by their 'type' field."""
    counts: Dict[str, int] = {}
    for edge in edges:
        edge_type = edge.get("type", "unknown")
        counts[edge_type] = counts.get(edge_type, 0) + 1
    return counts


def _count_anomalies_by_type(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count active anomalies by type across all nodes."""
    counts: Dict[str, int] = {}
    for node in nodes:
        info = _extract_anomaly_info(node.get("anomaly"))
        if info is not None:
            atype = info.get("type", "UNKNOWN")
            counts[atype] = counts.get(atype, 0) + 1
    return counts


def _generate_regression_report(diff: Dict[str, Any]) -> str:
    """Format the diff data into a human-readable markdown regression report."""
    lines: List[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines.append("# Causal DAG Regression Report")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Schema version: {diff.get('version', 'unknown')}")
    lines.append("")

    # Summary table
    new_anomalies = diff.get("new_anomalies", [])
    resolved_anomalies = diff.get("resolved_anomalies", [])
    metric_deltas = diff.get("metric_deltas", {})
    edge_changes = diff.get("edge_changes", {})

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- New anomalies: **{len(new_anomalies)}**")
    lines.append(f"- Resolved anomalies: **{len(resolved_anomalies)}**")
    lines.append(f"- Nodes with metric changes: **{len(metric_deltas)}**")
    lines.append(f"- Edge type changes: **{len(edge_changes)}**")
    lines.append("")

    # New anomalies
    if new_anomalies:
        lines.append("## New Anomalies (Regressions)")
        lines.append("")
        lines.append("| Node | Anomaly Type |")
        lines.append("|------|-------------|")
        for a in new_anomalies:
            lines.append(f"| `{a['node']}` | `{a['type']}` |")
        lines.append("")

    # Resolved anomalies
    if resolved_anomalies:
        lines.append("## Resolved Anomalies (Improvements)")
        lines.append("")
        lines.append("| Node | Anomaly Type |")
        lines.append("|------|-------------|")
        for a in resolved_anomalies:
            lines.append(f"| `{a['node']}` | `{a['type']}` |")
        lines.append("")

    # Metric deltas
    if metric_deltas:
        lines.append("## Metric Deltas by Node")
        lines.append("")
        lines.append("Positive delta = current is higher than previous.")
        lines.append("")
        lines.append("| Node | Metric | Delta |")
        lines.append("|------|--------|-------|")
        for node_id, deltas in sorted(metric_deltas.items()):
            for metric, delta in sorted(deltas.items()):
                sign = "+" if delta > 0 else ""
                lines.append(f"| `{node_id}` | {metric} | {sign}{delta:.4f} |")
        lines.append("")

    # Edge changes
    if edge_changes:
        lines.append("## Edge Count Changes")
        lines.append("")
        lines.append("| Edge Type | Previous | Current | Delta |")
        lines.append("|-----------|----------|---------|-------|")
        for edge_type, counts in sorted(edge_changes.items()):
            prev_n = counts["previous"]
            curr_n = counts["current"]
            delta = counts["delta"]
            sign = "+" if delta > 0 else ""
            lines.append(f"| {edge_type} | {prev_n} | {curr_n} | {sign}{delta} |")
        lines.append("")

    # Anomaly distribution shift
    dist = diff.get("anomaly_distribution_shift", {})
    if dist:
        prev_dist = dist.get("previous", {})
        curr_dist = dist.get("current", {})
        all_types = sorted(set(list(prev_dist.keys()) + list(curr_dist.keys())))
        if all_types:
            lines.append("## Anomaly Distribution Shift")
            lines.append("")
            lines.append("| Anomaly Type | Previous Count | Current Count |")
            lines.append("|-------------|----------------|---------------|")
            for atype in all_types:
                prev_count = prev_dist.get(atype, 0)
                curr_count = curr_dist.get(atype, 0)
                lines.append(f"| `{atype}` | {prev_count} | {curr_count} |")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def diff_versions(
    current_dag: Dict[str, Any],
    previous_dag: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two scored causal DAGs and return a structured diff.

    Args:
        current_dag: The newer scored DAG (from score_nodes.py).
        previous_dag: The older scored DAG to compare against.

    Returns:
        Diff dict with schema:
        {
          "version": "1.0",
          "compared": {"current": {...}, "previous": {...}},
          "metric_deltas": {"node_id": {"metric": delta, ...}, ...},
          "new_anomalies": [{"node": "...", "type": "...", "not_in_previous": true}],
          "resolved_anomalies": [{"node": "...", "type": "...", "was_in_previous": true}],
          "edge_changes": {"type": {"previous": N, "current": M, "delta": M-N}},
          "anomaly_distribution_shift": {"previous": {...}, "current": {...}},
          "deep_investigation": null,
        }
    """
    # Index nodes by ID in both DAGs
    current_nodes_by_id: Dict[str, Dict[str, Any]] = {
        n["id"]: n for n in current_dag.get("nodes", [])
    }
    previous_nodes_by_id: Dict[str, Dict[str, Any]] = {
        n["id"]: n for n in previous_dag.get("nodes", [])
    }

    # Matched nodes (present in both DAGs)
    matched_node_ids = set(current_nodes_by_id.keys()) & set(previous_nodes_by_id.keys())

    # --- Metric deltas ---
    metric_deltas: Dict[str, Dict[str, float]] = {}
    for node_id in sorted(matched_node_ids):
        deltas = _compute_metric_deltas(
            current_nodes_by_id[node_id],
            previous_nodes_by_id[node_id],
        )
        if deltas:
            metric_deltas[node_id] = deltas

    # --- New anomalies: in current but not in previous ---
    new_anomalies: List[Dict[str, Any]] = []
    for node_id in sorted(matched_node_ids):
        curr_node = current_nodes_by_id[node_id]
        prev_node = previous_nodes_by_id[node_id]

        curr_info = _extract_anomaly_info(curr_node.get("anomaly"))
        prev_info = _extract_anomaly_info(prev_node.get("anomaly"))

        if curr_info is not None and prev_info is None:
            new_anomalies.append({
                "node": node_id,
                "type": curr_info.get("type", "UNKNOWN"),
                "not_in_previous": True,
            })

    # Also detect nodes that are only in current (newly added nodes with anomalies)
    only_in_current = set(current_nodes_by_id.keys()) - set(previous_nodes_by_id.keys())
    for node_id in sorted(only_in_current):
        curr_node = current_nodes_by_id[node_id]
        curr_info = _extract_anomaly_info(curr_node.get("anomaly"))
        if curr_info is not None:
            new_anomalies.append({
                "node": node_id,
                "type": curr_info.get("type", "UNKNOWN"),
                "not_in_previous": True,
            })

    # --- Resolved anomalies: in previous but not in current ---
    resolved_anomalies: List[Dict[str, Any]] = []
    for node_id in sorted(matched_node_ids):
        curr_node = current_nodes_by_id[node_id]
        prev_node = previous_nodes_by_id[node_id]

        curr_info = _extract_anomaly_info(curr_node.get("anomaly"))
        prev_info = _extract_anomaly_info(prev_node.get("anomaly"))

        if prev_info is not None and curr_info is None:
            resolved_anomalies.append({
                "node": node_id,
                "type": prev_info.get("type", "UNKNOWN"),
                "was_in_previous": True,
            })

    # --- Edge count changes ---
    current_edge_counts = _count_edges_by_type(current_dag.get("edges", []))
    previous_edge_counts = _count_edges_by_type(previous_dag.get("edges", []))
    all_edge_types = set(list(current_edge_counts.keys()) + list(previous_edge_counts.keys()))

    edge_changes: Dict[str, Dict[str, int]] = {}
    for edge_type in sorted(all_edge_types):
        prev_count = previous_edge_counts.get(edge_type, 0)
        curr_count = current_edge_counts.get(edge_type, 0)
        delta = curr_count - prev_count
        if delta != 0:
            edge_changes[edge_type] = {
                "previous": prev_count,
                "current": curr_count,
                "delta": delta,
            }

    # --- Anomaly distribution shift ---
    previous_anomaly_dist = _count_anomalies_by_type(previous_dag.get("nodes", []))
    current_anomaly_dist = _count_anomalies_by_type(current_dag.get("nodes", []))

    # Build "compared" metadata
    compared = {
        "current": {
            "total_nodes": len(current_dag.get("nodes", [])),
            "total_edges": len(current_dag.get("edges", [])),
            "campaign": current_dag.get("campaign", {}),
        },
        "previous": {
            "total_nodes": len(previous_dag.get("nodes", [])),
            "total_edges": len(previous_dag.get("edges", [])),
            "campaign": previous_dag.get("campaign", {}),
        },
    }

    return {
        "version": "1.0",
        "compared": compared,
        "metric_deltas": metric_deltas,
        "new_anomalies": new_anomalies,
        "resolved_anomalies": resolved_anomalies,
        "edge_changes": edge_changes,
        "anomaly_distribution_shift": {
            "previous": previous_anomaly_dist,
            "current": current_anomaly_dist,
        },
        "deep_investigation": None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--current-dag",
        required=True,
        help="Path to current (newer) scored DAG JSON",
    )
    p.add_argument(
        "--previous-dag",
        required=True,
        help="Path to previous (older) scored DAG JSON",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output path for diff JSON",
    )
    p.add_argument(
        "--deep",
        action="store_true",
        default=False,
        help="Enable deep investigation mode (generates regression report)",
    )
    p.add_argument(
        "--skill-diff",
        default=None,
        help="Optional path to a skill diff JSON for additional context",
    )
    p.add_argument(
        "--regression-report",
        default=None,
        help="Output path for markdown regression report (requires --deep)",
    )

    args = p.parse_args()

    # Load DAG files
    for path_str, label in [
        (args.current_dag, "current-dag"),
        (args.previous_dag, "previous-dag"),
    ]:
        if not Path(path_str).exists():
            print(f"ERROR: {label} file not found: {path_str}", file=sys.stderr)
            return 1

    try:
        current_dag = json.loads(Path(args.current_dag).read_text(encoding="utf-8"))
        previous_dag = json.loads(Path(args.previous_dag).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR reading input files: {e}", file=sys.stderr)
        return 1

    # Optionally load skill diff
    skill_diff = None
    if args.skill_diff:
        skill_diff_path = Path(args.skill_diff)
        if not skill_diff_path.exists():
            print(f"WARNING: skill-diff file not found: {skill_diff_path}", file=sys.stderr)
        else:
            try:
                skill_diff = json.loads(skill_diff_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"WARNING: could not read skill-diff: {e}", file=sys.stderr)

    # Run diff
    diff = diff_versions(current_dag=current_dag, previous_dag=previous_dag)

    # Attach skill diff if provided
    if skill_diff is not None:
        diff["skill_diff"] = skill_diff

    # Deep mode: populate deep_investigation summary
    if args.deep:
        diff["deep_investigation"] = {
            "mode": "structural_diff",
            "new_anomaly_count": len(diff["new_anomalies"]),
            "resolved_anomaly_count": len(diff["resolved_anomalies"]),
            "regressed_metrics": [
                {"node": node_id, "metric": metric, "delta": delta}
                for node_id, deltas in diff["metric_deltas"].items()
                for metric, delta in deltas.items()
                if delta < 0
            ],
            "improved_metrics": [
                {"node": node_id, "metric": metric, "delta": delta}
                for node_id, deltas in diff["metric_deltas"].items()
                for metric, delta in deltas.items()
                if delta > 0
            ],
        }

    # Write diff JSON
    out_path = Path(args.output)
    try:
        out_path.write_text(
            json.dumps(diff, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError as e:
        print(f"ERROR writing output: {e}", file=sys.stderr)
        return 1

    print(
        f"Wrote diff to: {out_path} "
        f"({len(diff['new_anomalies'])} new anomalies, "
        f"{len(diff['resolved_anomalies'])} resolved)",
        file=sys.stderr,
    )

    # Write regression report if requested
    if args.regression_report:
        report_text = _generate_regression_report(diff)
        report_path = Path(args.regression_report)
        try:
            report_path.write_text(report_text, encoding="utf-8")
        except OSError as e:
            print(f"ERROR writing regression report: {e}", file=sys.stderr)
            return 1
        print(f"Wrote regression report to: {report_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
