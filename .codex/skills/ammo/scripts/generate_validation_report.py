#!/usr/bin/env python3
"""Render a Markdown validation summary from structured track evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fmt_num(value: Any, suffix: str = "") -> str:
    if isinstance(value, float):
        return f"{value:.6g}{suffix}"
    if isinstance(value, int):
        return f"{value}{suffix}"
    return str(value)


def _render_kill_criteria(kill_criteria: Dict[str, Any]) -> List[str]:
    lines = ["## Kill Criteria", ""]
    for name, result in sorted(kill_criteria.items()):
        if not isinstance(result, dict):
            continue
        status = result.get("status", "UNKNOWN")
        source = result.get("source_run_purpose", "unknown")
        note = result.get("note") or result.get("notes") or ""
        lines.append(f"- `{name}`: `{status}` from `{source}`")
        if note:
            lines.append(f"  note: {note}")
    lines.append("")
    return lines


def render_validation_md(evidence: Dict[str, Any]) -> str:
    correctness = evidence.get("correctness", {})
    kernel = evidence.get("kernel_bench", {})
    e2e = evidence.get("e2e", {})
    amdahl = evidence.get("amdahl", {})
    fastpath = e2e.get("fastpath_proof", {}) if isinstance(e2e, dict) else {}
    admissibility = e2e.get("admissibility", {}) if isinstance(e2e, dict) else {}

    lines = [
        f"# {evidence.get('track_id', 'track')} validation results",
        "",
        "This file is generated from `evidence.json`. Structured evidence is authoritative.",
        "",
        "## Baseline",
        "",
        f"- Source: `{evidence.get('baseline_source', {}).get('citation', 'unknown')}`",
        "",
        "## Correctness",
        "",
        f"- Status: `{correctness.get('status', 'UNKNOWN')}`",
        f"- Method: `{correctness.get('method', 'unknown')}`",
        f"- Tolerance (atol/rtol): `{_fmt_num(correctness.get('atol'))}` / `{_fmt_num(correctness.get('rtol'))}`",
        f"- Max abs diff: `{_fmt_num(correctness.get('max_abs_diff'))}`",
        f"- NaN/INF check: `{correctness.get('nan_inf_check')}`",
        f"- Graph replay check: `{correctness.get('graph_replay_check')}`",
        "",
        "## Kernel Perf",
        "",
        f"- Status: `{kernel.get('status', 'UNKNOWN')}`",
        f"- Measured under CUDA graphs: `{kernel.get('measured_under_cuda_graphs')}`",
        f"- Weighted speedup: `{_fmt_num(kernel.get('weighted_speedup'), 'x')}`",
        "",
        "## E2E Latency",
        "",
        f"- Status: `{e2e.get('status', 'UNKNOWN')}`",
        f"- Run purpose: `{e2e.get('run_purpose', 'unknown')}`",
        f"- Baseline avg: `{_fmt_num(e2e.get('baseline_avg_s'), ' s')}`",
        f"- Optimized avg: `{_fmt_num(e2e.get('optimized_avg_s'), ' s')}`",
        f"- Speedup: `{_fmt_num(e2e.get('speedup'), 'x')}`",
        f"- Improvement: `{_fmt_num(e2e.get('improvement_pct'), '%')}`",
        "",
        "## Fast-Path Proof",
        "",
        f"- Status: `{fastpath.get('status', 'UNKNOWN')}`",
        f"- Source: `{fastpath.get('source', 'unknown')}`",
        f"- Hits: `{fastpath.get('hits', 'unknown')}`",
        "",
        "## Admissibility",
        "",
        f"- Status: `{admissibility.get('status', 'UNKNOWN')}`",
    ]
    for issue in admissibility.get("issues", []) if isinstance(admissibility, dict) else []:
        lines.append(f"- Issue: `{issue}`")
    lines.extend(
        [
            "",
            "## Amdahl Sanity",
            "",
            f"- component share `f`: `{_fmt_num(amdahl.get('component_share_f'))}`",
            f"- kernel speedup `s`: `{_fmt_num(amdahl.get('kernel_speedup'), 'x')}`",
            f"- expected E2E improvement: `{_fmt_num(amdahl.get('expected_e2e_pct'), '%')}`",
            f"- actual E2E improvement: `{_fmt_num(amdahl.get('actual_e2e_pct'), '%')}`",
            "",
        ]
    )
    kill_criteria = evidence.get("kill_criteria", {})
    if isinstance(kill_criteria, dict) and kill_criteria:
        lines.extend(_render_kill_criteria(kill_criteria))

    contamination = evidence.get("cross_track_contamination", {})
    if isinstance(contamination, dict):
        lines.extend(
            [
                "## Cross-Track Contamination",
                "",
                f"- Status: `{contamination.get('status', 'UNKNOWN')}`",
                f"- Note: {contamination.get('note', '')}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument("--track", type=str, default=None)
    parser.add_argument("--evidence-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    args = parser.parse_args()

    if args.evidence_json:
        evidence_path = Path(args.evidence_json).expanduser().resolve()
        output_path = Path(args.output_md).expanduser().resolve() if args.output_md else evidence_path.with_name("validation_results.md")
    else:
        if not args.artifact_dir or not args.track:
            raise SystemExit("Either --evidence-json or both --artifact-dir and --track are required.")
        artifact_dir = Path(args.artifact_dir).expanduser().resolve()
        evidence_path = artifact_dir / "tracks" / args.track / "evidence.json"
        output_path = Path(args.output_md).expanduser().resolve() if args.output_md else artifact_dir / "tracks" / args.track / "validation_results.md"

    evidence = _read_json(evidence_path)
    _write_text(output_path, render_validation_md(evidence))
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
