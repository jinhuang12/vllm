#!/usr/bin/env python3
"""Aggregate multi-run eval results and rebuild the repository index.

For each version+target combination with multiple runs, computes
mean/stddev/min/max across runs. Rebuilds the master index.json
with leaderboards.

Usage:
  python aggregate_versions.py --repository ~/.claude/ammo-eval
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def calculate_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Compute mean, stddev, min, max for a list of values."""
    if not values:
        return {"mean": None, "stddev": None, "min": None, "max": None, "n": 0}

    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        stddev = math.sqrt(variance)
    else:
        stddev = 0.0

    return {
        "mean": round(mean, 4),
        "stddev": round(stddev, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": n,
    }


def _collect_metric(scorecards: List[Dict], *path: str) -> List[float]:
    """Extract a numeric metric from multiple scorecards by dot-path."""
    values = []
    for sc in scorecards:
        val = sc
        for key in path:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                val = None
                break
        if isinstance(val, (int, float)) and val is not None:
            values.append(float(val))
    return values


def aggregate_target_runs(target_dir: Path) -> Optional[Dict[str, Any]]:
    """Aggregate runs for a single version+target combination."""
    run_dirs = sorted(
        [d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
    )
    if not run_dirs:
        return None

    scorecards = []
    for rd in run_dirs:
        sc = _load_json(rd / "scorecard.json")
        if sc:
            scorecards.append(sc)

    if not scorecards:
        return None

    aggregate = {
        "target_slug": target_dir.name,
        "runs_count": len(scorecards),
        "overall_score": calculate_stats(
            _collect_metric(scorecards, "overall_score")
        ),
        "overall_score_without_transcript": calculate_stats(
            _collect_metric(scorecards, "overall_score_without_transcript")
        ),
        "e2e_speedup": calculate_stats(
            _collect_metric(scorecards, "raw_metrics", "cumulative_e2e_speedup")
        ),
        "shipped_count": calculate_stats(
            _collect_metric(scorecards, "raw_metrics", "shipped_optimizations")
        ),
        "total_rounds": calculate_stats(
            _collect_metric(scorecards, "raw_metrics", "total_rounds")
        ),
        "dimensions": {},
    }

    # Per-dimension aggregation
    for dim in ["e2e_outcome", "gate_pass_rates", "debate_quality",
                "campaign_efficiency", "transcript_quality"]:
        aggregate["dimensions"][dim] = calculate_stats(
            _collect_metric(scorecards, "dimensions", dim, "score")
        )

    return aggregate


def aggregate_version(version_dir: Path) -> Optional[Dict[str, Any]]:
    """Aggregate all targets for a single version."""
    runs_dir = version_dir / "runs"
    if not runs_dir.exists():
        return None

    target_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    if not target_dirs:
        return None

    meta = _load_json(version_dir / "meta.json") or {}
    targets = {}

    for td in sorted(target_dirs, key=lambda d: d.name):
        agg = aggregate_target_runs(td)
        if agg:
            targets[td.name] = agg
            # Write per-target aggregate
            (td / "aggregate.json").write_text(
                json.dumps(agg, indent=2) + "\n", encoding="utf-8"
            )

    if not targets:
        return None

    summary = {
        "version_id": meta.get("version_id", version_dir.name),
        "git_commit": meta.get("git_commit"),
        "description": meta.get("description"),
        "created_at": meta.get("created_at"),
        "targets_evaluated": list(targets.keys()),
        "summary": targets,
    }

    # Write version summary
    (version_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    return summary


def rebuild_index(repository: Path) -> Dict[str, Any]:
    """Rebuild the master index.json from all versions."""
    versions_dir = repository / "versions"
    if not versions_dir.exists():
        versions_dir.mkdir(parents=True, exist_ok=True)

    version_dirs = sorted(
        [d for d in versions_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    versions = []
    all_targets = set()

    for vd in version_dirs:
        summary = aggregate_version(vd)
        if summary:
            versions.append(summary)
            all_targets.update(summary.get("targets_evaluated", []))

    # Build leaderboard: best version per target by e2e_speedup mean
    leaderboard = {}
    for target_slug in sorted(all_targets):
        best_version = None
        best_speedup = 0.0
        best_score = 0.0

        for v in versions:
            target_data = v.get("summary", {}).get(target_slug)
            if not target_data:
                continue
            speedup_mean = (target_data.get("e2e_speedup") or {}).get("mean", 0)
            score_mean = (target_data.get("overall_score") or {}).get("mean", 0)
            if speedup_mean > best_speedup:
                best_speedup = speedup_mean
                best_score = score_mean
                best_version = v.get("version_id")

        if best_version:
            leaderboard[target_slug] = {
                "best_version": best_version,
                "best_e2e_speedup_mean": best_speedup,
                "best_overall_score_mean": best_score,
            }

    # Discover reference targets (all unique targets seen)
    reference_targets = []
    for slug in sorted(all_targets):
        # Try to extract model type from slug
        model_type = "moe" if any(kw in slug for kw in ["a3b", "moe", "mixtral"]) else "dense"
        parts = slug.rsplit("_tp", 1)
        tp = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        reference_targets.append({
            "slug": slug,
            "model_type": model_type,
            "tp": tp,
        })

    index = {
        "repository_version": 1,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "reference_targets": reference_targets,
        "versions": [
            {
                "version_id": v.get("version_id"),
                "git_commit": v.get("git_commit"),
                "description": v.get("description"),
                "created_at": v.get("created_at"),
                "targets_evaluated": v.get("targets_evaluated", []),
                "summary": {
                    target: {
                        "overall_score": data.get("overall_score"),
                        "e2e_speedup": data.get("e2e_speedup"),
                        "shipped_count": data.get("shipped_count"),
                    }
                    for target, data in v.get("summary", {}).items()
                },
            }
            for v in versions
        ],
        "leaderboard": leaderboard,
    }

    index_path = repository / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    return index


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repository", type=str,
        default=str(Path.home() / ".claude" / "ammo-eval"),
    )

    args = parser.parse_args()
    repository = Path(args.repository).expanduser().resolve()

    if not repository.exists():
        print(f"Repository does not exist: {repository}", file=sys.stderr)
        return 1

    index = rebuild_index(repository)
    n_versions = len(index.get("versions", []))
    n_targets = len(index.get("reference_targets", []))
    print(f"Rebuilt index: {n_versions} version(s), {n_targets} target(s)", file=sys.stderr)
    print(f"Wrote: {repository / 'index.json'}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
