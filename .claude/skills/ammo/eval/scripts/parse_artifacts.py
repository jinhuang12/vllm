#!/usr/bin/env python3
"""Parse completed AMMO campaign artifacts into a structured JSON snapshot.

Reads all files from a completed campaign artifact directory and emits a flat,
structured JSON snapshot suitable for scoring and archival.

All inputs are optional — missing files yield null fields, never errors.

Inputs (searched in artifact_dir):
  state.json, target.json, env.json, constraints.md,
  investigation/bottleneck_analysis.md,
  debate/summary.md, debate/proposals/*.md,
  debate/campaign_round_*/summary.md,
  tracks/op*/validation_results.md,
  tracks/op*/e2e_latency/e2e_latency_results.json,
  tracks/op*/validation_summary.json

Output:
  artifacts_snapshot.json (to --output or stdout)

Usage:
  python parse_artifacts.py --artifact-dir /path/to/artifacts
  python parse_artifacts.py --artifact-dir /path/to/artifacts --output snapshot.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file, returning None if not found."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _load_text(path: Path) -> Optional[str]:
    """Load text file, returning None if not found."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _safe_get(d: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dicts."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


# ---------------------------------------------------------------------------
# Target & Environment
# ---------------------------------------------------------------------------

def _parse_target(artifact_dir: Path) -> Optional[Dict[str, Any]]:
    target = _load_json(artifact_dir / "target.json")
    if target is None:
        # Fall back to state.json target
        state = _load_json(artifact_dir / "state.json")
        if state and "target" in state:
            t = state["target"]
            return {
                "model_id": t.get("model_id"),
                "hardware": t.get("hardware"),
                "dtype": t.get("dtype"),
                "tp": t.get("tp"),
            }
        return None
    t = target.get("target", {})
    return {
        "model_id": t.get("model_id"),
        "hardware": _safe_get(_load_json(artifact_dir / "state.json"), "target", "hardware"),
        "dtype": t.get("dtype"),
        "tp": t.get("tp"),
    }


def _parse_environment(artifact_dir: Path) -> Optional[Dict[str, Any]]:
    env = _load_json(artifact_dir / "env.json")
    if env is None:
        return None

    nvsmi = env.get("nvidia_smi", {})
    gpus = nvsmi.get("gpus", []) if isinstance(nvsmi, dict) and nvsmi.get("ok") else []
    gpu0 = gpus[0] if isinstance(gpus, list) and gpus else {}

    torch_info = env.get("torch", {})
    vllm_info = env.get("vllm", {})
    git_info = env.get("git", {})

    return {
        "gpu_model": gpu0.get("name"),
        "cuda_version": torch_info.get("cuda_version") if isinstance(torch_info, dict) else None,
        "vllm_version": vllm_info.get("version") if isinstance(vllm_info, dict) else None,
        "git_commit": git_info.get("head") if isinstance(git_info, dict) else None,
        "git_branch": git_info.get("branch") if isinstance(git_info, dict) else None,
        "git_dirty": git_info.get("dirty") if isinstance(git_info, dict) else None,
    }


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

def _parse_campaign(state: Dict[str, Any]) -> Dict[str, Any]:
    campaign = state.get("campaign", {})
    rounds_data = campaign.get("rounds", [])

    rounds = []
    for r in rounds_data:
        if not isinstance(r, dict):
            continue
        impl_results = r.get("implementation_results", {})
        shipped = r.get("shipped", [])
        failed_count = sum(
            1 for v in impl_results.values()
            if isinstance(v, dict) and v.get("status") == "FAILED"
        )
        total_candidates = len(impl_results)

        rounds.append({
            "round_id": r.get("round_id"),
            "top_bottleneck_share_pct": r.get("top_bottleneck_share_pct"),
            "candidates_proposed": len(r.get("selected_candidates", [])),
            "candidates_selected": len(r.get("selected_candidates", [])),
            "candidates_shipped": len(shipped),
            "candidates_failed": failed_count,
            "round_e2e_speedup": _best_e2e_from_results(impl_results),
            "cumulative_speedup_after": r.get("cumulative_speedup_after"),
        })

    shipped_opts = campaign.get("shipped_optimizations", [])
    return {
        "status": campaign.get("status"),
        "total_rounds": campaign.get("current_round", len(rounds)),
        "cumulative_e2e_speedup": campaign.get("cumulative_e2e_speedup", 1.0),
        "shipped_optimizations_count": len(shipped_opts),
        "shipped_optimization_ids": shipped_opts,
        "rounds": rounds,
    }


def _best_e2e_from_results(impl_results: Dict[str, Any]) -> Optional[float]:
    """Extract the best e2e_speedup from implementation_results."""
    best = None
    for v in impl_results.values():
        if isinstance(v, dict) and v.get("status") == "PASSED":
            speedup = v.get("e2e_speedup")
            if isinstance(speedup, (int, float)):
                if best is None or speedup > best:
                    best = speedup
    return best


# ---------------------------------------------------------------------------
# E2E Results
# ---------------------------------------------------------------------------

def _parse_e2e_results(artifact_dir: Path) -> Optional[Dict[str, Any]]:
    """Find and parse e2e_latency_results.json from tracks or root."""
    # Search order: tracks/op*/e2e_latency/, then root e2e_latency/
    e2e_files = list(artifact_dir.glob("tracks/*/e2e_latency/e2e_latency_results.json"))
    if not e2e_files:
        e2e_files = list(artifact_dir.glob("e2e_latency/e2e_latency_results.json"))
    if not e2e_files:
        return None

    # Aggregate across all tracks (take the best shipped track's results)
    all_results = []
    for f in e2e_files:
        data = _load_json(f)
        if data and isinstance(data.get("results"), list):
            all_results.append(data)

    if not all_results:
        return None

    # Use the first file's results as representative (typically the shipped track)
    data = all_results[0]
    results = data.get("results", [])

    per_batch = []
    speedups = []
    for row in results:
        if not isinstance(row, dict):
            continue
        speedup = row.get("speedup")
        per_batch.append({
            "batch_size": row.get("batch_size"),
            "input_len": row.get("input_len"),
            "output_len": row.get("output_len"),
            "baseline_avg_s": _safe_get(row, data.get("bench", {}).get("baseline_label", "baseline"), "avg_s"),
            "optimized_avg_s": _safe_get(row, data.get("bench", {}).get("opt_label", "opt"), "avg_s"),
            "speedup": speedup,
            "improvement_pct": row.get("improvement_pct"),
        })
        if isinstance(speedup, (int, float)):
            speedups.append(speedup)

    return {
        "batch_sizes": [r["batch_size"] for r in per_batch if r.get("batch_size") is not None],
        "per_batch": per_batch,
        "mean_speedup": sum(speedups) / len(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
        "min_speedup": min(speedups) if speedups else None,
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _parse_gates(artifact_dir: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse gate pass/fail information from state.json and validation files."""
    # Phase 1 baseline gate
    phase1_status = _safe_get(state, "verification_run", "stage1")
    phase1 = {
        "status": "PASS" if phase1_status else ("UNKNOWN" if phase1_status is None else "FAIL"),
        "first_attempt": _infer_first_attempt(state, "stage1"),
    }

    # Per-track validation gates
    tracks_state = state.get("parallel_tracks", {})
    track_gates = []
    for track_id, track_data in tracks_state.items():
        if not isinstance(track_data, dict):
            continue
        track_gates.append({
            "track_id": track_id,
            "status": track_data.get("status", "UNKNOWN"),
            "correctness": track_data.get("correctness"),
            "kernel_speedup": track_data.get("kernel_speedup"),
            "e2e_speedup": track_data.get("e2e_speedup"),
            "first_attempt": True,  # No retry tracking in current schema
        })

    # Integration gate
    integration = state.get("integration", {})

    return {
        "phase1_baseline": phase1,
        "validation_gates": track_gates,
        "integration": {
            "status": integration.get("status", "pending"),
            "final_decision": _safe_get(integration, "final_decision", "action"),
        },
    }


def _infer_first_attempt(state: Dict[str, Any], stage: str) -> bool:
    """Infer if a stage passed on first attempt from opportunity_attempts."""
    attempts = state.get("opportunity_attempts", [])
    if not attempts:
        return True  # No retry data = assume first attempt
    # If there's more than one attempt, it wasn't first-attempt
    return len(attempts) <= 1


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------

def _parse_debate(artifact_dir: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    """Parse debate quality metrics from debate artifacts and state."""
    debate_state = state.get("debate", {})
    campaign = state.get("campaign", {})
    total_rounds = campaign.get("current_round", 1)

    # Count proposals
    proposals_dir = artifact_dir / "debate" / "proposals"
    proposal_files = list(proposals_dir.glob("*.md")) if proposals_dir.exists() else []

    # Also check campaign round debate dirs
    for i in range(2, total_rounds + 1):
        round_proposals = artifact_dir / "debate" / f"campaign_round_{i}" / "proposals"
        if round_proposals.exists():
            proposal_files.extend(round_proposals.glob("*.md"))

    total_proposals = len(proposal_files)

    # Check micro-experiment backing and grounding
    proposals_with_micro = 0
    proposals_with_grounding = 0
    for pf in proposal_files:
        content = _load_text(pf)
        if content is None:
            continue
        # Check for micro-experiment sections
        if re.search(r"micro.?experiment|prototype|benchmark|roofline|ncu", content, re.IGNORECASE):
            proposals_with_micro += 1
        # Check for grounded data references
        if re.search(r"bottleneck_analysis|nsys|profil|µs|microsec|\d+\.?\d*\s*(µs|us|ms)", content, re.IGNORECASE):
            proposals_with_grounding += 1

    # Parse candidate scores from summary.md
    all_scores = []
    per_campaign_round = []

    for round_idx in range(1, total_rounds + 1):
        if round_idx == 1:
            summary_path = artifact_dir / "debate" / "summary.md"
        else:
            summary_path = artifact_dir / "debate" / f"campaign_round_{round_idx}" / "summary.md"

        scores = _extract_candidate_scores(summary_path)
        if scores:
            all_scores.extend(scores)

        # Count debate rounds for this campaign round
        debate_round_dirs = []
        base = artifact_dir / "debate" if round_idx == 1 else artifact_dir / "debate" / f"campaign_round_{round_idx}"
        if base.exists():
            debate_round_dirs = [d for d in base.iterdir() if d.is_dir() and re.match(r"round_\d+", d.name)]

        per_campaign_round.append({
            "campaign_round": round_idx,
            "champion_count": len(debate_state.get("candidates", [])) if round_idx == 1 else None,
            "proposals_count": len([p for p in proposal_files if str(p).find(f"campaign_round_{round_idx}") > -1]) if round_idx > 1 else len(list((artifact_dir / "debate" / "proposals").glob("*.md"))) if (artifact_dir / "debate" / "proposals").exists() else 0,
            "debate_rounds": len(debate_round_dirs),
            "winners_selected": len(debate_state.get("selected_winners", [])) if round_idx == 1 else None,
            "candidate_scores": scores if scores else None,
        })

    return {
        "campaign_rounds_with_debate": len([r for r in per_campaign_round if r.get("proposals_count", 0) > 0]),
        "total_proposals": total_proposals,
        "proposals_with_micro_experiments": proposals_with_micro,
        "proposals_with_grounded_data": proposals_with_grounding,
        "micro_experiment_rate": proposals_with_micro / total_proposals if total_proposals > 0 else None,
        "grounding_rate": proposals_with_grounding / total_proposals if total_proposals > 0 else None,
        "total_debate_rounds_across_campaign": sum(r.get("debate_rounds", 0) for r in per_campaign_round),
        "all_candidate_scores": all_scores if all_scores else None,
        "per_campaign_round": per_campaign_round,
    }


def _extract_candidate_scores(summary_path: Path) -> List[float]:
    """Regex-extract per-candidate final scores from debate summary.md tables."""
    content = _load_text(summary_path)
    if content is None:
        return []

    scores = []
    # Match patterns like "**8.4/10**" or "8.4/10" or "| 8.4/10 |" in table rows
    # Also match patterns like "Total (weighted): 8.4/10"
    for match in re.finditer(r"(\d+\.?\d*)\s*/\s*10", content):
        try:
            score = float(match.group(1))
            if 0 <= score <= 10:
                scores.append(score)
        except ValueError:
            continue

    # Deduplicate: the summary table often has the same score multiple times
    # (per-criterion + final). Take the scores from the Final Score column.
    # Heuristic: look for "Final Score" or "Total" rows
    final_scores = []
    for line in content.split("\n"):
        if re.search(r"(Final\s+Score|Total|SELECTED|REJECTED)", line, re.IGNORECASE):
            line_scores = []
            for m in re.finditer(r"\*?\*?(\d+\.?\d*)\s*/\s*10\*?\*?", line):
                try:
                    s = float(m.group(1))
                    if 0 <= s <= 10:
                        line_scores.append(s)
                except ValueError:
                    continue
            if line_scores:
                final_scores.append(line_scores[-1])  # Last score on line is usually the total

    return final_scores if final_scores else scores


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------

def _parse_tracks(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse per-track results from state.json."""
    tracks_state = state.get("parallel_tracks", {})
    campaign = state.get("campaign", {})

    tracks = []
    for track_id, track_data in tracks_state.items():
        if not isinstance(track_data, dict):
            continue

        # Determine which campaign round this track belongs to
        campaign_round = None
        for r in campaign.get("rounds", []):
            if isinstance(r, dict) and track_id in r.get("selected_candidates", []):
                campaign_round = r.get("round_id")
                break

        tracks.append({
            "op_id": track_id,
            "campaign_round": campaign_round,
            "status": track_data.get("status", "UNKNOWN"),
            "correctness": track_data.get("correctness"),
            "kernel_speedup": track_data.get("kernel_speedup"),
            "e2e_speedup": track_data.get("e2e_speedup"),
        })

    return tracks


# ---------------------------------------------------------------------------
# Master Parser
# ---------------------------------------------------------------------------

def parse_campaign(artifact_dir: Path) -> Dict[str, Any]:
    """Master parser. Returns the complete artifacts_snapshot dict."""
    state = _load_json(artifact_dir / "state.json") or {}

    target = _parse_target(artifact_dir)
    environment = _parse_environment(artifact_dir)
    campaign = _parse_campaign(state)
    e2e_results = _parse_e2e_results(artifact_dir)
    gates = _parse_gates(artifact_dir, state)
    debate = _parse_debate(artifact_dir, state)
    tracks = _parse_tracks(state)

    return {
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "target": target,
        "environment": environment,
        "campaign": campaign,
        "e2e_results": e2e_results,
        "gates": gates,
        "debate": debate,
        "tracks": tracks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artifact-dir", type=str, required=True,
        help="Path to completed campaign artifact directory",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for artifacts_snapshot.json (default: stdout)",
    )

    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()

    if not artifact_dir.exists():
        print(f"ERROR: Artifact directory does not exist: {artifact_dir}", file=sys.stderr)
        return 1

    snapshot = parse_campaign(artifact_dir)
    json_str = json.dumps(snapshot, indent=2) + "\n"

    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Wrote: {args.output}", file=sys.stderr)
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
