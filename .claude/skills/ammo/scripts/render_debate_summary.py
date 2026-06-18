#!/usr/bin/env python3
"""Render debate/summary.md from state.json.

The debate moderator writes the authoritative cross-agent contract to
state.json.campaign.rounds[N].debate.selected_candidates (array — typically 2-4
winners per round). This script turns that structure into a human-readable
markdown view. It is the ONLY writer of summary.md — no agent hand-authors the
file. If summary.md and state.json disagree, state.json wins; regenerate by
re-running this script.

By default the current (last) round is rendered. Pass --round to render a
historical round for audit or to render round N+1 while round N is still
implementing (sequential rounds — see SKILL.md for the round lifecycle).

Usage:
    python render_debate_summary.py --state <state.json> --out <summary.md>
    python render_debate_summary.py --state <state.json> --out <summary.md> --round 2
"""

import argparse
import json
import sys
from pathlib import Path


def _render_candidate(cand: dict) -> list:
    op_id = cand.get("op_id", "?")
    track = cand.get("track_assignment", "?")
    breakdown = cand.get("score_breakdown", {}) or {}
    obligations = cand.get("stage_4_validation_obligations", []) or []
    evidence = cand.get("cited_evidence", []) or []

    lines = [
        f"### `{op_id}` — track `{track}`",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Feasibility | {breakdown.get('feasibility', 'n/a')} |",
        f"| Evidence tier | `{breakdown.get('evidence_tier', 'n/a')}` |",
        f"| Expected E2E (%) | {breakdown.get('expected_e2e_pct', 'n/a')} |",
        f"| Weighted total | {breakdown.get('weighted_total', 'n/a')} |",
        "",
        "**Stage-4 validation obligations:**",
        "",
    ]
    if obligations:
        lines += [f"- `{o}`" for o in obligations]
    else:
        lines.append("_None._")
    lines += ["", "**Cited evidence:**", ""]
    if evidence:
        lines += [f"- `{e}`" for e in evidence]
    else:
        lines.append("_None._")
    lines.append("")
    return lines


def render(state: dict, round_index: int = -1) -> str:
    """Render a round's debate summary.

    round_index semantics match Python list indexing:
      -1 (default): last round
       0 .. N-1:    absolute round index
      -N .. -1:     relative to the end

    Raises IndexError with a human-readable message if the index is out of range.
    """
    rounds = state.get("campaign", {}).get("rounds", [])
    if not rounds:
        return "# Debate Summary\n\n_No rounds recorded in state.json._\n"
    try:
        round_entry = rounds[round_index]
    except IndexError:
        raise IndexError(
            f"--round {round_index} out of range; state.json has {len(rounds)} round(s)"
        )
    round_id = round_entry.get("round_id", round_index if round_index >= 0 else len(rounds) + round_index + 1)
    debate = round_entry.get("debate", {})
    selected = debate.get("selected_candidates") or []

    lines = [f"# Debate Summary — Round {round_id}", ""]

    if not selected:
        lines.append("_No selected_candidates recorded. Debate may still be in progress._")
        lines.append("")
        return "\n".join(lines)

    op_ids = [c.get("op_id", "?") for c in selected]
    lines += [
        f"**Selected winners ({len(selected)}):** " + ", ".join(f"`{o}`" for o in op_ids),
        "",
        "## Per-winner breakdown",
        "",
    ]
    for cand in selected:
        lines += _render_candidate(cand)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--round",
        dest="round_index",
        type=int,
        default=-1,
        help="Which round to render. Python list indexing: -1 (default) = last, 0 = first, -2 = second-to-last, etc.",
    )
    args = parser.parse_args()

    try:
        state = json.loads(args.state.read_text())
    except FileNotFoundError:
        print(f"error: state file not found: {args.state}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: state file is not valid JSON: {exc}", file=sys.stderr)
        return 2

    try:
        rendered = render(state, round_index=args.round_index)
    except IndexError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
