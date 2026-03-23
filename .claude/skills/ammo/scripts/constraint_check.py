#!/usr/bin/env python3
"""Constraint pre-screening gate for AMMO debate proposals (R1).

Parses debate proposal markdown files, extracts constraint-relevant claims,
and checks them against hardware limits and technique blacklists.

Usage:
    python constraint_check.py \\
        --proposals-dir /path/to/proposals \\
        --hardware L40S \\
        --blacklist "split-k,persistent warp" \\
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Hardware SMEM limits (bytes) — max shared memory per SM
# ---------------------------------------------------------------------------

SMEM_LIMITS: Dict[str, int] = {
    "L40S": 102400,   # 100 KB
    "H100": 233472,   # 228 KB
    "A100": 167936,   # 164 KB
    "B200": 262144,   # 256 KB
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConstraintResult:
    check: str
    passed: bool
    reason: str = ""
    skipped: bool = False


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_smem_budget(
    smem_bytes: Optional[int],
    sm_smem_limit: int,
    double_buffered: bool = False,
) -> ConstraintResult:
    """Check whether the claimed SMEM fits within the hardware limit.

    Args:
        smem_bytes: Claimed shared memory in bytes. If None, check is skipped.
        sm_smem_limit: Hardware SM shared memory limit in bytes.
        double_buffered: If True, effective SMEM = smem_bytes * 2.

    Returns:
        ConstraintResult with passed/skipped status and a KB-formatted reason on failure.
    """
    if smem_bytes is None:
        return ConstraintResult(check="smem_budget", passed=True, skipped=True)

    effective = smem_bytes * 2 if double_buffered else smem_bytes

    if effective > sm_smem_limit:
        effective_kb = effective / 1024
        limit_kb = sm_smem_limit / 1024
        reason = (
            f"SMEM {effective_kb:.1f} KB exceeds hardware limit {limit_kb:.1f} KB"
            + (" (double-buffered)" if double_buffered else "")
        )
        return ConstraintResult(check="smem_budget", passed=False, reason=reason)

    return ConstraintResult(check="smem_budget", passed=True)


def check_technique_blacklist(
    technique_desc: str,
    blacklist: List[str],
) -> ConstraintResult:
    """Check whether the technique description matches any blacklist entry.

    Matching is case-insensitive substring search.

    Args:
        technique_desc: Description of the proposed technique.
        blacklist: List of forbidden technique substrings.

    Returns:
        ConstraintResult indicating whether the technique is blocked.
    """
    lower_desc = technique_desc.lower()
    for entry in blacklist:
        if entry.lower() in lower_desc:
            return ConstraintResult(
                check="technique_blacklist",
                passed=False,
                reason=f"Technique matches blacklisted entry: '{entry}'",
            )
    return ConstraintResult(check="technique_blacklist", passed=True)


# ---------------------------------------------------------------------------
# Proposal parser
# ---------------------------------------------------------------------------

def parse_proposal_constraints(proposal_text: str) -> Dict:
    """Extract constraint-relevant information from a proposal markdown string.

    Extracts:
    - smem_bytes: int or None — shared memory claim converted to bytes
    - double_buffered: bool — whether double-buffering is mentioned
    - technique: str or None — technique name from Candidate Specification section

    Args:
        proposal_text: Raw markdown text of the proposal.

    Returns:
        Dict with keys: smem_bytes, double_buffered, technique
    """
    result: Dict = {
        "smem_bytes": None,
        "double_buffered": False,
        "technique": None,
    }

    # --- Extract SMEM claim ---
    # Match patterns like "48 KB", "48KB", "49152 bytes", "49152B"
    kb_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*KB", re.IGNORECASE)
    bytes_pattern = re.compile(r"(\d+)\s*(?:bytes|B)\b", re.IGNORECASE)

    # Search for shared memory context lines first
    smem_context_pattern = re.compile(
        r"shared[\s_-]*mem(?:ory)?\s*[:\-]?\s*"
        r"(\d+(?:\.\d+)?)\s*(KB|bytes?|B)\b",
        re.IGNORECASE,
    )
    smem_match = smem_context_pattern.search(proposal_text)
    if smem_match:
        value = float(smem_match.group(1))
        unit = smem_match.group(2).lower()
        if unit == "kb":
            result["smem_bytes"] = int(value * 1024)
        else:
            result["smem_bytes"] = int(value)
    else:
        # Fallback: grab first standalone KB mention
        kb_match = kb_pattern.search(proposal_text)
        if kb_match:
            result["smem_bytes"] = int(float(kb_match.group(1)) * 1024)
        else:
            bytes_match = bytes_pattern.search(proposal_text)
            if bytes_match:
                result["smem_bytes"] = int(bytes_match.group(1))

    # --- Detect double buffering ---
    if re.search(r"double[\s_-]*buffer", proposal_text, re.IGNORECASE):
        result["double_buffered"] = True

    # --- Extract technique from Candidate Specification section ---
    spec_section_pattern = re.compile(
        r"##\s*Candidate Specification.*?(?=\n##|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    spec_match = spec_section_pattern.search(proposal_text)
    if spec_match:
        section_text = spec_match.group(0)
        technique_pattern = re.compile(r"Technique\s*[:\-]\s*(.+)", re.IGNORECASE)
        tech_match = technique_pattern.search(section_text)
        if tech_match:
            result["technique"] = tech_match.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_checks(
    proposal_path: Path,
    hardware: str,
    blacklist: List[str],
) -> List[ConstraintResult]:
    """Run all constraint checks against a single proposal file.

    Args:
        proposal_path: Path to the proposal markdown file.
        hardware: Hardware name (must be in SMEM_LIMITS).
        blacklist: List of forbidden technique substrings.

    Returns:
        List of ConstraintResult objects.
    """
    if hardware not in SMEM_LIMITS:
        known = ", ".join(sorted(SMEM_LIMITS))
        raise ValueError(f"Unknown hardware '{hardware}'. Known: {known}")

    sm_smem_limit = SMEM_LIMITS[hardware]
    proposal_text = proposal_path.read_text()
    constraints = parse_proposal_constraints(proposal_text)

    results: List[ConstraintResult] = []

    results.append(
        check_smem_budget(
            smem_bytes=constraints["smem_bytes"],
            sm_smem_limit=sm_smem_limit,
            double_buffered=constraints["double_buffered"],
        )
    )

    if constraints["technique"]:
        results.append(
            check_technique_blacklist(constraints["technique"], blacklist)
        )
    else:
        results.append(
            ConstraintResult(
                check="technique_blacklist",
                passed=True,
                skipped=True,
                reason="No technique found in proposal",
            )
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constraint pre-screening gate for AMMO debate proposals."
    )
    parser.add_argument(
        "--proposals-dir",
        required=True,
        help="Directory containing proposal markdown files.",
    )
    parser.add_argument(
        "--hardware",
        default="L40S",
        choices=list(SMEM_LIMITS),
        help="Target hardware for SMEM limit lookup.",
    )
    parser.add_argument(
        "--blacklist",
        default="",
        help="Comma-separated list of blacklisted technique substrings.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON results.",
    )
    args = parser.parse_args()

    proposals_dir = Path(args.proposals_dir)
    if not proposals_dir.is_dir():
        print(f"ERROR: proposals-dir '{proposals_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    blacklist = [b.strip() for b in args.blacklist.split(",") if b.strip()]

    all_results: Dict[str, List[Dict]] = {}
    any_failed = False

    for proposal_path in sorted(proposals_dir.glob("*.md")):
        results = run_checks(proposal_path, args.hardware, blacklist)
        serialized = [
            {
                "check": r.check,
                "passed": r.passed,
                "reason": r.reason,
                "skipped": r.skipped,
            }
            for r in results
        ]
        all_results[proposal_path.name] = serialized

        for r in results:
            if not r.passed and not r.skipped:
                any_failed = True
                print(f"FAIL [{proposal_path.name}] {r.check}: {r.reason}")
            elif r.skipped:
                print(f"SKIP [{proposal_path.name}] {r.check}")
            else:
                print(f"PASS [{proposal_path.name}] {r.check}")

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nResults written to {output_path}")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
