#!/usr/bin/env python3
"""T5 gate: verify Stage 2 artifacts exist.

Checks:
1. bottleneck_analysis.md exists and contains a '## Technology Landscape' section
2. e2e_latency_results.json exists
3. bottleneck_analysis.md publishes the f_e2e dilution fields:
   `decode_busy`, `decode_share_of_e2e`, and an `f_e2e` column in the
   top-components table.

Exit 0 = PASS, exit 1 = FAIL (prints reason to stdout).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional


def _check_dilution_fields(content: str) -> list[str]:
    """Return a list of missing-field strings; empty list = all present.

    Uses word-boundary regex so "f_e2e" doesn't match the "_e2e" suffix of
    "decode_share_of_e2e" (and vice-versa). Token characters allowed: word
    chars + underscore (Python \\b uses \\w).

    For f_e2e: requires at least one markdown table row containing a numeric
    value in an f_e2e column (not just prose mention of the token).
    """
    missing: list[str] = []
    if not re.search(r"\bdecode_busy\b", content):
        missing.append("decode_busy")
    if not re.search(r"\bdecode_share_of_e2e\b", content):
        missing.append("decode_share_of_e2e")
    if not re.search(r"\bf_e2e\b", content):
        missing.append("f_e2e column")
    elif not _has_numeric_f_e2e_row(content):
        missing.append("f_e2e column (token present but no table row with numeric value)")
    return missing


def _has_numeric_f_e2e_row(content: str) -> bool:
    """Check that at least one markdown table row contains a numeric f_e2e value."""
    lines = content.splitlines()
    f_e2e_col_idx = None
    for i, line in enumerate(lines):
        if not line.strip().startswith("|"):
            f_e2e_col_idx = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # Detect header row containing f_e2e
        if any(c.strip().lower() == "f_e2e" for c in cells):
            f_e2e_col_idx = next(
                j for j, c in enumerate(cells) if c.strip().lower() == "f_e2e"
            )
            continue
        # Skip separator rows
        if f_e2e_col_idx is not None and re.match(r"^[\s|:-]+$", line):
            continue
        # Data row — check if f_e2e column has a numeric value
        if f_e2e_col_idx is not None and f_e2e_col_idx < len(cells):
            val = cells[f_e2e_col_idx].strip().rstrip("%")
            try:
                float(val)
                return True
            except ValueError:
                continue
    return False


def _is_v2_layout(artifact_dir: Path) -> bool:
    return (artifact_dir / "rounds").is_dir()


def _read_current_round(artifact_dir: Path) -> int:
    try:
        state = json.loads((artifact_dir / "state.json").read_text(encoding="utf-8"))
        return int(state["campaign"]["current_round"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError, ValueError):
        return 1


def _resolve_bottleneck_path(artifact_dir: Path, round_arg: Optional[int]) -> Path:
    """v2: rounds/{N}/mining/bottleneck_analysis.md
    legacy: {artifact_dir}/bottleneck_analysis.md
    """
    if _is_v2_layout(artifact_dir):
        n = round_arg if round_arg is not None else _read_current_round(artifact_dir)
        v2_path = artifact_dir / "rounds" / str(n) / "mining" / "bottleneck_analysis.md"
        if v2_path.exists():
            return v2_path
        # Fall through to legacy if v2 path doesn't yet exist (back-compat).
    return artifact_dir / "bottleneck_analysis.md"


def _find_e2e_results(artifact_dir: Path, round_arg: Optional[int]) -> list[Path]:
    """v2: rounds/{N}/sweeps/baseline/e2e_latency_results.json (preferred) +
    rounds/{N}/sweeps/{integration,opt/*}/e2e_latency_results.json
    legacy: {artifact_dir}/e2e_latency*/e2e_latency_results.json
    """
    found: list[Path] = []
    if _is_v2_layout(artifact_dir):
        n = round_arg if round_arg is not None else _read_current_round(artifact_dir)
        round_root = artifact_dir / "rounds" / str(n) / "sweeps"
        for slot in ("baseline", "integration"):
            rf = round_root / slot / "e2e_latency_results.json"
            if rf.exists():
                found.append(rf)
        if (round_root / "opt").is_dir():
            for op_dir in (round_root / "opt").iterdir():
                rf = op_dir / "e2e_latency_results.json"
                if rf.exists():
                    found.append(rf)
        if found:
            return found

    # Legacy fallback (or v2 with no v2 results yet).
    candidate_dirs: list[Path] = []
    for name in ("e2e_latency", "e2e_latency_gate", "e2e_latency_combined"):
        d = artifact_dir / name
        if d.is_dir():
            candidate_dirs.append(d)
    for d in artifact_dir.glob("e2e_latency*"):
        if d.is_dir() and d not in candidate_dirs:
            candidate_dirs.append(d)
    for d in candidate_dirs:
        rf = d / "e2e_latency_results.json"
        if rf.exists():
            found.append(rf)
    return found


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: verify_stage2_gate.py <artifact_dir> [--round N]")
        return 1

    # Lightweight arg parsing (preserve historical positional usage).
    artifact_dir = Path(sys.argv[1])
    round_arg: Optional[int] = None
    if "--round" in sys.argv:
        idx = sys.argv.index("--round")
        if idx + 1 < len(sys.argv):
            try:
                round_arg = int(sys.argv[idx + 1])
            except ValueError:
                print(f"FAIL: --round must be an int, got {sys.argv[idx + 1]!r}")
                return 1

    # Check 1: bottleneck_analysis.md with Technology Landscape
    ba = _resolve_bottleneck_path(artifact_dir, round_arg)
    if not ba.exists():
        print(f"FAIL: {ba} does not exist")
        return 1
    content = ba.read_text(encoding="utf-8")
    if "## Technology Landscape" not in content:
        print(f"FAIL: {ba} missing '## Technology Landscape' section")
        return 1

    # Check 2: e2e_latency_results.json exists (v2: rounds/{N}/sweeps/*; legacy fallback).
    result_files = _find_e2e_results(artifact_dir, round_arg)
    if not result_files:
        print(f"FAIL: no e2e_latency_results.json found under {artifact_dir}")
        return 1

    # Check 3: f_e2e dilution fields are present in bottleneck_analysis.md.
    missing = _check_dilution_fields(content)
    if missing:
        print(
            f"FAIL: {ba} missing required f_e2e dilution field(s): "
            f"{', '.join(missing)}"
        )
        return 1

    print("PASS: bottleneck_analysis.md + Technology Landscape + e2e_latency_results.json + dilution fields present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
