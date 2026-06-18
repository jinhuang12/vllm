#!/usr/bin/env python3
"""TDD tests for v2 round paths in verify_stage2_gate.py.

Spec: docs/superpowers/specs/2026-05-12-ammo-artifact-layout-design.md
Plan: .claude/plans/artifact-layout-v2-skills.md (Task 1D)

Verifies the gate finds:
  - rounds/{N}/mining/bottleneck_analysis.md
  - rounds/{N}/sweeps/baseline/e2e_latency_results.json
  - falls back to legacy root paths for v1 campaigns
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_GATE = _SCRIPTS_DIR / "verify_stage2_gate.py"


def _bottleneck_md(with_dilution: bool = False) -> str:
    """Generate a minimal bottleneck_analysis.md that passes Stage 2 checks."""
    base = (
        "# Bottleneck Analysis\n\n"
        "## Technology Landscape\n\n"
        "Some content.\n\n"
    )
    if with_dilution:
        base += (
            "## Dilution Table\n\n"
            "| Component | decode_busy | decode_share_of_e2e | f_e2e |\n"
            "|---|---|---|---|\n"
            "| GEMM | 0.5 | 0.6 | 0.3 |\n"
        )
    return base


def _e2e_json() -> str:
    return json.dumps({
        "summary": {"baseline": {}, "opt": {}},
    })


def _make_v2_artifact(tmp_path: Path, *, current_round: int = 1, with_dilution: bool = False) -> Path:
    artifact = tmp_path / "v2_artifact"
    rd = artifact / "rounds" / str(current_round)
    (rd / "mining").mkdir(parents=True)
    (rd / "sweeps" / "baseline").mkdir(parents=True)
    (rd / "mining" / "bottleneck_analysis.md").write_text(
        _bottleneck_md(with_dilution=with_dilution), encoding="utf-8"
    )
    (rd / "sweeps" / "baseline" / "e2e_latency_results.json").write_text(_e2e_json())
    state = {"schema_version": "4.1", "campaign": {"current_round": current_round, "schema_version": "4.0"}}
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return artifact


def _make_legacy_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "v1_artifact"
    (artifact / "e2e_latency").mkdir(parents=True)
    (artifact / "bottleneck_analysis.md").write_text(_bottleneck_md(), encoding="utf-8")
    (artifact / "e2e_latency" / "e2e_latency_results.json").write_text(_e2e_json())
    return artifact


def _run_gate(artifact_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_GATE), str(artifact_dir)],
        capture_output=True, text=True,
    )


class TestStage2GateV2:
    def test_finds_bottleneck_in_round_dir(self, tmp_path):
        artifact = _make_v2_artifact(tmp_path, current_round=1)
        result = _run_gate(artifact)
        assert result.returncode == 0, f"unexpected failure: {result.stdout} / {result.stderr}"
        assert "PASS" in result.stdout

    def test_finds_e2e_results_in_round_sweeps(self, tmp_path):
        artifact = _make_v2_artifact(tmp_path, current_round=2)
        result = _run_gate(artifact)
        assert result.returncode == 0, f"unexpected failure: {result.stdout} / {result.stderr}"

    def test_legacy_root_paths_still_work(self, tmp_path):
        artifact = _make_legacy_artifact(tmp_path)
        result = _run_gate(artifact)
        assert result.returncode == 0, f"legacy fallback failed: {result.stdout} / {result.stderr}"
        assert "PASS" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
