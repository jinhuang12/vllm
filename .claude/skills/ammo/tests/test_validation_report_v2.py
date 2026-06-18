#!/usr/bin/env python3
"""TDD tests for v2 --round/--slot flags on generate_validation_report.py.

Spec: docs/superpowers/specs/2026-05-12-ammo-artifact-layout-design.md
Plan: .claude/plans/artifact-layout-v2-skills.md (Task 1F)

Verifies:
  - --round 1 --slot baseline reads from rounds/1/sweeps/baseline/e2e_latency_results.json
  - Legacy default (no flags) still reads from {artifact_dir}/e2e_latency/.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_GEN = _SCRIPTS_DIR / "generate_validation_report.py"


def _e2e_payload(speedup: float = 1.05) -> str:
    return json.dumps({
        "model_id": "test/model",
        "tp": 1,
        "max_model_len": 4096,
        "workload": {"input_len": 64, "output_len": 64, "num_iters": 1},
        "results": [{
            "batch_size": 1,
            "speedup": speedup,
            "improvement_pct": (speedup - 1) * 100,
            "baseline": {"avg_s": 1.0},
            "opt": {"avg_s": 1.0 / speedup},
        }],
        "bench": {"baseline_label": "baseline", "opt_label": "opt"},
    })


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_GEN)] + args, capture_output=True, text=True
    )


class TestValidationReportV2:
    def test_round_and_slot_resolves_e2e_path(self, tmp_path):
        artifact = tmp_path / "v2_artifact"
        rd = artifact / "rounds" / "1" / "sweeps" / "baseline"
        rd.mkdir(parents=True)
        (rd / "e2e_latency_results.json").write_text(_e2e_payload(speedup=1.10))
        # Make it look like v2 (rounds dir present); state.json optional.
        result = _run([
            "--artifact-dir", str(artifact),
            "--round", "1",
            "--slot", "baseline",
        ])
        assert result.returncode == 0, f"failed: {result.stderr}"
        out = (artifact / "validation_results.md").read_text()
        # Realized 10% improvement should appear in the table.
        assert "1.1" in out or "10%" in out, \
            f"expected speedup data from rounds/1/sweeps/baseline; got:\n{out[:500]}"

    def test_legacy_path_still_works(self, tmp_path):
        artifact = tmp_path / "v1_artifact"
        e2e_dir = artifact / "e2e_latency"
        e2e_dir.mkdir(parents=True)
        (e2e_dir / "e2e_latency_results.json").write_text(_e2e_payload(speedup=1.20))
        result = _run(["--artifact-dir", str(artifact)])
        assert result.returncode == 0, f"failed: {result.stderr}"
        out = (artifact / "validation_results.md").read_text()
        assert "1.2" in out or "20%" in out, \
            f"expected legacy speedup data; got:\n{out[:500]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
