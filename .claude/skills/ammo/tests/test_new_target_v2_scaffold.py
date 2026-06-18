#!/usr/bin/env python3
"""TDD tests for v2 round-scoped scaffold in new_target.py.

Spec: docs/superpowers/specs/2026-05-12-ammo-artifact-layout-design.md (§Bootstrap)
Plan: .claude/plans/artifact-layout-v2-skills.md (Task 1A)

Verifies that `new_target.py` creates the v2 layout:
  rounds/1/{profiling/{nsys,ncu},
            sweeps/{baseline/{json,logs,status},opt,integration,golden_capture},
            mining, debate/{proposals,micro_experiments,monitor_audits},
            tracks, audits, _archive}
  blockers/                    (root, NOT round-scoped)

Legacy dirs (`investigation/`, `runs/`, `nsys/`) MUST NOT be created at root.
`constraints.md` MUST NOT be written at root.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_NEW_TARGET = _SCRIPTS_DIR / "new_target.py"


@pytest.fixture
def scaffolded_artifact_dir(tmp_path):
    """Run new_target.py to scaffold a fresh artifact dir, return its path."""
    artifact_dir = tmp_path / "artifact"
    cmd = [
        sys.executable,
        str(_NEW_TARGET),
        "--artifact-dir", str(artifact_dir),
        "--model-id", "test-model",
        "--hardware", "H100",
        "--dtype", "fp8",
        "--tp", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"new_target.py failed: {result.stderr}"
    return artifact_dir


def test_scaffold_creates_rounds_1_dir(scaffolded_artifact_dir):
    assert (scaffolded_artifact_dir / "rounds" / "1").is_dir()


def test_scaffold_creates_profiling_subdirs(scaffolded_artifact_dir):
    base = scaffolded_artifact_dir / "rounds" / "1" / "profiling"
    for sub in ("nsys", "ncu"):
        assert (base / sub).is_dir(), f"missing rounds/1/profiling/{sub}/"
    assert not (base / "probe").exists()
    assert not (base / "torch_profile").exists()


def test_scaffold_creates_sweeps_subdirs(scaffolded_artifact_dir):
    sweeps = scaffolded_artifact_dir / "rounds" / "1" / "sweeps"
    # baseline has nested json/logs/status
    for sub in ("json", "logs", "status"):
        assert (sweeps / "baseline" / sub).is_dir(), f"missing baseline/{sub}/"
    # other slot containers
    for slot in ("opt", "integration", "golden_capture"):
        assert (sweeps / slot).is_dir(), f"missing sweeps/{slot}/"


def test_scaffold_creates_debate_subdirs(scaffolded_artifact_dir):
    base = scaffolded_artifact_dir / "rounds" / "1" / "debate"
    for sub in ("proposals", "micro_experiments", "monitor_audits"):
        assert (base / sub).is_dir(), f"missing debate/{sub}/"


def test_scaffold_creates_mining_dir(scaffolded_artifact_dir):
    assert (scaffolded_artifact_dir / "rounds" / "1" / "mining").is_dir()


def test_scaffold_creates_tracks_dir(scaffolded_artifact_dir):
    assert (scaffolded_artifact_dir / "rounds" / "1" / "tracks").is_dir()


def test_scaffold_creates_audits_dir(scaffolded_artifact_dir):
    assert (scaffolded_artifact_dir / "rounds" / "1" / "audits").is_dir()


def test_scaffold_creates_archive_dir(scaffolded_artifact_dir):
    assert (scaffolded_artifact_dir / "rounds" / "1" / "_archive").is_dir()


def test_scaffold_creates_blockers_at_root(scaffolded_artifact_dir):
    """blockers/ is cross-round (campaign-level), NOT under rounds/{N}/."""
    assert (scaffolded_artifact_dir / "blockers").is_dir()
    assert not (scaffolded_artifact_dir / "rounds" / "1" / "blockers").exists()


def test_scaffold_no_legacy_dirs(scaffolded_artifact_dir):
    """Legacy flat layout dirs MUST NOT be created at the campaign root."""
    for legacy in ("investigation", "runs", "nsys", "e2e_latency",
                   "monitoring", "tracks", "debate", "mining", "audits"):
        assert not (scaffolded_artifact_dir / legacy).exists(), \
            f"legacy root dir {legacy}/ should not be scaffolded"


def test_no_constraints_md_at_root(scaffolded_artifact_dir):
    """constraints.md is round-scoped (rounds/{N}/constraints.md), not at root."""
    assert not (scaffolded_artifact_dir / "constraints.md").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
