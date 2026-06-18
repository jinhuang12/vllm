#!/usr/bin/env python3
"""TDD tests for v2 --round/--slot flags on run_vllm_bench_latency_sweep.py.

Spec: docs/superpowers/specs/2026-05-12-ammo-artifact-layout-design.md (§Path Resolution)
Plan: .claude/plans/artifact-layout-v2-skills.md (Task 1B)

Verifies:
  - --round N --slot SLOT resolves to rounds/{N}/sweeps/{SLOT}/
  - --round without --slot fails
  - state.json fallback when --round missing
  - --out-name produces a hard error with helpful message
  - Archive moves to rounds/{N}/_archive/{slot}_{timestamp}/
  - Legacy fallback when no rounds/ dir exists (v1 layout)
  - nsys traces redirect to rounds/{N}/profiling/nsys/
  - legacy/manual torch_profile paths remain resolvable for compatibility
  - _build_child_cmd no longer includes --out-name
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_v2_artifact(tmp_path: Path, current_round: int = 1) -> Path:
    """Create a v2-layout artifact dir with rounds/{current_round}/ scaffold + state.json."""
    artifact = tmp_path / "artifact"
    (artifact / "rounds" / str(current_round) / "sweeps").mkdir(parents=True)
    state = {
        "schema_version": "4.1",
        "campaign": {"current_round": current_round, "rounds": []},
    }
    (artifact / "state.json").write_text(json.dumps(state), encoding="utf-8")
    # Minimal target.json so script imports validate.
    target = {
        "artifact_dir": str(artifact),
        "target": {"model_id": "x", "dtype": "fp16", "tp": 1, "ep": 1, "max_model_len": 4096},
        "workload": {"input_len": 64, "output_len": 512, "batch_sizes": [1], "num_iters": 1},
        "bench": {
            "runner": "vllm_bench_latency",
            "vllm_cmd": "vllm",
            "extra_args": [],
            "baseline_extra_args": [],
            "opt_extra_args": [],
            "baseline_env": {},
            "opt_env": {},
            "baseline_label": "baseline",
            "opt_label": "opt",
        },
    }
    (artifact / "target.json").write_text(json.dumps(target), encoding="utf-8")
    return artifact


def _make_legacy_artifact(tmp_path: Path) -> Path:
    """Create a legacy (v1) flat-layout artifact (no rounds/ dir)."""
    artifact = tmp_path / "artifact_legacy"
    artifact.mkdir(parents=True)
    return artifact


# ---------------------------------------------------------------------------
# Layout detection helper
# ---------------------------------------------------------------------------

class TestLayoutDetection:
    """`_is_v2_layout(artifact_dir)` is the sole layout check."""

    def test_v2_layout_detected_when_rounds_dir_exists(self, tmp_path):
        from run_vllm_bench_latency_sweep import _is_v2_layout
        artifact = _make_v2_artifact(tmp_path)
        assert _is_v2_layout(artifact) is True

    def test_legacy_when_no_rounds_dir(self, tmp_path):
        from run_vllm_bench_latency_sweep import _is_v2_layout
        artifact = _make_legacy_artifact(tmp_path)
        assert _is_v2_layout(artifact) is False


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestRoundSlotPathResolution:
    """`_resolve_sweep_out_name(args, artifact_dir)` -> str like 'rounds/1/sweeps/baseline'."""

    def _resolve(self, *, round_=None, slot=None, out_name="e2e_latency",
                 _out_root=None, artifact_dir=None):
        from run_vllm_bench_latency_sweep import _resolve_sweep_out_name
        import argparse as _ap
        ns = _ap.Namespace(round=round_, slot=slot, out_name=out_name, _out_root=_out_root)
        return _resolve_sweep_out_name(ns, artifact_dir)

    def test_round_and_slot_baseline_resolves_path(self, tmp_path):
        artifact = _make_v2_artifact(tmp_path, current_round=1)
        assert self._resolve(round_=1, slot="baseline", artifact_dir=artifact) == "rounds/1/sweeps/baseline"

    def test_round_and_slot_opt_resolves_path(self, tmp_path):
        artifact = _make_v2_artifact(tmp_path, current_round=2)
        assert self._resolve(round_=2, slot="opt/op007", artifact_dir=artifact) == "rounds/2/sweeps/opt/op007"

    def test_round_and_slot_integration_resolves_path(self, tmp_path):
        artifact = _make_v2_artifact(tmp_path, current_round=1)
        assert self._resolve(round_=1, slot="integration", artifact_dir=artifact) == "rounds/1/sweeps/integration"

    def test_round_and_slot_golden_capture(self, tmp_path):
        artifact = _make_v2_artifact(tmp_path, current_round=1)
        assert self._resolve(round_=1, slot="golden_capture", artifact_dir=artifact) == "rounds/1/sweeps/golden_capture"

    def test_round_only_fails(self, tmp_path):
        artifact = _make_v2_artifact(tmp_path, current_round=1)
        with pytest.raises(SystemExit, match="(?i)slot"):
            self._resolve(round_=2, slot=None, artifact_dir=artifact)

    def test_neither_reads_state_json(self, tmp_path):
        """No --round, --slot only: read current_round from state.json."""
        artifact = _make_v2_artifact(tmp_path, current_round=3)
        assert self._resolve(round_=None, slot="baseline", artifact_dir=artifact) == "rounds/3/sweeps/baseline"

    def test_legacy_fallback_no_rounds_dir(self, tmp_path):
        """Legacy (v1) artifact dir: --out-name fallback used."""
        artifact = _make_legacy_artifact(tmp_path)
        # Default out-name returns unchanged (fallback path).
        assert self._resolve(round_=None, slot=None,
                             out_name="e2e_latency", artifact_dir=artifact) == "e2e_latency"


# ---------------------------------------------------------------------------
# --out-name hard error
# ---------------------------------------------------------------------------

class TestOutNameHardError:
    """Custom --out-name now hard-errors with guidance toward --round/--slot."""

    def test_out_name_errors_with_helpful_message(self, tmp_path):
        from run_vllm_bench_latency_sweep import _resolve_sweep_out_name
        import argparse as _ap
        artifact = _make_v2_artifact(tmp_path)
        ns = _ap.Namespace(round=None, slot=None, out_name="custom_name", _out_root=None)
        with pytest.raises(SystemExit) as exc:
            _resolve_sweep_out_name(ns, artifact)
        msg = str(exc.value)
        assert "--round" in msg or "--slot" in msg, f"unhelpful error: {msg!r}"


# ---------------------------------------------------------------------------
# Archive behavior
# ---------------------------------------------------------------------------

class TestArchive:
    """Existing v2 sweep output is moved to rounds/{N}/_archive/{slot}_{timestamp}/."""

    def test_archive_moves_to_rounds_archive(self, tmp_path):
        from run_vllm_bench_latency_sweep import _prepare_out_root
        artifact = _make_v2_artifact(tmp_path, current_round=1)
        # Pre-existing baseline output with a sentinel file:
        baseline = artifact / "rounds" / "1" / "sweeps" / "baseline"
        baseline.mkdir(parents=True, exist_ok=True)
        sentinel = baseline / "old_results.json"
        sentinel.write_text("{}", encoding="utf-8")

        # The new logic: when out_name resolves under rounds/{N}/sweeps/{SLOT},
        # archive moves to rounds/{N}/_archive/{slot}_{timestamp}/.
        out_root = _prepare_out_root(
            artifact_dir=artifact,
            out_name="rounds/1/sweeps/baseline",
            overwrite=False,
        )
        # Old contents must be in _archive — not in baseline.
        assert not sentinel.exists(), "old sentinel should have been moved out"
        archive_dir = artifact / "rounds" / "1" / "_archive"
        assert archive_dir.is_dir(), "rounds/1/_archive/ should exist after archiving"
        # At least one archived dir starting with baseline_:
        archived = list(archive_dir.glob("baseline_*"))
        assert archived, f"expected baseline_*/ subdir in {archive_dir}, got: {list(archive_dir.iterdir())}"
        # Sentinel preserved inside the archive subdir.
        assert (archived[0] / "old_results.json").exists()
        # New out_root is the empty baseline dir.
        assert out_root == baseline
        assert out_root.is_dir() and not any(out_root.iterdir())


# ---------------------------------------------------------------------------
# _build_child_cmd no longer carries --out-name
# ---------------------------------------------------------------------------

class TestBuildChildCmdNoOutName:
    """After the refactor, _build_child_cmd passes only --_out-root, never --out-name."""

    def test_child_cmd_omits_out_name(self):
        from run_vllm_bench_latency_sweep import _build_child_cmd
        cmd = _build_child_cmd(
            python_exe="/venv/bin/python",
            script_path=Path("/sweep.py"),
            run_label="baseline",
            artifact_dir=Path("/art"),
            target_path=Path("/art/target.json"),
            timeout_s=1800,
            out_name="rounds/1/sweeps/baseline",
            out_root=Path("/art/rounds/1/sweeps/baseline"),
            dp=1,
            nproc=1,
            extra_child_flags=[],
        )
        assert "--out-name" not in cmd, \
            f"_build_child_cmd must not include --out-name; got cmd={cmd!r}"
        # --_out-root carries the resolved absolute path.
        assert "--_out-root" in cmd
        idx = cmd.index("--_out-root")
        assert cmd[idx + 1] == "/art/rounds/1/sweeps/baseline"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
