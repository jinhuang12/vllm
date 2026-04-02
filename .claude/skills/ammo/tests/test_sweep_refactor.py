#!/usr/bin/env python3
"""Tests for the sweep script refactor: GPU lock, argparse, config validation.

Covers the three removed options and their replacements:
  1. --no-gpu-lock / --gpu-lock  → lock always on, children skip via --_child-label
  2. --allow-identical-config     → unconditional SystemExit on identical configs
  3. --execution-mode cli_per_bs  → only inproc_sweep exists

Also covers previously-untested pure functions: _is_placeholder, _require*,
_sanitize_filename, _parse_latency_metrics, _check_patterns, _format_cmd_for_md,
_bench_exe_tokens, _prepare_out_root.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Ensure the scripts directory is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Argparse: removed flags should not be recognized
# ---------------------------------------------------------------------------

class TestArgparseRemovedFlags:
    """Verify removed CLI flags are rejected by the argument parser."""

    def _parser(self):
        """Get the script's argument parser by importing main and intercepting."""
        from run_vllm_bench_latency_sweep import main
        import argparse as _ap

        # Capture the parser before it calls parse_args.
        original_parse = _ap.ArgumentParser.parse_args

        captured = {}

        def intercept(self, args=None, namespace=None):
            captured["parser"] = self
            raise SystemExit(0)  # Don't actually run main()

        with mock.patch.object(_ap.ArgumentParser, "parse_args", intercept):
            try:
                main()
            except SystemExit:
                pass

        return captured.get("parser")

    def test_no_gpu_lock_rejected(self):
        """--no-gpu-lock is no longer a valid argument."""
        p = self._parser()
        if p is None:
            pytest.skip("Could not capture parser")
        with pytest.raises(SystemExit):
            p.parse_args(["--artifact-dir", "/tmp/x", "--no-gpu-lock"])

    def test_gpu_lock_rejected(self):
        """--gpu-lock is no longer a valid argument."""
        p = self._parser()
        if p is None:
            pytest.skip("Could not capture parser")
        with pytest.raises(SystemExit):
            p.parse_args(["--artifact-dir", "/tmp/x", "--gpu-lock"])

    def test_allow_identical_config_rejected(self):
        """--allow-identical-config is no longer a valid argument."""
        p = self._parser()
        if p is None:
            pytest.skip("Could not capture parser")
        with pytest.raises(SystemExit):
            p.parse_args(["--artifact-dir", "/tmp/x", "--allow-identical-config"])

    def test_execution_mode_rejected(self):
        """--execution-mode is no longer a valid argument."""
        p = self._parser()
        if p is None:
            pytest.skip("Could not capture parser")
        with pytest.raises(SystemExit):
            p.parse_args(["--artifact-dir", "/tmp/x", "--execution-mode", "inproc_sweep"])

    def test_child_label_still_accepted(self):
        """Internal --_child-label is still accepted."""
        p = self._parser()
        if p is None:
            pytest.skip("Could not capture parser")
        ns = p.parse_args(["--artifact-dir", "/tmp/x", "--_child-label", "baseline"])
        assert ns._child_label == "baseline"


# ---------------------------------------------------------------------------
# Config identity check (unconditional)
# ---------------------------------------------------------------------------

class TestIdenticalConfigCheck:
    """The identical-config check is now unconditional — no bypass flag."""

    def _make_target_json(self, tmp_path, *, opt_env=None, opt_extra_args=None,
                          baseline_extra_args=None):
        """Write a minimal target.json and return its path."""
        target = {
            "artifact_dir": str(tmp_path),
            "target": {
                "model_id": "test/model",
                "dtype": "fp16",
                "tp": 1,
                "ep": 1,
                "max_model_len": 4096,
            },
            "workload": {
                "input_len": 64,
                "output_len": 512,
                "batch_sizes": [1],
                "num_iters": 1,
            },
            "bench": {
                "runner": "vllm_bench_latency",
                "vllm_cmd": "vllm",
                "extra_args": [],
                "baseline_extra_args": baseline_extra_args or [],
                "opt_extra_args": opt_extra_args or [],
                "baseline_env": {},
                "opt_env": opt_env or {},
                "baseline_label": "baseline",
                "opt_label": "opt",
            },
        }
        p = tmp_path / "target.json"
        p.write_text(json.dumps(target), encoding="utf-8")
        return p

    def test_identical_config_both_labels_raises(self, tmp_path):
        """When both labels selected and configs identical, script fails fast."""
        self._make_target_json(tmp_path)
        from run_vllm_bench_latency_sweep import main
        with mock.patch("sys.argv", [
            "sweep", "--artifact-dir", str(tmp_path),
            "--labels", "baseline,opt",
        ]):
            with pytest.raises(SystemExit, match="identical"):
                main()

    def test_baseline_only_skips_check(self, tmp_path):
        """With --labels baseline, the identity check is skipped (only one label)."""
        self._make_target_json(tmp_path)
        from run_vllm_bench_latency_sweep import main
        # This should NOT raise about identical config — it'll fail later
        # at the actual benchmark step, but not at the config check.
        with mock.patch("sys.argv", [
            "sweep", "--artifact-dir", str(tmp_path),
            "--labels", "baseline",
        ]):
            # We expect it to get past config validation. It'll fail at
            # GPU lock or benchmark execution, but not at config identity.
            try:
                main()
            except SystemExit as e:
                assert "identical" not in str(e)

    def test_different_opt_env_passes_check(self, tmp_path):
        """When opt_env is set, configs differ, so the check passes."""
        self._make_target_json(tmp_path, opt_env={"MY_OPT_FLAG": "1"})
        from run_vllm_bench_latency_sweep import main
        with mock.patch("sys.argv", [
            "sweep", "--artifact-dir", str(tmp_path),
            "--labels", "baseline,opt",
        ]):
            try:
                main()
            except SystemExit as e:
                assert "identical" not in str(e)

    def test_different_opt_extra_args_passes_check(self, tmp_path):
        """When opt_extra_args differ from baseline, check passes."""
        self._make_target_json(tmp_path, opt_extra_args=["--some-flag"])
        from run_vllm_bench_latency_sweep import main
        with mock.patch("sys.argv", [
            "sweep", "--artifact-dir", str(tmp_path),
            "--labels", "baseline,opt",
        ]):
            try:
                main()
            except SystemExit as e:
                assert "identical" not in str(e)


# ---------------------------------------------------------------------------
# Sanitize filename
# ---------------------------------------------------------------------------

class TestSanitizeFilename:
    """Tests for GPU lock key sanitization."""

    def _call(self, s):
        from run_vllm_bench_latency_sweep import _sanitize_filename
        return _sanitize_filename(s)

    def test_simple_device_list(self):
        assert self._call("0,1,2") == "0_1_2"

    def test_alpha_numeric(self):
        assert self._call("gpu-0") == "gpu-0"

    def test_special_chars(self):
        assert self._call("a/b:c") == "a_b_c"

    def test_empty_string(self):
        assert self._call("") == "default"

    def test_all_special(self):
        assert self._call("///") == "default"

    def test_preserves_dots_equals(self):
        """Dots and equals are kept (conservative but safe)."""
        assert self._call("key=val.1") == "key=val.1"


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

class TestIsPlaceholder:
    """Tests for detecting placeholder values in target.json."""

    def _call(self, v):
        from run_vllm_bench_latency_sweep import _is_placeholder
        return _is_placeholder(v)

    def test_fill_me_marker(self):
        assert self._call("<FILL_ME>") is True

    def test_angle_bracket_custom(self):
        assert self._call("<ENABLE_FLAG>") is True

    def test_angle_with_whitespace(self):
        assert self._call("  <FLAG>  ") is True

    def test_real_value(self):
        assert self._call("VLLM_ENABLE_OPT") is False

    def test_non_string(self):
        assert self._call(42) is False
        assert self._call(None) is False
        assert self._call(["<FOO>"]) is False


# ---------------------------------------------------------------------------
# Require field helpers
# ---------------------------------------------------------------------------

class TestRequireFields:
    """Tests for _require, _require_int, _require_list_int."""

    def test_require_present(self):
        from run_vllm_bench_latency_sweep import _require
        assert _require({"key": "val"}, "key", "ctx") == "val"

    def test_require_missing_raises(self):
        from run_vllm_bench_latency_sweep import _require
        with pytest.raises(SystemExit, match="Missing required field"):
            _require({}, "key", "ctx")

    def test_require_placeholder_raises(self):
        from run_vllm_bench_latency_sweep import _require
        with pytest.raises(SystemExit, match="still placeholder"):
            _require({"key": "<FILL_ME>"}, "key", "ctx")

    def test_require_int_valid(self):
        from run_vllm_bench_latency_sweep import _require_int
        assert _require_int({"n": 42}, "n", "ctx") == 42

    def test_require_int_wrong_type(self):
        from run_vllm_bench_latency_sweep import _require_int
        with pytest.raises(SystemExit, match="Expected int"):
            _require_int({"n": "42"}, "n", "ctx")

    def test_require_list_int_valid(self):
        from run_vllm_bench_latency_sweep import _require_list_int
        assert _require_list_int({"bs": [1, 4, 8]}, "bs", "ctx") == [1, 4, 8]

    def test_require_list_int_mixed_types(self):
        from run_vllm_bench_latency_sweep import _require_list_int
        with pytest.raises(SystemExit, match="Expected list"):
            _require_list_int({"bs": [1, "four"]}, "bs", "ctx")


# ---------------------------------------------------------------------------
# Parse latency metrics
# ---------------------------------------------------------------------------

class TestParseLatencyMetrics:
    """Tests for stdout parsing of vllm bench latency output."""

    def _call(self, stdout):
        from run_vllm_bench_latency_sweep import _parse_latency_metrics
        return _parse_latency_metrics(stdout)

    def test_typical_output(self):
        stdout = (
            "Avg latency: 0.1234 seconds\n"
            "50% percentile latency: 0.1200 seconds\n"
            "99% percentile latency: 0.1300 seconds\n"
        )
        m = self._call(stdout)
        assert m["avg_s"] == pytest.approx(0.1234)
        assert m["p50_s"] == pytest.approx(0.1200)
        assert m["p99_s"] == pytest.approx(0.1300)

    def test_empty_output(self):
        assert self._call("") == {}

    def test_garbage_output(self):
        assert self._call("ERROR: model not found\nSegfault") == {}

    def test_scientific_notation(self):
        stdout = "Avg latency: 1.5e-02 seconds\n"
        m = self._call(stdout)
        assert m["avg_s"] == pytest.approx(0.015)

    def test_partial_output(self):
        """Only avg, no percentiles."""
        stdout = "Avg latency: 2.5 seconds\n"
        m = self._call(stdout)
        assert m == {"avg_s": 2.5}


# ---------------------------------------------------------------------------
# Pattern matching (fastpath evidence)
# ---------------------------------------------------------------------------

class TestCheckPatterns:
    """Tests for _check_patterns used in fast-path evidence checking."""

    def _call(self, text, require, forbid):
        from run_vllm_bench_latency_sweep import _check_patterns
        return _check_patterns(text, require, forbid)

    def test_all_required_present(self):
        result = self._call("kernel launched", ["kernel"], [])
        assert result["ok"] is True
        assert result["require_hits"] == ["kernel"]
        assert result["require_miss"] == []

    def test_required_missing(self):
        result = self._call("idle", ["kernel"], [])
        assert result["ok"] is False
        assert result["require_miss"] == ["kernel"]

    def test_forbidden_found(self):
        result = self._call("eager fallback", [], ["eager"])
        assert result["ok"] is False
        assert result["forbid_hits"] == ["eager"]

    def test_empty_patterns(self):
        result = self._call("anything", [], [])
        assert result["ok"] is True

    def test_regex_patterns(self):
        result = self._call("compiled graph v2.1", [r"graph v\d+\.\d+"], [])
        assert result["ok"] is True

    def test_combined_require_and_forbid(self):
        text = "CUDA graph captured, no eager fallback"
        result = self._call(text, ["CUDA graph"], ["eager"])
        assert result["ok"] is False  # forbid hit overrides require hit


# ---------------------------------------------------------------------------
# Format command for markdown
# ---------------------------------------------------------------------------

class TestFormatCmdForMd:
    """Tests for shell command formatting."""

    def _call(self, cmd, env):
        from run_vllm_bench_latency_sweep import _format_cmd_for_md
        return _format_cmd_for_md(cmd, env)

    def test_simple_command(self):
        result = self._call(["vllm", "bench", "latency"], {})
        assert result == "vllm bench latency"

    def test_with_env(self):
        result = self._call(["vllm", "bench"], {"FOO": "1"})
        assert result.startswith("FOO=")
        assert "vllm bench" in result

    def test_quotes_special_chars(self):
        result = self._call(["cmd", "arg with spaces"], {})
        assert "arg with spaces" in result or "'arg with spaces'" in result


# ---------------------------------------------------------------------------
# Bench exe tokens
# ---------------------------------------------------------------------------

class TestBenchExeTokens:
    """Tests for vllm command parsing."""

    def _call(self, vllm_cmd):
        from run_vllm_bench_latency_sweep import _bench_exe_tokens
        return _bench_exe_tokens(vllm_cmd)

    def test_string_command(self):
        assert self._call("vllm") == ["vllm"]

    def test_string_with_spaces(self):
        assert self._call("python -m vllm") == ["python", "-m", "vllm"]

    def test_list_command(self):
        assert self._call(["python", "-m", "vllm"]) == ["python", "-m", "vllm"]

    def test_invalid_type_raises(self):
        with pytest.raises(SystemExit, match="must be str or list"):
            self._call(42)

    def test_list_with_non_string_raises(self):
        with pytest.raises(SystemExit, match="must be list"):
            self._call(["vllm", 42])


# ---------------------------------------------------------------------------
# Prepare output root
# ---------------------------------------------------------------------------

class TestPrepareOutRoot:
    """Tests for output directory setup logic."""

    def _call(self, *, artifact_dir, out_name, overwrite):
        from run_vllm_bench_latency_sweep import _prepare_out_root
        return _prepare_out_root(
            artifact_dir=artifact_dir, out_name=out_name, overwrite=overwrite,
        )

    def test_creates_fresh_dir(self, tmp_path):
        out = self._call(artifact_dir=tmp_path, out_name="e2e", overwrite=False)
        assert out.exists()
        assert out == tmp_path / "e2e"

    def test_archives_existing_nonempty(self, tmp_path):
        """Existing non-empty dir is archived (renamed), not deleted."""
        existing = tmp_path / "e2e"
        existing.mkdir()
        (existing / "old_data.json").write_text("{}")

        out = self._call(artifact_dir=tmp_path, out_name="e2e", overwrite=False)
        assert out.exists()
        # Original should be gone (renamed).
        assert not (existing / "old_data.json").exists()
        # An archive directory should exist.
        archives = [p for p in tmp_path.iterdir() if p.name.startswith("e2e_")]
        assert len(archives) == 1

    def test_overwrite_deletes_existing(self, tmp_path):
        """With overwrite=True, existing dir is removed."""
        existing = tmp_path / "e2e"
        existing.mkdir()
        (existing / "old_data.json").write_text("{}")

        out = self._call(artifact_dir=tmp_path, out_name="e2e", overwrite=True)
        assert out.exists()
        # No archive created.
        archives = [p for p in tmp_path.iterdir() if p.name.startswith("e2e_")]
        assert len(archives) == 0
        # Old file should be gone.
        assert not (out / "old_data.json").exists()


# ---------------------------------------------------------------------------
# Maybe list str
# ---------------------------------------------------------------------------

class TestMaybeListStr:
    """Tests for _maybe_list_str helper."""

    def _call(self, obj, key):
        from run_vllm_bench_latency_sweep import _maybe_list_str
        return _maybe_list_str(obj, key)

    def test_present_list(self):
        assert self._call({"a": ["x", "y"]}, "a") == ["x", "y"]

    def test_missing_key(self):
        assert self._call({}, "a") == []

    def test_none_value(self):
        assert self._call({"a": None}, "a") == []

    def test_invalid_type_raises(self):
        with pytest.raises(SystemExit, match="Expected list"):
            self._call({"a": "not_a_list"}, "a")


# ---------------------------------------------------------------------------
# _ensure_worktree_pythonpath: prepend CWD so worktree vllm is found first
# ---------------------------------------------------------------------------

class TestEnsureWorktreePythonpath:
    """Verify _ensure_worktree_pythonpath correctly sets PYTHONPATH."""

    def _call(self, env, cwd=None):
        from run_vllm_bench_latency_sweep import _ensure_worktree_pythonpath
        if cwd is not None:
            with mock.patch("run_vllm_bench_latency_sweep.os.getcwd", return_value=cwd):
                return _ensure_worktree_pythonpath(env)
        return _ensure_worktree_pythonpath(env)

    def test_no_existing_pythonpath(self):
        """CWD is set as PYTHONPATH when none exists."""
        env = {"FOO": "bar"}
        result = self._call(env, cwd="/tmp/worktree")
        assert result["PYTHONPATH"] == "/tmp/worktree"
        assert result["FOO"] == "bar"

    def test_existing_pythonpath_is_preserved(self):
        """CWD is prepended to existing PYTHONPATH."""
        env = {"PYTHONPATH": "/some/other/path"}
        result = self._call(env, cwd="/tmp/worktree")
        parts = result["PYTHONPATH"].split(":")
        assert parts[0] == "/tmp/worktree"
        assert "/some/other/path" in parts

    def test_cwd_already_first_in_pythonpath(self):
        """No duplication when CWD is already first entry."""
        env = {"PYTHONPATH": "/tmp/worktree:/other"}
        result = self._call(env, cwd="/tmp/worktree")
        # Should not duplicate
        parts = result["PYTHONPATH"].split(":")
        assert parts.count("/tmp/worktree") == 1
        assert parts[0] == "/tmp/worktree"

    def test_does_not_mutate_input(self):
        """Input dict is not modified in place."""
        env = {"PYTHONPATH": "/old"}
        result = self._call(env, cwd="/tmp/worktree")
        assert env["PYTHONPATH"] == "/old"
        assert result["PYTHONPATH"].startswith("/tmp/worktree")

    def test_empty_pythonpath(self):
        """Empty string PYTHONPATH is treated as absent."""
        env = {"PYTHONPATH": ""}
        result = self._call(env, cwd="/tmp/worktree")
        assert result["PYTHONPATH"] == "/tmp/worktree"
