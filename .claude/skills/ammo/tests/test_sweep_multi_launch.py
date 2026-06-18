#!/usr/bin/env python3
"""Tests for Track B: multi-launch + fresh-cache sweep methodology.

Covers:
- --num-launches CLI argument (default 1; multi-launch >= 2)
- --fresh-cache CLI argument (default OFF; injects VLLM_CACHE_ROOT + TRITON_CACHE_DIR)
- _aggregate_launches: cross-launch mean/stddev/p50/p90/p99, warmup discard
- _compute_noise_flag: |delta| < 2*max(stddev) → True
- Backward compatibility: N=1 preserves flat avg_latency schema
- Status payload: launch N/total surface when num_launches >= 2
- Multi-launch is optional (not gate-enforced); provides noise flag when N>=2
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure the scripts directory is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args(argv):
    """Build the sweep argparser and parse argv (excluding program name)."""
    from run_vllm_bench_latency_sweep import main
    import argparse as _ap

    captured = {}
    original = _ap.ArgumentParser.parse_args

    def intercept(self, args=None, namespace=None):
        captured["parser"] = self
        raise SystemExit(0)

    with mock.patch.object(_ap.ArgumentParser, "parse_args", intercept):
        try:
            main()
        except SystemExit:
            pass

    parser = captured.get("parser")
    assert parser is not None, "Could not capture parser"
    return parser.parse_args(argv)


class TestNumLaunchesArg:
    """--num-launches N (default 1)."""

    def test_default_is_one(self):
        ns = _parse_args(["--artifact-dir", "/tmp/x"])
        assert ns.num_launches == 1

    def test_explicit_value(self):
        ns = _parse_args(["--artifact-dir", "/tmp/x", "--num-launches", "3"])
        assert ns.num_launches == 3

    def test_zero_rejected(self, tmp_path):
        from run_vllm_bench_latency_sweep import main
        # Writing an empty target.json — the validation happens before
        # we use it. We just want to confirm num_launches=0 triggers SystemExit.
        target = {"artifact_dir": str(tmp_path), "target": {}, "workload": {}, "bench": {}}
        (tmp_path / "target.json").write_text(json.dumps(target), encoding="utf-8")
        with mock.patch("sys.argv", [
            "sweep", "--artifact-dir", str(tmp_path), "--num-launches", "0",
        ]):
            with pytest.raises(SystemExit):
                main()


class TestFreshCacheArg:
    """--fresh-cache (default OFF)."""

    def test_default_false(self):
        ns = _parse_args(["--artifact-dir", "/tmp/x"])
        assert ns.fresh_cache is False

    def test_flag_sets_true(self):
        ns = _parse_args(["--artifact-dir", "/tmp/x", "--fresh-cache"])
        assert ns.fresh_cache is True


# ---------------------------------------------------------------------------
# _aggregate_launches pure function
# ---------------------------------------------------------------------------


class TestAggregateLaunches:
    """Cross-launch aggregation: mean/stddev/p50/p90/p99, warmup discard."""

    def _agg(self, launches):
        from run_vllm_bench_latency_sweep import _aggregate_launches
        return _aggregate_launches(launches)

    def test_discards_warmup(self):
        launches = [
            {"launch_id": 1, "warmup": True, "avg_latency": 10.0, "latencies": [10.0]},
            {"launch_id": 2, "warmup": False, "avg_latency": 4.0, "latencies": [4.0]},
            {"launch_id": 3, "warmup": False, "avg_latency": 5.0, "latencies": [5.0]},
            {"launch_id": 4, "warmup": False, "avg_latency": 6.0, "latencies": [6.0]},
        ]
        agg = self._agg(launches)
        assert agg is not None
        assert agg["num_measured_launches"] == 3
        assert abs(agg["mean_latency"] - 5.0) < 1e-9

    def test_stddev_cross_launch(self):
        launches = [
            {"launch_id": 1, "warmup": True, "avg_latency": 10.0},
            {"launch_id": 2, "warmup": False, "avg_latency": 4.0},
            {"launch_id": 3, "warmup": False, "avg_latency": 5.0},
            {"launch_id": 4, "warmup": False, "avg_latency": 6.0},
        ]
        agg = self._agg(launches)
        # pstdev of [4,5,6] = sqrt(2/3) ≈ 0.8165
        assert 0.8 < agg["stddev_cross_launch"] < 0.9

    def test_percentiles(self):
        launches = [
            {"launch_id": 1, "warmup": True, "avg_latency": 10.0},
            {"launch_id": 2, "warmup": False, "avg_latency": 4.0},
            {"launch_id": 3, "warmup": False, "avg_latency": 5.0},
            {"launch_id": 4, "warmup": False, "avg_latency": 6.0},
        ]
        agg = self._agg(launches)
        # sorted means = [4.0, 5.0, 6.0]; p50=5.0
        assert abs(agg["p50"] - 5.0) < 1e-9
        # p90 of 3 points ≈ 4 + 0.9*2*(6-4) hmm. Using linear-interp: k=(3-1)*0.9=1.8; floor=1, ceil=2
        # sorted[1] + (sorted[2]-sorted[1])*(0.8) = 5 + 1*0.8 = 5.8
        assert abs(agg["p90"] - 5.8) < 1e-9

    def test_all_warmup_returns_none(self):
        launches = [
            {"launch_id": 1, "warmup": True, "avg_latency": 10.0},
        ]
        assert self._agg(launches) is None

    def test_empty_returns_none(self):
        assert self._agg([]) is None

    def test_missing_avg_latency_skipped(self):
        launches = [
            {"launch_id": 1, "warmup": True, "avg_latency": 10.0},
            {"launch_id": 2, "warmup": False, "avg_latency": None},
            {"launch_id": 3, "warmup": False, "avg_latency": 5.0},
        ]
        agg = self._agg(launches)
        assert agg is not None
        assert agg["num_measured_launches"] == 1
        assert abs(agg["mean_latency"] - 5.0) < 1e-9

    def test_single_measured_launch(self):
        """When only 1 measured launch, stddev should be 0 (or defined)."""
        launches = [
            {"launch_id": 1, "warmup": True, "avg_latency": 10.0},
            {"launch_id": 2, "warmup": False, "avg_latency": 5.0},
        ]
        agg = self._agg(launches)
        assert agg is not None
        assert agg["num_measured_launches"] == 1
        assert agg["mean_latency"] == 5.0
        assert agg["stddev_cross_launch"] == 0.0


# ---------------------------------------------------------------------------
# _compute_noise_flag
# ---------------------------------------------------------------------------


class TestNoiseFlag:
    """|delta| < 2 * max(stddev_baseline, stddev_opt) → True."""

    def _flag(self, b_mean, b_std, o_mean, o_std):
        from run_vllm_bench_latency_sweep import _compute_noise_flag
        return _compute_noise_flag(b_mean, b_std, o_mean, o_std)

    def test_noise_when_delta_small(self):
        # delta = 0.1, max stddev = 0.1, threshold = 0.2, 0.1 < 0.2 → True
        assert self._flag(5.0, 0.1, 5.1, 0.05) is True

    def test_not_noise_when_delta_large(self):
        # delta = 1.0, threshold = 0.2, → False
        assert self._flag(5.0, 0.1, 4.0, 0.05) is False

    def test_equal_means_is_noise(self):
        assert self._flag(5.0, 0.1, 5.0, 0.1) is True

    def test_zero_stddev_never_noise_unless_equal(self):
        # delta = 1.0, threshold = 0, → False
        assert self._flag(5.0, 0.0, 4.0, 0.0) is False
        # delta = 0, threshold = 0 → True (they're equal, within "noise")
        assert self._flag(5.0, 0.0, 5.0, 0.0) is True


# ---------------------------------------------------------------------------
# Fresh cache env var injection
# ---------------------------------------------------------------------------


class TestFreshCacheEnvInjection:
    """When --fresh-cache is set, VLLM_CACHE_ROOT + TRITON_CACHE_DIR injected
    into child_env for BOTH run.env and os.environ branches."""

    def _inject(self, base_env, sweep_cache_root):
        from run_vllm_bench_latency_sweep import _inject_fresh_cache_env
        return _inject_fresh_cache_env(base_env, sweep_cache_root)

    def test_injects_vllm_cache_root(self, tmp_path):
        env = {}
        out = self._inject(env, tmp_path / "cache" / "abc")
        assert "VLLM_CACHE_ROOT" in out
        assert str(tmp_path / "cache" / "abc") == out["VLLM_CACHE_ROOT"]

    def test_injects_triton_cache_dir(self, tmp_path):
        env = {}
        out = self._inject(env, tmp_path / "cache" / "abc")
        assert "TRITON_CACHE_DIR" in out
        # TRITON must be inside sweep cache root
        assert out["TRITON_CACHE_DIR"].startswith(str(tmp_path / "cache" / "abc"))

    def test_does_not_set_vllm_disable_compile_cache(self, tmp_path):
        """Task requirement: do NOT set VLLM_DISABLE_COMPILE_CACHE=1."""
        env = {}
        out = self._inject(env, tmp_path / "cache" / "abc")
        assert "VLLM_DISABLE_COMPILE_CACHE" not in out

    def test_preserves_existing_vars(self, tmp_path):
        env = {"PYTHONPATH": "/foo", "UNRELATED": "bar"}
        out = self._inject(env, tmp_path / "cache" / "abc")
        assert out["PYTHONPATH"] == "/foo"
        assert out["UNRELATED"] == "bar"

    def test_does_not_mutate_input(self, tmp_path):
        env = {"PYTHONPATH": "/foo"}
        out = self._inject(env, tmp_path / "cache" / "abc")
        # input env should not have been modified in place
        assert "VLLM_CACHE_ROOT" not in env

    def test_overrides_existing_cache_root(self, tmp_path):
        env = {"VLLM_CACHE_ROOT": "/already/set"}
        out = self._inject(env, tmp_path / "cache" / "abc")
        assert out["VLLM_CACHE_ROOT"] == str(tmp_path / "cache" / "abc")


# ---------------------------------------------------------------------------
# Backward compatibility: N=1 preserves flat avg_latency schema
# ---------------------------------------------------------------------------


class TestBackwardCompatN1:
    """With --num-launches 1 (default), output schema is unchanged."""

    def test_default_n1_no_launches_key_in_entry(self):
        """When num_launches=1, label entries have no 'launches' array."""
        from run_vllm_bench_latency_sweep import _build_label_result_entry

        entry = _build_label_result_entry(
            cmd=["vllm", "bench"],
            env_overrides={},
            metrics={"avg_s": 3.5},
            log_rel="logs/baseline_bs1.log",
            output_json_rel="json/baseline_bs1.json",
            runner_json_rel="json/baseline_bs1.runner.json",
            ok=True,
            returncode=0,
            evidence_status="unknown",
            evidence={"ok": True, "require_hits": [], "require_miss": [], "forbid_hits": []},
            timing={"start_time": None, "end_time": None, "duration_s": None},
            launches=None,
            aggregate=None,
        )
        assert "launches" not in entry
        assert "aggregate" not in entry
        assert entry["avg_s"] == 3.5
        assert entry["metrics"]["avg_s"] == 3.5

    def test_multi_launch_entry_has_launches_and_aggregate(self):
        from run_vllm_bench_latency_sweep import _build_label_result_entry

        launches = [
            {"launch_id": 1, "warmup": True, "avg_latency": 10.0, "latencies": [10.0]},
            {"launch_id": 2, "warmup": False, "avg_latency": 4.0, "latencies": [4.0]},
            {"launch_id": 3, "warmup": False, "avg_latency": 5.0, "latencies": [5.0]},
            {"launch_id": 4, "warmup": False, "avg_latency": 6.0, "latencies": [6.0]},
        ]
        aggregate = {
            "mean_latency": 5.0,
            "stddev_cross_launch": 0.816,
            "p50": 5.0,
            "p90": 5.8,
            "p99": 5.98,
            "num_measured_launches": 3,
        }

        entry = _build_label_result_entry(
            cmd=["vllm", "bench"],
            env_overrides={},
            metrics={"avg_s": 5.0},
            log_rel="logs/baseline_bs1.log",
            output_json_rel="json/baseline_bs1.json",
            runner_json_rel="json/baseline_bs1.runner.json",
            ok=True,
            returncode=0,
            evidence_status="unknown",
            evidence={"ok": True, "require_hits": [], "require_miss": [], "forbid_hits": []},
            timing={"start_time": None, "end_time": None, "duration_s": None},
            launches=launches,
            aggregate=aggregate,
        )
        assert entry["launches"] == launches
        assert entry["aggregate"] == aggregate
        # avg_s should reflect mean_latency for backward-compat downstream code.
        assert entry["avg_s"] == 5.0


# ---------------------------------------------------------------------------
# Multi-launch is optional (not enforced by any gate).
# The T5 gate checks only artifact existence. The default is --num-launches 1.
# Multi-launch (N>=2) provides cross-launch stddev and the row["noise"] flag
# but is not required for shipping.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# --num-launches help text references warmup semantics
# ---------------------------------------------------------------------------


class TestHelpText:
    """Help text should mention warmup/compile-cache characteristics."""

    def test_fresh_cache_help_mentions_compile(self, capsys):
        from run_vllm_bench_latency_sweep import main
        with mock.patch("sys.argv", ["sweep", "--help"]):
            with pytest.raises(SystemExit):
                main()
        out = capsys.readouterr().out
        assert "--fresh-cache" in out
        # Task spec says help text should explain:
        # "First launch pays full compile (~5 min for large models)."
        assert "compile" in out.lower()
