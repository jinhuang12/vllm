"""Unit tests for nsys_probe.py pure functions."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "nsys_probe",
    str(Path(__file__).resolve().parents[1] / "scripts" / "nsys_probe.py"),
)
nsys_probe = importlib.util.module_from_spec(_spec)
sys.modules["nsys_probe"] = nsys_probe
_spec.loader.exec_module(nsys_probe)


class TestOverheadModel:
    def test_zero_events(self):
        assert nsys_probe._estimate_time_min(0) == 0.0

    def test_low_events_green(self):
        t = nsys_probe._estimate_time_min(10_000)
        assert 0 < t < 5.0

    def test_30k_events_is_5_min(self):
        t = nsys_probe._estimate_time_min(30_000)
        assert abs(t - 5.0) < 0.1

    def test_200k_events_is_30_min(self):
        t = nsys_probe._estimate_time_min(200_000)
        assert abs(t - 30.0) < 0.1

    def test_interpolation(self):
        t = nsys_probe._estimate_time_min(115_000)
        assert 15.0 < t < 20.0

    def test_extrapolation_beyond_table(self):
        t = nsys_probe._estimate_time_min(5_000_000)
        assert t > 100.0


class TestRiskLevel:
    def test_green(self):
        assert nsys_probe._risk_level(3.0) == "GREEN"

    def test_yellow(self):
        assert nsys_probe._risk_level(10.0) == "YELLOW"

    def test_red(self):
        assert nsys_probe._risk_level(20.0) == "RED"

    def test_boundary_green_yellow(self):
        assert nsys_probe._risk_level(4.99) == "GREEN"
        assert nsys_probe._risk_level(5.0) == "YELLOW"

    def test_boundary_yellow_red(self):
        assert nsys_probe._risk_level(14.99) == "YELLOW"
        assert nsys_probe._risk_level(15.0) == "RED"


class TestComputeEstimates:
    def test_small_model_tp1(self):
        est = nsys_probe.compute_estimates(
            kernels_per_step=300, tp=1, batch_sizes=[1, 8, 32],
            num_layers=None, architecture=None,
        )
        assert est["suggested_sweep_args"]["--nsys-output-len"] == 32
        for bs_info in est["per_bucket"].values():
            assert bs_info["risk_level"] == "GREEN"

    def test_large_moe_tp8(self):
        est = nsys_probe.compute_estimates(
            kernels_per_step=1170, tp=8, batch_sizes=[1, 8, 32],
            num_layers=78, architecture="glm_moe_dsa",
        )
        assert est["suggested_sweep_args"]["--nsys-output-len"] == 2

    def test_total_sweep_calculated(self):
        est = nsys_probe.compute_estimates(
            kernels_per_step=500, tp=2, batch_sizes=[1, 8],
            num_layers=None, architecture=None,
        )
        assert est["total_sweep"]["num_buckets"] == 2
        per_bucket_events = list(est["per_bucket"].values())[0]["estimated_events"]
        assert est["total_sweep"]["total_events"] == per_bucket_events * 2

    def test_heuristic_crosscheck_uses_lower(self):
        est = nsys_probe.compute_estimates(
            kernels_per_step=5000, tp=1, batch_sizes=[1],
            num_layers=30, architecture="standard",
        )
        assert est["kernels_per_decode_step"] == 360

    def test_nsys_num_iters_always_1(self):
        est = nsys_probe.compute_estimates(
            kernels_per_step=300, tp=1, batch_sizes=[1],
            num_layers=None, architecture=None,
        )
        assert est["suggested_sweep_args"]["--nsys-num-iters"] == 1

    def test_total_sweep_time_is_additive(self):
        est = nsys_probe.compute_estimates(
            kernels_per_step=1170, tp=8, batch_sizes=[1, 8, 32],
            num_layers=78, architecture="glm_moe_dsa",
        )
        per_bs_time = list(est["per_bucket"].values())[0]["estimated_time_min"]
        expected_total = round(per_bs_time * 3, 1)
        assert abs(est["total_sweep"]["estimated_time_min"] - expected_total) < 0.5


class TestLoadTarget:
    def _make_target(self, tmp_path, **overrides):
        target_json = {
            "artifact_dir": str(tmp_path),
            "target": {"model_id": "test/model", "dtype": "fp8", "tp": 2, "ep": 1, "max_model_len": 4096},
            "workload": {"input_len": 64, "output_len": 512, "batch_sizes": [1, 8], "num_iters": 5},
            "bench": {
                "runner": "vllm_bench_latency", "vllm_cmd": "vllm",
                "extra_args": [], "baseline_extra_args": [], "opt_extra_args": [],
                "baseline_env": {}, "opt_env": {},
                "baseline_label": "baseline", "opt_label": "opt",
            },
        }
        for k, v in overrides.items():
            # Support dotted keys like "target.tp"
            parts = k.split(".")
            obj = target_json
            for p in parts[:-1]:
                obj = obj[p]
            obj[parts[-1]] = v
        (tmp_path / "target.json").write_text(json.dumps(target_json))
        return target_json

    def test_minimal_target(self, tmp_path):
        self._make_target(tmp_path)
        cfg = nsys_probe.load_target(tmp_path)
        assert cfg["model_id"] == "test/model"
        assert cfg["tp"] == 2
        assert cfg["batch_sizes"] == [1, 8]

    def test_missing_target_json(self, tmp_path):
        with pytest.raises(SystemExit):
            nsys_probe.load_target(tmp_path)

    def test_vllm_path_fallback(self, tmp_path):
        self._make_target(tmp_path)
        with patch("nsys_probe.shutil.which", return_value=None):
            cfg = nsys_probe.load_target(tmp_path)
        assert cfg["vllm_exe"][0] == sys.executable
        assert "vllm.entrypoints.cli.main" in cfg["vllm_exe"][-1]
