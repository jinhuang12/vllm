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


class TestProbeRecommendation:
    """Tests for tiered profiling recommendation logic.

    The probe output should include a 'recommendation' field that maps
    risk_level to a tiered profiling strategy:
      - GREEN  -> tier0_nsys_node (nsys --cuda-graph-trace=node is safe)
      - YELLOW -> tier0_nsys_node (proceed with caution, safe_nsys_OL reduced)
      - RED    -> tier1_torch_primary (use torch.profiler as primary)
      - RED + TP>1 or SM100 -> tier1_plus_tier2 (torch.profiler + nsys graph enrichment)
    """

    def test_probe_recommendation_tier0_green(self):
        """When probe risk is GREEN, recommendation should be tier0_nsys_node."""
        est = nsys_probe.compute_estimates(
            kernels_per_step=300, tp=1, batch_sizes=[1, 8, 32],
            num_layers=None, architecture=None,
        )
        # GREEN risk -> nsys node mode is safe
        for bs_info in est["per_bucket"].values():
            assert bs_info["risk_level"] == "GREEN"
        recommendation = _derive_recommendation(est)
        assert recommendation == "tier0_nsys_node"

    def test_probe_recommendation_tier1_red(self):
        """When any bucket is RED, recommendation should be tier1_torch_primary."""
        # Use high kernel count without heuristic crosscheck (num_layers=None)
        # to ensure RED risk. 20000 kernels * TP=8 = 160k effective, safe_ol=1,
        # events=160k -> ~24 min per bucket -> RED.
        est = nsys_probe.compute_estimates(
            kernels_per_step=20000, tp=8, batch_sizes=[1, 8, 32],
            num_layers=None, architecture=None,
        )
        has_red = any(
            info["risk_level"] == "RED" for info in est["per_bucket"].values()
        ) or est["total_sweep"]["risk_level"] == "RED"
        assert has_red, "Expected RED risk for high kernel count TP=8"
        recommendation = _derive_recommendation(est)
        assert recommendation in ("tier1_torch_primary", "tier1_plus_tier2")

    def test_probe_recommendation_tier1_plus_tier2(self):
        """When probe is RED on SM100 GPU with TP>1, recommend tier1_plus_tier2."""
        # Use high kernel count without heuristic to ensure RED at total_sweep
        # level. 20000 kernels * TP=4 = 80k effective -> total_sweep RED.
        est = nsys_probe.compute_estimates(
            kernels_per_step=20000, tp=4, batch_sizes=[1, 8, 32],
            num_layers=None, architecture=None,
        )
        target_ctx = {"tensor_parallel_size": 4, "gpu_arch": "sm_100"}
        recommendation = _derive_recommendation(est, target_ctx)
        assert recommendation == "tier1_plus_tier2"

    def test_probe_recommendation_tier0_yellow(self):
        """YELLOW risk still recommends tier0 -- nsys node mode is feasible."""
        est = nsys_probe.compute_estimates(
            kernels_per_step=600, tp=2, batch_sizes=[1],
            num_layers=None, architecture=None,
        )
        recommendation = _derive_recommendation(est)
        # YELLOW means proceed with caution but nsys node is still feasible
        assert recommendation == "tier0_nsys_node"


class TestProbeFailureRecommendation:
    """Test tier recommendations when the probe itself fails (exit code != 0).

    This exercises compute_tier_recommendation directly, which is the code path
    taken when nsys probe times out or crashes.
    """

    def test_probe_failure_tp1_non_sm100(self):
        """Probe failure at TP=1 on non-SM100 -> tier1_torch_primary."""
        rec = nsys_probe.compute_tier_recommendation(
            estimates=None, tp=1, gpu_arch="sm_89", probe_exit_code=3,
        )
        assert rec["recommendation"] == "tier1_torch_primary"

    def test_probe_failure_sm100_tp1(self):
        """Probe failure on SM100 at TP=1 -> tier1_plus_tier2 (SM100 enrichment)."""
        rec = nsys_probe.compute_tier_recommendation(
            estimates=None, tp=1, gpu_arch="sm_100", probe_exit_code=3,
        )
        assert rec["recommendation"] == "tier1_plus_tier2"

    def test_probe_failure_tp4_non_sm100(self):
        """Probe failure at TP>1 on non-SM100 -> tier1_plus_tier2 (multi-rank)."""
        rec = nsys_probe.compute_tier_recommendation(
            estimates=None, tp=4, gpu_arch="sm_89", probe_exit_code=3,
        )
        assert rec["recommendation"] == "tier1_plus_tier2"

    def test_probe_failure_sm100_tp4(self):
        """Probe failure on SM100 at TP>1 -> tier1_plus_tier2."""
        rec = nsys_probe.compute_tier_recommendation(
            estimates=None, tp=4, gpu_arch="sm_100", probe_exit_code=3,
        )
        assert rec["recommendation"] == "tier1_plus_tier2"

    def test_probe_failure_exit_code_2(self):
        """Prewarm failure (exit code 2) also produces recommendation."""
        rec = nsys_probe.compute_tier_recommendation(
            estimates=None, tp=1, gpu_arch=None, probe_exit_code=2,
        )
        assert rec["recommendation"] == "tier1_torch_primary"


def _derive_recommendation(
    estimates: dict, target_ctx: dict | None = None
) -> str:
    """Derive tiered profiling recommendation from probe estimates.

    This helper encodes the tiered strategy:
      - If any per-bucket risk or total sweep risk is RED -> Tier 1+
      - If target has TP>1 OR SM100 arch -> upgrade to tier1_plus_tier2
      - Otherwise -> tier0_nsys_node

    TODO: Call nsys_probe.compute_tier_recommendation() once that function
    is implemented in nsys_probe.py. This local helper is a stopgap.
    """
    has_red = any(
        info["risk_level"] == "RED"
        for info in estimates["per_bucket"].values()
    ) or estimates["total_sweep"]["risk_level"] == "RED"

    if not has_red:
        return "tier0_nsys_node"

    # RED risk: check if target context warrants Tier 2 enrichment
    if target_ctx:
        tp = target_ctx.get("tensor_parallel_size", 1)
        arch = target_ctx.get("gpu_arch", "")
        # Tier 2 enrichment recommended when SM100 OR TP>1
        if tp > 1 or "sm_100" in arch:
            return "tier1_plus_tier2"

    return "tier1_torch_primary"


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
