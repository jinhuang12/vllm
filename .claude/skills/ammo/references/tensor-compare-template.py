#!/usr/bin/env python3
"""TEMPLATE: Compare optimized module outputs against baseline for Gate 5.1b.

REFERENCE TEMPLATE — the impl-validator adapts this for the specific component.
The validator writes this script INDEPENDENTLY — not by reading the champion's
capture script. Both follow the same template. The state_dict + inputs + metadata
are the shared contract.

== PATTERN ==
1. Read baseline metadata.json
2. Init vLLM distributed state (same as capture)
3. Instantiate component from OPTIMIZED code
4. Load baseline state_dict with strict=False
5. Run forward with saved inputs, compare outputs
6. Write gate_5_1b_results.json
"""

import json
import os
import sys
from pathlib import Path

import torch

# ─── Dtype-scaled tolerances ────────────────────────────────────────────
# torch.allclose: |baseline - optimized| <= atol + rtol * |baseline|
# See references/validation-defaults.md
TOLERANCES = {
    "float32":       {"atol": 1e-5, "rtol": 1e-4},
    "float16":       {"atol": 1e-3, "rtol": 1e-2},
    "bfloat16":      {"atol": 1e-2, "rtol": 1e-1},
    "float8_e4m3fn": {"atol": 5e-1, "rtol": 5e-1},
}


# ─── CONFIG (ADAPT) ─────────────────────────────────────────────────────

BASELINE_DIR = Path("ADAPT_ME")  # {artifact_dir}/tracks/{op_id}/baseline_tensors
OUTPUT_DIR = Path("ADAPT_ME")    # {artifact_dir}/tracks/{op_id}/validator_tests/gate_5_1b
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

metadata = json.loads((BASELINE_DIR / "metadata.json").read_text())
DTYPE = metadata["dtype"]
dtype = getattr(torch, DTYPE)
device = torch.device("cuda")


# ─── Step 1: Init distributed (same boilerplate as capture) ─────────────

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

if torch.cuda.is_available():
    from vllm.platforms import cuda as _cuda_plat
    import vllm.platforms as _plat_mod
    if not getattr(_plat_mod._current_platform, "is_cuda", lambda: False)():
        _plat_mod.current_platform = _cuda_plat.CudaPlatform()

from vllm.distributed import init_distributed_environment, initialize_model_parallel
init_distributed_environment(world_size=1, rank=0, distributed_init_method="env://",
                             local_rank=0, backend="nccl")
initialize_model_parallel(tensor_model_parallel_size=1)


# ─── Step 2: Instantiate module from OPTIMIZED code ─────────────────────
#
# ADAPT: same constructor pattern as capture, but running on the optimized
# worktree. Read ctor_kwargs from metadata if needed.

from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config

model_config = ModelConfig(model=metadata["model_id"], dtype=DTYPE,
                           max_model_len=metadata["max_model_len"], enforce_eager=True)
vllm_config = VllmConfig(model_config=model_config,
                         device_config=DeviceConfig(device="cuda"))

with set_current_vllm_config(vllm_config):
    module = ...  # ADAPT: instantiate the component (same as capture)

module = module.to(device=device, dtype=dtype)
module.eval()


# ─── Step 3: Load baseline state_dict ────────────────────────────────────
#
# strict=False: matching keys get baseline values, missing keys (new params
# in optimized) keep defaults, unexpected keys (removed params) are skipped.

baseline_state = torch.load(BASELINE_DIR / metadata["state_dict_file"],
                            map_location="cpu", weights_only=True)
load_result = module.load_state_dict(baseline_state, strict=False)
print(f"Missing keys (new in optimized): {load_result.missing_keys}")
print(f"Unexpected keys (removed in optimized): {load_result.unexpected_keys}")

# ADAPT: post-init processing if needed (same as capture)


# ─── Step 4: Load inputs + forward ──────────────────────────────────────
#
# ADAPT: forward call and any required vLLM infrastructure (same as capture)

inputs = {}
for name, info in metadata["inputs"].items():
    inputs[name] = torch.load(BASELINE_DIR / info["file"],
                              map_location="cpu", weights_only=True).to(device=device, dtype=dtype)

input_tensors = [inputs[f"arg_{i}"] for i in range(len(inputs))]

with torch.no_grad():
    opt_output = module(*input_tensors)  # ADAPT: match forward signature

if isinstance(opt_output, (tuple, list)):
    opt_outputs = {f"output_{i}": o for i, o in enumerate(opt_output)
                   if isinstance(o, torch.Tensor)}
elif isinstance(opt_output, torch.Tensor):
    opt_outputs = {"output_0": opt_output}


# ─── Step 5: Compare + report ────────────────────────────────────────────

tol = TOLERANCES.get(DTYPE, TOLERANCES["bfloat16"])
results = {
    "gate": "5.1b",
    "module_class": metadata["module_class"],
    "dtype": DTYPE,
    "tolerance": tol,
    "missing_keys": list(load_result.missing_keys),
    "unexpected_keys": list(load_result.unexpected_keys),
    "per_output": {},
    "overall": "PENDING",
}

all_pass = True
for name, info in metadata["outputs"].items():
    baseline = torch.load(BASELINE_DIR / info["file"],
                          map_location="cpu", weights_only=True).to(device=device, dtype=dtype)
    optimized = opt_outputs.get(name)

    if optimized is None:
        results["per_output"][name] = {"status": "FAIL", "reason": "missing_output"}
        all_pass = False
        continue

    if baseline.shape != optimized.shape:
        results["per_output"][name] = {"status": "FAIL", "reason": "shape_mismatch",
                                        "baseline": list(baseline.shape),
                                        "optimized": list(optimized.shape)}
        all_pass = False
        continue

    if torch.isnan(optimized).any().item() or torch.isinf(optimized).any().item():
        results["per_output"][name] = {"status": "FAIL", "reason": "nan_or_inf"}
        all_pass = False
        continue

    close = torch.allclose(baseline, optimized, atol=tol["atol"], rtol=tol["rtol"])
    diff = (baseline.float() - optimized.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    status = "PASS" if close else "FAIL"
    if not close:
        all_pass = False

    results["per_output"][name] = {"status": status, "max_abs_diff": max_diff,
                                    "mean_abs_diff": mean_diff}
    print(f"{name}: {status} — max_abs_diff={max_diff:.6g}, mean={mean_diff:.6g}")

results["overall"] = "PASS" if all_pass else "FAIL"
(OUTPUT_DIR / "gate_5_1b_results.json").write_text(json.dumps(results, indent=2) + "\n")
print(f"\nGate 5.1b: {results['overall']}")

if not all_pass:
    sys.exit(1)
