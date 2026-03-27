#!/usr/bin/env python3
"""TEMPLATE: Capture baseline tensors from a vLLM nn.Module for Gate 5.1b.

REFERENCE TEMPLATE — the ammo-delegate adapts this for the specific component.
Each component has different constructor args, forward signatures, and vLLM
infrastructure needs. The delegate writes a concrete script following this
pattern and saves it as capture_script.py alongside the captured tensors.

== CONTRACT ==
The capture script must produce these artifacts:
  {output_dir}/
    metadata.json       # Schema below — validator reads this to reconstruct
    state_dict.pt       # Module weights (random, deterministic)
    inputs/arg_0.pt     # Forward input tensor(s)
    outputs/output_0.pt # Baseline output tensor(s)
    capture_script.py   # Copy of this script for reproducibility

== PATTERN ==
1. Init vLLM distributed state (single-process)
2. Instantiate the target nn.Module (ADAPT constructor for the component)
3. Fill parameters with seeded random values, save state_dict
4. Create random inputs, run forward, save inputs + outputs
5. Write metadata.json

== metadata.json SCHEMA ==
{
  "model_id": "org/model-name",
  "module_class": "ClassName",
  "module_path": "model.layers.N.some_submodule",
  "import_path": "vllm.model_executor.models.foo.ClassName",
  "dtype": "bfloat16",
  "seed": 42,
  "batch_size": 1,
  "input_len": 64,
  "hidden_size": 4096,
  "max_model_len": 4096,
  "state_dict_file": "state_dict.pt",
  "state_dict_keys": ["key1", "key2"],
  "inputs": {"arg_0": {"shape": [...], "dtype": "...", "file": "inputs/arg_0.pt"}},
  "outputs": {"output_0": {"shape": [...], "dtype": "...", "file": "outputs/output_0.pt"}},
  "ctor_kwargs": {}  // Optional: constructor kwargs for validator to reconstruct
}
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# ─── CONFIG (ADAPT all of these) ────────────────────────────────────────

MODEL_ID = "org/model-name"      # From target.json
DTYPE = "bfloat16"               # From target.json
MAX_MODEL_LEN = 4096             # From target.json
SEED = 42
BATCH_SIZE = 1                   # Smallest target BS
INPUT_LEN = 64                   # From target.json
HIDDEN_SIZE = 4096               # From model's HF config
OUTPUT_DIR = Path("ADAPT_ME")    # {artifact_dir}/tracks/{op_id}/baseline_tensors
MODULE_CLASS = "ADAPT_ME"        # e.g., "Llama4MoE", "MixtralMoE"
MODULE_PATH = "ADAPT_ME"         # e.g., "model.layers.4.feed_forward"
IMPORT_PATH = "ADAPT_ME"         # e.g., "vllm.model_executor.models.llama4.Llama4MoE"


# ─── Step 1: Init vLLM distributed state ────────────────────────────────
#
# vLLM modules call distributed APIs during __init__. This boilerplate
# sets up single-process distributed for standalone module instantiation.

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

# Workaround: force CudaPlatform if torch.cuda works but pynvml fails
if torch.cuda.is_available():
    from vllm.platforms import cuda as _cuda_plat
    import vllm.platforms as _plat_mod
    if not getattr(_plat_mod._current_platform, "is_cuda", lambda: False)():
        _plat_mod.current_platform = _cuda_plat.CudaPlatform()

from vllm.distributed import init_distributed_environment, initialize_model_parallel
init_distributed_environment(world_size=1, rank=0, distributed_init_method="env://",
                             local_rank=0, backend="nccl")
initialize_model_parallel(tensor_model_parallel_size=1)


# ─── Step 2: Instantiate the module ─────────────────────────────────────
#
# ADAPT: each component has a different constructor. Read the class's
# __init__ signature and provide the right args. Common patterns:
#
#   # Pattern A: vllm_config-first (Llama4MoE, Qwen3MoE, etc.)
#   module = SomeModule(vllm_config, prefix="model.layers.0.mlp")
#
#   # Pattern B: explicit kwargs (MixtralMoE, older modules)
#   module = SomeModule(num_experts=8, hidden_size=4096, ...)
#
#   # Pattern C: HF config object (DeepseekV2MoE)
#   module = SomeModule(config=hf_config, prefix="...")
#
# Wrap instantiation in set_current_vllm_config context if needed.

from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config

model_config = ModelConfig(model=MODEL_ID, dtype=DTYPE, max_model_len=MAX_MODEL_LEN,
                           enforce_eager=True)
vllm_config = VllmConfig(model_config=model_config,
                         device_config=DeviceConfig(device="cuda"))

with set_current_vllm_config(vllm_config):
    module = ...  # ADAPT: instantiate your component here

device = torch.device("cuda")
dtype = getattr(torch, DTYPE)
module = module.to(device=device, dtype=dtype)
module.eval()


# ─── Step 3: Random weights + save state_dict ───────────────────────────

torch.manual_seed(SEED)
for p in module.parameters():
    p.data.normal_()

# ADAPT: some modules need post-init processing before forward works.
# Check if your module or its submodules need process_weights_after_loading
# or similar setup. Example:
#   for submod in module.modules():
#       if hasattr(submod, "quant_method"):
#           submod.quant_method.process_weights_after_loading(submod)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "inputs").mkdir(exist_ok=True)
(OUTPUT_DIR / "outputs").mkdir(exist_ok=True)
torch.save(module.state_dict(), OUTPUT_DIR / "state_dict.pt")


# ─── Step 4: Random inputs + forward + save ─────────────────────────────
#
# ADAPT: input shapes and forward call depend on the component.
#   MoE wrappers:    forward(hidden_states)           → [num_tokens, hidden_size]
#   Attention layers: forward(positions, hidden_states) → [num_tokens, hidden_size]
#   Decoder layers:  forward(positions, hidden_states, residual)
#
# ADAPT: some components need vLLM infrastructure (ForwardContext,
# WorkspaceManager, etc.) to run forward. Check what your component's
# forward path requires and set it up. The delegate should trace the
# forward call chain to determine this.

num_tokens = BATCH_SIZE * INPUT_LEN
torch.manual_seed(SEED + 1)
hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, device=device, dtype=dtype)
torch.save(hidden_states, OUTPUT_DIR / "inputs" / "arg_0.pt")

t0 = time.perf_counter()
with torch.no_grad():
    output = module(hidden_states)  # ADAPT: match the component's forward signature
t1 = time.perf_counter()

# Save outputs (handle tuple/list returns)
if isinstance(output, (tuple, list)):
    outputs = {f"output_{i}": o for i, o in enumerate(output) if isinstance(o, torch.Tensor)}
elif isinstance(output, torch.Tensor):
    outputs = {"output_0": output}
for name, tensor in outputs.items():
    torch.save(tensor.detach().cpu(), OUTPUT_DIR / "outputs" / f"{name}.pt")


# ─── Step 5: Write metadata ─────────────────────────────────────────────

metadata = {
    "model_id": MODEL_ID,
    "module_class": MODULE_CLASS,
    "module_path": MODULE_PATH,
    "import_path": IMPORT_PATH,
    "dtype": DTYPE,
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "input_len": INPUT_LEN,
    "hidden_size": HIDDEN_SIZE,
    "max_model_len": MAX_MODEL_LEN,
    "forward_time_ms": round((t1 - t0) * 1000, 1),
    "capture_timestamp": datetime.now(timezone.utc).isoformat(),
    "state_dict_file": "state_dict.pt",
    "state_dict_keys": list(module.state_dict().keys()),
    "inputs": {
        "arg_0": {"shape": list(hidden_states.shape), "dtype": str(hidden_states.dtype),
                  "file": "inputs/arg_0.pt"},
    },
    "outputs": {
        name: {"shape": list(t.shape), "dtype": str(t.dtype), "file": f"outputs/{name}.pt"}
        for name, t in outputs.items()
    },
    # ADAPT: include ctor_kwargs if the validator needs them to reconstruct
    "ctor_kwargs": {},
}
(OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"Baseline capture complete: {OUTPUT_DIR}")
