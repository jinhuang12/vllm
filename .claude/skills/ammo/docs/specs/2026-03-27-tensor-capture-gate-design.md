> **DEPRECATED (2026-04-01)**: This design was superseded by the Gate 5.1b Redesign.
> See `docs/superpowers/specs/2026-04-01-gate-5-1b-redesign-design.md` for the current design.

# Tensor Capture Gate Design (v3)

**Date**: 2026-03-27
**Status**: Draft
**Scope**: Gate 5.1b — component-level correctness validation via baseline tensor comparison

## Problem Statement

Gate 5.1 validates kernel correctness using synthetic test inputs. During a gpt-oss-120b campaign, an optimization passed Gate 5.1 despite skipping per-expert bias entirely. The validator's synthetic test used manually constructed weights without bias — both paths agreed on the wrong answer. Synthetic kernel tests catch math bugs but miss integration bugs where surrounding computation is altered.

## Design

Instantiate the higher-level component that wraps the target kernel with random weights, run a forward pass with random inputs, save the outputs. After implementation, instantiate the same component on the optimized codebase with the same weights and inputs. If the optimization is correct, outputs match. If something is missing (like bias), outputs diverge.

No model loading. No checkpoint downloading. No vLLM engine startup. Just direct `nn.Module` instantiation and `forward()` calls.

### Why This Catches The Bias Skip

The module's `__init__` creates its structure from the HF model config. If the config says `has_bias=True`, the module registers bias parameters — even with random weights. The baseline capture (on unmodified code) includes bias computation. The 5.1b check (on optimized code) does not — outputs diverge.

### Why This Is Different From Current 5.1a

Current 5.1a: validator **manually constructs** kernel inputs (and forgot bias).
New 5.1b: the **module itself** creates its parameters from the model config (including bias). The module knows its own structure — humans don't have to remember every parameter.

## Workflow

### Stage 4 Start: Baseline Capture

Before making any code changes, the impl-champion spawns a delegate:

```
Agent(
  subagent_type="ammo-delegate",
  run_in_background=True,
  prompt="""
  Capture baseline tensors for Gate 5.1b.

  Component: {module_class} (e.g., Llama4MoE)
  Module path in model: {module_path} (e.g., model.layers.4.feed_forward)
  Model config: {model_id} (load HF config only, not weights)
  Seed: 42, Batch size: {smallest_bs}, Input len: {input_len}

  Steps:
  1. Load HF config for {model_id} (no weights)
  2. Construct minimal VllmConfig from HF config + target.json params
  3. Initialize vLLM distributed state (see "Module Instantiation Setup" below)
  4. Instantiate the component: module = {module_class}(vllm_config, prefix=...)
  5. Initialize parameters: seed RNG, fill all params with normal_(), save state_dict
  6. Create random input: hidden_states = torch.randn(bs * input_len, hidden_size)
  7. Run module.forward(hidden_states) on GPU
  8. Save state_dict, inputs, outputs to {artifact_dir}/tracks/{op_id}/baseline_tensors/

  Save capture_script.py for reproducibility.
  """
)
```

### Module Instantiation Setup

vLLM modules call distributed state APIs (e.g., `get_tensor_model_parallel_world_size()`, `get_ep_group()`) and access the global VllmConfig during `__init__`. The capture script must initialize this state before instantiating the module:

```python
import torch.distributed as dist
from vllm.distributed import initialize_model_parallel
from vllm.config import set_current_vllm_config

# 1. Init process group (single-process, no GPU communication needed)
dist.init_process_group(backend="gloo", world_size=1, rank=0)
initialize_model_parallel(tensor_model_parallel_size=1)

# 2. Construct VllmConfig from HF config + target.json
vllm_config = make_minimal_vllm_config(model_id, dtype, max_model_len)

# 3. Instantiate module within config context
with set_current_vllm_config(vllm_config):
    module = Llama4MoE(vllm_config, prefix="model.layers.4.feed_forward")
    module = module.to(device).to(dtype)
```

This is boilerplate — the capture script (`scripts/tensor_capture.py`) handles it.

### Weight Initialization and Reuse

vLLM creates parameters via `torch.empty()` (uninitialized). The capture script fills them deterministically and **saves the state_dict** so the exact same weights can be loaded during comparison:

```python
# Baseline capture:
torch.manual_seed(42)
for p in module.parameters():
    p.data.normal_()
torch.save(module.state_dict(), baseline_dir / "state_dict.pt")

# Gate 5.1b comparison (on optimized codebase):
optimized_module.load_state_dict(
    torch.load(baseline_dir / "state_dict.pt"),
    strict=False  # handles added/removed parameters gracefully
)
```

Using `state_dict` save/load instead of re-seeding avoids RNG-shift issues: if the optimization adds or removes parameters, `strict=False` loads matching keys with identical values. Missing keys (e.g., removed bias) are left at their default, and extra keys are ignored. The output comparison catches any computation difference.

### Artifacts

```
{artifact_dir}/tracks/{op_id}/baseline_tensors/
  metadata.json       # module_class, module_path, model_id, seed, shapes, dtypes
  state_dict.pt       # baseline module weights (random, deterministic)
  inputs/arg_0.pt     # random hidden_states
  outputs/output_0.pt # baseline output
  capture_script.py   # reproducible script
```

### Gate 5.1b: Validator Comparison

After Gate 5.1a passes:

1. Read `baseline_tensors/metadata.json`
2. Instantiate the same component from the **optimized** codebase (same distributed setup)
3. Load `baseline_tensors/state_dict.pt` with `strict=False`
4. Feed saved `inputs/arg_0.pt` through `module.forward()`
5. Compare outputs with dtype-scaled tolerance
6. Check for NaN/Inf
7. Report pass/fail with max diff values

### Dtype-Scaled Tolerances

`torch.allclose` semantics: `|baseline - optimized| <= atol + rtol * |baseline|`

| Dtype | atol | rtol |
|-------|------|------|
| FP32  | 1e-5 | 1e-4 |
| FP16  | 1e-3 | 1e-2 |
| BF16  | 1e-2 | 1e-1 |
| FP8   | 5e-1 | 5e-1 |

Stored in `references/validation-defaults.md`, tunable per-campaign.

### Gate 5.1 Structure

```
Gate 5.1 = 5.1a (validator's synthetic kernel test) AND 5.1b (baseline tensor comparison)
```

## Integration Points

1. **`agents/ammo-impl-champion.md`**: Baseline tensor capture as first step (before code changes)
2. **`agents/ammo-impl-validator.md`**: Gate 5.1b workflow
3. **`references/validation-defaults.md`**: Gate 5.1b definition + tolerance table
4. **`orchestration/parallel-tracks.md`**: Updated Gate 5.1 description
5. **New template**: `references/tensor-capture-template.py` — delegate adapts per-component
6. **New template**: `references/tensor-compare-template.py` — validator adapts per-component

## Edge Cases

**Distributed state**: vLLM modules require initialized distributed process groups and VllmConfig context. The capture script handles this with `gloo` backend (no GPU communication) and `initialize_model_parallel(1)`. See "Module Instantiation Setup" above.

**No clear wrapper module**: If the model calls FusedMoE directly from the decoder layer, capture at the decoder layer. Champion decides the appropriate level.

**TP models**: Baseline capture runs with TP=1 regardless of production TP. Single-shard forward is sufficient to verify the kernel's computation. TP correctness is validated by Gate 5.3 (E2E).

**`__init__` side effects**: This gate validates numerical correctness of the forward pass only. Side effects of `__init__` (registration in `static_forward_context`, distributed state mutations) are not validated. These are covered by E2E gates.
