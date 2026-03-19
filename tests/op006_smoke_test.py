"""op006 smoke test: BV=8 vs BV=32 correctness comparison.

Runs the fused_recurrent_gated_delta_rule_fwd kernel with both BV=32 (baseline)
and BV=8 (optimized) and compares output and state for numerical equivalence.
"""
import os
import importlib
import torch
import pytest


def _run_kernel(use_bv8: bool, B: int, device: str = "cuda"):
    """Run the recurrent kernel with specified BV setting."""
    # Temporarily set/unset the env var and reimport
    if use_bv8:
        os.environ["VLLM_GDN_BV8_DECODE"] = "1"
    else:
        os.environ.pop("VLLM_GDN_BV8_DECODE", None)

    import vllm.model_executor.layers.fla.ops.fused_recurrent as mod
    importlib.reload(mod)
    fwd = mod.fused_recurrent_gated_delta_rule_fwd

    # Production shapes for Qwen3.5-4B decode
    H, HV, K, V = 16, 32, 128, 128
    T = B  # decode: 1 token per sequence, varlen packs B tokens
    N = B  # number of sequences

    torch.manual_seed(42)
    q = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, T, H, K, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, T, HV, V, device=device, dtype=torch.bfloat16)
    g = torch.randn(1, T, HV, device=device, dtype=torch.float32) * 0.1
    beta = torch.rand(1, T, HV, device=device, dtype=torch.bfloat16)

    max_batch = max(B, 64)
    initial_state = torch.randn(max_batch, HV, V, K, device=device, dtype=torch.float32) * 0.01
    state_copy = initial_state.clone()

    cu_seqlens = torch.arange(0, N + 1, device=device, dtype=torch.long)
    ssm_state_indices = torch.arange(0, N, device=device, dtype=torch.long)

    scale = K ** -0.5

    o, final_state = fwd(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=state_copy,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
    )

    return o.clone(), final_state.clone()


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_bv8_matches_bv32(batch_size):
    """BV=8 output and state must match BV=32 within tolerance."""
    o_baseline, state_baseline = _run_kernel(use_bv8=False, B=batch_size)
    o_opt, state_opt = _run_kernel(use_bv8=True, B=batch_size)

    # Output comparison (BF16 precision)
    o_diff = (o_baseline - o_opt).abs().max().item()
    print(f"BS={batch_size} output max diff: {o_diff:.2e}")
    assert o_diff < 1e-2, f"Output diff {o_diff} exceeds 1e-2 at BS={batch_size}"

    # State comparison (FP32 precision, kill criterion: 1e-4)
    state_diff = (state_baseline[:batch_size] - state_opt[:batch_size]).abs().max().item()
    print(f"BS={batch_size} state max diff: {state_diff:.2e}")
    assert state_diff < 1e-4, f"State diff {state_diff} exceeds 1e-4 at BS={batch_size}"


if __name__ == "__main__":
    for bs in [1, 8, 32]:
        print(f"\n=== BS={bs} ===")
        test_bv8_matches_bv32(bs)
    print("\nAll smoke tests PASSED")
