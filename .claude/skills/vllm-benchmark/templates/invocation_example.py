"""
Template for invocation_example() function

This template provides the standard structure for demonstrating kernel usage.
Customize the sections marked with {{PLACEHOLDERS}} based on kernel requirements.
"""

def invocation_example(small: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Demonstrate correct usage of the kernel.

    Args:
        small: If True, use minimal sizes (B=2, H=128) for quick validation
               If False, use realistic production sizes (B=4, H=4096)

    Returns:
        (output_tensor, metadata_dict)
    """
    device = torch.device("cuda")

    # ========================================================================
    # SETUP PHASE
    # ========================================================================

    if small:
        batch_size, seq_len, hidden_size = 2, 64, 128
    else:
        # Use dimensions from execution context or model config
        batch_size, seq_len, hidden_size = 4, 2048, 4096

    print(f"Creating inputs: B={batch_size}, S={seq_len}, H={hidden_size}")

    # Create primary input tensor
    input_tensor = torch.randn(
        batch_size, seq_len, hidden_size,
        dtype=torch.float16,
        device=device,
        requires_grad=False  # CRITICAL: No gradient tracking
    )

    # ========================================================================
    # METADATA GENERATION (kernel-specific)
    # ========================================================================

    # === For MoE Kernels ===
    # num_experts = 8
    # top_k = 2
    # topk_ids = torch.randint(0, num_experts, (batch_size, top_k), device=device)
    # topk_weights = torch.softmax(torch.randn(batch_size, top_k, device=device), dim=-1)
    # expert_ids = torch.arange(num_experts, device=device)

    # === For PagedAttention Kernels ===
    # num_blocks = 1024
    # block_size = 16
    # max_blocks_per_seq = seq_len // block_size + 1
    # block_tables = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq), device=device)
    # context_lens = torch.randint(seq_len // 2, seq_len, (batch_size,), device=device)
    # seq_lens = torch.full((batch_size,), seq_len, device=device)

    # === For Quantization Kernels ===
    # group_size = 128
    # num_groups = hidden_size // group_size
    # scales = torch.randn(batch_size, num_groups, device=device, dtype=torch.float32)
    # zero_points = torch.randint(0, 255, (batch_size, num_groups), device=device, dtype=torch.uint8)

    # === For Activation Kernels (SiLU, GELU, etc.) ===
    # No additional metadata needed

    # === For Matmul/GEMM Kernels ===
    # weight = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float16)
    # bias = torch.randn(hidden_size, device=device, dtype=torch.float16)

    # ========================================================================
    # KERNEL INVOCATION
    # ========================================================================

    print("Launching kernel...")

    output = {{KERNEL_WRAPPER_FUNCTION}}(
        input_tensor,
        # Add kernel-specific arguments here
        # For MoE: topk_ids=topk_ids, topk_weights=topk_weights, expert_ids=expert_ids
        # For attention: key=key, value=value, attention_mask=mask
        # For quantization: scales=scales, zero_points=zero_points
    )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    print(f"Output shape: {output.shape}, dtype: {output.dtype}")

    # Shape check
    expected_shape = (batch_size, seq_len, hidden_size)  # Adjust as needed
    assert output.shape == expected_shape, f"Shape mismatch: {output.shape}"

    # Dtype check
    assert output.dtype == torch.float16, f"Dtype mismatch: {output.dtype}"

    # NaN/Inf checks
    if torch.isnan(output).any():
        print("⚠ Warning: Output contains NaN values")
    if torch.isinf(output).any():
        print("⚠ Warning: Output contains Inf values")

    print("✓ Validation passed")

    # ========================================================================
    # METADATA
    # ========================================================================

    metadata = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "dtype": str(output.dtype),
        "device": str(output.device),
    }

    return output, metadata


# ============================================================================
# KERNEL-SPECIFIC INVOCATION PATTERNS
# ============================================================================

# === Pattern 1: Simple Activation Kernel ===
# def invocation_example(small: bool = False):
#     input_tensor = torch.randn(B, S, H, device="cuda", dtype=torch.float16)
#     output = silu_and_mul_kernel(input_tensor)
#     return output, {"shape": output.shape}

# === Pattern 2: MoE Kernel ===
# def invocation_example(small: bool = False):
#     input_tensor = torch.randn(B, S, H, device="cuda")
#     topk_ids = torch.randint(0, num_experts, (B, top_k), device="cuda")
#     topk_weights = torch.softmax(torch.randn(B, top_k, device="cuda"), dim=-1)
#     output = fused_moe_kernel(input_tensor, topk_ids, topk_weights, expert_weights)
#     return output, {"num_experts": num_experts, "top_k": top_k}

# === Pattern 3: Attention Kernel ===
# def invocation_example(small: bool = False):
#     query = torch.randn(B, num_heads, S, head_dim, device="cuda", dtype=torch.float16)
#     key = torch.randn(B, num_heads, S, head_dim, device="cuda", dtype=torch.float16)
#     value = torch.randn(B, num_heads, S, head_dim, device="cuda", dtype=torch.float16)
#     output = attention_kernel(query, key, value)
#     return output, {"num_heads": num_heads, "head_dim": head_dim}

# === Pattern 4: Quantization Kernel ===
# def invocation_example(small: bool = False):
#     input_tensor = torch.randn(B, S, H, device="cuda", dtype=torch.float16)
#     scales = torch.randn(B, num_groups, device="cuda", dtype=torch.float32)
#     zero_points = torch.randint(0, 255, (B, num_groups), device="cuda", dtype=torch.uint8)
#     output = quantize_kernel(input_tensor, scales, zero_points)
#     return output, {"group_size": group_size}

# === Pattern 5: Paged Attention Kernel ===
# def invocation_example(small: bool = False):
#     query = torch.randn(B, num_heads, 1, head_dim, device="cuda")  # Decoding
#     key_cache = torch.randn(num_blocks, num_heads, block_size, head_dim, device="cuda")
#     value_cache = torch.randn(num_blocks, num_heads, block_size, head_dim, device="cuda")
#     block_tables = torch.randint(0, num_blocks, (B, max_blocks), device="cuda")
#     context_lens = torch.randint(1, S, (B,), device="cuda")
#     output = paged_attention_kernel(query, key_cache, value_cache, block_tables, context_lens)
#     return output, {"block_size": block_size, "num_blocks": num_blocks}
