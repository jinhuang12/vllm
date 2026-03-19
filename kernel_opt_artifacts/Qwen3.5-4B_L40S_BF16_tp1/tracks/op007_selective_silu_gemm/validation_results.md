# Validation Results: op007_selective_silu_gemm

## Implementation Summary

Fused SiLU GEMM kernel + Triton GEMM dispatch for all decode projections in Qwen3.5-4B.

- **Fused kernel**: `_fused_gate_up_silu_kernel` — Triton kernel that computes `SiLU(x @ W_gate.T) * (x @ W_up.T)` in one pass using paired tile addressing and two FP32 accumulators, eliminating the [M, 18432] intermediate write+read and separate SiLU kernel
- **Standard GEMM**: `_gemm_m32_kernel` — Triton GEMM with BLOCK_M=32 for all BF16 decode projections with M in [2, 32]
- **Dispatch**: `selective_triton_gemm()` in `dispatch_unquantized_gemm()` replaces cuBLAS for all BF16 shapes with M in [2, 32]
- **MLP integration**: `Qwen2MoeMLP.forward()` calls fused SiLU directly, bypassing gate_up_proj + act_fn
- **Env var**: `VLLM_TRITON_GEMM_SELECTIVE=1`
- **Commit**: f67d26e3c on worktree branch

### Design Evolution

The original plan was **selective dispatch** — only use Triton for 3 warm-winning shapes (gate_up, out_proj, qkv_proj) and keep cuBLAS for warm-regressing shapes (down_proj, in_proj_qkvz).

**This was abandoned** after E2E testing showed only 0.5% improvement at BS=32. Root cause: the cold-cache improvements on down_proj (~48 us/step) and in_proj_qkvz (~192 us/step) outweighed their warm regressions in production. SiLU fusion savings (~128 us/step) did not compensate.

The final implementation dispatches Triton GEMM to ALL shapes (like op004) plus fused SiLU, giving 1.4% E2E improvement — matching op004's result exactly.

## Environment

- GPU: NVIDIA L40S (SM89, 46GB, 864 GB/s HBM, 96 MB L2)
- vLLM commit: 54419bb16 (main) + op007 changes
- Model: Qwen/Qwen3.5-4B (BF16, TP=1)
- Production parity: CUDA graphs + torch.compile (VLLM_TORCH_COMPILE_LEVEL=3)

## Champion Gate 5.1: Correctness — PASS

### Fused SiLU GEMM (gate_up_proj)

Three-way comparison (M=32, K=2560, N_half=9216):

| Comparison | Max diff | Mean diff | Closer elements |
|---|---|---|---|
| Fused vs FP32 truth | 62.38 | 1.14 | 59.8% of elements |
| cuBLAS vs FP32 truth | 112.23 | 1.87 | 0.1% of elements |
| Fused vs cuBLAS | 128.00 | 1.53 | — |

The fused Triton kernel is **more accurate** than cuBLAS because it keeps gate and up accumulators in FP32 until the final BF16 store, avoiding the intermediate BF16 round-trip that cuBLAS requires.

Max diff of 128 = 1 BF16 ULP at output magnitude ~23680 (exponent 14, ULP = 2^7 = 128).

### Standard Triton GEMM

- out_proj (M=32, K=4096, N=2560): max_diff = 0.0 (exact match with cuBLAS)
- qkv_proj (M=32, K=2560, N=6144): max_diff = 0.0 (exact match with cuBLAS)

### CUDA Graph Capture

Both fused SiLU and standard GEMM kernels capture and replay correctly under CUDA graphs (verified using non-default stream capture).

## Champion Gate 5.2: Kernel Benchmarks

### BS=32 Warm Cache (CUDA graphs, no L2 flush)

| Shape | cuBLAS (us) | Triton/Fused (us) | Speedup | Calls/step | Saved/step (us) |
|---|---|---|---|---|---|
| gate_up (fused SiLU) | 73.0 | 35.8 | **2.039x** | 32 | 1190.4 |
| out_proj | 21.0 | 17.2 | **1.219x** | 32 | 121.6 |
| qkv_proj | 19.9 | 15.8 | **1.262x** | 8 | 32.8 |
| **Total warm savings** | | | | | **1344.8** |

### BS=32 Cold Cache (CUDA graphs, 240MB L2 flush)

| Shape | cuBLAS (us) | Triton/Fused (us) | Speedup | Calls/step | Saved/step (us) |
|---|---|---|---|---|---|
| gate_up (fused SiLU) | 515.8 | 491.2 | **1.050x** | 32 | 787.2 |
| out_proj | 377.4 | 375.7 | 1.005x | 32 | 54.4 |
| qkv_proj | 392.2 | 391.8 | 1.001x | 8 | 3.2 |
| **Total cold savings** | | | | | **844.8** |

### Warm Regression Check (all dispatched shapes)

| Shape | M=8 | M=16 | M=32 |
|---|---|---|---|
| gate_up (fused) | 1.125x | 1.317x | 2.039x |
| out_proj | fallback | 1.043x | 1.219x |
| qkv_proj | fallback | 1.002x | 1.262x |

No warm regression on any dispatched shape at any batch size.

## Champion Gate 5.3: E2E Sweep

**Baseline source: Stage 1 (NOT re-run)**

### E2E Results (VLLM_TRITON_GEMM_SELECTIVE=1, all shapes)

| BS | Baseline avg (s) | Opt avg (s) | Opt P50 (s) | Delta avg | Delta P50 |
|---|---|---|---|---|---|
| 1 | 6.567 | 6.493 | 6.494 | -74ms (1.1%) | -73ms (1.1%) |
| 8 | 7.854 | 7.740 | 7.737 | -114ms (1.5%) | -117ms (1.5%) |
| 32 | 10.286 | 10.141 | 10.140 | -145ms (1.4%) | -146ms (1.4%) |

### BS=32 Per-Iteration Detail

Per-iteration: [10.126, 10.140, 10.137, 10.163, 10.141]s — std dev = 13ms (0.13%), no outliers.

Compare to op004's [10.135, 10.139, 10.930, 11.715, 10.139]s which had 2 outliers.

### Selective vs All-Shapes Comparison

| Dispatch Strategy | BS=32 Avg (s) | Improvement |
|---|---|---|
| Selective (3 shapes + fused SiLU) | 10.234 | -52ms (0.5%) |
| All shapes + fused SiLU | 10.141 | -145ms (1.4%) |
| Op004 all shapes (no SiLU fusion) | 10.139 (P50) | -147ms (1.4%) |

The selective dispatch performs worse because excluding down_proj and in_proj_qkvz loses ~240 us/step of cold improvements that outweigh their warm regressions in production.

### Why SiLU Fusion Provides Negligible E2E Benefit

The gate_up intermediate tensor [32, 18432] = 1.18 MB fits entirely in L2 (96 MB). The SiLU kernel reads this from L2 (not HBM), so the fusion saves at most 1.37 us from eliminating the HBM write. At 32 calls/step, that's ~44 us/step — negligible vs the ~800 us/step total improvement from Triton GEMM.

### Amdahl's Law Analysis

| Factor | Value |
|---|---|
| f_GEMM (BS=32) | 79.6% |
| Cold kernel savings | 844.8 us/step |
| Amdahl's projection (512 steps) | 433 ms = 4.2% |
| Measured E2E (avg) | 145 ms = 1.4% |
| Translation factor | 0.33 |

Translation factor of 0.33 matches op004's 0.37 within measurement noise. The ~60% gap between projected and measured is consistent with production L2 cache behavior where warm weights partially offset cold-cache kernel improvements.

## Validator's Independent Results

### Validator Gate 5.1: Correctness — PASS (23/23)

- All 5 shapes × BS=[1,8,32]: PASS
- down_proj BS=32: max_diff=2.0 (1 BF16 ULP at scale ~260; Triton more accurate vs FP32)
  - Triton vs FP32: 1.003, cuBLAS vs FP32: 1.634
- CUDA graph capture+replay: PASS
- NaN/INF: PASS (15/15)

### Validator Gate 5.2: Kernel Benchmarks (BS=32, CUDA graphs, 240MB L2 flush)

| Shape | base_warm (us) | opt_warm (us) | Warm speedup | base_cold (us) | opt_cold (us) | Cold speedup |
|---|---|---|---|---|---|---|
| gate_up_proj | 67.3 | 35.3 | **1.906x** | 161.8 | 145.1 | **1.115x** |
| out_proj | 20.9 | 17.1 | **1.222x** | 39.2 | 38.4 | 1.021x |
| qkv_proj | 19.1 | 13.7 | **1.394x** | 54.2 | 53.8 | 1.007x |
| down_proj | 26.9 | 35.7 | **0.754x** ← | 78.6 | 76.3 | 1.030x |
| in_proj_qkvz | 21.0 | 22.7 | **0.925x** ← | 106.5 | 98.2 | 1.085x |
| Aggregate/step | 4337 | 3476 | **1.248x** | 11938 | 11105 | **1.075x** |

Fused SiLU (Part B):

| BS | unfused_warm (us) | fused_warm (us) | unfused_cold (us) | fused_cold (us) | Cold speedup |
|---|---|---|---|---|---|
| 32 | 72.1 | 36.6 | 166.3 | 147.8 | **1.125x** |

### Validator Gate 5.3: E2E Sweep

| BS | Baseline avg (s) | Opt avg (s) | Delta | Improvement |
|---|---|---|---|---|
| 1 | 6.567 | 6.495 | -72ms | 1.1% |
| 8 | 7.854 | 7.732 | -122ms | 1.6% |
| 32 | 10.286 | 10.136 | **-150ms** | **1.46%** |

Validator script paths:
- `tracks/op007_selective_silu_gemm/validator_tests/test_correctness.py`
- `tracks/op007_selective_silu_gemm/validator_tests/benchmark_kernel.py`
- `tracks/op007_selective_silu_gemm/e2e_latency_op007_validator/`

### Cross-Check: Champion vs Validator (BS=32)

| Metric | Champion (GPU 0) | Validator (GPU 0) | Delta |
|---|---|---|---|
| E2E avg | 10.141s | 10.136s | 5ms (0.05%) |
| E2E improvement | -145ms (1.41%) | -150ms (1.46%) | 0.05% |
| gate_up fused cold | 1.050x | 1.125x | 7.5% |
| Aggregate warm | — | 1.248x | — |
| Aggregate cold | — | 1.075x | — |

E2E measurements are consistent (5ms difference). The gate_up cold speedup discrepancy (1.050x vs 1.125x) is likely due to different baseline measurement methodology (champion measured cuBLAS GEMM+SiLU as a unit; validator measured cuBLAS GEMM separately plus SiLU).

## Kill Criteria Evaluation

| # | Criterion | Threshold | Champion | Validator | Verdict |
|---|---|---|---|---|---|
| 1 | Fused kernel cold speedup (BS=32) | >= 1.05x | 1.050x | 1.125x | **PASS** |
| 2 | E2E improvement BS=32 | >= 1.5% | 1.41% | 1.46% | **FAIL** |
| 3 | Correctness (fused vs cuBLAS) | <= 1 BF16 ULP | 1 ULP; more accurate | 1 ULP; more accurate | **PASS** |
| 4 | CUDA graph capture | No failure | PASS | PASS | **PASS** |
| 5 | No warm regression (dispatched shapes) | All >= 1.0x | — | down_proj 0.754x, in_proj 0.925x | **FAIL** |

## Overall Determination: FAIL

**Two kill criteria fail:**
- **Kill #2**: E2E improvement at BS=32 is 1.41-1.46%, below the 1.5% threshold (by 0.04-0.09%)
- **Kill #5**: down_proj (0.754x warm) and in_proj_qkvz (0.925x warm) regress in warm cache

### The Catch-22

The optimization faces an unsolvable dilemma:
- **Selective dispatch** (3 shapes, avoid warm regressions): Passes criterion #5 but fails criterion #2 (0.5% E2E)
- **All-shapes dispatch** (5 shapes): Fails criterion #5 (warm regressions) but gets closest on criterion #2 (1.46% E2E)

No dispatch configuration satisfies both criteria simultaneously.

### Root Cause Analysis

The op007 proposal hypothesized that:
1. Selective dispatch (excluding warm-regressing shapes) would increase the translation factor from op004's 0.37 to ~0.55
2. SiLU fusion would add ~183 us/step savings

Both hypotheses were falsified:
1. **Selective dispatch performs WORSE**: The cold-cache improvements on down_proj and in_proj_qkvz (240 us/step) outweigh their warm regressions in production. The kernel-to-E2E translation factor for GEMM optimizations is fundamentally ~0.33, regardless of shape selection.
2. **SiLU fusion provides negligible E2E benefit**: The intermediate tensor (1.18 MB) fits in L2 and is read from L2 by the SiLU kernel. Eliminating this L2 read saves <50 us/step — within noise.

### Implications for Future GEMM Optimizations

The Triton GEMM kernel for small-M decode is validated across 3 independent attempts (op004, op007 champion, op007 validator) to produce 1.4-1.5% E2E improvement at BS=32 on L40S. This appears to be the ceiling for GEMM-only optimizations at this model/hardware configuration:
- Kernel-level savings are ~833-845 us/step (cold)
- Translation factor is ~0.33-0.37
- Resulting E2E: 1.3-1.5%

To exceed 1.5% E2E, a fundamentally different approach is needed (e.g., reducing non-GEMM overhead, or targeting a different bottleneck).

## Repro Commands

```bash
# Kernel benchmark
cd /home/jinhun/vllm/.claude/worktrees/op007-selective-silu-gemm
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -c "import vllm.model_executor.layers.triton_selective_gemm; ..."

# E2E sweep (opt only, compare against Stage 1 baseline)
CUDA_VISIBLE_DEVICES=0 python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir kernel_opt_artifacts/Qwen3.5-4B_L40S_BF16_tp1 \
  --target-json kernel_opt_artifacts/Qwen3.5-4B_L40S_BF16_tp1/tracks/op007_selective_silu_gemm/target_op007.json \
  --labels opt \
  --out-name tracks/op007_selective_silu_gemm/e2e_sweep_v3

# Validator's independent tests
CUDA_VISIBLE_DEVICES=0 VLLM_TRITON_GEMM_SELECTIVE=1 python \
  kernel_opt_artifacts/Qwen3.5-4B_L40S_BF16_tp1/tracks/op007_selective_silu_gemm/validator_tests/test_correctness.py
CUDA_VISIBLE_DEVICES=0 VLLM_TRITON_GEMM_SELECTIVE=1 python \
  kernel_opt_artifacts/Qwen3.5-4B_L40S_BF16_tp1/tracks/op007_selective_silu_gemm/validator_tests/benchmark_kernel.py
```
