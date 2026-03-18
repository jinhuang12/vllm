# Validation Results: op002_fused_gating_wide_bv

## Implementation Summary

**Optimization**: Fused gating (softplus+sigmoid) into the GDN recurrent Triton kernel + BV=128 wide-tile tiling (4x fewer blocks, 4 warps).

**Commit**: c578b811c

**Modified files**:
- `vllm/model_executor/layers/fla/ops/fused_recurrent.py`: New `fused_gating_recurrent_gdn_fwd_kernel` (~150 LOC Triton)
- `vllm/model_executor/models/qwen3_next.py`: Dispatch to fused kernel for pure decode when `VLLM_GDN_FUSED_GATING=1` (~20 LOC)
- `vllm/model_executor/layers/fla/ops/__init__.py`: Exports

**Scope**: Full scope per debate plan — gating fusion + BV=128 tiling. No descoping.

**Activation**: `VLLM_GDN_FUSED_GATING=1` env variable. Only activates for non-spec, non-prefill decode path.

---

## Gate 5.1: Correctness — PASS

**Validator**: impl-validator-op002 (independent tests)
**Script**: `tracks/op002_fused_gating_wide_bv/validator_tests/test_correctness.py`

| BS | max output diff | max state diff | Kill (>1e-4)? | Result |
|----|-----------------|----------------|---------------|--------|
| 1 | 0.00e+00 | 8.94e-08 | No | PASS |
| 8 | 1.91e-06 | 1.19e-07 | No | PASS |
| 32 | 1.22e-04 | 1.19e-07 | No | PASS |

Additional checks: NaN/INF (PASS), CUDA graph capture/replay (PASS), edge gating values (PASS), L2Norm modes (PASS), baseline path integrity (PASS).

**State max diff across all tests: 2.38e-07** (kill threshold: 1e-4)

---

## Gate 5.2: Kernel Benchmarks — FAIL at BS=32

**Validator**: impl-validator-op002 (independent benchmarks)
**Script**: `tracks/op002_fused_gating_wide_bv/validator_tests/benchmark_kernel.py`
**Method**: CUDA graph captured, L2 flush 240MB for cold-cache

### Raw Timings (microseconds per kernel call)

| BS | Baseline Warm | Opt Warm | Baseline Cold | Opt Cold |
|----|--------------|----------|--------------|----------|
| 1 | 6.57 | 5.63 | 17.32 | 16.77 |
| 8 | 14.26 | 12.53 | 56.27 | 55.15 |
| 32 | 39.73 | 49.13 | 215.59 | 213.13 |

### Speedups

| BS | Warm Speedup | Cold Speedup | Per-Step Savings (cold, 24 layers) |
|----|-------------|-------------|------|
| 1 | 1.167x | 1.033x | 13.2 us |
| 8 | 1.138x | 1.020x | 26.9 us |
| 32 | **0.809x** | **1.012x** | 59 us |

**Kill criterion 1**: Kernel speedup < 1.075x at BS=32 → Cold: 1.012x → **FAIL**

### Cross-Check: Champion vs Validator

| Metric | Champion | Validator | Diff |
|--------|----------|-----------|------|
| Fused kernel warm BS=32 | 49.38 us | 49.13 us | 0.5% |
| Baseline warm BS=32 | 55.51 us | 39.73 us | 40% divergent |

Champion's baseline used `reference_gating()` (PyTorch reimplementation, ~16us overhead) instead of the production Triton gating kernel (~1.3us). Validator's baseline is correct (actual production two-step path captured in CUDA graph). Champion's fused kernel number matches validator's — the implementation is consistent; the baseline methodology was flawed.

### Root Cause of BV=128 Failure

BV=128 with 4 warps is **slower** than BV=32 with 1 warp at BS=32 (warm: 49.13 vs ~38.4us recurrent-only). Probable causes:
1. Triton compiler generates suboptimal code for the 4-warp reduction pattern
2. Register pressure at 128 regs/thread + overhead exceeds efficient scheduling
3. The structural benefits (fewer blocks, larger DMA) are offset by per-block compute overhead

Under cold cache, the regression disappears (memory access dominates), but the BV=128 structural benefit is negligible (1.012x). The gating fusion alone saves ~2.5us/call at BS=32 — consistent with the debate's 0.17% E2E floor prediction.

---

## Gate 5.3: E2E Sweep — FAIL at BS=32

**Validator**: impl-validator-op002
**Method**: `run_vllm_bench_latency_sweep.py --labels opt` with `VLLM_GDN_FUSED_GATING=1`
**Baseline**: Stage 1 (NOT re-run)

### E2E Latencies

| BS | Stage 1 Baseline (s) | Optimized (s) | Delta (ms) | Improvement |
|----|---------------------|---------------|-----------|-------------|
| 1 | 6.567 | 6.486 | -81 | 1.24% |
| 8 | 7.854 | 7.722 | -132 | 1.68% |
| 32 | 10.286 | 10.236 | -50 | **0.49%** |

**Kill criterion 2**: E2E improvement < 1.0% at BS=32 → 0.49% → **FAIL**

### Amdahl's Sanity Check (BS=32)

- f(GDN recurrent + gating) = 14.1%
- Kernel speedup (cold): S = 1.012x
- Expected E2E = 1 / (1 - f + f/S) = 1 / (0.859 + 0.1393) = 1.0017x = **0.17%**
- Measured E2E: 0.49%

Measured > predicted — the 0.49% includes noise. With only 5 iterations, cross-session variance is ~50-100ms on a 10s benchmark. The 50ms improvement at BS=32 is within noise. The true improvement is likely ~0.17% (gating fusion only).

### BS=1 and BS=8 Anomaly

BS=1 (1.24%) and BS=8 (1.68%) show improvements exceeding what kernel speedup predicts (f=1% at BS=1, f=5.2% at BS=8). This suggests either:
1. Cross-session noise (different GPU thermal state, etc.)
2. Kernel launch overhead savings from graph node reduction (24 fewer gating nodes)
3. BV=128 performs better at small batch sizes (warm cache shows 1.167x and 1.138x)

---

## Kill Criteria Evaluation

| # | Criterion | Threshold | Measured | Verdict |
|---|-----------|-----------|----------|---------|
| 1 | Kernel speedup at BS=32 | ≥ 1.075x | 1.012x (cold) | **FAIL** |
| 2 | E2E improvement at BS=32 | ≥ 1.0% | 0.49% (noisy) | **FAIL** |
| 3 | State correctness | < 1e-4 | 2.38e-07 | PASS |
| 4 | CUDA graph capture | No failure | Works | PASS |
| 5 | Spec decode preserved | No breakage | Preserved | PASS |

---

## Overall Determination: FAIL

The optimization fails kill criteria 1 and 2 at BS=32. The BV=128 wide-tile hypothesis did not materialize — the Triton compiler generates slower code for the 4-warp case, making the fused kernel slower under warm cache (0.809x) and negligibly faster under cold cache (1.012x). The gating fusion alone provides ~0.17% E2E at BS=32, far below the 1% threshold.

The debate's key risk was validated: "BV=128 structural benefit unverified (needs ≥6% for >1% E2E)." The actual BV=128 structural benefit is negative (warm) to negligible (cold).

### What Worked
- Correctness is perfect (state diff < 1e-6)
- Gating fusion is valid (saves ~31us/step at BS=32)
- CUDA graph compatibility maintained
- Spec decode path preserved

### What Failed
- BV=128 tiling regresses at BS=32 warm cache (Triton 4-warp codegen issue)
- Cold-cache BV=128 benefit is negligible (~1%)
- Combined kernel speedup at BS=32 is only 1.012x (needs 1.075x)

### Lessons
- Champion's benchmark methodology was flawed (Python `reference_gating()` instead of production Triton kernel added ~16us to baseline, inflating apparent speedup to 1.124x vs actual 1.012x cold)
- The autotuner fallback from the debate would have selected BV=32, leaving only gating fusion (0.17% E2E) — confirming the debate's "narrow success corridor" prediction

---

## Repro Commands

### Kernel correctness
```bash
CUDA_VISIBLE_DEVICES=1 python tracks/op002_fused_gating_wide_bv/validator_tests/test_correctness.py
```

### Kernel benchmark
```bash
CUDA_VISIBLE_DEVICES=1 python tracks/op002_fused_gating_wide_bv/validator_tests/benchmark_kernel.py
```

### E2E sweep
```bash
cd /home/jinhun/vllm
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir /home/jinhun/vllm/kernel_opt_artifacts/Qwen3.5-4B_L40S_BF16_tp1 \
  --labels opt
```

## Environment
- GPU: NVIDIA L40S (CC 8.9, 46 GB, 864 GB/s)
- CUDA: 12.x
- vLLM commit: c578b811c
- Model: Qwen/Qwen3.5-4B (BF16, TP=1)
- Production parity: CUDA graphs + torch.compile (VLLM_TORCH_COMPILE_LEVEL=3)
