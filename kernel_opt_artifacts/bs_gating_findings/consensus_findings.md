# Consensus Findings: Batch-Size-Gated Optimization in AMMO

## 1. Verdict

**Replace the current "no regression" hard gates with a tiered gating workflow.** Both sides converged on a three-tier verdict system with a mechanism decision tree. The gating-first approach captures value that the current binary pass/fail discards, while noise tolerance prevents over-engineering for measurement-level effects. The mechanism is prescribed via decision tree (not a single pattern), addressing both the advocate's concern about agent discretion failure and the DA's concern about torch.compile compatibility.

**Recommendation**: Implement the tiered verdict + decision tree approach described below as a replacement for the current Stage 5.2/5.3 regression gates and a new Stage 6 sub-workflow for `GATED_PASS` tracks.

---

## 2. What Both Sides Agreed On

| Finding | Evidence |
|---------|----------|
| `GATED_PASS` as a first-class terminal track status | Both papers propose this independently. Currently no status between PASS and FAIL for partially-beneficial optimizations. |
| Gating must be mandatory (not an escape hatch) when mixed results detected | OP-001-relu2 was killed despite +2.53% at BS=1 because the agent had no structured path — both sides cite this as proof the current vague guidance fails. |
| Runtime dispatch overhead is negligible | Expert confirmed: ~20-50ns per call, 0ns under CUDA graphs. DA fully withdrew runtime overhead argument. |
| torch.compile requires mechanism-aware dispatch | Plain Python `if M <= 32` breaks fullgraph torch.compile. `torch.cond()` is needed for compiled paths. Both sides incorporated this into their decision trees. |
| Noise-level regressions should NOT trigger gating | Both incorporated a tolerance tier. Gating is expensive — don't trigger it for measurement noise. |
| The mechanism should vary by optimization context | Advocate expanded from single mechanism to 3-variant decision tree. DA added structure to their flexible approach. Both converged on a decision tree. |
| Crossover probing uses binary search when triggered | 3-5 iterations, 6-25 min. Both agree this is acceptable cost when gating is needed. |
| Dedicated resolver agent handles merge conflicts | Pre-decided by user, both sides incorporated. |

---

## 3. Key Dispute and Resolution

### Core Disagreement: How Prescriptive Should the Mechanism Decision Tree Be?

**Advocate's final position**: A 3-branch decision tree prescribing specific patterns:
1. Fullgraph-compiled path -> `torch.cond()`
2. Custom op / layer forward -> plain Python `if/else` on M
3. Module init -> function pointer selection

**DA's final position**: A more open-ended decision tree with guiding questions:
1. Is the path torch.compiled? -> `torch.cond()` or custom op
2. Is it model-specific? -> layer-level dispatch
3. Is the threshold architectural or empirical? -> hardcode vs env-var + runtime check

**Resolution**: These are functionally nearly identical. Both are decision trees; both prescribe mechanism based on compilation context; both cite the same codebase examples. The difference is presentation, not substance.

**Adopted approach**: Use the advocate's concrete 3-variant structure (clearer for agents to follow) but include the DA's "architectural vs empirical threshold" distinction as a sub-decision for variant 2 (the most common case). This gives agents a deterministic path while acknowledging that threshold type matters.

### Secondary Disagreement: Noise Tolerance Threshold

**Advocate**: 0.5% tolerance (regressions >= -0.5% trigger gating)
**DA**: 1.0% tolerance (regressions >= -1.0% trigger gating)

**Resolution**: Use **0.5%** as the default. Rationale:
- MEMORY.md reports 1-3% run-to-run variance, which argues for a larger tolerance
- However, E2E benchmarks in AMMO use 20 iterations with production parity, which reduces variance significantly
- The advocate's compounding argument is valid: 10 optimizations x -0.9% = -9% cumulative at one BS
- 0.5% is conservative but safe. Operators can override via `target.json` config.

### Tertiary Disagreement: Is Crossover Probing Always Required?

**Advocate**: Mandatory for all `GATE_REQUIRED` tracks
**DA**: Only for regressions > 1%; sub-1% absorbed by budget

**Resolution**: Crossover probing is **mandatory when gating is triggered**. With the 0.5% noise tolerance, gating only triggers for regressions > 0.5%, which are material enough to warrant precise thresholds. The cost (6-25 min) is small relative to the alternative (discarding the optimization entirely).

---

## 4. Risks and Mitigations

| Risk | Severity | Mitigation | Residual Risk |
|------|----------|------------|---------------|
| torch.compile graph breaks from Python `if` on tensor shape | High | Decision tree mandates `torch.cond()` for compiled paths | Low — requires agents to correctly identify compilation context |
| Threshold drift across CUDA/PyTorch versions | Medium | All gated optimizations ship behind env vars (disable path always exists). Thresholds documented in `evidence.json`. | Medium — no automated re-calibration mechanism exists |
| Nested conditionals from overlapping call sites | Low | Priority dispatch chain pattern for rare same-site overlap. AMMO debate process selects different components per round. | Low — 2 campaigns produced 0 overlapping call sites |
| Crossover probing noise | Medium | Conservative bias: use last known-beneficial BS as threshold. Binary search with 3-iter minimum per probe point. 30-min timeout. | Low — conservative bias means gating is slightly too aggressive, not too lenient |
| Cumulative regressions from noise tolerance | Low | 0.5% tolerance bounds worst-case to -0.5% per optimization. Track cumulative regression in campaign state. | Low — 10 optimizations x -0.5% = -5% worst case, which would be flagged in campaign evaluation |

---

## 5. Implementation Specification

### 5.1 Tiered Verdict System (Replaces Current Hard Gates)

Replace Stage 5.2 and 5.3 regression gates with:

```
Per Batch Size E2E Verdict:

speedup >= 1.0:             PASS (improvement)
speedup >= 0.995 (-0.5%):   NOISE (within noise, treated as neutral)
speedup >= 0.95 (-5%):      REGRESSED (material regression, gating required)
speedup < 0.95:             CATASTROPHIC (too large to gate — track FAILs)
```

Track-Level Verdict:
```
IF all BS are PASS or NOISE:
    track_status = PASS
ELIF any BS is CATASTROPHIC:
    track_status = FAIL
ELIF any BS is REGRESSED AND any BS is PASS:
    track_status = GATING_REQUIRED
    -> Implementer must gate -> GATED_PASS or FAIL
ELIF all BS are REGRESSED or NOISE (none PASS):
    track_status = FAIL (nothing to gate FOR)
```

### 5.2 Mechanism Decision Tree

When `GATING_REQUIRED` is triggered, the implementer follows this decision tree:

```
1. Is the dispatch site inside a fullgraph-compiled region?
   YES -> Use torch.cond(condition, optimized_fn, baseline_fn, operands)
          Both branches must return same-shape tensors.
          Reference: vllm/.../fp8_utils.py:308-315

2. Is the dispatch site inside a custom op, layer forward(), or CUDA-graphed path?
   YES -> Use plain Python if/else on M dimension.
          CUDA graph capture freezes the branch per batch-size bucket.
          Combine with env-var gating (two-level dispatch).
          Reference: vllm/.../triton_selective_gemm.py:330-355

          Sub-decision: Is the threshold architectural or empirical?
            ARCHITECTURAL (e.g., BLOCK_M determines max M) -> Hardcode threshold
            EMPIRICAL (determined by crossover probing) -> Use probed threshold
              with conservative bias (last known-beneficial BS)

3. Is the dispatch site at module init time or platform level?
   YES -> Use init-time function pointer selection (zero per-call cost).
          Reference: vllm/.../utils.py:298-308
```

Every gated optimization also gets:
- An env var in `vllm/envs.py`: `VLLM_{OP_NAME}=1` (opt-in enable)
- A fallback to baseline when disabled or outside beneficial BS range

### 5.3 Crossover Probing Protocol

**Note**: This binary search protocol has been superseded by kernel-informed crossover probing. See Amendment 1 below and `references/crossover-probing.md` for the operational protocol.

Triggered when track verdict is `GATING_REQUIRED`. Run by the existing impl-validator agent.

```
Input: beneficial_bs_set (PASS verdicts), regressed_bs_set (REGRESSED verdicts)
Let lo = max(beneficial_bs_set)
Let hi = min(regressed_bs_set)

While hi - lo > 1:
    mid = (lo + hi) // 2
    Run E2E benchmark at BS=mid (num_iters from target.json, production parity)
    If speedup >= 0.995:   # Within noise tolerance
        lo = mid
    Else:
        hi = mid

crossover_threshold = lo    # Conservative: last known-good BS
Time budget: 30 minutes max. If not converged, use lo from last iteration.
```

### 5.4 State Tracking

New fields in `parallel_tracks.{track_id}`:

```json
{
  "status": "GATED_PASS",
  "gating": {
    "mechanism": "python_if_else | torch_cond | init_time",
    "env_var": "VLLM_{OP_NAME}",
    "dispatch_condition": "M <= 16",
    "crossover_threshold_bs": 16,
    "crossover_probing": {
      "method": "binary_search",
      "probed_points": [
        {"bs": 16, "speedup": 1.012, "verdict": "beneficial"},
        {"bs": 24, "speedup": 0.994, "verdict": "regressed"}
      ],
      "converged": true,
      "time_minutes": 12.5
    },
    "pre_gating_results": {"bs1": 1.025, "bs8": 1.012, "bs32": 0.985},
    "post_gating_results": {"bs1": 1.025, "bs8": 1.012, "bs32": 1.000}
  }
}
```

New terminal status: `GATED_PASS` added to `verify_validation_gates.py`.

New optional `gating` section in `evidence.json` (required when `status == GATED_PASS`).

### 5.5 Pipeline Placement

**Stage 5 (worktree, implementer-owned)**:
1. Run E2E benchmarks at all campaign batch sizes
2. Apply tiered verdict per BS
3. If `GATING_REQUIRED`: run crossover probing
4. Implement gating using decision tree
5. Re-validate gated version at all BS
6. Record gating metadata in `evidence.json`
7. Mark track as `GATED_PASS`

**Stage 6 (main session, orchestrator-owned)**:
1. Cherry-pick `GATED_PASS` tracks from worktrees
2. If merge conflicts: spawn dedicated resolver agent + DA reviewer
3. Run combined E2E validation with all passing + gated tracks
4. Confirm no interaction effects between multiple gated optimizations

### 5.6 Conflict Resolver Agent (for merge conflicts)

When cherry-picking a `GATED_PASS` track produces merge conflicts:

1. Orchestrator spawns a **resolver agent** (Opus) with:
   - Conflicting files and both tracks' gating metadata
   - The gating decision tree for reference
   - Both tracks' dispatch conditions and env vars
2. Resolver proposes merged code preserving both gating dispatches
3. Orchestrator spawns a **DA reviewer** (Sonnet) to verify:
   - Correct dispatch ordering (more specific conditions first)
   - No accidental interaction between gating conditions
   - Env var namespace conflicts
   - torch.compile safety of the merged dispatch
4. If DA approves: merged version committed
5. If DA rejects: resolver revises or escalates to orchestrator

For overlapping call sites (rare): resolver uses priority dispatch chain pattern:
```python
AMMO_DISPATCH_CHAIN = [
    (lambda M: 2 <= M <= 16, fused_qkv_fn, "op012"),
    (lambda M: 2 <= M <= 32, selective_fn, "op007"),
]
# First match wins, evaluated in order
```

### 5.7 Noise Tolerance Configuration

Default: 0.5% (configurable in `target.json`):
```json
{
  "gating": {
    "noise_tolerance_pct": 0.5,
    "catastrophic_regression_pct": 5.0
  }
}
```

---

## 6. Areas Requiring Empirical Testing

| Question | Why Debate Can't Resolve It | Suggested Test |
|----------|---------------------------|----------------|
| Is 0.5% the right noise tolerance? | Depends on actual E2E variance with 20-iter production-parity benchmarks on target hardware | Run 10 identical E2E benchmarks, measure variance. If p95 variance > 0.5%, increase tolerance. |
| How often do optimizations produce mixed BS results? | No historical data — both campaigns had all-positive or all-failed tracks | Track over next 3 campaigns. If < 10% of tracks are mixed, gating rarely triggers and the workflow cost is negligible. |
| Do crossover thresholds drift across CUDA versions? | Depends on cuBLAS/Triton autotuner stability | Re-run crossover probing for shipped gated optimizations after major CUDA/PyTorch upgrade. |
| Does the priority dispatch chain pattern work under torch.compile? | Depends on torch.compile's handling of lambda chains and list iteration | Write a micro-benchmark: dispatch chain under `torch.compile(fullgraph=True)` and verify no graph breaks. |

---

## 7. Appendix: Debate Artifact Index

| Artifact | Path |
|----------|------|
| Advocate position paper | `kernel_opt_artifacts/bs_gating_findings/advocate_position.md` |
| DA position paper | `kernel_opt_artifacts/bs_gating_findings/da_position.md` |
| Expert research | `kernel_opt_artifacts/bs_gating_findings/expert_research.md` |
| Advocate rebuttal | `kernel_opt_artifacts/bs_gating_findings/advocate_rebuttal.md` |
| DA rebuttal | `kernel_opt_artifacts/bs_gating_findings/da_rebuttal.md` |
| Consensus (this document) | `kernel_opt_artifacts/bs_gating_findings/consensus_findings.md` |

---

## 8. Post-Debate Amendments

### Amendment 1: Kernel-Informed Crossover Probing (supersedes Section 5.3)

**Date**: 2026-03-19
**Rationale**: The original binary search protocol (Section 5.3) runs E2E benchmarks at every intermediate BS, costing 6-25 minutes. Post-debate analysis identified that kernel-level benchmarks (~seconds each) combined with E2E delta math prediction can find the crossover BS with only 1-2 E2E confirmation runs, reducing probing time by ~80%.

**Operational authority**: `references/crossover-probing.md` supersedes Section 5.3 for the probing protocol. Section 5.3 is preserved as historical record.

### Amendment 2: evidence.json Superseded

Section 5.4 references `evidence.json` as a gating metadata container. In the current agent architecture, gating metadata is recorded in `validation_results.md` (champion-authored) and `state.json:parallel_tracks.{op_id}.gating` (orchestrator-managed). There is no separate `evidence.json` file.
