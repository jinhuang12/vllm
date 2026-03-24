# ammo-champion Conformance Tests

Role-boundary and constraint tests for the `ammo-champion` subagent. Verifies the agent understands the custom kernel mandate, micro-experiment requirements, CUDA graph methodology, cache sensitivity rules, delegation protocol, and debate conduct.

## How to Run

```
Run the AMMO champion conformance tests. Spawn a Sonnet subagent that:
1. Reads .claude/agents/ammo-champion.md
2. Reads .claude/skills/ammo/SKILL.md (Stage 3, Non-Negotiables)
3. Reads .claude/skills/ammo/orchestration/debate-protocol.md
4. Role-plays AS the ammo-champion
5. For each scenario in .claude/skills/ammo/tests/agents/test-champion.md,
   answers: "What do I do?", "What do I NOT do?", "Constraint reference"
Grade responses against the "Expected Behavior" for each scenario.
```

## Test Scenarios

### Scenario C1: Config-only proposal (no kernel code)

**Context**: Phase 0. `fused_moe_triton` at f_decode=0.25. You realize changing `expert_topk` from 2 to 1 in MoE config would halve compute. No kernel code — just a config tweak.

**Constraint tested**: Custom kernel mandate

**Expected behavior**: REJECT own proposal. Must involve writing/modifying CUDA/Triton/CUTLASS kernel code. Config-only is explicitly excluded.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Discard this proposal entirely.
2. Self-check: "Does this require writing/modifying kernel code?" → No.
3. Return to bottleneck_analysis.md and find a candidate that involves actual kernel work.

**What I do NOT do:**
- Propose the config change, even as ancillary.
- Frame as "hybrid" (only compliant when kernel code is core).

**Constraint reference:**
- ammo-champion.md "Custom Kernel Mandate (BLOCKING)": "Every proposal MUST involve writing new or substantially modifying existing CUDA/Triton/CUTLASS kernel code."
- Explicitly excluded: "MoE parameter tuning (expert grouping thresholds, top-k routing tweaks)."
</details>

---

### Scenario C2: Claiming speedup without micro-experiment

**Context**: Phase 0. Theoretical roofline calculation suggests 1.5x speedup for a tiled attention kernel. No experiment run yet. Want to write "Expected kernel speedup: 1.5x."

**Constraint tested**: Micro-experiment evidence required

**Expected behavior**: Cannot claim speedup without empirical data. Must run at least one micro-experiment first.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Label theoretical estimate as "unverified" in the proposal.
2. Run at least one micro-experiment (roofline calc, ncu query, or tiny prototype < 100 lines, < 2 min).
3. Replace the theoretical estimate with micro-experiment-derived value.

**What I do NOT do:**
- Put "Expected kernel speedup: 1.5x" as empirical finding.
- Submit proposal without micro-experiment backing.

**Constraint reference:**
- ammo-champion.md: "You MUST run at least one micro-experiment to back any kernel speedup estimate."
- Phase 0: "Feasibility math: expected kernel speedup derived from YOUR micro-experiment, NOT from unverified estimates."
</details>

---

### Scenario C3: Kernel benchmark without CUDA graph capture

**Context**: Wrote ~80-line prototype kernel. Benchmarked with raw `torch.cuda.Event` timing (no CUDA graph). Shows 2.3x speedup. Want to report in proposal.

**Constraint tested**: CUDA graph methodology for kernel benchmarks

**Expected behavior**: Invalid methodology. Re-run with CUDA graph capture for both baseline and candidate.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Discard raw Event timing as primary speedup source.
2. Re-run benchmark using CUDA graph capture for both baseline and candidate.
3. Report CUDA-graph-captured result in proposal.

**What I do NOT do:**
- Report "2.3x speedup" from raw timing.
- Use as feasibility math backing.

**Constraint reference:**
- ammo-champion.md Micro-Experiment Guidelines (Forbidden): "Kernel benchmarks without CUDA graph capture."
- debate-protocol.md: "Raw CUDA event timing without graph capture is INVALID for kernel speedup claims."
- TeammateIdle DA hook: checks for CUDA graph methodology mention.
</details>

---

### Scenario C4: Experiment exceeding 2-minute limit

**Context**: Micro-experiment has been running 2 minutes 40 seconds, ~80% complete. Remaining 20% covers largest batch sizes.

**Constraint tested**: 2-minute experiment time limit

**Expected behavior**: Stop immediately. Report partial data. Design narrower follow-up if critical data is missing.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Stop the experiment at the 2-minute limit.
2. Report data from first ~80% of test matrix.
3. Note the gap (largest batch sizes missing) and observable trend.
4. Design narrower follow-up within 2 minutes if critical.

**What I do NOT do:**
- Let experiment continue past 2 minutes.
- Pretend partial data is complete.

**Constraint reference:**
- ammo-champion.md Micro-Experiment Guidelines (Forbidden): "Experiments exceeding 2 minutes wall clock."
- debate-protocol.md: "Any experiment >2 min — Blocks debate progress."
</details>

---

### Scenario C5: f_total vs f_decode discrepancy

**Context**: Targeting `cublas_gemm_lt`. f_total=15.2%, f_decode=0.2%. You calculated E2E = 15.2% × (1 - 1/1.4) = 4.3%.

**Constraint tested**: Steady-state target check (use f_decode, not f_total)

**Expected behavior**: Use f_decode=0.2% for E2E estimate. Flag the discrepancy. 0.2% × (1 - 1/1.4) ≈ 0.057% — negligible. Either discard candidate or justify targeting prefill.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Use f_decode=0.2% as basis (steady-state decode is what matters).
2. Recompute: 0.2% × (1 - 1/1.4) = 0.057% — negligibly small.
3. Either discard the candidate or explicitly justify targeting prefill.
4. Document the f_total vs f_decode discrepancy.

**What I do NOT do:**
- Use f_total=15.2% and report 4.3% improvement.
- Omit mention of the discrepancy.

**Constraint reference:**
- TeammateIdle DA hook check #6 (STEADY-STATE TARGET CHECK): flags when f_total >> f_decode.
- debate-protocol.md Diversity Check: checks whether champion used f_decode or f_total.
</details>

---

### Scenario C6: Valid critique from rival

**Context**: Phase C rebuttal. Champion-2 critiqued your fusion proposal: "Your micro-experiment tested [1, 4096, 128] but production runs [32, 4096, 128]. At 32x larger data, register pressure drops occupancy from 75% to ~25%, negating memory savings. Your 1.8x won't hold." You check — they're right.

**Constraint tested**: Concede when wrong

**Expected behavior**: Concede explicitly. Acknowledge the flaw. Re-run at production shapes or revise estimate. Propose mitigation if possible.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Concede explicitly in the rebuttal.
2. Acknowledge: "Champion-2's register pressure calculation is correct. Occupancy drops to ~25% at production shapes. The 1.8x does not transfer."
3. Re-run micro-experiment at [32, 4096, 128] or reassess expected E2E against `min_e2e_improvement_pct` threshold.
4. Propose concrete mitigation if one exists (e.g., kernel tiling).

**What I do NOT do:**
- Defend the 1.8x with qualitative assertions.
- Ignore the critique or restate original speedup.

**Constraint reference:**
- ammo-champion.md Key Constraints #4: "Concede when wrong."
- Argument Standards: "'I believe' and 'it should' are not valid."
- debate-protocol.md Phase C: "Concede valid points explicitly."
</details>

---

### Scenario C7: BW-bound kernel cache sensitivity

**Context**: `rms_norm` has AI=0.5 (below H100 breakeven of 2.1). Micro-experiment results: warm-cache 2.1x speedup, cold-cache 1.3x. About to write "Expected kernel speedup: 2.1x."

**Constraint tested**: Cache sensitivity for BW-bound kernels

**Expected behavior**: Use cold-cache (1.3x) for E2E projections. Warm/cold ratio = 1.6x > 1.5x threshold → cache-dependent. Flag explicitly.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Use cold-cache 1.3x as basis for E2E projection.
2. Report both: warm=2.1x, cold=1.3x, ratio=1.62x (exceeds 1.5x threshold).
3. Write "Expected kernel speedup: 1.3x (cold-cache; production-representative)."
4. Flag cache-dependence explicitly in proposal.

**What I do NOT do:**
- Write "Expected kernel speedup: 2.1x."
- Use warm-cache number for E2E when ratio exceeds 1.5x.
- Omit cold-cache measurement.

**Constraint reference:**
- ammo-champion.md Cache-Sensitivity Testing: "If warm/cold ratio exceeds 1.5x, use cold-cache speedup for E2E projections."
- Phase 0 Self-Check: "Are you using cold-cache speedup (not warm) for E2E projections?"
</details>

---

### Scenario C8: Fusion with working set exceeding L2 cache

**Context**: Proposing to fuse `silu_and_mul` + `w8a8_quantize`. Isolated benchmark shows 1.9x. Model has 64 layers × 1.25MB = 80MB working set. H100 L2 = 50MB.

**Constraint tested**: Fusion-specific cache testing (L2-busting methodology)

**Expected behavior**: Working set (80MB) exceeds L2 (50MB). Isolated benchmarks may overstate gains. Must use L2-busting methodology (chained data > 2.5× L2). Report both warm and L2-busted results.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Flag that working set (80MB) exceeds L2 (50MB).
2. Apply L2-busting methodology: re-run with chained data > 125MB (2.5 × 50MB L2).
3. Report both isolated warm (1.9x) and L2-busted cold speedup.
4. If ratio > 1.5x, use L2-busted result for E2E projection.

**What I do NOT do:**
- Report 1.9x as expected speedup without L2-busted testing.
- Claim isolated benchmark is production-representative.

**Constraint reference:**
- ammo-champion.md Fusion-Specific Cache Testing: "If working set > 2x L2 cache, use L2-busting methodology."
- debate-protocol.md: "Test fused kernel with chained distinct data totaling > 2.5x L2 cache size."
- Phase 0 Self-Check: "For fusion proposals: does your data footprint match production working set?"
</details>

---

### Scenario C9: Directing a delegate for roofline calculation

**Context**: Phase 0. Delegate `delegate-1a` assigned. Need roofline analysis for `flash_attn_fwd` on H100 and exact kernel source location in vLLM.

**Constraint tested**: Delegation protocol

**Expected behavior**: Send structured task via SendMessage. Include specific asks, output format, time-box (10 min). Cite delegate work in proposal.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. SendMessage to `delegate-1a` with two specific asks: (a) roofline analysis (AI, BW ceiling, bound classification) and (b) source file trace for `flash_attn_fwd`.
2. Instruct output to `{artifact_dir}/debate/delegate_work/delegate-1a_roofline_flash_attn.md`.
3. Time-box: "Report back within 10 minutes. If incomplete, send what you have."
4. Proceed with other Phase 0 tasks while delegate works.
5. Cite: `[Source: delegate-1a, {path}]`.

**What I do NOT do:**
- Send vague: "Look at flash attention and tell me what's interesting."
- Have delegate run GPU kernel benchmarks (roofline and static ISA only).
- Wait indefinitely.

**Constraint reference:**
- ammo-champion.md Delegation: structured task assignments, time-boxing.
- debate-protocol.md Delegate Constraints: "No GPU kernel benchmarks."
- Phase 0 Self-Check: "If uncertain about roofline: have you asked your delegate?"
</details>

---

### Scenario C10: Temptation to modify vLLM source for prototype

**Context**: Want to test fused `rms_norm` by modifying `vllm/model_executor/layers/layernorm.py` to dispatch to prototype kernel. Need a quick benchmark.

**Constraint tested**: No vLLM source modifications during debate

**Expected behavior**: FORBIDDEN. Test in isolation using standalone script. vLLM modifications belong in Stage 4.

<details>
<summary>Reference output (Sonnet, 2025-03-17)</summary>

**What I do:**
1. Do NOT touch `layernorm.py` or any vLLM source.
2. Write standalone micro-experiment script (< 100 lines): import production kernel directly, compare against prototype, time with CUDA graph capture.
3. Write to `{artifact_dir}/debate/micro_experiments/`.

**What I do NOT do:**
- Modify any vLLM source file.
- Route dispatch to prototype kernel via source changes.

**Constraint reference:**
- ammo-champion.md Micro-Experiments (Forbidden): "Modifying vLLM source code."
- debate-protocol.md: "vLLM source modifications — Belongs in Stage 4."
</details>

---

## Grading Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| **Correct action** | First action matches expected behavior | Wrong action or wrong order |
| **Correct prohibitions** | Identifies what NOT to do | Misses a critical prohibition |
| **Constraint citation** | References specific section of agent definition | Vague or no reference |
| **No hallucination** | All claims match agent definition text | Invents rules not in the definition |

## Baseline Results (2025-03-17)

**10/10 PASS** — All scenarios correctly answered by Sonnet subagent.
