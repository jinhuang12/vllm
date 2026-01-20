# Verification Commands

Run these commands BEFORE marking any phase complete.

**BLOCKING**: If any verification fails, do NOT proceed to the next phase.

---

## Phase 1 → Phase 2 Gate

### Automated Verification
```bash
python .claude/skills/moe-monokernel-optimizer/scripts/verify_phase1_baseline.py {artifact_dir}
# Exit code 0 = PASS, proceed to Phase 2
# Exit code 1 = BLOCKED, fix issues first
```

### Manual Verification (if script unavailable)
```bash
# 1. Verify nsys profile files exist
ls -la {artifact_dir}/runs/*.nsys-rep
# Expected: At least one .nsys-rep file per profiled batch size

# 2. Verify baseline timing data in constraints.md (not just commands)
grep -E "[0-9]+\.?[0-9]*\s*(µs|us|ms)" {artifact_dir}/constraints.md
# Expected: Lines with actual timing numbers

# 3. Verify vLLM kernel references (not naive PyTorch)
grep -iE "fused_moe|fused_experts|triton|CUTLASS" {artifact_dir}/constraints.md
# Expected: References to vLLM's production kernels

# 4. Verify production parity documented
grep -iE "cuda.*graph|torch.*compile|VLLM_TORCH_COMPILE" {artifact_dir}/constraints.md
# Expected: Production parity settings documented
```

---

## Phase 2 → Phase 3 Gate

No automated verification script. Manual checks:

```bash
# 1. Verify route decision is documented with rationale
grep -iE "Route.*[ABC]|Route decision" {artifact_dir}/optimization_plan.md
# Expected: Clear route selection with justification

# 2. Verify kill criteria are defined
grep -iE "kill.*criter|stop.*if" {artifact_dir}/optimization_plan.md
# Expected: 2-3 measurable kill criteria

# 3. Verify feasibility math uses actual baseline numbers
grep -E "[0-9]+\.?[0-9]*\s*(µs|us|ms)" {artifact_dir}/optimization_plan.md
# Expected: Calculations with real numbers from constraints.md
```

---

## Phase 3 → Phase 4 Gate

No automated verification script. Manual checks:

```bash
# 1. Verify CUDA files compile
cmake --build --preset release --target install 2>&1 | grep -c "error:"
# Expected: 0 errors

# 2. Verify no TODOs in GEMM kernels
grep -rE "TODO|FIXME" {cuda_dir}/*.cu | grep -v "^#" | wc -l
# Expected: 0 (or documented exceptions)

# 3. Verify MMA calls present (not placeholder)
grep -cE "mma|wmma|MMA" {cuda_dir}/*.cu
# Expected: > 0 for each kernel file with GEMM operations
```

---

## Phase 4 → Phase 5 Gate

### Automated Verification
```bash
python .claude/skills/moe-monokernel-optimizer/scripts/verify_phase4_gates.py {artifact_dir}
# Exit code 0 = PASS, proceed to Phase 5
# Exit code 1 = BLOCKED, do NOT declare SHIP
```

### Manual Verification (if script unavailable)
```bash
# 1. Verify vLLM baseline (not naive PyTorch)
grep -iE "fused_experts|fused_moe|from vllm.*import" {artifact_dir}/benchmark_*.py
# Expected: Imports from vllm.model_executor.layers.fused_moe

# 2. Verify NO naive PyTorch baseline
grep -E "for.*expert.*in.*range|for\s+e\s+in\s+range" {artifact_dir}/benchmark_*.py
# Expected: No matches (or only in comments)

# 3. Verify numerical comparison exists
grep -iE "torch\.allclose|assert_close|atol.*rtol" {artifact_dir}/benchmark_*.py
# Expected: At least one numerical comparison

# 4. Verify NO production disabling
grep -iE "TORCH_COMPILE_DISABLE|enforce_eager.*True" {artifact_dir}/benchmark_*.py
# Expected: No matches

# 5. Verify all kill criteria evaluated
grep -iE "TODO|optional|skip" {artifact_dir}/state.json | grep -i "kill_criteria"
# Expected: No matches (all criteria must have results)
```

---

## Phase 5 → Complete Gate

```bash
# 1. Verify envelope guard exists
grep -iE "envelope|guard|should_use_monokernel" vllm/model_executor/models/*.py
# Expected: Guard function in model file

# 2. Verify fallback mechanism
grep -iE "fallback|except|try:" vllm/model_executor/models/*.py | head -10
# Expected: try/except with fallback to baseline

# 3. Verify environment variable toggle
grep -iE "VLLM_USE_MOE_MONOKERNEL" vllm/envs.py
# Expected: Environment variable defined

# 4. Verify test file exists
ls tests/kernels/moe/test_*monokernel*.py
# Expected: At least one test file
```

---

## Quick Reference: Verification Script Exit Codes

| Script | Exit 0 | Exit 1 | Exit 2 |
|--------|--------|--------|--------|
| `verify_phase1_baseline.py` | All gates pass | Blocker found | Script error |
| `verify_phase4_gates.py` | All gates pass | Blocker found | Script error |

---

## When Verification Fails

1. **Read the blocker message** from script output
2. **Fix the issue** (don't ignore it)
3. **Re-run verification** until it passes
4. **Do NOT manually override** by marking phase complete

If you cannot fix the issue:
1. Update `state.json` with `"status": "blocked"`
2. Create blocker file using `orchestration/blocker-template.md`
3. Escalate per `SKILL.md` § Escalation Protocol
