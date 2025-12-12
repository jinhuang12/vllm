# Success Criteria

This document defines the success criteria for each stage of the vllm-benchmark workflow. Use this as a checklist to verify completion before proceeding to the next stage.

## Pre-Stage: Input Validation

**Before Stage 0 begins, verify**:
- [ ] SYMBOL_NAME validated (alphanumeric + underscores only)
- [ ] No path traversal characters (`..`, `/`) in SYMBOL_NAME
- [ ] Symbol exists in vLLM codebase (verified via Grep)
- [ ] HARDWARE is one of: H200, A100, H100
- [ ] PRECISION is one of: FP8, BF16, FP16
- [ ] MODEL is provided (or default "qwen2.5" used)
- [ ] SKIP_REMOTE flag parsed correctly
- [ ] Success message printed with all parsed values

**Failure Handling**: If any check fails, STOP and report error to user.

---

## Stage 0: Initialization

**Success Criteria**:
- [ ] Directory `bundled_benchmarks/` created (if not exists)
- [ ] Directory `artifacts/` created (if not exists)
- [ ] Old bundle files removed:
  - `bundled_benchmarks/{{SYMBOL_NAME}}.py`
  - `bundled_benchmarks/{{SYMBOL_NAME}}_bench.py`
- [ ] Old plan removed: `/tmp/plan_{{SYMBOL_NAME}}.md`
- [ ] TodoWrite initialized with 4 stages:
  - Stage 1: Planning
  - Stage 2.1: Kernel Extraction
  - Stage 2.2: Benchmark Generation
  - Stage 2.3: Remote Verification (or marked as skipped)

**Verification Commands**:
```bash
ls -la bundled_benchmarks/
ls -la artifacts/
test ! -f bundled_benchmarks/{{SYMBOL_NAME}}.py && echo "Bundle removed ✓"
test ! -f /tmp/plan_{{SYMBOL_NAME}}.md && echo "Plan removed ✓"
```

---

## Stage 1: Planning Phase

**Success Criteria**:
- [ ] Plan file created at `/tmp/plan_{{SYMBOL_NAME}}.md`
- [ ] Plan contains ALL required sections (no "TODO" or "Unknown"):
  - **Kernel Analysis**: Location, type, entry point, wrapper function
  - **Dependency Map**: Concrete source file paths listed
  - **Input/Output Contract**: Exact tensor shapes specified
  - **Kernel Configuration**: Compile-time constants documented
  - **Benchmark Strategy**: Realistic dimension ranges for {{MODEL}}/{{HARDWARE}}
  - **Reference Implementation Strategy**: PyTorch equivalent or algorithm
  - **Gotchas & Edge Cases**: Known issues documented
- [ ] Dependency Map either:
  - Lists specific dependencies with file paths, OR
  - Explicitly states "No dependencies" if kernel is self-contained
- [ ] Plan summary printed to console
- [ ] Plan reviewed for completeness (no missing sections)
- [ ] TodoWrite updated: Stage 1 marked as completed

**Verification Commands**:
```bash
test -f /tmp/plan_{{SYMBOL_NAME}}.md && echo "Plan exists ✓"
grep -q "Kernel Analysis" /tmp/plan_{{SYMBOL_NAME}}.md && echo "Kernel Analysis section ✓"
grep -q "Dependency Map" /tmp/plan_{{SYMBOL_NAME}}.md && echo "Dependency Map section ✓"
! grep -q "TODO\|Unknown" /tmp/plan_{{SYMBOL_NAME}}.md && echo "No TODOs ✓"
```

**Failure Handling**: If plan is incomplete, iterate with the /plan agent until all sections are populated.

---

## Stage 2.1: Kernel Extraction & Bundle Creation

**Success Criteria**:
- [ ] Bundle file created: `bundled_benchmarks/{{SYMBOL_NAME}}.py`
- [ ] File size > 1 KB (not empty)
- [ ] Contains required sections:
  - **Module docstring** explaining kernel purpose
  - **Imports section** (only torch, triton, stdlib)
  - **Inlined Dependencies section** with source attribution
  - **Kernel Implementation** (Triton @jit or CUDA wrapper)
  - **Wrapper Function** that launches kernel
  - **`invocation_example()` function** with proper signature
  - **`if __name__ == "__main__":` block**
- [ ] NO vLLM imports anywhere:
  - No `import vllm`
  - No `from vllm.` anywhere in file
- [ ] All dependencies inlined with attribution comments like:
  - `# Inlined from vllm/utils/foo.py:123`
- [ ] `invocation_example()` function complete with:
  - Setup phase (tensor creation)
  - Metadata generation (kernel-specific)
  - Kernel invocation
  - Validation (NaN/Inf checks)
  - Return tuple: `(output, metadata)`
- [ ] File runs successfully: `python bundled_benchmarks/{{SYMBOL_NAME}}.py`
- [ ] Prints "✓ Bundle execution successful!" or similar
- [ ] No Python exceptions during execution
- [ ] TodoWrite updated: Stage 2.1 marked as completed

**Verification Commands**:
```bash
test -f bundled_benchmarks/{{SYMBOL_NAME}}.py && echo "Bundle exists ✓"
! grep -q "^import vllm\|^from vllm" bundled_benchmarks/{{SYMBOL_NAME}}.py && echo "No vLLM imports ✓"
grep -q "def invocation_example" bundled_benchmarks/{{SYMBOL_NAME}}.py && echo "invocation_example() ✓"
python bundled_benchmarks/{{SYMBOL_NAME}}.py 2>&1 | grep -q "successful\|✓" && echo "Execution test ✓"
```

**Failure Handling**:
- If vLLM imports found: Identify and inline missing dependencies
- If execution fails: Debug error, check dependencies, verify tensor shapes
- If missing sections: Add them following the template structure

---

## Stage 2.2: Benchmark Suite Generation

**Success Criteria**:
- [ ] Benchmark file created: `bundled_benchmarks/{{SYMBOL_NAME}}_bench.py`
- [ ] File size > 3 KB (substantial content)
- [ ] Contains required components:
  - **`BenchmarkContext` dataclass** with proper fields
  - **`BenchmarkTensors` dataclass** with `make()` and `as_kernel_kwargs()` methods
  - **`reference_implementation()` function** with actual implementation
  - **`test_correctness()` function** with validation logic
  - **`benchmark_configuration()` function** with CUDA event timing
  - **`generate_benchmark_contexts()` function** returning List[BenchmarkContext]
  - **`main()` function** with argparse CLI
- [ ] `reference_implementation()` is NOT just `raise NotImplementedError`:
  - Must have actual PyTorch implementation
  - Can use `torch.nn.functional` operations
  - Or explicit loops if no PyTorch equivalent
- [ ] `generate_benchmark_contexts()` returns > 0 contexts:
  - "quick" mode: ≥ 2 contexts
  - "full" mode: ≥ 8 contexts
- [ ] CLI supports required flags:
  - `--quick`: Quick validation run
  - `--validate`: Correctness checking
  - `--sweep [quick|full|custom]`: Sweep type
  - `--output`: JSON output file path
  - `--num-iters`: Timing iterations
  - `--rtol`, `--atol`: Tolerance values
- [ ] Running `python {{SYMBOL_NAME}}_bench.py --quick --validate` completes without errors
- [ ] JSON output file created with correct structure:
  - `kernel`, `model`, `hardware`, `precision` fields
  - `results` list with per-config data
  - `num_successful` count
- [ ] TodoWrite updated: Stage 2.2 marked as completed

**Verification Commands**:
```bash
test -f bundled_benchmarks/{{SYMBOL_NAME}}_bench.py && echo "Benchmark exists ✓"
grep -q "class BenchmarkContext" bundled_benchmarks/{{SYMBOL_NAME}}_bench.py && echo "BenchmarkContext ✓"
grep -q "def reference_implementation" bundled_benchmarks/{{SYMBOL_NAME}}_bench.py && echo "reference_implementation ✓"
! grep -q "raise NotImplementedError" bundled_benchmarks/{{SYMBOL_NAME}}_bench.py && echo "reference implemented ✓"
python bundled_benchmarks/{{SYMBOL_NAME}}_bench.py --quick --validate 2>&1 | grep -q "✓\|success" && echo "Quick test ✓"
test -f benchmark_results.json && echo "JSON output ✓"
```

**Failure Handling**:
- If reference not implemented: Add PyTorch equivalent for kernel operation
- If contexts empty: Check `generate_benchmark_contexts()` logic
- If execution fails: Debug with `--quick` first, check tensor generation
- If JSON missing: Verify `main()` function writes output file

---

## Stage 2.3: Remote Verification (Conditional)

**Pre-check**:
- [ ] If `SKIP_REMOTE == true`: Skip this stage entirely, print "⏭ Skipping Stage 2.3 (--skip-remote flag set)"
- [ ] If `SKIP_REMOTE == false`: Proceed with remote deployment

**Success Criteria**:
- [ ] Files successfully uploaded to p5e-cmh:
  - `~/bundled_benchmarks/{{SYMBOL_NAME}}.py`
  - `~/bundled_benchmarks/{{SYMBOL_NAME}}_bench.py`
- [ ] Upload verified: `ssh p5e-cmh "ls -lh ~/bundled_benchmarks/{{SYMBOL_NAME}}*"` shows both files
- [ ] Remote execution completes without Python exceptions
- [ ] If `--validate` used: Correctness validation passes
- [ ] Latency numbers printed to console (e.g., "Latency: 0.234 ms")
- [ ] JSON results file created on remote machine
- [ ] Remote log saved locally: `artifacts/{{SYMBOL_NAME}}_remote.log`
- [ ] Log contains success indicators:
  - "✓ Correctness validated" (if --validate used)
  - "Latency: X.XXX ms" for each config
  - No traceback or exception messages
- [ ] TodoWrite updated: Stage 2.3 marked as completed (or skipped)

**Verification Commands**:
```bash
ssh p5e-cmh "test -f ~/bundled_benchmarks/{{SYMBOL_NAME}}.py && echo 'Bundle uploaded ✓'"
ssh p5e-cmh "test -f ~/bundled_benchmarks/{{SYMBOL_NAME}}_bench.py && echo 'Benchmark uploaded ✓'"
grep "Latency:" artifacts/{{SYMBOL_NAME}}_remote.log && echo "Latency reported ✓"
! grep "Traceback\|Error:" artifacts/{{SYMBOL_NAME}}_remote.log && echo "No errors ✓"
```

**Failure Handling**:
- If upload fails: Check SSH connection, file permissions
- If execution fails: Follow Iterative Debugging Loop (see troubleshooting.md)
- If correctness fails: Adjust tolerances or fix reference implementation
- After 3+ failed iterations: Review Stage 1 plan, consult troubleshooting guide

---

## Final Verification: Complete Workflow

**All Stages Complete**:
- [ ] All TodoWrite items marked as completed
- [ ] Both bundle and benchmark files exist locally
- [ ] Local execution successful (bundle runs, benchmark runs)
- [ ] Remote execution successful (if not skipped)
- [ ] Final summary printed to console (see below)

**Final Summary Template**:
```
================================================================================
✓ Benchmark Bundle Generated Successfully!
================================================================================

PLAN SUMMARY:
-------------
  Kernel: {{SYMBOL_NAME}}
  Type: {{KERNEL_TYPE}}
  Location: {{KERNEL_FILE_PATH}}
  Dependencies inlined: {{NUM_DEPS}}
  Primary function: {{ENTRY_POINT_FUNCTION}}

OUTPUT FILES:
-------------
  📄 bundled_benchmarks/{{SYMBOL_NAME}}.py ({{FILE_SIZE_KB}} KB)
     - Standalone execution bundle
     - Zero vLLM dependencies

  📄 bundled_benchmarks/{{SYMBOL_NAME}}_bench.py
     - Benchmark suite with parameter sweeps
     - Correctness validation included

  📄 /tmp/plan_{{SYMBOL_NAME}}.md
     - Detailed execution plan

  📄 artifacts/{{SYMBOL_NAME}}_remote.log (if remote testing done)

USAGE:
------
  # Quick validation
  python bundled_benchmarks/{{SYMBOL_NAME}}_bench.py --quick --validate

  # Full benchmark
  python bundled_benchmarks/{{SYMBOL_NAME}}_bench.py --sweep full --output results.json

  # Remote execution
  scp -r bundled_benchmarks p5e-cmh:~/
  ssh p5e-cmh "cd ~/bundled_benchmarks && python {{SYMBOL_NAME}}_bench.py --quick"

BENCHMARK SWEEP:
----------------
  Total configurations: {{TOTAL_CONFIGS}}
  Estimated runtime: {{ESTIMATED_MINUTES}} minutes

VALIDATION STATUS:
------------------
  ✓ Local execution: Passed
  ✓ Correctness check: Passed
  {{REMOTE_STATUS}}
================================================================================
```

---

## Quality Checklist

Before considering the workflow complete, verify:

### Code Quality
- [ ] No TODO comments in generated code
- [ ] All functions have docstrings
- [ ] Inline comments explain non-obvious operations
- [ ] Code follows PEP 8 style
- [ ] No hardcoded magic numbers (use named constants)

### Correctness
- [ ] All tensor shapes validated
- [ ] Dtype consistency throughout
- [ ] NaN/Inf checks in place
- [ ] Reference implementation matches kernel semantics
- [ ] Numerical tolerances appropriate for precision (FP16/BF16/FP8)

### Performance
- [ ] CUDA synchronization before/after timing
- [ ] Warmup iterations used
- [ ] Multiple timing iterations for averaging
- [ ] Memory usage reported
- [ ] TFLOPS calculated (if applicable)

### Documentation
- [ ] Bundle has clear module docstring
- [ ] Benchmark has usage examples in docstring
- [ ] README-style comments for complex sections
- [ ] Plan document saved for reference

### Security
- [ ] No eval() or exec() calls
- [ ] No arbitrary file path access
- [ ] Input validation performed
- [ ] No shell injection vulnerabilities
