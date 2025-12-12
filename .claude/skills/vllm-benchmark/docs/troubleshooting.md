# Troubleshooting Guide

This document provides solutions to common issues encountered during kernel extraction and benchmarking.

## Common Issues & Solutions

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `ModuleNotFoundError: vllm` | Missed an import during inlining | Grep bundle for `vllm`, inline the missing dependency |
| `ImportError: cannot import name 'X'` | Incomplete dependency resolution | Trace X back to source, inline recursively |
| `CUDA illegal memory access` | Out-of-bounds indexing in kernel | Check grid/block math, add bounds checks, verify metadata shapes |
| `CUDA out of memory` | Batch size too large | Reduce batch in sweep, check for memory leaks, add `.empty_cache()` |
| `RuntimeError: sizes mismatch` | Metadata shape incorrect | Print shapes in `BenchmarkTensors.make()`, compare to vLLM tests |
| `AssertionError in test_correctness` | Numerical precision issue | Use FP32 for reference, relax tolerances (rtol=1e-2 for FP16) |
| Benchmark returns 0 ms or NaN | Missing CUDA synchronization | Add `torch.cuda.synchronize()` before and after timing |
| Compilation takes forever | Autotune space too large | Reduce `@triton.autotune` configs, or remove decorator temporarily |
| `TritonError: compilation failed` | Syntax or version incompatibility | Check Triton version, verify decorator syntax, test locally |
| SSH timeout during upload | File too large (>100MB) | Check if accidentally including datasets, use compression |
| "Reference may be slow" not printed | Missing implementation | Actually implement `reference_implementation()` - it's required! |
| All tests pass but perf is 10x slower | Kernel not actually running | Verify kernel is being called, check for fallback to reference |
| `AttributeError: 'NoneType'` in bundle | Forgot to return value from function | Check all wrapper functions return tensors |
| Gradient computation error | `requires_grad=True` on inputs | Set `requires_grad=False` on all benchmark tensors |
| `RuntimeError: Expected all tensors to be on the same device` | Device mismatch | Ensure all tensors created with `device="cuda"` or same device |
| `TypeError: unsupported operand type` | Dtype mismatch in operations | Check dtype consistency, add explicit `.to(dtype)` conversions |

## Iterative Debugging Loop

When errors occur during remote execution (Stage 2.3), follow this systematic workflow:

### Step 1: Capture Full Error

**Goal**: Get complete error output for analysis

```bash
# Save complete output for analysis
ssh p5e-cmh "cd ~/bundled_benchmarks && python {{SYMBOL_NAME}}_bench.py --quick 2>&1" | tee artifacts/{{SYMBOL_NAME}}_error.log

# Examine the error log
cat artifacts/{{SYMBOL_NAME}}_error.log
```

### Step 2: Classify Error Type

| Error Pattern | Classification | Go To Step |
|---------------|----------------|------------|
| `ModuleNotFoundError: No module named 'vllm'` | Missing dependency | Step 3a |
| `ImportError: cannot import name 'X' from 'vllm'` | Missing dependency | Step 3a |
| `RuntimeError: CUDA error: illegal memory access` | Hardware/memory issue | Step 3b |
| `RuntimeError: CUDA error: out of memory` | Hardware/memory issue | Step 3b |
| `AssertionError: Shape mismatch` | Logic/shape issue | Step 3c |
| `ValueError: invalid dimensions` | Logic/shape issue | Step 3c |
| `TritonError: compilation failed` | Kernel syntax issue | Step 3d |

### Step 3a: Fix Missing Dependencies

**Process**:
1. Identify the missing import from error message
2. Search vLLM codebase for the source file:
   ```bash
   # Example: find where 'reshape_and_cache' is defined
   grep -r "def reshape_and_cache" vllm/
   ```
3. Read the source file and inline the function
4. Check if the inlined function has its own vLLM imports → recursively inline those
5. Add to "Inlined Dependencies" section with attribution comment:
   ```python
   # Inlined from vllm/utils/tensor_ops.py:45-67
   def reshape_and_cache(...):
       # ... exact vLLM implementation ...
   ```
6. Verify no more vLLM imports:
   ```bash
   grep -n "vllm" bundled_benchmarks/{{SYMBOL_NAME}}.py
   ```

**Success Criteria**: No output from the grep command above

### Step 3b: Fix CUDA Errors

**Process**:
1. Add debug logging before kernel launch:
   ```python
   print(f"Launching kernel with:")
   print(f"  grid={grid}, block={block}")
   print(f"  input.shape={input.shape}, input.device={input.device}")
   print(f"  input.is_contiguous()={input.is_contiguous()}")
   ```

2. Check tensor properties:
   - ✅ Device placement: All tensors must be on CUDA
   - ✅ Contiguity: Call `.contiguous()` if needed
   - ✅ Dtype consistency: All tensors have expected dtype

3. Verify grid/block dimensions are valid:
   - Grid dims must be > 0
   - Block dims must be ≤ 1024 (typically)
   - Total threads (product of block dims) must be valid for {{HARDWARE}}

4. Try smaller batch size to rule out OOM:
   ```python
   # In generate_benchmark_contexts(), temporarily use:
   batch_sizes = [1]  # Minimal batch for debugging
   ```

5. Add CUDA memory cleanup:
   ```python
   torch.cuda.empty_cache()
   torch.cuda.synchronize()
   ```

### Step 3c: Fix Logic/Shape Errors

**Process**:
1. Add shape debugging throughout the pipeline:
   ```python
   print(f"Input shapes: {[t.shape for t in [input, weight, bias]]}")
   print(f"Metadata shapes: {[m.shape for m in [topk_ids, topk_weights]]}")
   print(f"Expected output shape: {expected_shape}")
   ```

2. Check metadata generation in `BenchmarkTensors.make()`:
   - ✅ Verify index tensors are within valid ranges
   - ✅ Check metadata dimensions match batch/sequence sizes
   - ✅ Ensure dtype consistency (int32 for indices, float16 for weights)

3. Compare against vLLM's test cases:
   - Find tests in `tests/kernels/test_{{kernel_type}}.py`
   - Check how vLLM generates test inputs
   - Match their approach

4. Verify kernel wrapper signature matches call site:
   ```python
   # Print actual function signature
   import inspect
   print(inspect.signature({{SYMBOL_NAME}}_wrapper))
   ```

### Step 3d: Fix Compilation Errors

**Process**:
1. Check Triton version compatibility:
   ```bash
   ssh p5e-cmh "python -c 'import triton; print(triton.__version__)'"
   ```

2. Verify `@triton.jit` decorator syntax:
   - ✅ Ensure all type hints are valid
   - ✅ Check `tl.constexpr` usage is correct
   - ✅ Verify no Python 3.11+ syntax if using older Triton

3. Try compiling locally first:
   ```bash
   # Generate Triton IR to check for syntax errors
   python -c "import {{SYMBOL_NAME}}; print('Compilation successful')"
   ```

4. Check for compute capability issues:
   - H200 supports SM90
   - A100 supports SM80
   - Some Triton features may be hardware-specific

### Step 4: Retest Cycle

After applying fixes locally:

```bash
# 1. Verify fix locally if possible
python bundled_benchmarks/{{SYMBOL_NAME}}.py  # Should work without errors

# 2. Upload fixed version
scp bundled_benchmarks/{{SYMBOL_NAME}}*.py p5e-cmh:~/bundled_benchmarks/

# 3. Test with verbose output
ssh p5e-cmh "cd ~/bundled_benchmarks && python -u {{SYMBOL_NAME}}_bench.py --quick --validate" 2>&1 | tee artifacts/{{SYMBOL_NAME}}_test.log

# 4. Check for success indicators
grep "✓" artifacts/{{SYMBOL_NAME}}_test.log
grep "Latency:" artifacts/{{SYMBOL_NAME}}_test.log
```

### Step 5: Verify Success

**Success Indicators**:
- ✅ No Python exceptions in output
- ✅ Prints "✓ Correctness validated"
- ✅ Prints latency numbers (e.g., "Latency: 0.234 ms")
- ✅ JSON output file created at specified path
- ✅ All configurations in sweep completed successfully

**If Still Failing After 3 Iterations**:
1. Review the full plan from Stage 1 - may have missed critical dependencies
2. Check if the kernel requires vLLM-specific CUDA extensions
3. Consider testing with an even simpler input configuration
4. Consult vLLM's kernel test suite for additional clues
5. Use AskUserQuestion to ask the user for kernel-specific insights

## Debugging Best Practices

### DO:
- ✅ Add extensive `print()` statements to trace execution
- ✅ Check intermediate tensor shapes after each operation
- ✅ Test with minimal batch sizes first (B=1, H=128)
- ✅ Compare against vLLM's test suite for expected behavior
- ✅ Save error logs to `artifacts/` for analysis
- ✅ Verify locally before deploying to remote hardware

### DON'T:
- ❌ Skip validation steps to "save time"
- ❌ Assume error messages are wrong - they're usually accurate
- ❌ Modify kernel logic without documenting changes
- ❌ Test with production-size inputs during debugging
- ❌ Give up after first failure - iterate systematically

## Getting Help

If stuck after exhausting these troubleshooting steps:

1. **Review Stage 1 Plan**: Re-read `/tmp/plan_{{SYMBOL_NAME}}.md` for insights
2. **Check vLLM Tests**: Look in `tests/kernels/` for similar kernel tests
3. **Search vLLM Issues**: GitHub issues may have similar problems solved
4. **Use AskUserQuestion**: Ask the user for kernel-specific domain knowledge
5. **Document Unknown Issues**: Add new patterns to this troubleshooting guide
