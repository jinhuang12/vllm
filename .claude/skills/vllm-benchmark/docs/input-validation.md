# Input Validation

**SECURITY CRITICAL**: Execute these validations BEFORE Stage 0. If any validation fails, STOP immediately and report the issue to the user.

## Validation Checklist

### 1. SYMBOL_NAME Validation

**Regex Check**:
```
Pattern: ^[a-zA-Z0-9_]+$
```
- Must contain only alphanumeric characters and underscores
- No special characters, spaces, or punctuation allowed
- Examples:
  - ✅ Valid: `fused_moe_kernel`, `paged_attention`, `flash_attn_v2`
  - ❌ Invalid: `fused-moe`, `paged attention`, `../kernel`, `kernel/impl`

**Path Traversal Prevention**:
- Must NOT contain `..` (parent directory references)
- Must NOT contain `/` or `\` (path separators)
- This prevents malicious paths like `../../etc/passwd` or `../vllm/config`

**Existence Check**:
1. Use Grep tool to search for the symbol in the vLLM codebase
2. Pattern: `def {SYMBOL_NAME}` or `class {SYMBOL_NAME}` or `{SYMBOL_NAME}_kernel`
3. If not found:
   - Use AskUserQuestion to ask the user to confirm spelling
   - Suggest similar symbols found via fuzzy search
   - Example: "Symbol 'fused_mo' not found. Did you mean 'fused_moe'?"

### 2. HARDWARE Validation

**Allowed Values**:
- Must be EXACTLY one of: `H200`, `A100`, `H100`
- Case-sensitive (must be uppercase)

**If Invalid**:
```
Error: Invalid hardware "{HARDWARE}"
Valid options: H200, A100, H100
```
- Report error message to user
- Do NOT proceed to Stage 0
- Ask user to retry with valid hardware

### 3. PRECISION Validation

**Allowed Values**:
- Must be one of: `FP8`, `BF16`, `FP16`
- Case-sensitive (must be uppercase)

**If Invalid**:
```
Error: Invalid precision "{PRECISION}"
Valid options: FP8, BF16, FP16
```
- Report error message to user
- Do NOT proceed to Stage 0
- Ask user to retry with valid precision

### 4. MODEL Validation

**Advisory Only**:
- MODEL is used for context and documentation only
- No strict validation required
- Can be any string value (e.g., HuggingFace model names)

**Recommendations**:
- Prefer full HuggingFace model IDs: `meta-llama/Llama-2-7b-hf`
- Short names are acceptable: `llama-70b`, `gpt-j`, `qwen2.5`
- No validation errors for MODEL parameter

## When to Use AskUserQuestion

Invoke the AskUserQuestion tool in these specific scenarios:

### Scenario 1: Symbol Name Ambiguity
```
Trigger: Grep finds multiple matches for SYMBOL_NAME
Example: Found "fused_moe" in:
  - vllm/kernels/fused_moe.py:45
  - vllm/model_executor/layers/fused_moe.py:123
  - vllm/ops/triton/fused_moe.py:78

Action: Ask which implementation to use
Question: "Found multiple 'fused_moe' implementations. Which one should I extract?"
Options:
  - "vllm/kernels/fused_moe.py (Triton kernel)"
  - "vllm/model_executor/layers/fused_moe.py (Model layer)"
  - "vllm/ops/triton/fused_moe.py (Triton ops)"
```

### Scenario 2: Kernel Type Unclear
```
Trigger: Cannot determine if kernel is Triton, CUDA, or hybrid from source inspection
Example: File contains both @triton.jit decorators AND torch.ops calls

Action: Ask user to clarify
Question: "Is this kernel Triton, CUDA, or a hybrid implementation?"
Options:
  - "Triton (focus on .py files with @triton.jit)"
  - "CUDA (focus on csrc/ and torch.ops bindings)"
  - "Hybrid (extract both Triton and CUDA components)"
```

### Scenario 3: Missing Critical Metadata
```
Trigger: Plan stage cannot determine input/output shapes or metadata requirements

Action: Ask user for clarification
Question: "Cannot determine input tensor shapes for {SYMBOL_NAME}. What are the expected dimensions?"
Options:
  - "Batch-first: (B, S, H)"
  - "Sequence-first: (S, B, H)"
  - "Custom (specify in text)"
```

### Scenario 4: Existing Bundle File Conflict
```
Trigger: bundled_benchmarks/{SYMBOL_NAME}.py already exists

Action: Ask if user wants to regenerate
Question: "Bundle for '{SYMBOL_NAME}' already exists. What should I do?"
Options:
  - "Regenerate (overwrite existing files)"
  - "Skip (use existing bundle, only regenerate benchmark)"
  - "Cancel (abort operation)"
```

### Scenario 5: Remote Hardware Unavailable
```
Trigger: --skip-remote NOT set but p5e-cmh is unreachable

Action: Ask if user wants to skip remote verification
Question: "Cannot reach p5e-cmh. How should I proceed?"
Options:
  - "Skip remote verification for now"
  - "Wait and retry connection"
  - "Cancel operation"
```

## When NOT to Use AskUserQuestion

**Do NOT use AskUserQuestion for**:
1. **Standard workflow decisions**: Follow the stages sequentially
2. **Implementation details**: Make reasonable choices based on vLLM patterns
3. **Minor formatting preferences**: Follow vLLM code style
4. **Trivial choices**: Use defaults when appropriate
5. **Already documented decisions**: Refer to the plan from Stage 1

## Validation Failure Handling

If validation fails:

1. **Print clear error message**:
   ```
   ❌ Validation failed: {REASON}

   Issue: {DETAILED_EXPLANATION}
   Expected: {WHAT_WAS_EXPECTED}
   Received: {WHAT_WAS_RECEIVED}
   ```

2. **Provide guidance**:
   ```
   To fix this:
   - {ACTION_ITEM_1}
   - {ACTION_ITEM_2}

   Example: /bundle-benchmark {CORRECTED_COMMAND}
   ```

3. **Do NOT proceed**: Exit immediately without executing any stages

## Success Message

After all validations pass:
```
✓ Input validation passed
  - Symbol: {SYMBOL_NAME}
  - Model: {MODEL}
  - Hardware: {HARDWARE}
  - Precision: {PRECISION}
  - Skip remote: {SKIP_REMOTE}

Proceeding to Stage 0: Initialization...
```
