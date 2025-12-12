# Argument Processing

This document explains how to parse and process command-line arguments for the vllm-benchmark skill.

## Command Format

```bash
/bundle-benchmark <SYMBOL_NAME> [--model MODEL_NAME] [--hardware DEVICE] [--precision DTYPE] [--skip-remote]
```

## Parsing Steps

When the skill is invoked with arguments like:
```
/bundle-benchmark fused_moe_kernel --model llama-70b --hardware H200 --skip-remote
```

Parse the arguments as follows:

### 1. Extract Positional Argument

**SYMBOL_NAME**: First positional argument
- Example: `"fused_moe_kernel"` from the command above
- This is the name of the kernel to extract and benchmark

### 2. Extract Optional Flags

**--model VALUE** → `MODEL` variable
- Default: `"qwen2.5"`
- Example: `--model llama-70b` → MODEL="llama-70b"
- Used for context in benchmark generation

**--hardware VALUE** → `HARDWARE` variable
- Default: `"H200"`
- Valid values: `H200`, `A100`, `H100`
- Example: `--hardware A100` → HARDWARE="A100"

**--precision VALUE** → `PRECISION` variable
- Default: `"FP16"`
- Valid values: `FP8`, `BF16`, `FP16`
- Example: `--precision BF16` → PRECISION="BF16"

**--skip-remote** → `SKIP_REMOTE` variable (boolean flag)
- Default: `false`
- If present: SKIP_REMOTE=true
- Controls whether to skip Stage 2.3 (remote verification)

## Variable Replacement Throughout Templates

After parsing, mentally replace these placeholders in all prompts and templates:
- `{{SYMBOL_NAME}}` → actual symbol name value
- `{{MODEL}}` → actual model value
- `{{HARDWARE}}` → actual hardware value
- `{{PRECISION}}` → actual precision value

## Examples

### Example 1: Basic Usage
```
/bundle-benchmark fused_moe_kernel
```
Results in:
- SYMBOL_NAME="fused_moe_kernel"
- MODEL="qwen2.5" (default)
- HARDWARE="H200" (default)
- PRECISION="FP16" (default)
- SKIP_REMOTE=false (default)

### Example 2: Full Specification
```
/bundle-benchmark paged_attention --model gpt-j --hardware A100 --precision BF16 --skip-remote
```
Results in:
- SYMBOL_NAME="paged_attention"
- MODEL="gpt-j"
- HARDWARE="A100"
- PRECISION="BF16"
- SKIP_REMOTE=true

### Example 3: Partial Specification
```
/bundle-benchmark flash_attn_varlen --hardware H100
```
Results in:
- SYMBOL_NAME="flash_attn_varlen"
- MODEL="qwen2.5" (default)
- HARDWARE="H100"
- PRECISION="FP16" (default)
- SKIP_REMOTE=false (default)

## Implementation Notes

- Parse arguments before Stage 0 (Initialization)
- Store parsed values for use throughout all stages
- Use default values when flags are not provided
- Validate all parsed values (see input-validation.md)
