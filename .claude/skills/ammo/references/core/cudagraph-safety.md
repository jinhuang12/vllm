# CUDA Graph Safety Checklist

Use this checklist when optimized paths run under graph capture/replay modes.

## Safety requirements

- no hidden allocations during capture/replay
- stable tensor shapes/layouts inside a capture envelope
- deterministic stream usage
- consistent workspace ownership and lifetime

## Validation steps

1. baseline correctness in non-capture mode
2. correctness under capture mode
3. activation proof for optimized path in capture mode
4. fallback behavior outside capture envelope

## Failure handling

If graph safety is uncertain:
- mark gate as blocked or failed,
- document reason,
- do not promote candidate.
