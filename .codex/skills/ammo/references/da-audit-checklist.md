# Devil's Advocate Audit Checklist

Use this to audit a completed implementer track before integration. Default behavior: inspect artifacts first and re-run E2E only if red flags appear.

## Environment (Blocking)

- Use the track worktree: `cd {worktree_path}`
- Activate the existing environment: `source .venv/bin/activate`
- Never install packages
- Never create a new virtual environment

## Checklist

### 1. Worktree and venv isolation

- `which python` points to the worktree `.venv`
- `.so` provenance matches the expected build state
- `git log --oneline | head` shows only the track branch on top of main

### 2. Baseline is Stage 1

- `validation_results.md` cites `Baseline source: Stage 1 (not re-run)`
- reported baseline numbers match `{artifact_dir}/runs/baseline_bs{N}.json`
- no worktree baseline command appears without the optimization enablement flag

### 3. Production parity

Flag any use of:

- `TORCH_COMPILE_DISABLE=1`
- `--enforce-eager`
- `VLLM_TORCH_COMPILE_LEVEL=0`

Require explicit production-parity evidence.

### 4. Amdahl sanity

- read component share `f`
- read kernel speedup `s`
- compute `expected = f x (1 - 1/s)`
- compare against actual E2E improvement

Flag if actual is more than 1.5x above expected or less than half expected without a concrete explanation.

### 5. Compile cache and warmup

Check whether a cold `torch.compile` cache or insufficient warmup can explain the measurements.

### 6. CUDA graph claims

If the track claims node-count reduction or graph-overhead savings, require specific evidence.

### 7. Cross-track contamination

If multiple tracks exist:

- verify the worktree does not contain another track's source changes
- if this track is Python-only and another track changed `csrc/`, audit `.so` provenance explicitly

### 8. Conditional re-run

Re-run a locked optimized-only E2E benchmark if any red flag remains unresolved.

## Verdicts

| Verdict | Meaning | Action |
|---|---|---|
| CLEAN | no issues found | track may proceed |
| SUSPICIOUS | red flags not yet proven | lead investigates before integration |
| CONTAMINATED | methodology error proven | track fails until revalidated |
