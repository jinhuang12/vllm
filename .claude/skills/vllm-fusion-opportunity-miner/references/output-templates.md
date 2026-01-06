# Output templates (copy/paste)

## Table of contents
1. `state.json` schema
2. `baseline_snapshot.md` template
3. `fusion_opportunities.md` template
4. `patch_plan.md` template

## 1) `state.json` schema

```json
{
  "run_id": "",
  "artifact_dir": "",
  "stage": "",
  "status": "in_progress",
  "commands": [],
  "artifacts": [],
  "notes": []
}
```

## 2) `baseline_snapshot.md` template

```markdown
# vLLM baseline snapshot

## Environment
- vLLM commit:
- PyTorch:
- CUDA + driver:
- GPU(s):
- NCCL (if multi-GPU):
- Quantization:
- Parallelism: TP=?, PP=?, EP=?

## Production parity knobs (confirm from vLLM config/code; do not guess)
- CUDA graphs:
  - enabled?:
  - capture buckets/shapes:
- torch.compile:
  - enabled?:
  - mode/flags/caches:

## Workloads
### Decode-heavy
- command:
- batch sizes:
- input_len:
- output_len:
- results (p50/p95 or mean):

### Prefill-heavy
- command:
- batch sizes:
- input_len:
- output_len:
- results:

## Notes
- warnings (fallback kernels, missing tuned configs, graph breaks):
```

## 3) `fusion_opportunities.md` template

```markdown
# vLLM fusion opportunities (ranked)

## Summary (top 5)
| Rank | Candidate | Regime | Evidence | Why it should help | Risk |
|------|----------|--------|----------|--------------------|------|
| 1 | | | | | |

## Evidence
Attach:
- kernel ranking (per regime)
- repeated chains (per regime)
- nsys report tags used

## Candidate details (repeat per candidate)
### Candidate: <name>
- Regime: decode or prefill; batch sizes
- Kernel(s) / chain signature:
- Total GPU time share:
- Attribution (where in vLLM):
  - file:line / module / op
- Fusion boundary:
- Expected savings:
- Primary risks:
- Validation plan:
```

## 4) `patch_plan.md` template

```markdown
# Patch plan (top fusion candidates)

## Candidate 1: <name>
### Files to change
- path/to/file.py: function(...)
- path/to/kernel.(cu|py): kernel(...)

### Proposed change
- Replace kernel chain A->B->C with fused kernel F
- Keep fallback behind env var / flag

### Rollback / safety
- Env var / flag:
- Default behavior unchanged unless enabled

### Validation
- Correctness: tolerance + tests
- Performance: regimes + metrics + parity (graphs/compile)

## Candidate 2: <name>
...
```

