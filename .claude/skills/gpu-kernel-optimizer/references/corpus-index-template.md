# Kernel Corpus Index Template

Use this template for every `references/corpus.<name>.md` so Codex can quickly route queries.

Keep the file **skimmable**: prefer short bullets, curated file lists, Insight Cards, and grep recipes over long prose.

---

# Corpus: <name>

- **Location:** `assets/corpora/<name>/`
- **Upstream:** <URL or repo>
- **License:** <license id>
- **Last indexed:** <YYYY-MM-DD>

## What this corpus is useful for

- <primary use case 1>
- <primary use case 2>
- <primary use case 3>

## Where to look first (curated)

Map common optimization tasks to the best starting files/dirs.

- **Attention (decode / prefill / paged / ragged):** <paths>
- **KV-cache layout / paging:** <paths>
- **Sampling / logits processing (top-k / top-p / min-p):** <paths>
- **Quantization / low precision (fp8/int8/fp4):** <paths>
- **MoE / routed GEMM / all-to-all:** <paths>
- **Norm / activation / fused epilogues:** <paths>
- **Benchmarks / repro harnesses:** <paths>
- **Tests / correctness:** <paths>

## High-level insights (required)

5–12 Insight Cards. Keep each card short and evidence-backed.

### Insight N — <motif name>
- Why it matters: <1 sentence>
- Where it lives (evidence): <2–5 file paths>
- Core mechanism: <2–4 bullets>
- Porting recipe: <2–4 bullets>
- Constraints: <arch/dtype/shape/runtime assumptions>
- How to validate: <metrics + benchmark/test hooks>

## Delta vs existing corpora (required)

- Most similar to: <existing corpus> because …
- New value add: <new family/layout/scheduler/bench>
- What not to import / intentionally omitted: <dirs/files>

## Trace recipes (required)

2–4 concrete traces from API → binding → kernel (+ grep hints).

## Directory map (top-level)

| Path | What’s inside | Why you’d read it |
|---|---|---|
| `docs/` | | |
| `csrc/` | | |
| `include/` | | |
| `<python_pkg>/` | | |
| `benchmarks/` | | |
| `tests/` | | |

## Key entry points

### Docs
- `<path>` — <why it matters>

### Python API / integration
- `<path>` — <what it exports / wraps>

### CUDA/C++ kernels
- `<path>` — <kernel family>

### Bindings (PyTorch / C++)
- `<path>` — <how python calls into kernels>

### Benchmarks
- `<path>` — <what it measures / how to run>

### Tests
- `<path>` — <what it validates>

## Grep recipes

Provide 8–15 “ready-to-run” searches.

## Notes & gotchas

- <JIT compilation? generated code? shape constraints? arch specialization?>
- <common pitfalls when porting ideas into other codebases?>
