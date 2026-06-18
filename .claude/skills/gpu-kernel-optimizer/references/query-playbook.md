# Query Playbook for Kernel Corpora

Goal: find the *smallest* set of files needed to answer “how is this kernel implemented / scheduled / fused?”

**Path convention:** paths are written **relative to the skill folder**.
If you are operating from your repo root, prefix with: `.codex/skills/gpu-kernel-optimizer/`

## The 30-second flow

1. Write the **problem signature** (op, shapes, dtype, layout, runtime constraints).
2. Open `references/corpus-registry.md` and select a corpus.
3. Open `references/corpus.<name>.md`.
4. Jump to:
   - **Where to look first**
   - **Key entry points**
   - **Grep recipes**
5. Open only the exact files/functions you need inside `assets/corpora/<name>/...`.

## Tactics that keep you fast

- Prefer searching by **API name** first (python wrapper → binding → kernel).
- Prefer searching by **layout keywords** (`paged`, `ragged`, `kv_layout`, `NHD`, `HND`, etc.).
- Prefer searching by **kernel family** (`batch_decode`, `prefill`, `fmha`, `rope`, `topk`, `sampling`, `quant`).

## Example: “Paged attention decode is slow”

1. Corpus registry → `flashinfer`.
2. Open `references/corpus.flashinfer.md` → “Attention” section.
3. Use grep recipes to find `batch_decode` + `paged` implementations under:
   - `assets/corpora/flashinfer/csrc/`
   - `assets/corpora/flashinfer/flashinfer/`

## Example: “Top-p sampling kernel is slow”

1. Start at:
   - `assets/corpora/flashinfer/flashinfer/sampling.py`
   - `assets/corpora/flashinfer/csrc/flashinfer_sampling_binding.cu`
2. Then follow the binding to the underlying CUDA implementation.

## If the repo uses code generation / JIT

Many LLM kernel repos (including FlashInfer) generate kernels from templates.

Rule: **Find the generator first**, then the generated artifact.

- Look for `jit/`, `templates/`, `.jinja`, or “build_and_load” style APIs.
- The index’s “Notes & gotchas” section should call out where codegen lives.
