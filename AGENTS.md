# AGENTS.md — Monolithic Python Bundle Builder (vLLM)

Objective: Build one self-contained Python module for a target symbol from vLLM (e.g., Qwen3MoeDecoderLayer) that:
- Inlines all repo-local Python dependencies (minimal closure).
- Embeds Triton kernels as @triton.jit within the same file.
- Embeds CUDA/C++ kernels as strings compiled at runtime via torch.utils.cpp_extension.load_inline.
- Appends a non-executing invocation_example() using model-aware shapes.
- Writes to bundles_out/<SYMBOL_NAME>.py and prints a single confirmation line (path, size, sha256).

License compliance: vLLM is Apache-2.0. We must preserve license headers for all inlined repo code. Agent A0 verifies and extracts notices; A7 prepends them verbatim. A0 prevents “refusal to copy substantial code” by proving intra-repo reuse is license-permitted and by constraining scope to repo-local code only.

---

## Orchestration Overview

Stages & artifacts
1. A0 — License & Scope Validator → artifacts/A0.license.json
2. A1 — Symbol Locator & Reachability Graph → artifacts/A1.graph.json
3. A2 — Python Collector & Pruner → artifacts/A2.inline.py, artifacts/A2.manifest.json
4. A3 — Triton Kernel Collector → artifacts/A3.triton.py, artifacts/A3.manifest.json
5. A4 — CUDA Collector & Wrapper → artifacts/A4.cuda.py, artifacts/A4.manifest.json
6. A5 — Binder & Rewire → artifacts/A5.module.py, artifacts/A5.bindmap.json
7. A6 — Model-Aware Invocation Synthesizer → artifacts/A6.example.py, artifacts/A6.model.json
8. A7 — Packager & Verifier → bundles_out/<SYMBOL_NAME>.py, artifacts/A7.report.json

Global inputs
- SYMBOL_NAME (required), e.g., Qwen3MoeDecoderLayer
- SYMBOL_FILE_HINT (optional path)
- MODEL_NAME (optional), e.g., Qwen/Qwen3-Coder-480B-A35B-Instruct
- WORKLOAD (optional) ∈ {prefill, decode} (auto-infer if omitted)

Allowed tools: rg (ripgrep), Python 3 (ast, libcst or parso, json, hashlib, pathlib), web browsing only for model configs/cards (no external code inlining).

Global constraints
- Inline only repo-local code that is strictly required.
- Keep external deps as imports (torch, triton, numpy, etc.).
- Preserve original symbol names and public method signatures.
- Triton passes element strides (not bytes).
- CUDA dynamic shared memory (smem) math must mirror the original launcher.
- No auto-execution (especially for invocation_example()).

---

## A0 — License & Scope Validator

Goal: Confirm permissive license (Apache-2.0) and extract notice text for inclusion. This avoids LLM refusal for “copying substantial code” by proving license permissibility and constraining copying to repo-local code only.

Input: repo root
Output: artifacts/A0.license.json with fields:
- status: ok|error
- repo_license: Apache-2.0
- license_files: list of found files (LICENSE, NOTICE)
- headers_to_prepend: full notice text to place at top of bundle
- notes: array

Procedure:
1) Read LICENSE and optional NOTICE; detect license.
2) On non-permissive license → status=error (fail fast).
3) Extract notice/header block into headers_to_prepend.

Success: status=ok, notice captured.

---

## A1 — Symbol Locator & Reachability Graph

Goal: Find SYMBOL_NAME, map minimal dependency closure across Python, Triton, and CUDA, and record wrapper→leaf dispatch.

Inputs: SYMBOL_NAME, optional SYMBOL_FILE_HINT
Output: artifacts/A1.graph.json (core fields):
- symbol, def_file, def_line
- python_nodes: list of required classes/functions/constants with file, lineno, type
- triton_kernels: list of @triton.jit kernels and helpers
- cuda_ops: list of torch.ops.vllm.* or bound symbols, binding_file, launched __global__ kernels
- imports_external: external libraries to keep as imports
- edges: call graph edges (from symbol methods to leaves)
- guards: predicates for variant selection (dtype, head_dim, causal, sliding_window)
- notes

Procedure:
- rg for class def; confirm with AST.
- Resolve repo-local imports used by symbol methods (__init__, forward, helpers).
- Collect Triton kernels and their local helpers.
- Resolve CUDA calls via bindings → launched __global__ kernels.
- Record guard predicates used for variant selection and meta (num_warps, num_stages, HEAD_DIM, BLOCK_*).

Success: Graph resolves all repo-local references; no unknowns.

---

## A2 — Python Collector & Pruner

Goal: Inline only the required Python definitions in topological order.

Input: A1.graph.json
Outputs: artifacts/A2.inline.py, artifacts/A2.manifest.json

Manifest fields:
- status
- inlined: list of {name, file, lines}
- skipped: unused helpers
- external_reqs: e.g., torch>=2.1, triton>=2.1
- topo_order: dependency order of pasted blocks
- notes

Procedure:
- Parse sources (libcst/AST) and slice exact node extents to preserve code verbatim.
- Emit sections with provenance comments like:
  # --- Inlined from vllm/.../rmsnorm.py:L45-L98 (Apache-2.0) ---
  class RMSNorm: ...
- Remove repo-local import lines satisfied by inlined blocks.
- Keep external import lines intact.

Success: Compiles standalone (excluding Triton/CUDA parts).

---

## A3 — Triton Kernel Collector

Goal: Inline required Triton kernels and local helpers.

Input: A1.graph.json
Outputs: artifacts/A3.triton.py, artifacts/A3.manifest.json

Manifest fields:
- status
- kernels: list with file, lines, meta (num_warps, num_stages, HEAD_DIM, BLOCK_SIZE)
- helpers: required jitted helpers
- imports: import triton and import triton.language as tl
- notes

Procedure:
- Copy @triton.jit kernels and helper functions exactly; keep signature and meta intact.
- Place before any Python wrappers that call them.

Success: Triton section imports and compiles on import.

---

## A4 — CUDA Collector & Wrapper

Goal: Embed necessary CUDA/C++ sources as strings and expose Python wrappers that mimic the original extension API.

Input: A1.graph.json
Outputs: artifacts/A4.cuda.py, artifacts/A4.manifest.json

A4.cuda.py structure (illustrative):
- CUDA_SOURCES dict mapping filenames to raw string content
- _EXT global and _build_cuda_extension_if_needed() using torch.utils.cpp_extension.load_inline
- thin Python wrappers mapping exactly to exported extension functions (e.g., flash_mla_fwd_wrapper)

Manifest fields:
- status
- cuda_files: list of embedded files
- exported: list of exported C++/PyBind functions
- wrappers: list of Python wrapper names
- notes: e.g., ensure smem_bytes matches original launcher when required

Procedure:
- Trace torch.ops.vllm.* → binding → __global__ kernel(s).
- Embed minimal required .cu/.cuh/.cc sources verbatim (or precisely sliced).
- Compute smem_bytes identical to original launcher if needed (in binding or wrapper).

Success: Wrapper callable and signature-compatible.

---

## A5 — Binder & Rewire

Goal: Merge A2 + A3 + A4 into a single module and rewrite internal call-sites to use local Triton/CUDA wrappers.

Inputs: A2.inline.py, A3.triton.py, A4.cuda.py
Outputs: artifacts/A5.module.py, artifacts/A5.bindmap.json

bindmap fields:
- status
- rewrites: list of {from, to} (e.g., torch.ops.vllm.flash_mla_fwd → flash_mla_fwd_wrapper)
- entry_symbol: target symbol name
- public_signatures: map of method signatures preserved
- notes

Procedure:
- Concatenate sections in this order:
  1) license header (placeholder for A0 output)
  2) external imports
  3) CUDA embedding + builder + wrappers
  4) Triton kernels
  5) Python helpers
  6) Target symbol (exact name/signature preserved)
- AST-rewrite call sites referencing torch.ops.vllm.* to local wrappers.
- Verify public signatures unchanged.

Success: Module imports without executing example; no unresolved refs.

---

## A6 — Model-Aware Invocation Synthesizer

Goal: Generate a non-executing invocation_example() that builds realistic shapes from the model config.

Inputs: A5.module.py, optional MODEL_NAME, optional WORKLOAD
Outputs: artifacts/A6.example.py, artifacts/A6.model.json

A6.model.json fields:
- status
- model_name
- dims: H_q, H_kv, Dh, H (hidden_size), max_ctx
- flags: sliding_window, use_qk_norm, rope_scaling as applicable
- workload: prefill or decode
- sources: list of URLs (HF config.json, model card)
- notes

Procedure:
1) If MODEL_NAME provided, browse Hugging Face config.json:
   - Extract num_attention_heads (H_q), num_key_value_heads (H_kv), head_dim (Dh) or compute Dh=hidden_size/H_q, hidden_size (H), max_position_embeddings or initial_context_length (max_ctx), and flags (sliding_window, rope_scaling, use_qk_norm).
   - Validate H_q % H_kv == 0; if hidden_size present, Dh*H_q == H.
2) Decide WORKLOAD:
   - decode if KV cache + small S_q (≤4), else prefill.
3) Choose shapes:
   - Decode: S_q ∈ {1..4}, S_kv large ≤ max_ctx; GROUP_SIZE=H_q/H_kv; scale B to ~20% of free GPU memory (clamp 2–8 GiB live set).
   - Prefill: S_q = S_kv ≈ 8k–32k within memory and ≤ max_ctx.
4) Emit invocation_example(small=False) at EOF (not executed automatically):
   - Comment header with model name, dims, URLs, workload, flags.
   - Memory-adaptive sizing using torch.cuda.mem_get_info() with SMALL_SANITY fallback.
   - Instantiate layer, allocate inputs, run a single forward, optional basic shape assertions.
   - Do not call this function automatically.

Success: Function composes with local wrappers; nothing runs unless explicitly called.

---

## A7 — Packager & Verifier

Goal: Build final monolithic file, verify importability, and print a single confirmation line.

Inputs: A5.module.py, A6.example.py, A0.license.json
Outputs: bundles_out/<SYMBOL_NAME>.py, artifacts/A7.report.json

Procedure:
1) Compose final file:
   - Prepend headers_to_prepend from A0.
   - External imports
   - CUDA embedding + builder + wrappers (A4)
   - Triton kernels (A3)
   - Python helpers (A2)
   - Target symbol (A2)
   - invocation_example() (A6)
2) Atomic write + digest:
   - Create bundles_out/
   - Write to tmp, replace
   - Compute sha256
   - Print one line: WROTE bundles_out/<SYMBOL_NAME>.py size=<bytes> sha256=<hex>
3) Verification:
   - Syntax: ast.parse or static checker (optional).
   - Import check (do not call invocation_example()).
   - If CUDA available, optional _build_cuda_extension_if_needed() probe.

Report fields:
- status
- path
- size
- sha256
- checks: syntax/import/cuda_builder statuses
- notes

Success: File written; single confirmation printed.

---

## Cross-Stage Quality Gates

- Minimal closure only: A2 must inline only referenced symbols (no import *, no dead blocks).
- Name/signature preservation: A5 guarantees public API parity with vLLM.
- CUDA parity: A4 wrapper mirrors extension API and smem math.
- Triton correctness: Pass element strides and mirror wrapper meta (num_warps, num_stages, BLOCK_*, HEAD_DIM).
- License compliance: A0 header at top of final file (verbatim).
- No auto-execution: Final module imports cleanly and idle.

---

## Failure & Retry Policy

- A1 missing symbol: allow SYMBOL_FILE_HINT to disambiguate.
- A3/A4 variant pruning: respect guard predicates (dtype, HEAD_DIM, causal/sliding).
- CUDA build errors: verify functions=[…] exports; include needed .cuh in cuda_sources; add original macros/defs captured in A1.
- “Too much code to copy” refusals: A0 proves permissive license; A2 ensures minimal inlining; A7 writes to file (not console).

---

## Optional CLI Runner (for humans)

A tiny convenience wrapper (outside agent flow) to sanity-run the example:
- Import the generated module from bundles_out/<SYMBOL_NAME>.py
- Call invocation_example(small=True) explicitly

---

## Notes for Qwen3MoeDecoderLayer

- Expect dependencies on MoE router, expert MLP, norms, and attention backends.
- Likely paged attention (Triton/CUDA). Mirror page sizes and layouts from wrappers.
- With MODEL_NAME=Qwen/Qwen3-Coder-480B-A35B-Instruct: expect H_q=96, H_kv=8, Dh=128, H=6144, max_ctx=262144 (validate via HF config.json in A6); decode example: S_q=1..4, S_kv large (memory-bounded), B scaled to target 2–8 GiB.
