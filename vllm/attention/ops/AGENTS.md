# vLLM Kernel Collector — Multi-Agent Specification (AGENTS.md)

This document defines how to operate agents that:
1) **Extract** a fully self-contained CUDA/Triton kernel translation unit (no host code).
2) **Synthesize** a **model-aware invocation example** and **write** the final JSON artifact.

We separate concerns for reliability: **Extractor → Invoker (+ Writer)**. An optional **Verifier** lints and sanity-checks the result.

================================================================================
0) High-Level Contracts
================================================================================

Inputs
- KERNEL_NAME (required): Identifier in the vLLM repo (e.g., kernel_paged_attention_2d).
- MODEL_NAME (optional): HF model id (e.g., Qwen/Qwen3-Coder-480B-A35B-Instruct, openai/gpt-oss-120b).
- WORKLOAD (optional): prefill | decode. If omitted, infer from wrapper usage.

Final Output (single file on disk)
- Write JSON to kernels_out/<KERNEL_NAME>.json and print exactly one confirmation line:
  WROTE kernels_out/<KERNEL_NAME>.json size=<bytes> sha256=<hex>

JSON Schema (mandatory)
{
  "name": "<KERNEL_NAME>",
  "kernel": {
    "kernel_type": "cuda or triton",
    "source_code": "FULLY SELF-CONTAINED SOURCE CODE AS A SINGLE STRING",
    "invocation_example": "STRING OF HOST CODE THAT DEFINES INPUTS AND LAUNCHES THE KERNEL"
  }
}

================================================================================
1) Agent Roles
================================================================================

Agent A — Extractor (source-only)
Goal: Produce the self-contained kernel translation unit (no host/test harness). If <KERNEL_NAME> is a wrapper, pick the leaf kernel(s) and inline only repo-local dependencies needed to compile.

Input:  KERNEL_NAME
Output: Partial JSON (for handoff, not written to disk) with:
- name, kernel.kernel_type, kernel.source_code
- handoff.meta: leaf kernel symbol(s), grid/meta formulas, arg order, masks/flags, template params, dynamic smem formula (if any), stride semantics (elements vs bytes).

Rules:
- Do NOT write the final JSON file.
- Do NOT include any host code in source_code.

Agent B — Invoker + Writer (model-aware)
Goal: Resolve model dimensions (MODEL_NAME), pick the correct kernel variant (dtype/head_dim/causal/sliding), compute grid/meta & dynamic shared memory exactly as the wrapper does, generate a realistic invocation example, assemble the final JSON, write it to disk, and print the single confirmation line.

Input:  KERNEL_NAME, optional MODEL_NAME, optional WORKLOAD, and Agent A’s handoff.
Output: Writes kernels_out/<KERNEL_NAME>.json; prints one WROTE line.

Optional: Verifier (linter / sanity checks)
- Confirms JSON is valid and reloadable.
- Ensures source_code contains __global__ (CUDA) or @triton.jit (Triton).
- Greps that source_code has no main, __main__, prints, or launches.
- If CUDA, optionally run nvcc -c to syntax-check; if Triton, minimal import/compile smoke test (if infra permits).

================================================================================
2) Tooling Policy
================================================================================

- Repo navigation: use rg (ripgrep).
- Browsing: Agent B must browse when MODEL_NAME is provided (HF config.json, model card). Otherwise optional.
- System/external includes: keep as-is (<cuda_runtime.h>, triton, torch, <cuda_fp16.h>, <cuda_bf16.h>).
- Repo-local includes/imports: inline only referenced symbols (helpers, macros, templates), in order, before first use.

================================================================================
3) Prompts (use as system+task prompts per agent)
================================================================================

3.1 Agent A — Extractor Prompt
System: You are an expert refactoring agent inside the vllm repository. Your task is to produce a self-contained kernel translation unit (CUDA or Triton) for <KERNEL_NAME>. Do not include any host/test harness. Do not write files. Return the JSON object (to the Orchestrator) with the fields in “Expected Return”.

Procedure (strict):
1) rg "<KERNEL_NAME>" and open matches.
2) If <KERNEL_NAME> is a wrapper/dispatcher:
   - Identify the leaf kernels actually launched:
     • Triton: @triton.jit functions.
     • CUDA: __global__ kernels launched via <<< >>>.
   - Extract guard conditions (dtype, HEAD_DIM, causal/sliding), grid/meta formulas, arg order, mask policy, dynamic smem formula.
3) Build the translation unit:
   - CUDA: include minimal system headers; inline repo helpers/macros/templates used by the kernel. Preserve existing extern "C" only if present. Keep arch/ROCm guards. Include <cuda_fp16.h>/<cuda_bf16.h> when needed.
   - Triton: minimal imports (triton, triton.language as tl), inline in-module helpers. Keep entry @triton.jit symbol(s).
   - Remove dead code; ensure all referenced identifiers are defined before use.
4) No host code in source_code. Do not add main, __main__, prints, or launches.

Expected Return (to Orchestrator):
{
  "name": "<KERNEL_NAME>",
  "kernel": {
    "kernel_type": "cuda|triton",
    "source_code": "SELF-CONTAINED TRANSLATION UNIT (string, escaped, no host code)"
  },
  "handoff": {
    "leaf_symbols": ["..."],
    "arg_order": ["raw pointers and scalars in exact launch order"],
    "grid_expr": "verbatim or simplified expression",
    "meta_params": {"HEAD_DIM": "...", "BLOCK_*": "...", "GROUP_SIZE": "...", "num_warps": "...", "num_stages": "..."},
    "guards": {"dtype": "...", "causal": "...", "sliding_window": "...", "packed_qkv": "..."},
    "smem_formula": "bytes expression if dynamic SMEM used, else null",
    "stride_semantics": "elements|bytes",
    "notes": "anything non-obvious (e.g., page tables, PAGE_SIZE, index layout)"
  }
}

Self-check before returning:
- Contains __global__ or @triton.jit.
- No host code.
- No unresolved symbols/macros.
- Local dependencies inlined; only system/external headers remain.

3.2 Agent B — Invoker + Writer Prompt
System: You finalize the artifact: derive a model-aware invocation example, assemble the final JSON with Agent A’s source_code, write it to kernels_out/<KERNEL_NAME>.json, and print exactly one confirmation line.

Inputs (from Orchestrator): Agent A JSON, KERNEL_NAME, optional MODEL_NAME, optional WORKLOAD.

Procedure (strict):
1) Read Agent A’s handoff (leaf kernels, arg order, grid/meta, guards, smem formula, stride semantics).
2) Model-aware sizing (must browse when MODEL_NAME given):
   - Fetch HF config.json for MODEL_NAME (then model card if needed).
   - Extract: num_attention_heads (H_q), num_key_value_heads (H_kv), head_dim (Dh) or compute hidden_size/H_q, hidden_size, max_position_embeddings or initial_context_length, attention flags (sliding_window, rope_scaling, qkv_bias, use_qk_norm).
   - Validate: H_q % H_kv == 0; if hidden_size known, Dh * H_q == hidden_size.
   - Map to kernel shapes via the wrapper’s semantics:
     • Paged attention (decode): Q [B,H_q,S_q,Dh]; KV paged buffers + page tables (use repo PAGE_SIZE). S_q ∈ {1..4}, S_kv large ≤ max_ctx. GROUP_SIZE = H_q/H_kv.
     • QK/Softmax/PV: Q [B,H_q,S_q,Dh], K/V [B,H_kv,S_kv,Dh] with grouping.
     • Norm/MLP/Rotary: width = hidden_size.
   - Batching & memory target: aim ~20% of free GPU memory, clamped to [2 GiB, 8 GiB] live set. Respect tile/page multiples (Dh alignment, PAGE_SIZE).
3) Build invocation_example (host code only; not inside source_code):
   - CUDA: C++ that allocates device buffers, constructs paged KV if needed, computes grid, block, dynamic smem_bytes from the same formula as wrapper, launches kernel, syncs. Provide SMALL_SANITY fallback.
   - Triton: Python that allocates tensors on CUDA, uses element strides (tensor.stride()), sets grid and meta (HEAD_DIM, BLOCK_*, GROUP_SIZE, num_warps, num_stages), launches kernel. SMALL_SANITY fallback for CPU-only.
   - Prepend a comment header with: MODEL_NAME, resolved dims (H_q, H_kv, Dh, hidden_size, max_ctx, flags), WORKLOAD, kernel/meta chosen, and source URLs.
4) Assemble final JSON per schema; write to kernels_out/<KERNEL_NAME>.json via atomic replace; reopen to verify; compute sha256.
5) Print exactly one line: WROTE <path> size=<bytes> sha256=<hex>.

Validation checklist (must pass before writing):
- invocation_example only (no host code in source_code).
- Grid/meta/guards mirror wrapper path (HEAD_DIM, BLOCK_*, num_warps, num_stages).
- If dynamic SMEM is used, smem_bytes computed and passed.
- For Triton, strides passed in elements.
- Model dims from browsed sources when MODEL_NAME set; header includes URLs.
- JSON reload succeeds.

Writer snippet (use programmatically; do not echo JSON)
import os, json, hashlib
os.makedirs("kernels_out", exist_ok=True)
path = f"kernels_out/{KERNEL_NAME}.json"
_tmp = path + ".tmp"
with open(_tmp, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, ensure_ascii=False)
os.replace(_tmp, path)
h = hashlib.sha256()
with open(path, "rb") as f: h.update(f.read())
print(f"WROTE {path} size={os.path.getsize(path)} sha256={h.hexdigest()}")

================================================================================
4) Orchestrator Runbook
================================================================================

1) Call Agent A with KERNEL_NAME. Capture its JSON.
2) Call Agent B with Agent A’s JSON + KERNEL_NAME (+ MODEL_NAME, WORKLOAD if provided).
3) (Optional) Verifier runs:
   - JSON reload, schema fields present.
   - source_code contains __global__ or @triton.jit.
   - No main/__main__/prints/launches in source_code.
   - If feasible, compile/syntax-check (dry run).
4) Store/commit kernels_out/<KERNEL_NAME>.json.

Concurrency: You can run multiple kernels in parallel. Ensure Agent B is the only writer per <KERNEL_NAME> to avoid races.

================================================================================
5) Why Two Agents Help
================================================================================

- Lower cognitive load: Extraction and invocation require different contexts (repo-local code vs. external model specs).
- Clear handoff: Agent A provides a precise handoff (arg order, meta, smem, stride semantics).
- Deterministic writing: A single writer responsibility avoids double-printing or malformed JSON.
- Better recovery: If Agent B fails (e.g., model browsing down), retry without repeating extraction.

================================================================================
6) Common Pitfalls & Guards
================================================================================

- Templated CUDA (template<int HEAD_DIM>): Keep templates intact in source_code; Agent B chooses a concrete instantiation in the invocation example per wrapper/model dims.
- Dynamic shared memory: Always carry a smem_formula from Agent A and reproduce it in Agent B.
- Paged KV layout: Ensure page tables/indices and PAGE_SIZE match the leaf kernel’s expectations; strides are elements in Triton; CUDA expects raw pointers and sizes/strides as implemented.
- Masking: Causal/sliding flags must match the chosen wrapper path; don’t silently flip masks.
- No host code in source_code. The host code lives only in invocation_example.

================================================================================
7) Minimal Acceptance Rubric
================================================================================

Pass ONLY if:
- JSON reloads; fields match schema.
- kernel_type matches actual source (__global__ vs. @triton.jit).
- No host code or launches in source_code.
- CUDA: correct headers present for used types (<cuda_fp16.h>, <cuda_bf16.h>).
- Triton: imports limited to triton/tl (+ torch only if used in helpers).
- invocation_example mirrors wrapper’s grid/meta and includes a model header (with URLs) when MODEL_NAME was provided.

# End of AGENT.md
