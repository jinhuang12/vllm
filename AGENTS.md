# Repository Guidelines

Quick-start for vLLM contributors; keep changes scoped, documented, tested.

## Codex (OpenAI) — repo-local MoE monokernel workflow

This repo may include a repo-local Codex bundle under `.codex/` that provides:
- Custom prompt(s) as slash commands under `$CODEX_HOME/prompts/*.md`
- Additional agent guidance under `$CODEX_HOME/AGENTS.md`
- Helper scripts for non-interactive runs

Codex reads `AGENTS.md` files before doing work and builds an instruction chain from:
- **Codex home** (defaults to `~/.codex`, or set `CODEX_HOME=...`)
- **Repo path** (repo-root `AGENTS.md`, plus nested overrides when starting Codex from subdirs)  
Keep this file concise: instruction chains truncate at a size cap (32 KiB default). Prefer referencing `.codex/**` docs over inlining everything here. :contentReference[oaicite:6]{index=6}

### Recommended: activate repo-local prompts + instructions
From the vLLM repo root:

```bash
export CODEX_HOME="$PWD/.codex"
````

Prompts are discovered from `$CODEX_HOME/prompts/*.md` (file name becomes the prompt name), and are loaded when a session starts—restart Codex after edits. ([GitHub][2])

### Interactive (Codex TUI)

Launch Codex with the repo-local bundle enabled:

```bash
CODEX_HOME="$PWD/.codex" codex
```

Then invoke the monokernel workflow prompt via:

```text
/prompts:moe-monokernel-optimizer MODEL_ID="..." HARDWARE="..." DTYPE="fp8|bf16|fp16" TP=1 TOPK=8 [MODE=full] [PHASE=1..5] [ARTIFACT_DIR="..."]
```

### Non-interactive (CI / scripting)

Use `codex exec` for non-interactive runs; it can read the prompt from stdin using `-`. ([OpenAI Developers][3])

For the MoE monokernel workflow, prefer the repo helper script (if present):

* `./.codex/scripts/moe-monokernel-optimizer-exec.sh ...`
  This ensures the same prompt content/arguments are used as the interactive slash command.

### Codex behavior expectations for MoE monokernel tasks

If the task involves “monokernel”, “fused MoE”, “router→GEMM fusion”, or similar:

* Follow the `.codex/prompts/moe-monokernel-optimizer.md` 5-phase workflow (constraints → plan → implementation → validation → integration).
* Keep work resumable and auditable: write artifacts + logs to an artifact dir; prefer patch files over drive-by refactors.
* For non-interactive runs: never invoke interactive editors/pagers; avoid commands that prompt for auth.

---

## Project Structure & Module Organization

* `vllm/`: Core Python + API; `_custom_ops.py` holds custom bindings.
* `csrc/`: CUDA/C++ kernels and bindings via `CMakeLists.txt`.
* `tests/`: Pytest suite mirroring modules.
* `docs/`: MkDocs content; config in `mkdocs.yaml`.
* `examples/`, `benchmarks/`, `optimization-guides/`: Samples, perf scripts, tuning notes.
* `docker/`, `tools/`, `requirements/`: Images, scripts, pinned deps.

## Build, Test, and Development Commands

* Install (Python only): `VLLM_USE_PRECOMPILED=1 uv pip install -e .`
* Install (full CUDA/C++): `uv pip install -e .`
* Lint/format: `pre-commit run -a`
* Tests (GPU recommended): `pytest tests/`; focused `pytest -s -v tests/test_logger.py`
* Docs preview: `mkdocs serve`

### CUDA Kernel Development

- CUDA kernels in `csrc/`
- Python bindings in `vllm/_custom_ops.py`
- Incremental compilation is configured via `CMakeUserPresets.json` ([docs](https://docs.vllm.ai/en/latest/contributing/incremental_build/))

**After editing `csrc/` code:**
```bash
cmake --build --preset release --target install
```

**After adding new `csrc/` files:**
```bash
cmake --preset release && cmake --build --preset release --target install
```

Python changes take effect immediately (editable install).

## Coding Style & Naming Conventions

* Follow Google Python/C++ style; 4-space Python indents; `.clang-format` for C++/CUDA.
* Pre-commit tools: `ruff`, `clang-format`, `typos`, `shellcheck`, SPDX checks.
* Naming: snake_case modules/functions, PascalCase classes, `test_*` files mirroring source paths.
* Avoid new direct `import triton`; rely on existing wrappers.

## Testing Guidelines

* Pytest is standard; many suites expect GPU.
* Add tests in `tests/` alongside features; mirror source subpackages.
* Keep runs reproducible: minimize external I/O, seed randomness, document required hardware flags.

## MoE Monokernel (template for any MoE)

* Llama4 Scout/Maverick 8×H200 demo (see `optimization-guides/MOE_MONOKERNEL_OPTIMIZATION_GUIDE.md`, `optimization-guides/LLAMA4_MONOKERNEL_PATCH.md`) shows fused routing→quantize→GEMM→SiLU→GEMM for BS≤8/≤64; use it as the blueprint for any vLLM MoE.
* Constraints: CUDA≥12, SM 8.9+/9.0a (FP8); CMake `CUDA_ARCHS=9.0a`; allocate scratchpad; guard runtime with `moe_monokernel_supported()`.
* Preserve invariants when porting: tiny vs normal paths, 8:4 warp split with named barriers, triple buffering, bank-conflict mitigation (32B swizzle/padding), bitfield expert dedup + scale folding, speculative compute + mask filtering, branchless expert selection.
* Extend to new MoE: define `MoEDimensions`, instantiate `MOEMONOKERNEL_WRAPPER_IMPLEMENTATION`, register Torch bindings + Python wrappers with scratchpad sizing/timing helpers; reuse the guide as a checklist.

## Commit & Pull Request Guidelines

* Sign every commit with DCO: `git commit -s "message"`.
* Prefix PR titles with scope (e.g., `[Bugfix]`, `[CI/Build]`, `[Doc]`, `[Model]`, `[Frontend]`, `[Kernel]`, `[Core]`, `[Hardware][Vendor]`, `[Misc]`).
* Include a clear description, linked issues, and test results; add docs when user-facing behavior changes.
* Keep PRs tight; open an RFC/issue for architectural changes (>500 LOC) before implementation.
* Let CI run; run manual hooks like `mypy`/`markdownlint` locally if relevant.

## Security & Reporting

* Do not commit secrets or model credentials. For vulnerabilities, follow `SECURITY.md` and use GitHub Security Advisories rather than filing a public issue.

## Sanity Check

```bash
CODEX_HOME="$PWD/.codex" codex --ask-for-approval never "Summarize the current instructions you loaded."
````