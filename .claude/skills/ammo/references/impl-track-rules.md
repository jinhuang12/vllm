# Implementation Track Rules

Agent-facing rules for Stages 4-5 parallel implementation tracks. The impl-champion must follow these rules.

## Worktree Build Rules

| Change Type | Required Action | Time |
|-------------|----------------|------|
| **Pure Python** (model code, Triton kernels, CuTeDSL kernels, configs) | Edit, test, commit. **No rebuild.** | Immediate (CuTeDSL JIT-compiles on first import — cache under `$CUTE_DSL_CACHE_DIR` or `/tmp/{user}/cutlass_python_cache`) |
| **C++ kernel** (csrc/ changes, including CUTLASS C++ templates) | `cmake --preset release && cmake --build --preset release --target install` | ~5-55s (ccache) |

The champion compiles. The kernel correctness & speedup checks run against the committed, compiled code.

## Source Modification Rules

- Only the champion modifies source files (`csrc/`, `vllm/`, etc.).
- The champion writes its kernel-gate artifacts to `{artifact_dir}/rounds/{CR}/tracks/{op_id}/validator_tests/` (where `{CR}` is `campaign.current_round`). The `validator_tests/` dirname is retained as a historical label — downstream consumers (state.json merge, dashboard tabs, eval scorer) read from this exact path.
- These gates run before the E2E sweep — no GPU coordination needed.

## Kernel-Gate Validation Principle

The champion writes its own correctness tests and benchmarks for the kernel correctness & speedup checks from the **optimization plan and debate summary** — deriving tests from what the optimization SHOULD do (the plan) rather than only what it DOES do (the implementation). Hold yourself to the plan's intent: test every batch size, keep assertions tight, and report benchmark numbers straight.

## Champion-Owned Validation

The champion owns all Stage 5 validation, running the kernel-level gates:

```
Kernel-Level:
  Gate 5.1a: Kernel correctness tests (champion-authored, from the plan)
  Gate 5.2: Kernel speedup benchmark under production parity

E2E-Level (Champion):
  Gate 5.1b: Sweep --verify-correctness  [separate invocation; NEVER combine with --nsys-profile]
  Gate 5.3a: Sweep --nsys-profile (kernel proof)  [separate invocation from Gate 5.1b]
  Gate 5.3b: Sweep E2E latency (per-BS verdicts)
  Writes final validation_results.md with evidence chain
```

Gates 5.1b and 5.3a MUST be separate sweep invocations. Combining
`--verify-correctness` and `--nsys-profile` cgroup-OOMs EngineCore mid-correctness
because nsys wraps the entire child and its trace buffers grow across all 1319
GSM8K questions; `--nsys-output-len` only bounds the bench loop, not correctness.
The sweep script enforces this with a hard `SystemExit`. See
`nsys-profiling-guide.md` §3.14 for the full mechanism and recovery procedure
(stale EngineCore on the GPU after OOM).

## Handling Validation Failures

When a kernel correctness or speedup check fails:

1. Champion records the gate failure details
2. Champion diagnoses root cause
3. Champion fixes implementation, recompiles if needed
4. Champion completes the Self-Validation Gate checklist (root cause reasoning, smoke test, fix-attempt counter)
5. Champion commits and re-runs the kernel gates (no E2E sweep until 5.1a PASSes)

Resist the temptation to "fix" a failure by loosening your own test. The gate exists to catch a broken kernel before the expensive E2E sweep; weakening the assertion just defers the failure to Gate 5.1b (GSM8K) or 5.3b (latency), where it surfaces anyway — without the kernel-level diagnosis you'd have had here.

## Track State Reconciliation

A failure or infrastructure blocker is not recorded until `state.json` is reconciled with the evidence. After writing `rounds/{CR}/tracks/{op_id}/evidence.json` and `rounds/{CR}/tracks/{op_id}/validation_results.md`, run:

```bash
python .claude/skills/ammo/scripts/reconcile_track_state.py \
  --artifact-dir <artifact_dir> --track-id {op_id} --write

python .claude/skills/ammo/scripts/reconcile_track_state.py \
  --artifact-dir <artifact_dir> --track-id {op_id} --check
```

The reconciled track state must carry the same `status` as the evidence, a schema-valid `verdict` (`null` for `GPU_BLOCKED`), and `kill_criteria_results` from structured `evidence.json`. `GPU_BLOCKED` is a lead-triage blocker, not a terminal pass/fail verdict, and must not be counted as complete for Stage 6. Do not let a failed or blocked track remain `IN_PROGRESS` with stale verdict fields.

## GATING_REQUIRED Workflow

> **This is the canonical definition.** Other files reference this section.

When per-BS verdicts show mixed results (some PASS + some REGRESSED), the track enters GATING_REQUIRED:

1. Sweep reports per-BS verdict table showing mixed results
2. Champion evaluates gating feasibility (is the dispatch site compatible with a gating mechanism?)
3. If feasible: champion runs crossover probing benchmarks itself
4. Champion runs the kernel sweep + E2E confirmation per `crossover-probing.md`
5. Champion implements gating mechanism per `code-templates.md` dispatch decision tree
6. Champion registers env var in `vllm/envs.py`, defaulting off (`=0`). **The flag name is the PR-facing public name of the optimization** — derive it from the mechanism per § Env Flag Naming (PR-Ready) below, never from the internal `op_id`.
7. Champion commits gated implementation
8. Champion re-runs the kernel gates on the gated kernel (correctness & speedup)
9. Champion re-runs sweep on gated code (5.1b + 5.3a + 5.3b) — all BS must be PASS or NOISE
10. If both kernel re-validation and sweep pass: verdict = `GATED_PASS`. If either fails: verdict = `FAIL`.

One gating attempt per track — no nested gating.

## Env Flag Naming (PR-Ready)

> **This is the canonical naming rule.** Other files reference this section.

Whenever you register a `VLLM_*` env flag in `vllm/envs.py` — whether for a GATED_PASS dispatch gate or any opt-in optimization — that flag is **not an internal label**. It ships verbatim in the `vllm/envs.py` diff, gets promoted into `target.json:bench.baseline_env` on SHIP, and is copied straight into the PR's enable instructions (the PR workflow reads `baseline_env` keys verbatim — it cannot rename a flag without rewriting your merged code). So the name a reviewer reads in the PR is the name you type here. A name that doesn't communicate what the optimization does reads as low-effort and gets the PR bounced.

The internal `op_id` (`op007`, `OP-003`) is a campaign tracking handle for wiring up agents, tracks, and artifact dirs. It is meaningless to a vLLM maintainer. **Never let it become the flag name.**

**Convention** — describe the optimization, not its tracking number:

```
VLLM_<SCOPE>_<MECHANISM>[_<ARCH>]
```

- `<SCOPE>` — the model family or subsystem the flag governs (`NEMOTRON3`, `MAMBA2`, `MOE`, `ATTN`). Use the model family when the optimization is checkpoint-specific; use the subsystem when it's general.
- `<MECHANISM>` — what the kernel/dispatch actually does (`FP8_PREFILL_GEMM`, `GATED_RMS_NORM_FUSION`, `TWO_STREAM`, `SSD_FUSED_STATE`).
- `<ARCH>` — optional hardware tag when the path is architecture-gated (`SM100`, `SM90`).

**Example 1 (good):**
Optimization: a CUTLASS SM100 FP8 dense GEMM for the prefill path of Nemotron-3.
Flag: `VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100`

**Example 2 (bad — rejected):**
Same optimization, named off the tracking id.
Flag: `VLLM_OP004` — a maintainer cannot tell what it does, what it touches, or whether it's safe to enable. This is exactly the leak that gets a PR bounced on naming.

**Rule of thumb**: if you deleted the campaign's `state.json`, would the flag name still tell a stranger what the optimization does? If not, rename it before you commit. A flag whose name is just `VLLM_OP<n>` / `VLLM_OPT<n>` (the op_id with a `VLLM_` prefix) is never acceptable — the Stage 4-5 audit treats it as a BLOCKING finding.

## Stage 1 Baseline Reuse

All E2E comparisons use Stage 1 baseline numbers. Never run a new baseline during implementation.

Baseline data locations (round-scoped, where `{CR}` is `campaign.current_round`):
- Per-BS E2E latency: `{artifact_dir}/rounds/{CR}/sweeps/baseline/json/baseline_bs{N}.json`
- Summary table: `{artifact_dir}/rounds/{CR}/constraints.md` — "Baseline E2E latency" section
- Kernel breakdown: `{artifact_dir}/rounds/{CR}/constraints.md` — "Baseline Truth Snapshot" section

## Track Constraints

These constraints apply to the champion, including its kernel gates:

1. **All batch sizes.** Test every batch size in target.json. No exceptions. No cherry-picking.
2. **Production parity.** CUDA graphs + torch.compile in ALL measurements. NEVER use `--enforce-eager`, `TORCH_COMPILE_DISABLE=1`, or `VLLM_TORCH_COMPILE_LEVEL=0`.
3. **vLLM baseline.** Compare against vLLM's production kernel, NOT naive PyTorch.

## References

- `validation-defaults.md` — verdict thresholds (noise_tolerance_pct, catastrophic_regression_pct) and per-BS classification logic
- `crossover-probing.md` — crossover probing protocol for GATING_REQUIRED tracks
- `code-templates.md` — dispatch patterns and kernel templates
- `gpu-pool.md` — GPU reservation pattern
- `cudagraph-safety.md` — CUDA graph capture checklist
