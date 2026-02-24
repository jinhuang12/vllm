# Optimization Plan Template + Quality Bar (Stage 3)

Use this file to write `{artifact_dir}/optimization_plan.md`.

This is intentionally strict: Stage 3 is where most "paper wins" and parity drift are introduced.

## Non-negotiable quality bar

Your plan is **not acceptable** if any of the following are true:

- It contains placeholder bullets like "1. 2. 3." or empty sections.
- It proposes fusion work without citing Stage 1 evidence (dominant kernels, bucket regime, component share `f`).
- It does not state **production parity** knobs (CUDA graphs mode, torch.compile mode, TP/EP, bucketing).
- It does not include a concrete measurement protocol (exact bucket set + commands) for:
  - Stage 5.2 kernel perf gate, and
  - Stage 5.3 full-model E2E.
- It does not specify how you will prove the intended fast-path actually executed (log line / kernel name / profiler evidence).
- It has no rollback story (env var/config switch) and no bounded enablement envelope.

If you cannot fill required sections because the model is not cached locally, the correct action is:
- download weights (preferred), or
- mark the target `blocked` and ask the user for an explicit waiver (gated/auth/terms/network/disk).

## Copy/paste template

```markdown
# Optimization Plan (Stage 3)

## 0A) Previous Attempts (iteration N > 1 only; delete this section for first attempt)

> Include this section when writing an optimization plan after a KILL. It helps the planner
> and reviewers understand why prior approaches failed and how this attempt differs.

| Attempt | Opportunity ID | Status | Kill Reason | Date |
|---------|---------------|--------|-------------|------|
| 1 | OP-003 | KILLED | (e.g., FlashInfer uses identical kernel — 0% improvement possible) | 2026-02-15 |
| 2 | OP-002 | KILLED | (e.g., GEMM autotuner already optimal for these shapes) | 2026-02-17 |

**Lessons learned from previous attempts:**
- (What failure modes were discovered?)
- (What constraints do previous failures impose on this attempt?)

**Why this attempt is different:**
- (How does the new opportunity avoid the failure modes above?)

## 0) Context (from Stage 1; cite evidence)

- Model / dtype / hardware:
- TP / EP:
- Bucket regime targeted (decode vs prefill; BS set):
- Component share of end-to-end `f` (or why currently blocked):
- Baseline dominant kernels (top 3-5 by GPU time) and where time goes:
- Baseline "already optimized?" facts:

## 1) Optimization approach

- Approach chosen: (describe the optimization strategy)
- Rationale tied to evidence:
  - why this approach targets the dominant cost
  - what you are *not* trying to optimize (and why)

## 2) Correctness invariants (do not break)

- Component semantics: input/output contracts, numerical requirements
- Dtype/quantization invariants (scale layout, saturation rules, etc.)
- Graph safety constraints (no allocations in capture, stable shapes per bucket)

## 3A) Ranked optimization opportunities (top 10; evidence-backed)

**Required (unless you state a valid reason):** run the `vllm-fusion-opportunity-miner` workflow on your Stage 1 `nsys` trace and use its outputs to populate the ranked list below.

Valid reasons to not run `nsys_mine.py` (must be stated explicitly):
- No Stage 1 `.nsys-rep` available (must re-capture).
- `nsys export --type sqlite` fails in your environment.
- Python runtime unavailable (e.g., no `python3`) and you cannot use a venv.
- Trace is corrupted/unusable and cannot be re-captured.

Required artifacts (link paths):
- Stage 1 trace: `{artifact_dir}/.../*.nsys-rep`
- SQLite export: `{artifact_dir}/.../*_sqlite.sqlite`
- Mining outputs: `{artifact_dir}/nsys_mining.md` and `{artifact_dir}/nsys_mining.json` (or your chosen output dir)

Copy/paste commands (adjust paths):

```bash
# Export sqlite from an existing trace
nsys export --type sqlite -o {artifact_dir}/nsys/decode_sqlite {artifact_dir}/nsys/decode.nsys-rep

# Mine top kernels + repeated chains → ranked opportunities scaffold
FUSION_MINER_SKILL_DIR="<path-to-vllm-fusion-opportunity-miner>"
python3 "${FUSION_MINER_SKILL_DIR}/scripts/nsys_mine.py" \
  --sqlite {artifact_dir}/nsys/decode_sqlite.sqlite \
  --out-dir {artifact_dir}
```

Fallback if mining is blocked (must justify why):
- Export `cuda_gpu_kern_sum` and `cuda_gpu_trace` CSVs via `nsys stats`, then build the ranking and chains manually.

**Ranking rule (use consistently):** `Priority = (Impact + Feasibility)` (higher is better).
- Impact/Risk use 0–5 rubric from `vllm-fusion-opportunity-miner/references/fusion_patch_plan_rubric.md`, Rank feasibility from `time_saved_max_us` using `references/fusion-feasibility-heuristics.md`.
- Each item must cite evidence (kernel name(s) / chain signature) and link to the mining output row/section.
- Include opportunities across all kernel categories; if non-target-component items dominate, explicitly explain why you are still focusing on the target component (or adjust scope).
- You **must** prioritize the approach with highest potential improvement (ignore risk of failure) given the feasibility quantification math checks out.

Technique checklist (always evaluate; include as OP items if applicable):
- [ ] **W1 epilogue fusion (activation + quant)**: if activation/quant kernels exist between W1 and W2 and quantization metadata/layout can be matched. Reference: `references/optimization-techniques.md` → T11.
- [ ] **FlashInfer kernel survey + feasibility** (optional, for attention optimization): enumerate applicable FlashInfer kernel families and score feasibility/ROI. Reference: `references/optimization-techniques.md` → T12.

Ranked top-10 table (sorted by Priority; keep entries short):

| Rank | ID | Candidate | Regime (decode/prefill + buckets) | Evidence (kernel/chain + link) | Impact | Feas. | Risk | Priority | Next action |
|------|----|-----------|-----------------------------------|---------------------------------|--------|-------|------|----------|------------|
| 1 | OP-001 | | | | | | | | |
| 2 | OP-002 | | | | | | | | |
| 3 | OP-003 | | | | | | | | |
| 4 | OP-004 | | | | | | | | |
| 5 | OP-005 | | | | | | | | |
| 6 | OP-006 | | | | | | | | |
| 7 | OP-007 | | | | | | | | |
| 8 | OP-008 | | | | | | | | |
| 9 | OP-009 | | | | | | | | |
| 10 | OP-010 | | | | | | | | |

Feasibility sanity-check (per item, 1-2 lines each):
- What disappears or becomes cheaper?
- Conservative upper bound (e.g., time share x plausible reduction); relate to component share `f` for E2E plausibility.

## 3B) Active hypotheses (pick 2–3 from 3A; each testable)

Pick the top 2–3 backlog items you will actually implement/validate next. Each hypothesis must reference a backlog ID. Activate `gpu-kernel-optimizer` skill to expand optimization variety when considering potential changes.

For each hypothesis:
- Backlog ID: OP-???
- Change: what code/kernels you will modify (file/symbol)
- Why it should help: what bottleneck it targets (bytes, occupancy, barriers, launch)
- Expected signature in profiling: which kernel(s) get faster or disappear
- Kill criteria: what metric would falsify it quickly

## 4) Measurement plan (production parity)

### 4.1 Correctness gate
- Tests to run (and tolerances; source of tolerances):
- Additional edge cases (component-specific edge cases):

### 4.2 Kernel perf gate (CUDA graphs)
- Bucket set:
- Exact commands for baseline vs opt:
- Evidence collection:
  - `nsys` trace(s) and which report(s) you will use
  - `ncu` sanity checks for the dominant GEMM(s)
- Acceptance criteria:
  - default: no per-bucket regressions inside the enablement envelope

### 4.3 Full-model E2E gate (`vllm bench latency`)
- Weight availability plan:
  - if not cached locally, download weights (or mark `blocked` and ask for explicit waiver)
- Workload (decode-heavy vs prefill-heavy) + bucket sweep:
- Exact commands for baseline vs opt (same parity knobs):
- Acceptance criteria:
  - state expected E2E delta given component share `f` (use `references/e2e-delta-math.md`)

## 4B) Acceptance Criteria (for verifier)

The verifier uses this section to derive test methodology independently. Be specific and measurable.

### Correctness criteria
- Numerical tolerance: (e.g., `atol=1e-3, rtol=1e-3` for fp8; source of tolerance)
- Required test buckets: (list all batch sizes that must pass)
- Edge cases to test: (e.g., BS=1, max batch, empty input if applicable)
- Baseline for comparison: (must be vLLM production kernel, NOT naive PyTorch)

### Performance criteria
- Kernel-level: optimized must be ≤ baseline GPU time for ALL buckets under CUDA graphs
- E2E: expected improvement given component share `f` (cite `references/e2e-delta-math.md`)
- No-regression buckets: (list any buckets where regression is acceptable vs not)

### Kill criteria (pass/fail, no ambiguity)
- Kill criterion 1: {metric} {comparison} {threshold} — FAIL action: {what to do}
- Kill criterion 2: {metric} {comparison} {threshold} — FAIL action: {what to do}
- Kill criterion 3 (optional): ...

### What the verifier should NOT read
- Do NOT use `implementation_notes.md` to derive test methodology
- Derive all test inputs, expected outputs, and tolerances from this section and `references/validation-defaults.md`

## 5) Enablement envelope + rollback

- Enablement envelope (exact):
  - model id(s)
  - dtype/quant format(s)
  - TP/EP
  - hidden sizes/top_k/E
  - bucket set
- Proof of activation (how you will detect fast-path)
- Rollback switch:
- Fallback behavior outside envelope:

## 6) Implementation checklist (concrete steps)

1.
2.
3.
```
