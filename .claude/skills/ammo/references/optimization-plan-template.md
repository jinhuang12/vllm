# Optimization Plan Template + Quality Bar (AMMO Stage 3)

Use this template to write `{artifact_dir}/optimization_plan.md`.

This is intentionally strict. Most bad outcomes happen when Stage 3 allows paper wins, missing constraints evidence, or ambiguous rollback scope.

## Non-negotiable quality bar

Your plan is not acceptable if any of the following are true:

- It contains placeholders (`1.`, `2.`, `TODO`) in execution-critical sections.
- It remains a scaffold/reference-only stub and does not synthesize run evidence.
- It does not cite Stage 1 constraints evidence (`constraints.md`, snapshot ID, profile files + per-bucket metrics).
- It does not state production parity knobs (CUDA graphs, compile/runtime mode, TP/EP, bucket policy).
- It lacks exact reproducible command entrypoints (AMMO scripts + adapter manifests).
- It has no incumbent comparison or dependency map.
- It lacks kill criteria for each selected hypothesis.
- It has no bounded enablement envelope and rollback switch.
- It does not define explicit Stage 5 pass/fail acceptance criteria.
- It does not define a concrete baseline-vs-candidate differentiation contract for the first implementation candidate.
- It does not define a top-3 parallel execution slate with one worktree/subagent per candidate.
- It does not define per-candidate subagent evidence package requirements.
- It does not define bounded-exhaustion continuation/termination criteria when autonomy mode is enabled.
- It leaves placeholder execution commands unresolved (for example `<candidate parity nsys command>`).

If Stage 1 constraints evidence is missing because model/runtime access is blocked:
- mark `status=blocked`,
- document the blocker in `{artifact_dir}/investigation/blocker.md`,
- request explicit waiver before proceeding.

## Command discipline

Use AMMO command sources of truth:
- `scripts/new_run.py`
- `scripts/run_adapter_bench.py`
- `references/adapters/<adapter>.manifest.json`

Do not invent flags. If a flag is not present in script help or manifest template, mark it as unknown and do not assume it.

## Copy/paste template

```markdown
# Optimization Plan

## 0) Constraints citations (mandatory)

- Constraints file: `constraints.md`
- Constraints snapshot ID:
- Constraints status from bundle:
- Sections used from constraints:
- Evidence links imported from Stage 1:
- Stage 2 mining evidence (required for adapter-specific ranking):
  - list adapter-declared mining artifacts

## 1) Context and envelope (from Stage 1 constraints)

- Framework + adapter:
- Model / hardware / dtype:
- TP / EP:
- Target regime:
- Bucket set:
- Production parity knobs (CUDA graphs, compile/runtime, serving flags):
- Evidence links (profiles/logs/json, copied from constraints):

## 2) Baseline truth snapshot summary (from constraints)

- Baseline per-bucket metrics (latency/throughput):
- Dominant hotspot groups (kernel/graph/runtime):
- Launch vs kernel split (if relevant):
- Active incumbent optimizations already in path:
- Current incumbent variant + metrics:

## 3) Ranked opportunity backlog (top 10)

Scoring rule (default):
- `Priority = Impact + Feasibility - Risk`
- Each score on `0..5`, justified by evidence.

| Rank | ID | Scope (kernel/graph/runtime) | Buckets | Evidence link | Impact | Feasibility | Risk | Priority | Incumbent interaction | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| 1 | OP-001 |  |  |  |  |  |  |  | preserve/modify/disable |  |
| 2 | OP-002 |  |  |  |  |  |  |  | preserve/modify/disable |  |
| 3 | OP-003 |  |  |  |  |  |  |  | preserve/modify/disable |  |
| 4 | OP-004 |  |  |  |  |  |  |  | preserve/modify/disable |  |
| 5 | OP-005 |  |  |  |  |  |  |  | preserve/modify/disable |  |
| 6 | OP-006 |  |  |  |  |  |  |  | preserve/modify/disable |  |
| 7 | OP-007 |  |  |  |  |  |  |  | preserve/modify/disable |  |
| 8 | OP-008 |  |  |  |  |  |  |  | preserve/modify/disable |  |
| 9 | OP-009 |  |  |  |  |  |  |  | preserve/modify/disable |  |
| 10 | OP-010 |  |  |  |  |  |  |  | preserve/modify/disable |  |

## 4) Selected hypotheses (choose top 3 feasible)

For each selected ID:

### Hypothesis H1 (ID: OP-xxx)
- Change scope (files/symbols/components):
- Why it should help (bottleneck mechanism):
- Expected profiler signature (what gets faster/disappears):
- Kill criteria (falsify fast):
- Incumbent dependency impact (preserve/modify/disable):
- Safety constraints (graph/runtime/correctness):

### Hypothesis H2 (ID: OP-yyy)
- Same fields as H1.

### Hypothesis H3 (optional)
- Same fields as H1.

### 4.1 Implementation order and fallback (mandatory)
- Parallel execution set: top 3 feasible IDs from ranked backlog.
- Priority order inside the parallel set: primary, secondary, tertiary.
- Blocker policy: if one candidate is blocked in Stage 4, continue remaining candidates and record blocker evidence.

### 4.2 Parallel execution slate (mandatory)

| Candidate ID | Rank | Worktree path | Branch name | Subagent owner | Allowed scope | Kill criteria summary |
|---|---:|---|---|---|---|---|
| OP-001 | 1 | `{artifact_dir}/parallel/worktrees/OP-001` | `ammo/<run>/op-001` | subagent-op001 |  |  |
| OP-002 | 2 | `{artifact_dir}/parallel/worktrees/OP-002` | `ammo/<run>/op-002` | subagent-op002 |  |  |
| OP-003 | 3 | `{artifact_dir}/parallel/worktrees/OP-003` | `ammo/<run>/op-003` | subagent-op003 |  |  |

### 4.3 Autonomy continuation policy (mandatory for bounded_exhaustion)

- Re-mining trigger after each completed wave:
- Minimum priority threshold for next-wave eligibility:
- Termination conditions:
  - `no_new_above_threshold`
  - `all_remaining_blocked`
  - `all_remaining_non_improving`
- Manual stop criteria:
- Convergence signature definition `(candidate, gate, command_fingerprint, failure_signature)`:

## 5) Dependency map and nullification risk (mandatory)

List current incumbent optimizations and activation conditions, then map each selected hypothesis.

| Incumbent optimization | Activation condition | H1 impact | H2 impact | H3 impact | Mitigation |
|---|---|---|---|---|---|
|  |  | preserve/modify/disable | preserve/modify/disable | preserve/modify/disable |  |

## 6) Pre-mortem expected deltas (mandatory)

For each selected hypothesis, predict directional deltas before implementation:

| Hypothesis | Win-source kernels/path | New overhead risk | Net kernel-time expectation by bucket | Confidence |
|---|---|---|---|---|
| H1 |  |  |  | low/med/high |
| H2 |  |  |  | low/med/high |
| H3 |  |  |  | low/med/high |

## 7) Measurement protocol (Stage 5 ready)

### 7.1 Correctness gate
- Tests and tolerances:
- Edge-case scenarios:
- Failure triage trigger:

### 7.2 Kernel-time gate
- Bucket set:
- Exact command entrypoints:
- Evidence artifacts to capture:
- Acceptance criteria:

### 7.3 E2E gate
- Workload mix and buckets:
- Exact command entrypoints:
- Acceptance criteria tied to ROI policy:

### 7.4 Candidate differentiation plan for Stage 4 (mandatory)
- Primary candidate ID for Stage 4:
- Baseline command/env/flags (exact):
- Candidate command/env/flags (planned exact shape; finalize after Stage 4 wiring):
- Explicit delta to introduce in Stage 4 (feature flag, dispatch switch, build artifact, or config key):
- Activation proof evidence to collect in Stage 4/5 (required log/profiler signatures):
- Rollback/disable switch:
- Invalid A/B condition: if normalized command+env+flags are identical, Stage 5 kernel/e2e promotion gates must be `blocked`.

### 7.5 Subagent evidence package contract (mandatory for parallel runs)

Each candidate subagent must produce:
- command provenance (baseline/candidate commands + env + flags),
- activation proof signatures,
- correctness evidence (tolerances, edge buckets, failures if any),
- kernel-time and e2e artifacts with parity proof,
- `subagent_evidence_manifest` path for checker input.

Promotion requires:
- subagent local gates `pass`,
- orchestrator acceptance gates `pass`.

## 8) Enablement envelope and rollback

- Enablement envelope (exact): model(s), dtype/quant, TP/EP, bucket conditions, runtime flags.
- Activation proof method:
- Rollback switch (env/config):
- Fallback path outside envelope:

## 9) Implementation checklist (concrete)

1. Create worktrees and spawn one subagent per selected candidate with explicit ownership boundaries.
2. Implement H1/H2/H3 in parallel with exact file/function list and dispatch/guard updates.
3. Add or update correctness tests for changed semantics and boundary buckets per candidate.
4. Capture per-candidate kernel-time evidence for baseline vs candidate vs incumbent on validated buckets.
5. Run per-candidate E2E parity benchmarks and record local pass/fail against Stage 5 acceptance criteria.
6. Re-run orchestrator acceptance validation for each candidate.
7. If multiple candidates pass, stack sequentially and revalidate after each merge.

## 10) Stop conditions

Stop and re-scope if any of the following occurs:
- Correctness gate fails twice on same failure mode without new evidence.
- Kernel-time regresses vs incumbent in validated envelope.
- E2E upper-bound math indicates target gain is unrealistic.
- Candidate requires envelope narrowing so severe it no longer serves target workload.
```
