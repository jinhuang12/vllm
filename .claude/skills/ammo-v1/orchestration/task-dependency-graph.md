# AMMO Task Dependency Graph

Full T1-T22 task graph with dependency relationships, TaskCreate templates, and stage transition rules.

## Dependency Overview

```
STAGE 1: CONSTRAINTS + BASELINE CAPTURE
T1:    Scaffold artifact directory              [lead]         → blocks T2, T3
T2:    Collect environment                      [verifier]     ← blocked by T1
T3:    Read vLLM source for target component    [planner]      ← blocked by T1
T4:    Run nsys baseline profiling              [verifier]     ← blocked by T2
T5:    Extract dominant kernel timings          [verifier]     ← blocked by T4
T6:    Write constraints.md                     [verifier]     ← blocked by T3, T4, T5
T7:    GATE: verify_stage1_baseline.py          [lead]         ← blocked by T6

STAGE 2: BOTTLENECK MINING AND RANKING
T8:    Run bottleneck mining (nsys analysis)    [verifier]     ← blocked by T7
T9:    Rank optimization opportunities          [planner]      ← blocked by T8
T10:   GATE: Stage 2 review (ranked opps)       [lead]         ← blocked by T9

STAGE 3: CANDIDATE SELECTION AND OPTIMIZATION PLAN
T11:   Select optimization approach             [planner]      ← blocked by T10
T12:   Write optimization_plan.md               [planner]      ← blocked by T11
T13:   GATE: Stage 3 plan review                [lead]         ← blocked by T12

STAGE 4: IMPLEMENTATION
T14:   Implement optimization                   [implementer]  ← blocked by T13
T15:   GATE: compilation check                  [lead]         ← blocked by T14

STAGE 5: VALIDATION
T16:   Write correctness test script            [implementer]  ← blocked by T15
T16.5: Review and extend test script            [verifier]     ← blocked by T16
T17:   Run correctness tests (Gate 5.1)         [verifier]     ← blocked by T16.5
T18:   Run kernel perf benchmarks (Gate 5.2)    [verifier]     ← blocked by T15
T19:   Run E2E latency benchmarks (Gate 5.3)    [verifier]     ← blocked by T18
T20:   Evaluate kill criteria                   [verifier]     ← blocked by T17, T18, T19
T21:   Write validation_results.md              [verifier]     ← blocked by T17, T18, T19
T22:   GATE: verify_validation_gates.py         [lead]         ← blocked by T20, T21
```

**Parallelism windows**:
- T2 ‖ T3 (Stage 1 start)
- T16.5 ‖ T18 (Stage 5 start — T16.5 blocks T17, T18 only needs T15)
- T17 ‖ T19 (Stage 5 mid — T17 unblocked by T16.5, T19 unblocked by T18)

---

## TaskCreate Templates

### T1: Scaffold artifact directory
```
TaskCreate:
  subject: "Scaffold artifact directory for {component}-{model_short}"
  description: |
    Run `python .claude/skills/ammo/scripts/new_target.py --artifact-dir kernel_opt_artifacts/{component}_{model_short}_{hardware}_{dtype}_tp{tp} --model-id "{model_id}" --hardware "{hardware}" --dtype "{dtype}" --tp {tp}`.
    Verify artifact directory exists with state.json initialized.
    Update state.json with team info.
  activeForm: "Scaffolding artifact directory"
```
Owner: lead

### T2: Collect environment
```
TaskCreate:
  subject: "Collect environment info"
  description: |
    Run `python .claude/skills/ammo/scripts/collect_env.py {artifact_dir}`.
    Captures GPU info, CUDA version, vLLM version, Python environment for reproducibility.
    Output: {artifact_dir}/environment.json
  activeForm: "Collecting environment info"
```
Owner: verifier. Blocked by: T1.

### T3: Read vLLM source for target component
```
TaskCreate:
  subject: "Read vLLM source and extract target component semantics"
  description: |
    Find implementation of the target component in vLLM source code.
    Trace forward path through the target component.
    Document: input/output shapes, intermediate allocations, control flow,
    correctness invariants, numerical tolerance requirements.
    Use references/component-constraints-template.md as guide.
    Output: notes to be incorporated into constraints.md by the verifier.
    Send findings to verifier via SendMessage when done.
  activeForm: "Reading vLLM source code"
```
Owner: planner. Blocked by: T1.

### T4: Run nsys baseline profiling
```
TaskCreate:
  subject: "Run nsys baseline profiling"
  description: |
    Run nsys profile with production parity settings per SKILL.md Non-Negotiable #1.
    Read references/nsys-profiling-guide.md FIRST.
    Use fast Stage 1 bucket set: batch_size ∈ {8, 64}, input_len=1024, output_len=32.
    Verify: {artifact_dir}/runs/baseline.nsys-rep exists.
  activeForm: "Running nsys baseline profiling"
```
Owner: verifier. Blocked by: T2.

### T5: Extract dominant kernel timings
```
TaskCreate:
  subject: "Extract dominant kernel timings from nsys"
  description: |
    Run: nsys stats --report cuda_gpu_kern_sum {artifact_dir}/runs/baseline.nsys-rep
    Extract target component kernel times and top-K kernels by GPU time.
    Record ACTUAL NUMBERS (not commands) for each bucket.
    Also run E2E baseline: vllm bench latency for representative bucket.
    Compute component share: f = T_component / T_total.
  activeForm: "Extracting dominant kernel timings"
```
Owner: verifier. Blocked by: T4.

### T6: Write constraints.md
```
TaskCreate:
  subject: "Write constraints.md"
  description: |
    Write {artifact_dir}/constraints.md using references/component-constraints-template.md.
    Include:
    - Target envelope (model, hardware, dtype, TP)
    - Component semantics from planner (check SendMessage inbox or read shared notes)
    - Baseline Truth Snapshot with ACTUAL NUMBERS from profiling
    - "Already optimized?" checklist
  activeForm: "Writing constraints.md"
```
Owner: verifier. Blocked by: T3, T4, T5.

### T7: GATE: verify_phase1_baseline.py
```
TaskCreate:
  subject: "GATE: Run Stage 1 baseline verification"
  description: |
    Run: python .claude/skills/ammo/scripts/verify_phase1_baseline.py {artifact_dir}
    Exit code MUST be 0 to proceed.
    If non-zero: update state.json with "status": "blocked", create blocker file,
    send fix instructions to verifier.
    If pass: update state.json to stage "2_bottleneck_mining", status "pending".
    This is a LEAD-ONLY task — do not delegate.
  activeForm: "Running Stage 1 verification gate"
```
Owner: lead. Blocked by: T6.

### T8: Run bottleneck mining
```
TaskCreate:
  subject: "Run bottleneck mining (nsys analysis)"
  description: |
    Read {artifact_dir}/constraints.md and references/bottleneck-mining-guide.md.
    Export nsys trace to sqlite, extract top-K kernels by GPU time.
    Identify repeated kernel chains (fusion candidates).
    Map kernel names back to vLLM code paths.
    Compute feasibility bounds using references/fusion-feasibility-heuristics.md.
    Output: {artifact_dir}/bottleneck_analysis.md
  activeForm: "Running bottleneck mining"
```
Owner: verifier. Blocked by: T7.

### T9: Rank optimization opportunities
```
TaskCreate:
  subject: "Rank optimization opportunities"
  description: |
    Read {artifact_dir}/bottleneck_analysis.md.
    Rank opportunities by (Impact, Feasibility) score.
    Use references/optimization-techniques.md for technique catalog.
    Define 2-3 measurable kill criteria.
    Output: ranked opportunity list in bottleneck_analysis.md.
  activeForm: "Ranking optimization opportunities"
```
Owner: planner. Blocked by: T8.

### T10: GATE: Stage 2 review
```
TaskCreate:
  subject: "GATE: Stage 2 bottleneck review"
  description: |
    Review {artifact_dir}/bottleneck_analysis.md for completeness:
    - Top-K kernels identified with timing data
    - Opportunities ranked with evidence
    - Feasibility bounds computed
    If pass: update state.json to stage "3_optimization_plan", status "pending".
    This is a LEAD-ONLY task.
  activeForm: "Reviewing Stage 2 bottleneck analysis"
```
Owner: lead. Blocked by: T9.

### T11: Select optimization approach
```
TaskCreate:
  subject: "Select optimization approach"
  description: |
    Read {artifact_dir}/bottleneck_analysis.md and constraints.md.
    Select optimization approach based on profiling evidence.
    Document rationale with profiling evidence.
  activeForm: "Selecting optimization approach"
```
Owner: planner. Blocked by: T10.

### T12: Write optimization_plan.md
```
TaskCreate:
  subject: "Write optimization_plan.md"
  description: |
    Write {artifact_dir}/optimization_plan.md using references/optimization-plan-template.md.
    Must include: optimization approach + rationale, feasibility math,
    ranked top-10 opportunity list (section 3A) and 2-3 active hypotheses (section 3B).
    NO placeholders or empty sections.
  activeForm: "Writing optimization plan"
```
Owner: planner. Blocked by: T11.

### T13: GATE: Stage 3 plan review
```
TaskCreate:
  subject: "GATE: Stage 3 plan review"
  description: |
    Review {artifact_dir}/optimization_plan.md for completeness:
    - Optimization approach with profiling evidence
    - Feasibility math (required savings vs upper bound)
    - Kill criteria defined
    - No placeholders
    If pass: update state.json to stage "4_implementation", status "pending".
    This is a LEAD-ONLY task.
  activeForm: "Reviewing Stage 3 plan"
```
Owner: lead. Blocked by: T12.

### T14: Implement optimization
```
TaskCreate:
  subject: "Implement optimization per plan"
  description: |
    Read {artifact_dir}/optimization_plan.md for implementation plan.
    Use references/code-templates.md and references/tiling-config.md.
    Do NOT change plan mid-stage — if profiling indicates plan is wrong, STOP and notify lead.
    Output: kernel/code changes + {artifact_dir}/implementation_notes.md.
  activeForm: "Implementing optimization"
```
Owner: implementer. Blocked by: T13.

### T15: GATE: compilation check
```
TaskCreate:
  subject: "GATE: Compilation check"
  description: |
    Build the code changes.
    Verify clean compilation with no errors.
    If fails: send error details to implementer for fixing.
    If pass: unblocks validation tasks.
    This is a LEAD-ONLY task.
  activeForm: "Running compilation check"
```
Owner: lead. Blocked by: T14.

### T16: Write correctness test script
```
TaskCreate:
  subject: "Write correctness test script"
  description: |
    Write {artifact_dir}/benchmark_validation.py with:
    - Import the production vLLM kernel for the target component as baseline (NOT naive PyTorch loops)
    - torch.allclose() numerical comparison with tolerances from validation-defaults.md
    - Test ALL batch buckets: {batch_buckets}
    - Production parity environment (VLLM_TORCH_COMPILE_LEVEL=3, no TORCH_COMPILE_DISABLE)
    See SKILL.md Non-Negotiables #2, #5, #6 and validation/E2E_LATENCY_GUIDE.md.
  activeForm: "Writing correctness test script"
```
Owner: implementer. Blocked by: T15.

### T16.5: Review and extend correctness test script
```
TaskCreate:
  subject: "Review and extend correctness test script"
  description: |
    Read planner's acceptance criteria from {artifact_dir}/optimization_plan.md § Acceptance Criteria.
    Review implementer's test script ({artifact_dir}/benchmark_validation.py) for coverage gaps.
    Add adversarial test cases:
    - Edge batch sizes (BS=1, max batch)
    - Numerical edge cases (near-zero inputs, saturation values)
    - Verify test uses vLLM production kernel as baseline (NOT naive PyTorch)
    - Verify tolerances match validation-defaults.md
    Do NOT read implementation_notes.md — derive methodology from acceptance criteria only.
  activeForm: "Reviewing and extending test script"
```
Owner: verifier. Blocked by: T16.

### T17: Run correctness tests (Gate 5.1)
```
TaskCreate:
  subject: "Run correctness tests (Gate 5.1)"
  description: |
    Run {artifact_dir}/benchmark_validation.py correctness suite.
    Verify torch.allclose passes for ALL batch buckets.
    Record: max_abs_diff, tolerance used, pass/fail per bucket.
    If fails: STOP. Use investigation-playbook.md § Correctness. Notify lead.
  activeForm: "Running correctness tests"
```
Owner: verifier. Blocked by: T16.5.

### T18: Run kernel perf benchmarks (Gate 5.2)
```
TaskCreate:
  subject: "Run kernel perf benchmarks (Gate 5.2)"
  description: |
    Run kernel-level benchmarks under CUDA graphs (BLOCKING — Non-Negotiable #6).
    Both baseline and optimized MUST be captured in CUDA graphs.
    Time graph replays, NOT individual kernel launches.
    Baseline = vLLM's production kernel for the target component (NOT naive PyTorch).
    Gate: optimized ≤ baseline for ALL buckets.
    Run NCU to verify occupancy.
    If fails: STOP. Use investigation-playbook.md § Kernel Perf. Notify lead.
  activeForm: "Running kernel perf benchmarks"
```
Owner: verifier. Blocked by: T15.

### T19: Run E2E latency benchmarks (Gate 5.3)
```
TaskCreate:
  subject: "Run E2E latency benchmarks (Gate 5.3)"
  description: |
    GPU SEQUENCING: This task is blocked by T18 to prevent GPU contention.
    Use `scripts/run_vllm_bench_latency_sweep.py` for all E2E measurements.

    Run E2E latency sweep with production settings (CUDA graphs + torch.compile).
    Both baseline and optimized must use same environment.
    Calculate improvement percentage with actual numbers.
    See validation/E2E_LATENCY_GUIDE.md for methodology.
    If fails or regresses: notify lead with data.
  activeForm: "Running E2E latency benchmarks"
```
Owner: verifier. Blocked by: T18.

### T20: Evaluate kill criteria
```
TaskCreate:
  subject: "Evaluate kill criteria from optimization plan"
  description: |
    Read {artifact_dir}/optimization_plan.md kill criteria section.
    For EVERY kill criterion: evaluate as PASS or FAIL with measured data.
    "TODO", "optional", "skip" are NOT valid results.
    Use results from T17 (correctness), T18 (kernel perf), T19 (E2E).
    Use references/e2e-delta-math.md to bound expectations mathematically.
    Update state.json at `route_decision.kill_criteria_results` (exact path required).
    If any FAIL: recommend action to lead (pivot approach, restrict envelope, etc.).
  activeForm: "Evaluating kill criteria"
```
Owner: verifier. Blocked by: T17, T18, T19.

### T21: Write validation_results.md
```
TaskCreate:
  subject: "Write validation_results.md"
  description: |
    Write {artifact_dir}/validation_results.md with:
    - Correctness: pass/fail, max diff, tolerance used (from T17)
    - Kernel perf: per-bucket timings vs vLLM baseline under CUDA graphs (from T18)
    - E2E: latency comparison, improvement % (from T19)
    All data must be ACTUAL NUMBERS, not estimates.
  activeForm: "Writing validation results"
```
Owner: verifier. Blocked by: T17, T18, T19.

### T22: GATE: verify_validation_gates.py
```
TaskCreate:
  subject: "GATE: Run verify_validation_gates.py"
  description: |
    Run: python .claude/skills/ammo/scripts/verify_validation_gates.py {artifact_dir}
    Exit code MUST be 0 to complete.
    Record in state.json: verification_run.validation with status and gate results.
    If non-zero: update state.json with "status": "blocked", create blocker file.
    If pass: update state.json to stage "complete", status "shipped".
    This is a LEAD-ONLY task.
  activeForm: "Running validation verification gate"
```
Owner: lead. Blocked by: T20, T21.

---

## Conditional Task Creation

Some tasks are only created when a gate fails:

### Investigation task (on gate failure)
```
TaskCreate:
  subject: "Investigate {gate_name} failure"
  description: |
    Gate {gate_name} failed with: {failure_details}.
    Use investigation-playbook.md § {relevant_section}.
    Bounded to 3 hypothesis cycles.
    Report findings to lead via SendMessage.
  activeForm: "Investigating {gate_name} failure"
```

### Re-plan task (on kill criteria failure — single-cycle fix)
```
TaskCreate:
  subject: "Re-plan: approach pivot after kill criteria failure"
  description: |
    Kill criterion "{criterion}" failed: {details}.
    Re-read constraints.md and bottleneck analysis.
    Propose alternative approach or envelope restriction.
    Write updated optimization_plan.md.
  activeForm: "Re-planning after kill criteria failure"
```

---

## Opportunity Iteration Loop (Post-T22)

When T22 results in a KILL decision, the lead creates iteration loop tasks to pivot to the next ranked opportunity. This loop repeats until an opportunity is SHIPPED or `max_attempts` is exhausted.

### Iteration Loop Overview

```
T22 KILL → T23 → T24 → T25 → T26+ (fresh T14-T22 chain)
                                  ↓
                           T22' KILL → T23' → T24' → T25' → T26'+ ...
                                  ↓
                           T22'' KILL + at limit → EXHAUSTED → shutdown
```

### T23: Record KILL and select next opportunity
```
TaskCreate:
  subject: "Record KILL for {opportunity_id} and select next opportunity (attempt {N})"
  description: |
    T22 resulted in KILL for {opportunity_id}.

    1. Record the KILL in state.json:
       - Append to `opportunity_attempts` array:
         {"attempt": {N}, "opportunity_id": "{opportunity_id}", "status": "KILLED",
          "kill_criteria_results": {results_from_T20}, "kill_reason": "{reason}", "date": "{today}"}
       - Clear `route_decision` for next attempt
       - Check: len(opportunity_attempts) vs max_attempts

    2. If at limit (len >= max_attempts): declare EXHAUSTED, update state.json
       with status "exhausted", broadcast to team, begin shutdown. STOP here.

    3. If under limit: read {artifact_dir}/bottleneck_analysis.md ranked list.
       Select the next untried opportunity (skip IDs already in opportunity_attempts).
       Update state.json: current_opportunity_id = new ID, stage = "3_optimization_plan".

    4. Send selected opportunity to planner via SendMessage.
  activeForm: "Recording KILL and selecting next opportunity"
```
Owner: lead. Blocked by: T22.

### T24: Write updated optimization_plan.md (iteration)
```
TaskCreate:
  subject: "Write optimization_plan.md for {new_opportunity_id} (attempt {N})"
  description: |
    Previous attempt ({prev_opportunity_id}) was KILLED: {kill_reason}.

    Write updated {artifact_dir}/optimization_plan.md:
    - Include "0A) Previous Attempts" section listing ALL prior attempts with kill reasons
      (use references/optimization-plan-template.md § 0A as guide)
    - Select approach for {new_opportunity_id} from bottleneck_analysis.md
    - Explicitly address why this opportunity avoids the failure mode of previous attempts
    - All other sections per optimization-plan-template.md

    HARD RULE: Do NOT skip feasibility math or kill criteria just because this is an iteration.
  activeForm: "Writing optimization plan for next opportunity"
```
Owner: planner. Blocked by: T23.

### T25: GATE: Stage 3 plan review (iteration)
```
TaskCreate:
  subject: "GATE: Stage 3 plan review (iteration {N} — {new_opportunity_id})"
  description: |
    Review {artifact_dir}/optimization_plan.md for iteration {N}:
    - Previous Attempts section documents prior KILLs with evidence
    - New approach addresses different bottleneck than killed attempts
    - Feasibility math is complete (not recycled from prior attempt)
    - Kill criteria are specific to the new approach
    If pass: update state.json to stage "4_implementation", status "pending".
    This is a LEAD-ONLY task.
  activeForm: "Reviewing Stage 3 plan (iteration)"
```
Owner: lead. Blocked by: T24.

### T26+: Fresh T14-T22 chain
After T25 passes, the lead creates fresh copies of T14 through T22 for the new opportunity. These follow the same templates as the original T14-T22 but with updated task subjects indicating the iteration number (e.g., "Implement optimization (attempt 2 — OP-002)").

The new T22 feeds back into the iteration loop: if KILL, create T23' → T24' → T25' → T26'+.

---

## GPU Sequencing Rules

GPU benchmark tasks (T18 kernel perf, T19 E2E latency) MUST run sequentially to prevent
GPU memory contention. Text-based HARD RULES in agent briefings are insufficient for
enforcement — the only reliable mechanism is Claude Code's `blockedBy` task dependency system.

### Why This Exists

During the OLMo-3-7B-Instruct verification run on L40S, multiple agents launched concurrent
`vllm bench latency` processes on the same GPU, causing:
- OOM errors (3+ vllm processes competing for 44 GiB GPU memory)
- Unreliable benchmark results (GPU contention inflated latencies from 1.37s to 2.48s)
- Broadcast STOP messages were ignored; shutdown requests were slow to take effect

### Rules

1. **First iteration**: T19 is blocked by T18 (not T15). This is set in the dependency
   graph above and in the T19 TaskCreate template.

2. **Iteration loop**: When the lead creates fresh T14'-T22' chains after T25 passes,
   T19' MUST be blocked by T18' (not T15'). The lead must set this explicitly when
   creating iteration tasks.

3. **Cross-iteration sequencing**: Normally safe because T22 gates depend on both T18
   and T19, so a new iteration's T18' cannot start until the previous T22 completes.

4. **Sweep script required**: All E2E measurements MUST use
   `scripts/run_vllm_bench_latency_sweep.py`, which holds a system-wide GPU lock in
   `/tmp/ammo_gpu_locks/`. Do NOT run `vllm bench latency` directly.

---

## Iteration Termination Rules

| Condition | Action |
|-----------|--------|
| T22 PASS (SHIP) | Update state.json to `"status": "shipped"`, shut down team |
| T22 KILL + `len(opportunity_attempts) < max_attempts` | Create T23-T26+, pivot to next opportunity |
| T22 KILL + `len(opportunity_attempts) >= max_attempts` | Declare EXHAUSTED, update state.json to `"status": "exhausted"`, shut down team |
| T22 BLOCKED | Follow blocker protocol (investigation task, escalation) |

---

## Stage Transition Rules

The lead updates state.json at these transition points:

| Transition | Trigger Task | state.json Update |
|-----------|-------------|-------------------|
| → Stage 2 | T7 passes | `"stage": "2_bottleneck_mining", "status": "pending"` |
| → Stage 3 | T10 passes | `"stage": "3_optimization_plan", "status": "pending"` |
| → Stage 4 | T13 passes | `"stage": "4_implementation", "status": "pending"` |
| → Stage 5 | T15 passes | `"stage": "5_validation", "status": "pending"` |
| → Complete | T22 passes | `"stage": "complete", "status": "shipped"` |
| → Stage 3 (loop) | T22 KILL + under limit | `"stage": "3_optimization_plan", "status": "pending"`, append to `opportunity_attempts` |
| → Exhausted | T22 KILL + at limit | `"stage": "exhausted", "status": "exhausted"` |
