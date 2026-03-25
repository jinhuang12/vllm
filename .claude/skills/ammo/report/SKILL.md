---
name: ammo-report
description: >
  Generate a detailed GPU kernel optimization report after an AMMO campaign completes.
  Reads campaign artifacts (profiling data, candidate evaluations, implementation results)
  and produces a reader-friendly technical report with matplotlib charts, code listings,
  and lessons learned. Use this skill whenever the user asks to generate a report, summarize
  results, create a write-up of optimization findings, or document what happened during an
  AMMO run — even if they just say "write up the results" or "make a report from the artifacts."
  Invoke after Stage 7 (campaign_complete/campaign_exhausted) or manually on any completed
  campaign artifact directory.
---

# AMMO Report Generator

Produce a publication-quality optimization report from a completed AMMO campaign. The report
explains the profiling methodology, bottleneck analysis, optimization approach, results, and
lessons learned — written for engineers who deploy vLLM in production and have **no familiarity
with AMMO's internal workflow**.

The report needs to stand on its own: a reader should be able to understand the full optimization
story without knowing what AMMO is, what "champions" are, or what "Stage 3" means.

## When to Use

- Automatically at the end of an AMMO campaign (Stage 7 `campaign_complete` or `campaign_exhausted`)
- Manually via `/ammo-report <artifact-dir>` on any completed campaign

## Inputs

The skill reads from the campaign's artifact directory:

| File | Purpose |
|------|---------|
| `state.json` | Campaign metadata, round results, shipped optimizations |
| `constraints.md` | Model architecture, hardware specs, baseline latency |
| `bottleneck_analysis.md` | Kernel breakdown, BW utilization, per-GEMM timings |
| `debate/summary.md` | Candidate evaluation and selection rationale |
| `debate/proposals/*.md` | Optimization proposals with micro-experiment data |
| `tracks/*/validation_results.md` | Per-candidate correctness, kernel, and E2E results |
| `e2e_latency/json/*.json` | Raw baseline benchmark JSON files |
| `nsys/*.csv` | Nsys kernel summary data (if exported) |

### Key JSON Paths in state.json

These are the specific fields you need to extract. Don't guess — use these paths:

```
Target info:       $.target.{model_id, hardware, dtype, tp}
Shipped ops:       $.campaign.shipped_optimizations        (list of OP-IDs like ["OP-003"])
Cumulative gain:   $.campaign.cumulative_e2e_speedup       (e.g., 1.06)
Candidate names:   $.debate.candidates[*].{id, name, score, status}
Round results:     $.campaign.rounds[*].implementation_results.{OP-ID}.{status, e2e_speedup, reason}
Worktree paths:    $.parallel_tracks.{OP-ID}.worktree_path (location of shipped code)
Selection logic:   $.debate.selection_rationale
```

**Important**: Candidate names may evolve between evaluation and implementation (e.g., a secondary
contribution like weight fusion gets added during implementation). Use the name from
`tracks/{OP-ID}/validation_results.md` title for the final name.

### Key Sections in Markdown Artifacts

| File | Section Header | What to Extract |
|------|---------------|-----------------|
| `bottleneck_analysis.md` | `## Executive Summary` | f_decode, ceiling, Amdahl bound per bottleneck |
| `bottleneck_analysis.md` | `### Per-GEMM breakdown` | Per-GEMM BW utilization table |
| `bottleneck_analysis.md` | `### BS=1 vs BS=8 comparison` | Key insight driving optimization decisions |
| `constraints.md` | `## Baseline E2E Latency` | Latency table for all batch sizes |
| `constraints.md` | `## Baseline Truth Snapshot` | Top-15 kernel breakdown, nsys profile paths |
| `constraints.md` | `## Per-decode-step kernel sequence` | Layer-by-layer kernel timeline (code block format) |
| `debate/summary.md` | `## Decisions` | Why each candidate was selected/eliminated |
| `debate/summary.md` | `## Conceded Weaknesses` | Original projections (for "met expectations" analysis) |
| `tracks/{OP-ID}/validation_results.md` | `## Gate 5.3` | E2E comparison tables (vs Stage 1 AND in-sweep) |

### Finding E2E Results

Optimized E2E results are in directories named `e2e_latency_{opid}/` (with possible version
suffixes like `_v2`, `_v3`). The OP-ID has hyphens stripped in directory names (e.g., `OP-003`
becomes `e2e_latency_op003_v3`). Use the **highest version number** for the final results.

The canonical comparison tables are in `tracks/{OP-ID}/validation_results.md` — prefer those
over raw JSON when both exist.

## Output

Writes to `{artifact_dir}/REPORT.md` with embedded chart references, plus PNG charts in
`{artifact_dir}/report_assets/`. Include a Table of Contents with section links.

## Terminology Translation

The report is for engineers who have never heard of AMMO. Translate ALL internal terms:

| AMMO Internal Term | Report Language |
|--------------------|----------------|
| Champion | Optimization candidate / approach / proponent |
| Debate / adversarial debate | Candidate evaluation and comparison / structured peer review |
| Stage 1-2 | Profiling and bottleneck analysis |
| Stage 3 | Candidate evaluation phase |
| Stage 4-5 | Implementation and validation |
| Stage 6-7 | Integration and final evaluation |
| Campaign | Optimization effort / study |
| Round | Optimization iteration |
| f_decode | Component share of decode latency (define on first use) |
| Ship / shipped | Merged / accepted / deployed |
| Minimum E2E improvement threshold | Campaign-wide threshold for optimization viability (min_e2e_improvement_pct, default 1%) |
| Stage 1 baseline | Original baseline measurement (do not re-run) |
| Co-located baseline | Same-session baseline comparison |
| Gate 5.1 / 5.2 / 5.3 | Correctness / kernel performance / E2E validation |
| Worktree path | Development working copy / branch |
| GATED_PASS | Optimization accepted with batch-size dispatch gate — active only for BS where it improves performance |
| Crossover probing | Kernel-informed threshold determination — finding the exact batch size where the optimization transitions from beneficial to harmful |

When discussing the evaluation process, describe it as: "We evaluated multiple optimization
approaches through structured analysis, micro-experiments, and peer review" — not "adversarial
debate between champion agents."

## Report Structure

Use this fixed template. Each section exists because readers need it — the executive summary
must stand alone (many readers stop there), the bottleneck analysis needs charts because visual
presentation of a 88%+ GEMM dominance is far more compelling than text, and the lessons section
exists because methodology pitfalls recur across optimization efforts.

Every section is required unless marked optional.

### 1. Executive Summary (~500 words)
- Target: model, hardware, dtype, TP
- Brief methodology overview (1-2 sentences explaining the pipeline: profile → analyze → evaluate candidates → implement → validate)
- Key finding: what bottleneck dominates and why
- What shipped: technique, E2E improvement, key numbers
- What didn't ship and why (brief)
- Remaining opportunities (brief)

### 2. Model & Hardware Context
- Model architecture (layer types, dimensions, unique features like hybrid attention)
- Hardware specs table (GPU, BW, TFLOPS, SM count, L2, VRAM)
- Per-decode-step compute profile (what happens in one forward pass)
- Memory-bound vs compute-bound regime analysis

### 3. Profiling Methodology
- Production parity requirements (CUDA graphs, torch.compile) — explain WHY this matters (kernel behavior under graph replay differs from eager)
- Nsys profiling setup (flags, capture mode, why `--cuda-graph-trace=node` matters)
- E2E benchmark methodology (sweep script, workload, iteration count)
- Baseline E2E latency table (all batch sizes)

### 4. Bottleneck Analysis
- **Chart**: Kernel time breakdown pie chart
- **Chart**: Synthetic nsys timeline (one decode step)
- **Chart**: Per-component BW utilization bar chart
- **Chart**: Roofline plot
- Key datapoints that drove optimization decisions (e.g., "BS=8 slower than BS=1 despite same weight data")
- Per-GEMM or per-component table with measured times and utilization

### 5. Optimization Approaches Evaluated
- **Diagram** (mermaid): Decision flow (candidates → evaluation → selection)
- Table: all candidates with target, technique, projected impact
- For each selected candidate: why it was chosen over alternatives, with specific evidence
- Methodology pitfalls caught during evaluation (as standalone callout boxes)

### 6. Implementation & Results
- What shipped: files modified, LOC, technique description, enablement mechanism (env var / flag)
- **Chart**: E2E results (baseline vs optimized per batch size)
- Kernel-level results table (per-shape speedup, BW utilization)
- **E2E baseline source**: ALWAYS use the original baseline numbers from `e2e_latency/json/baseline_bs*.json`, captured before any optimization. These are the campaign's official reference point.
- **Multi-component decomposition**: When an optimization has multiple parts (e.g., new kernel + weight fusion), break down each part's contribution. The in-sweep comparison (same worktree, env var toggle) isolates the kernel contribution alone; the Stage 1 comparison captures the full effect including structural changes. Explain why E2E improvement may exceed or differ from kernel-level speedup.
- Did results meet projections? Original projections are in `debate/summary.md` under "Conceded Weaknesses" (revised estimates post-debate). Compare projected vs actual, enumerate specific reasons for any gap, and frame honestly.
- For batch sizes outside the optimization's active range, explain why (e.g., "M=1 bypassed because cuBLAS GEMV is already optimal at 70.6% BW utilization")
- For `GATED_PASS` optimizations, include both pre-gating and post-gating E2E tables. Show which batch sizes benefit and which are gated off. Explain the dispatch mechanism in reader-friendly language (e.g., "This optimization activates only for batch sizes <= 16 via runtime dispatch").
- Rollback instructions (disable the env var or revert the code)

### 7. What Failed (optional — include if any candidate was IMPLEMENTED but failed validation)
- For each failed candidate: what it was, kernel-level results, why E2E didn't materialize
- Lessons from the failure (e.g., kernel-to-E2E translation gap)
- Include `GATED_PASS` tracks that failed during gating implementation (crossover probing inconclusive or dispatch mechanism broke torch.compile). Explain why gating was attempted and why it failed.
- Note: candidates eliminated during evaluation (before implementation) belong in section 5, not here

### 8. Remaining Opportunities
- Table: untried bottlenecks with estimated impact
- Why the optimization effort concluded (mechanical stop threshold: `f < min_e2e_improvement_pct`)
- Brief description of most promising remaining candidates

### 9. Key Lessons (standalone callout boxes)
- 3-5 methodology lessons as `> **Lesson N: Title**` blockquote callouts
- Each lesson: what happened, why it matters, how to avoid it
- Focus on things that would save other engineers time

### 10. Appendix
- **Full code listings**: Read from the worktree at `$.parallel_tracks.{OP-ID}.worktree_path` in state.json. Find the file list in `tracks/{OP-ID}/validation_results.md` under "Files modified" or "Scope Adherence". Include complete file contents (not diffs) with inline comments explaining key design decisions.
- **Reproduction commands**: Include cd to worktree path + activate venv, environment variable for enabling the optimization, full `vllm bench latency` command with all flags, and how to disable/revert
- Future work (brief)

## Chart Generation

There is no static chart generation script. You write a small bespoke Python script for each
chart, tailored to whatever data is actually available in this campaign's artifacts. This
approach works for any model and any profiling setup because you — the agent — understand the
data and can adapt.

### Workflow

For each chart:
1. Inspect the available artifacts (see Data Sources below)
2. Decide the best data source for this specific chart
3. Write a minimal Python script (~50-100 lines) that extracts the data and renders the chart
4. Save the script to `{artifact_dir}/report_assets/` alongside the PNG (for reproducibility)
5. Run the script, verify the PNG looks correct

### Required Charts

Produce these 5 PNGs in `{artifact_dir}/report_assets/`:

| PNG filename | Purpose |
|-------------|---------|
| `kernel_breakdown_pie.png` | GPU time breakdown by kernel category (GEMM, attention, normalization, etc.) |
| `bw_utilization_bar.png` | Per-GEMM HBM bandwidth utilization showing how close each operation is to peak |
| `e2e_results_bar.png` | Before/after E2E latency comparison across batch sizes, with improvement % |
| `roofline_plot.png` | Arithmetic intensity vs throughput for decode GEMM operations |
| `nsys_timeline_synthetic.png` | Kernel execution sequence for one decode step showing relative durations |

### Data Sources

Pick the best source for each chart based on what's available. The artifacts directory may
contain any combination of these:

**Structured (prefer these when available):**
- `e2e_latency_*/nsys/baseline_profile.sqlite` — nsys sqlite export with per-kernel timing
- `e2e_latency/json/baseline_bs*.json` — E2E benchmark results as JSON
- `state.json` — campaign metadata, shipped optimizations, target config
- `target.json` — workload config (batch sizes, model ID, hardware)

**Semi-structured (agent-written markdown from Stage 2):**
- `bottleneck_analysis.md` — kernel breakdown, GEMM shapes, BW utilization analysis
- `constraints.md` — model architecture, hardware specs, baseline latency tables

The nsys sqlite is the most accurate source for kernel timing data. Its schema:

```
CUPTI_ACTIVITY_KIND_KERNEL:
  start, end          — kernel timestamps in nanoseconds
  demangledName       — INTEGER foreign key into StringIds(id, value)
  gridX, gridY, gridZ — grid dimensions
  graphId             — CUDA graph ID (non-null when CUDA graphs are used)
  gridId              — unique node ID within a CUDA graph

StringIds:
  id    — INTEGER PRIMARY KEY
  value — TEXT (demangled kernel name)
```

To query kernel names: `JOIN StringIds s ON k.demangledName = s.id`, then use `s.value`.
To find the sqlite: look for `e2e_latency_20*Z/nsys/baseline_profile.sqlite` (timestamp-
named directories are the baseline; avoid `e2e_latency_op*/` which are optimization runs).

### Script Guidelines

- Use `matplotlib` for plotting. Import check: `try: import matplotlib...`
- Each script should be self-contained and runnable independently
- Read data, compute what's needed, render one chart, save PNG. No fallback logic — you
  already know what data is available before writing the script
- Name scripts descriptively: `gen_kernel_pie.py`, `gen_bw_bar.py`, etc.
- If a data source doesn't exist for a chart, note it in the report and skip that chart

## Callout Box Format

Use GitHub-flavored blockquote admonitions for lessons:

```markdown
> **Lesson 1: Descriptive Title**
>
> What happened, why it matters, and how to avoid it in the future.
> Include specific numbers and measured data where possible.
```

## Diagrams

Diagrams (decision flows, architecture overviews) must be rendered as PNG images, not
inline mermaid code blocks. Mermaid only renders in specific viewers — the report should
work in any markdown renderer.

Approach: write a small Python script using matplotlib to draw the diagram (boxes, arrows,
labels) and save it as a PNG in `report_assets/`. Reference the image in the report with
`![Decision Flow](report_assets/decision_flow.png)`.

For simple flows, `matplotlib.patches.FancyBboxPatch` + `matplotlib.patches.FancyArrowPatch`
work well. Keep the visual style consistent with the other charts (same font, dark backgrounds
optional).

## Quality Checklist

Before finalizing the report, verify:

- [ ] No AMMO-specific jargon (champion, debate, campaign, stage N, gate 5.x, etc.)
- [ ] All numbers cite their source (nsys trace, benchmark JSON, roofline calculation)
- [ ] Charts have descriptive titles, axis labels, and legends
- [ ] E2E results compare against original baseline (not re-run baselines)
- [ ] Multi-component optimizations have per-component contribution breakdowns
- [ ] Code listings are complete and runnable
- [ ] Reproduction commands include all environment variables, flags, and worktree path
- [ ] Lessons are actionable (not just "we learned X" but "do Y to avoid X")
- [ ] Executive summary can stand alone (someone reading only that section gets the key story)
- [ ] Table of Contents with section links is present
- [ ] "f_decode" and other technical terms are defined on first use
