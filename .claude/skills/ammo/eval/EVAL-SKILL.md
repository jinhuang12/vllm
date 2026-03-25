---
name: ammo-eval
description: Post-mortem evaluation for AMMO GPU kernel optimization campaigns. Scores completed campaign artifacts on a 5-dimension scorecard (E2E speedup, gates, debate quality, efficiency, transcript quality), archives versioned results, and generates a trend dashboard. Use after an /ammo campaign finishes to measure skill performance. Triggers on requests to evaluate, score, or measure ammo skill performance.
---

# AMMO Eval — Post-Mortem Campaign Evaluation

Score a completed AMMO campaign, archive the result, and generate a performance dashboard.

## When to Use

After an `/ammo` campaign completes (or is interrupted with results), invoke this skill to:
1. Parse all campaign artifacts into structured metrics
2. Score the run on a 5-dimension scorecard
3. Optionally grade transcript quality via LLM agent
4. Archive the scored run into a versioned repository
5. Rebuild aggregates and generate an HTML dashboard

## Arguments

The user provides:
- `--session-id` (required): Claude Code session UUID for the campaign. This is the only required argument — everything else can be auto-detected.
- `--artifact-dir` (auto-detected): Path to the campaign's artifact directory. **Auto-detection**: scan `kernel_opt_artifacts/*/state.json` for a matching `session_id` field. If not found, scan the session JSONL for file writes to `kernel_opt_artifacts/` and infer the directory from the path.
- `--description` (optional, default: auto-generated from target info): Human-readable description of what skill changes were made
- `--skip-transcript-grading` (optional): Skip the LLM grader for speed
- `--skip-deep-analysis` (optional): Skip the causal LLM deep dive for speed
- `--repository` (optional, default `~/.claude/ammo-eval`): Eval repository path

### Minimal Invocation

The user can simply say: `Run /ammo-eval for session ba11c39a-...`

The orchestrator resolves everything:
1. Locate JSONL at `~/.claude/projects/-home-jinhun-vllm/<session-id>.jsonl`
2. Find artifact dir by scanning `kernel_opt_artifacts/*/state.json` for matching session_id
3. If no match in state.json, grep the JSONL for file writes to `kernel_opt_artifacts/` and extract the directory
4. Auto-generate description from `target.json` (model + hardware + dtype)
5. Run the full pipeline (Steps 0-8) including causal analysis unless `--skip-*` flags are given

## Pipeline

Run these scripts in sequence from `.claude/skills/ammo/eval/scripts/`:

### Step 0: Parse Session Logs
```bash
python .claude/skills/ammo/eval/scripts/parse_session_logs.py \
  --session-id <SESSION_ID> \
  --artifact-dir <ARTIFACT_DIR> \
  --output /tmp/ammo_eval_session_data.json
```

Extracts ground-truth timing and token cost data from the session JSONL. This replaces manual `stage_timestamps` and `agent_costs` tracking in state.json — no manual recording by the lead is needed.

### Step 1: Parse Artifacts
```bash
python .claude/skills/ammo/eval/scripts/parse_artifacts.py \
  --artifact-dir <ARTIFACT_DIR> \
  --session-data /tmp/ammo_eval_session_data.json \
  --output /tmp/ammo_eval_snapshot.json
```

The `--session-data` flag provides session-log-derived timing and cost data. When omitted, falls back to state.json (backward compatible).

### Step 2: Snapshot Changes
```bash
python .claude/skills/ammo/eval/scripts/snapshot_changes.py \
  --artifact-dir <ARTIFACT_DIR> \
  --output /tmp/ammo_eval_changes_snapshot
```

This captures a full record of all code produced during the campaign:
- Git diffs (patches) from every campaign worktree listed in `state.json`'s `parallel_tracks`
- Session-member worktrees that shared the same session ID
- A complete copy of the artifact directory (minus large binary profiling files)
- Skill diff from the main branch
- A manifest summarizing worktree metadata, merge status, and changed files

The output directory is passed to `archive_run.py` in Step 5.

### Step 3: Score Campaign
```bash
python .claude/skills/ammo/eval/scripts/score_campaign.py \
  --snapshot /tmp/ammo_eval_snapshot.json \
  --output /tmp/ammo_eval_scorecard.json \
  --report /tmp/ammo_eval_report.md
```

### Step 3a: Create Eval Team

Before spawning LLM agents, create a team that will persist for follow-up questions:

```
TeamCreate:
  team_name: ammo-eval-{SESSION_ID_SHORT}
  description: "AMMO eval agents for session {SESSION_ID}"
```

This team hosts the causal analyzer and transcript grader. Do NOT shut down the team after the eval completes — it remains alive so the user can ask follow-up questions to either agent.

### Step 3b: Extract Events + Build Causal DAG

```bash
python .claude/skills/ammo/eval/causal/extract_events.py \
  --session-jsonl ~/.claude/projects/-home-jinhun-vllm/<SESSION_ID>.jsonl \
  --session-data /tmp/ammo_eval_session_data.json \
  --output /tmp/ammo_eval_events.json

python .claude/skills/ammo/eval/causal/build_dag.py \
  --events /tmp/ammo_eval_events.json \
  --artifact-dir <ARTIFACT_DIR> \
  --output /tmp/ammo_eval_causal_dag.json
```

### Step 3c: Score Nodes + Detect Anomalies

```bash
python .claude/skills/ammo/eval/causal/score_nodes.py \
  --dag /tmp/ammo_eval_causal_dag.json \
  --events /tmp/ammo_eval_events.json \
  --snapshot /tmp/ammo_eval_snapshot.json \
  --output /tmp/ammo_eval_scored_dag.json \
  --anomalies /tmp/ammo_eval_anomalies.json
```

### Step 3d: LLM Deep Dive (Optional)

If `--skip-deep-analysis` is NOT set, spawn as a team member:

```
Spawn Agent (general-purpose type) with:
  team_name: ammo-eval-{SESSION_ID_SHORT}
  name: causal-analyzer
  prompt: Read the causal analyzer rubric at .claude/skills/ammo/eval/causal/agents/causal_analyzer.md,
          then analyze anomalies at /tmp/ammo_eval_anomalies.json
          with events at /tmp/ammo_eval_events.json
          and artifacts at <ARTIFACT_DIR>.
          Write deep_analysis.json to /tmp/ammo_eval_deep_analysis.json.
```

### Step 3e: Generate Post-Mortem

```bash
python .claude/skills/ammo/eval/causal/generate_postmortem.py \
  --scored-dag /tmp/ammo_eval_scored_dag.json \
  --deep-analysis /tmp/ammo_eval_deep_analysis.json \
  --events /tmp/ammo_eval_events.json \
  --output-dag /tmp/ammo_eval_causal_dag_final.json \
  --output-narrative /tmp/ammo_eval_postmortem.md \
  --output-viz /tmp/ammo_eval_causal_viz.html
```

### Step 3f: Cross-Version Diff (if previous version exists in archive)

```bash
python .claude/skills/ammo/eval/causal/diff_versions.py \
  --current-dag /tmp/ammo_eval_causal_dag_final.json \
  --previous-dag ~/.claude/ammo-eval/versions/<prev>/runs/<target>/run_1/causal_dag.json \
  --output /tmp/ammo_eval_version_diff.json \
  [--deep --regression-report /tmp/ammo_eval_regression_report.md]
```

### Step 4: Transcript Grading (Optional)

If `--skip-transcript-grading` is NOT set, spawn as a team member:

```
Spawn Agent (general-purpose type) with:
  team_name: ammo-eval-{SESSION_ID_SHORT}
  name: transcript-grader
  prompt: Read the grader rubric at .claude/skills/ammo/eval/agents/transcript_grader.md,
          then evaluate the campaign artifacts at <ARTIFACT_DIR>.
          Write transcript_grading.json to /tmp/ammo_eval_transcript_grading.json.
```

Steps 3d and 4 can run in parallel since they are independent.

Then re-score with transcript quality:
```bash
python .claude/skills/ammo/eval/scripts/score_campaign.py \
  --snapshot /tmp/ammo_eval_snapshot.json \
  --output /tmp/ammo_eval_scorecard.json \
  --report /tmp/ammo_eval_report.md \
  --enrich-from /tmp/ammo_eval_transcript_grading.json
```

### Step 5: Archive
```bash
python .claude/skills/ammo/eval/scripts/archive_run.py \
  --scorecard /tmp/ammo_eval_scorecard.json \
  --snapshot /tmp/ammo_eval_snapshot.json \
  --description "<DESCRIPTION>" \
  --transcript-grading /tmp/ammo_eval_transcript_grading.json \
  --changes-snapshot /tmp/ammo_eval_changes_snapshot \
  --causal-dag /tmp/ammo_eval_causal_dag_final.json \
  --postmortem-narrative /tmp/ammo_eval_postmortem.md \
  --causal-viz /tmp/ammo_eval_causal_viz.html
```

The `--changes-snapshot` flag stores the full changes snapshot (worktree patches, artifact copies, manifest) alongside the scored run in the archive. This ensures you can always reconstruct what code was produced, even after cleanup.

The `--causal-dag`, `--postmortem-narrative`, and `--causal-viz` flags are optional. When provided, these causal engine outputs are stored alongside the scorecard in the archive run directory.

### Step 6: Aggregate & Dashboard
```bash
python .claude/skills/ammo/eval/scripts/aggregate_versions.py \
  --repository ~/.claude/ammo-eval

python .claude/skills/ammo/eval/scripts/generate_dashboard.py \
  --repository ~/.claude/ammo-eval
```

### Step 7: Present Results

1. Read and display `/tmp/ammo_eval_report.md` to the user
2. Tell the user: "Dashboard generated at `~/.claude/ammo-eval/dashboard.html`"
3. If there are previous versions in the repository, highlight the delta vs the most recent previous version
4. **Auto-rank against prior runs**: Query the eval repository (`index.json`) for all prior runs with the same target slug. If any exist, display a comparative ranking:
   ```
   This run's +X% E2E improvement ranks Nth of M runs for {target_slug}.
   Best: +Y% ({version_id}) | Median: +Z% | Worst: +W% ({version_id})
   ```
   Include the full ranked table if 3+ prior runs exist.
5. **Auto-offer deeper comparison**: After presenting results, tell the user: "There are N prior runs for this target. Want me to dig deeper into what drove the differences?" This is an open-ended offer — the orchestrator uses judgment on what to investigate based on user questions. The eval team (causal-analyzer, transcript-grader) remains alive to assist.

### Step 8: Cleanup

After presenting results, ask the user whether to clean up campaign artifacts. Use `AskUserQuestion` with a single multi-select question:

**Question**: "The eval run has been archived with a full snapshot. Which artifacts should I clean up?"

**Options**:
1. **Campaign worktrees** — Remove all worktrees listed in the changes snapshot manifest via `git worktree remove <path> --force` and delete their tracking branches
2. **Artifact directory** — Remove the `kernel_opt_artifacts/<campaign_dir>/` directory
3. **Temp files** — Remove `/tmp/ammo_eval_*.json`, `/tmp/ammo_eval_*.md`, and `/tmp/ammo_eval_changes_snapshot/`
4. **Eval team** — Shut down the eval team (causal-analyzer and transcript-grader)

If the user selects nothing (or cancels), skip cleanup entirely. For each selected category, execute the cleanup and report what was removed.

**Team persistence**: Do NOT shut down the eval team unless the user explicitly selects option 4. The team remains alive for follow-up questions — the user can ask the `causal-analyzer` or `transcript-grader` directly via SendMessage.

Cleanup commands:
```bash
# Worktrees (for each path in manifest.worktrees[*].path):
git worktree remove <path> --force
git branch -D <branch>   # only for local-only branches, skip remote-tracking

# Artifact directory:
rm -rf <ARTIFACT_DIR>

# Temp files:
rm -rf /tmp/ammo_eval_snapshot.json /tmp/ammo_eval_scorecard.json \
       /tmp/ammo_eval_report.md /tmp/ammo_eval_transcript_grading.json \
       /tmp/ammo_eval_changes_snapshot \
       /tmp/ammo_eval_events.json /tmp/ammo_eval_causal_dag.json \
       /tmp/ammo_eval_scored_dag.json /tmp/ammo_eval_anomalies.json \
       /tmp/ammo_eval_deep_analysis.json /tmp/ammo_eval_causal_dag_final.json \
       /tmp/ammo_eval_postmortem.md /tmp/ammo_eval_causal_viz.html \
       /tmp/ammo_eval_version_diff.json /tmp/ammo_eval_regression_report.md

# Eval team (only if user selected option 4):
# SendMessage to each team member with {type: "shutdown_request"}
```

## Scoring Dimensions

| Dimension | Weight | What it Measures |
|-----------|--------|------------------|
| E2E Outcome | 40% | Cumulative speedup, shipped optimization count |
| Gate Pass Rates | 15% | First-attempt pass rate across all verification gates |
| Debate Quality | 15% | Proposal grounding, micro-experiment backing, filtering |
| Campaign Efficiency | 15% | Rounds to completion, failure rate, convergence |
| Transcript Quality | 15% | LLM-graded: wasted retries, hallucinated data, off-track reasoning, delegation causality |

When transcript grading is skipped, the remaining 4 dimensions redistribute proportionally (E2E→47%, others→18%/18%/17%).

When delegation is enabled, the transcript grader additionally scores:
- **Delegation causality bonus** (+0.5 per verified causal chain, max +1.5): delegate research that demonstrably improved proposals
- **Delegation failures** (-1.0 each, max -2.0): wrong delegate data that went uncaught
- **Delegation efficiency** (-0.25 each, max -1.0): redundant delegate work
- **Delegation utilization failures** (-0.5 each, max -1.5): champions ignoring correct delegate data

**Philosophy**: Speedup is king. Guardrail violations are scored and reported but never automatically fail the eval.

## Repository Structure

```
~/.claude/ammo-eval/
  index.json                     # Master index with leaderboards
  versions/{version_id}/
    meta.json                    # Git commit, description, skill diff
    runs/{target_slug}/
      run_1/scorecard.json       # Scored run
      run_1/report.md
      run_1/changes_snapshot/    # Full code snapshot
        manifest.json            #   Worktree metadata, merge status
        patches/                 #   Git diffs for each worktree
          agent-xxx.patch
          agent-xxx_uncommitted.patch
        campaign_artifacts/      #   Copy of kernel_opt_artifacts dir
        skill.patch              #   Skill diff vs main
      aggregate.json             # Mean/stddev across runs
    summary.json                 # Cross-target summary
  dashboard.html                 # Static HTML dashboard
```

## Reference Targets

The standard evaluation targets (run each after a skill change):

1. **MoE**: Qwen3-30B-A3B on L40S (fp8, tp=1)
2. **Dense**: Llama-3.1-8B on L40S (fp8, tp=1)

Run 2-3 campaigns per target per skill revision (sequential) to handle non-determinism. The aggregation script computes mean/stddev across runs.

## Resume After Interruption

If the eval pipeline is interrupted, check which files exist:
- `/tmp/ammo_eval_snapshot.json` → Step 1 complete, continue from Step 2
- `/tmp/ammo_eval_changes_snapshot/manifest.json` → Step 2 complete, continue from Step 3
- `/tmp/ammo_eval_scorecard.json` → Step 3 complete, continue from Step 4/5
- Check repository for existing runs → Step 5 may already be done

## Dashboard Views

The HTML dashboard has 4 tabs:
1. **Trend**: Line chart of score and speedup across skill versions
2. **Leaderboard**: Best version per target by E2E speedup
3. **Change Correlation**: Delta vs previous version (green/red)
4. **Run Detail**: Drill-down into individual scorecards
