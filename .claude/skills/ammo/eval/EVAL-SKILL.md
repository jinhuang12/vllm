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
- `--artifact-dir` (required): Path to the completed campaign's artifact directory
- `--description` (required): Human-readable description of what skill changes were made
- `--skip-transcript-grading` (optional): Skip the LLM grader for speed
- `--repository` (optional, default `~/.claude/ammo-eval`): Eval repository path

## Pipeline

Run these scripts in sequence from `.claude/skills/ammo/eval/scripts/`:

### Step 1: Parse Artifacts
```bash
python .claude/skills/ammo/eval/scripts/parse_artifacts.py \
  --artifact-dir <ARTIFACT_DIR> \
  --output /tmp/ammo_eval_snapshot.json
```

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

### Step 4: Transcript Grading (Optional)

If `--skip-transcript-grading` is NOT set, spawn an LLM grader subagent:

```
Spawn a subagent (general-purpose type, NOT a team member) with:
  prompt: Read the grader rubric at .claude/skills/ammo/eval/agents/transcript_grader.md,
          then evaluate the campaign artifacts at <ARTIFACT_DIR>.
          Write transcript_grading.json to /tmp/ammo_eval_transcript_grading.json.
```

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
  --changes-snapshot /tmp/ammo_eval_changes_snapshot
```

The `--changes-snapshot` flag stores the full changes snapshot (worktree patches, artifact copies, manifest) alongside the scored run in the archive. This ensures you can always reconstruct what code was produced, even after cleanup.

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

### Step 8: Cleanup

After presenting results, ask the user whether to clean up campaign artifacts. Use `AskUserQuestion` with a single multi-select question:

**Question**: "The eval run has been archived with a full snapshot. Which artifacts should I clean up?"

**Options**:
1. **Campaign worktrees** — Remove all worktrees listed in the changes snapshot manifest via `git worktree remove <path> --force` and delete their tracking branches
2. **Artifact directory** — Remove the `kernel_opt_artifacts/<campaign_dir>/` directory
3. **Temp files** — Remove `/tmp/ammo_eval_*.json`, `/tmp/ammo_eval_*.md`, and `/tmp/ammo_eval_changes_snapshot/`

If the user selects nothing (or cancels), skip cleanup entirely. For each selected category, execute the cleanup and report what was removed.

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
       /tmp/ammo_eval_changes_snapshot
```

## Scoring Dimensions

| Dimension | Weight | What it Measures |
|-----------|--------|------------------|
| E2E Outcome | 40% | Cumulative speedup, shipped optimization count |
| Gate Pass Rates | 15% | First-attempt pass rate across all verification gates |
| Debate Quality | 15% | Proposal grounding, micro-experiment backing, filtering |
| Campaign Efficiency | 15% | Rounds to completion, failure rate, convergence |
| Transcript Quality | 15% | LLM-graded: wasted retries, hallucinated data, off-track reasoning |

When transcript grading is skipped, the remaining 4 dimensions redistribute proportionally (E2E→47%, others→18%/18%/17%).

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
