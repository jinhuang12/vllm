# Multi-Round Deliberation Example

This example shows the expected artifact layout and console output for a 2‑round deliberation.

## Scenario

A proposer is about to implement a risky CUDA kernel optimization. The council reviews the plan in Round 1, identifies issues, the proposer revises `.llm-council/context.md`, and Round 2 re-evaluates the revised plan.

## Running the deliberation

From your repo root:

```bash
# Point this to wherever you installed the skill
export LLM_COUNCIL_ROOT="${LLM_COUNCIL_ROOT:-$HOME/.codex/skills/llm-council}"

# Setup with fingerprint (optional but recommended)
bash "$LLM_COUNCIL_ROOT/scripts/setup_council.sh" . --fingerprint "cuda_kernel_atomicAdd_optimization"

# Fill in .llm-council/context.md with proposal + real code snippets

# Round 1 (parallel default)
bash "$LLM_COUNCIL_ROOT/scripts/run_deliberation.sh" 1 3

# ...revise context.md based on feedback...

# Round 2
bash "$LLM_COUNCIL_ROOT/scripts/run_deliberation.sh" 2 3
```

If you want Critic #2 to explicitly build on Critic #1’s same-round feedback, run sequential mode:

```bash
bash "$LLM_COUNCIL_ROOT/scripts/run_deliberation.sh" 1 3 --sequential
```

## Example console output (parallel default)

```
════════════════════════════════════════════════════════════════════
  Round 1 of 3 - Parallel Deliberation
════════════════════════════════════════════════════════════════════
  (parallel default: independent critiques; Critic #2 does NOT see Critic #1's same-round output)

[Round 1] Launching Critic #1 (Gemini) in parallel...
[Round 1] Launching Critic #2 (Claude Code) in parallel...

[Round 1] Critic #1 complete. VOTE: REJECT
[Round 1] Critic #2 complete. VOTE: REJECT
[Round 1] Round feedback appended to history.md

════════════════════════════════════════════════════════════════════
  Round 1 Summary
════════════════════════════════════════════════════════════════════

  Critic #1 (Gemini): VOTE: REJECT
  Critic #2 (Claude Code): VOTE: REJECT

📝 REVISION NEEDED: One or more critics rejected (or returned UNKNOWN).

Full feedback saved to:
  - .llm-council/tmp/critic_1_r1.md
  - .llm-council/tmp/critic_2_r1.md
  - .llm-council/tmp/round_1_aggregate.md
  - .llm-council/history.md
```

## Where to read results

- **Fastest “one file” view:** `.llm-council/tmp/round_<N>_aggregate.md`
- **Full multi-round log:** `.llm-council/history.md`

## Key observations

1. **Parallel default**: both critics run concurrently, so wall‑clock time is lower.
2. **Optional sequential mode**: use `--sequential` when you want Critic #2 to directly react to Critic #1’s same-round feedback.
3. **Multi-round loop**: reviewers are only as good as your updated proposal—revise `.llm-council/context.md`, then rerun the next round.
