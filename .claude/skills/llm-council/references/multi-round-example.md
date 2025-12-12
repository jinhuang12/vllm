# Multi-Round Deliberation Example

This example shows the expected output format and history structure for a 2-round sequential deliberation.

## Scenario

Claude proposes optimizing a CUDA kernel. Critics review in Round 1, identify issues, Claude revises, and Round 2 produces consensus.

## Running the Deliberation

```bash
# Setup with fingerprint
bash scripts/setup_council.sh . --fingerprint "cuda_kernel_atomicAdd_optimization"

# Fill in context.md with proposal...

# Round 1
bash scripts/run_deliberation.sh 1 3

# After reviewing feedback, update context.md with revisions...

# Round 2
bash scripts/run_deliberation.sh 2 3
```

## Example Console Output

### Round 1

```
════════════════════════════════════════════════════════════════════
  Round 1 of 3 - Sequential Deliberation
════════════════════════════════════════════════════════════════════

[Round 1] Running Critic #1 (Gemini)...
  Reading: critic_prompt.md, context.md, history.md
[Round 1] Critic #1 complete. VOTE: REJECT
[Round 1] Critic #1 feedback appended to history.md

[Round 1] Running Critic #2 (Codex)...
  Mode: Fresh session (Round 1)
  Reading: critic_prompt.md, context.md, history.md (with Critic #1 feedback)
[Round 1] Critic #2 complete. VOTE: REJECT

════════════════════════════════════════════════════════════════════
  Round 1 Summary
════════════════════════════════════════════════════════════════════

  Critic #1 (Gemini): VOTE: REJECT
  Critic #2 (Codex):  VOTE: REJECT

📝 REVISION NEEDED: One or more critics rejected.

Next steps:
  1. Review feedback in .llm-council/history.md
  2. Update .llm-council/context.md with revised proposal
  3. Run: bash run_deliberation.sh 2 3

Full feedback saved to:
  - /tmp/critic_1_r1.md
  - /tmp/critic_2_r1.md
  - .llm-council/history.md (cumulative)
```

### Round 2

```
════════════════════════════════════════════════════════════════════
  Round 2 of 3 - Sequential Deliberation
════════════════════════════════════════════════════════════════════

[Round 2] Running Critic #1 (Gemini)...
  Reading: critic_prompt.md, context.md, history.md
[Round 2] Critic #1 complete. VOTE: ACCEPT
[Round 2] Critic #1 feedback appended to history.md

[Round 2] Running Critic #2 (Codex)...
  Mode: Resume session (Round 2)
  Passing: Updated context + Critic #1's new feedback
[Round 2] Critic #2 complete. VOTE: ACCEPT

════════════════════════════════════════════════════════════════════
  Round 2 Summary
════════════════════════════════════════════════════════════════════

  Critic #1 (Gemini): VOTE: ACCEPT
  Critic #2 (Codex):  VOTE: ACCEPT

✅ CONSENSUS: Both critics ACCEPT. Proposal approved!

Next: Proceed with implementation.

Full feedback saved to:
  - /tmp/critic_1_r2.md
  - /tmp/critic_2_r2.md
  - .llm-council/history.md (cumulative)
```

## Example history.md After 2 Rounds

```markdown
# Deliberation History

<!-- Append each round's results here -->

## Round 1 - Critic #1 (Gemini)

VOTE: REJECT

## Blocking Issues
1. **Warp shuffle mask incorrect** - Using `0xFFFFFFFF` assumes all 32 threads participate, but only threads in warp 0 are active. This will cause undefined behavior.
   - Fix: Use `__activemask()` or calculate the correct mask based on active threads.

2. **k_idx0 bounds check is after the shuffle** - The shuffle operations will execute even if `k_idx0 >= Dims::HIDDEN_STATES`, potentially reading garbage data.
   - Fix: Move the bounds check before the shuffle reduction.

## Non-Blocking Improvements
- Consider using `__shfl_down_sync` instead of `__shfl_xor_sync` for potentially better performance on some architectures.

## Verification Notes
- Unable to verify if `CoreDims::CALC_WARP_COUNT` value affects the partial_result indexing.

## What Looks Good
- The overall approach of reducing atomicAdd operations via warp shuffle is sound.
- The 4x reduction in atomic operations is a reasonable estimate.

## Round 1 - Critic #2 (Codex)

VOTE: REJECT

## Blocking Issues
1. **Agree with Critic #1 on shuffle mask** - The `0xFFFFFFFF` mask is indeed incorrect. Additionally, I note that `__shfl_xor_sync` requires all threads in the mask to execute the instruction, which may not be guaranteed here.

2. **Race condition in output accumulation** - If multiple CTAs are processing the same token_idx, the atomicAdd operations may still race. The proposal doesn't address inter-CTA synchronization.
   - Suggestion: Consider a two-phase reduction: intra-CTA reduction first, then inter-CTA.

## Non-Blocking Improvements
- The variable naming (`d0`, `w_row`, `t_row`) could be clearer for maintainability.

## Verification Notes
- Need to verify the MMA output layout to confirm the shuffle pattern is correct.

## What Looks Good
- Agree with Critic #1 that the warp shuffle approach is fundamentally sound.
- The reduction from 4 atomicAdd per tile to 1 is correct arithmetic.

## Round 2 - Critic #1 (Gemini)

VOTE: ACCEPT

## Blocking Issues
None - previous blocking issues have been addressed.

## Non-Blocking Improvements
- The fallback to original code when `lane_count < 4` is conservative but safe. Could be optimized later.

## Verification Notes
- The use of `__activemask()` is correct for the shuffle operations.
- Bounds check now correctly precedes the shuffle reduction.

## What Looks Good
- Correctly addressed the shuffle mask issue with `__activemask()`.
- Bounds check moved before shuffle operations.
- Added clear comments explaining the reduction pattern.
- Inter-CTA synchronization addressed via existing scratchpad barrier.

## Round 2 - Critic #2 (Codex)

VOTE: ACCEPT

## Blocking Issues
None remaining.

## Non-Blocking Improvements
- Consider adding a performance comment noting expected speedup.

## Verification Notes
- Reviewed Critic #1's Round 2 assessment and concur.

## What Looks Good
- All blocking issues from Round 1 addressed.
- The revision maintains correctness while achieving the optimization goal.
- Ready for implementation and benchmarking.
```

## Key Observations

1. **Sequential Flow**: Critic #2 explicitly references Critic #1's feedback ("Agree with Critic #1...")
2. **Building on Analysis**: Critics build on each other rather than duplicating work
3. **Focused Round 2**: Critics focus on whether blocking issues were addressed, not re-reviewing everything
4. **History Accumulation**: All feedback is preserved in history.md for full context
5. **Codex Resume**: Round 2's Critic #2 uses `resume --last`, so it has full Round 1 context natively
