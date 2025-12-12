---
name: llm-council
description: Use Gemini CLI and Codex CLI as a council of critics to review Claude's proposals, plans, and approaches. Triggers when (1) needing a second opinion on architecture or design decisions, (2) stuck on debugging and want fresh perspectives, (3) reviewing complex implementation plans before execution, (4) wanting to validate assumptions across multiple AI models. Features read-only sandbox, web search verification, session ID tracking for robust resume, and graceful degradation when one CLI is unavailable.
---

# LLM Council

Consult Gemini and Codex as critics to review your proposals before implementation.

## Features

- **Read-only sandbox**: Critics can read files but cannot modify your codebase
- **Web search**: Critics can verify technical claims online (Google Search for Gemini, web_search for Codex)
- **Session ID tracking**: Robust multi-round resume using explicit session IDs (not `--last`)
- **Graceful degradation**: Continues with available critics if one CLI is unavailable
- **Automatic validation**: Checks critic output exists, is non-empty, and is fresh

## Core Principle: Rich Context is Essential

**Critics can only review what they can see.** The #1 failure mode is providing sparse context.

### What Critics Need

1. **Original user request** - Verbatim, including any file references
2. **Full session history** - All Explore/Plan agent findings since session start
3. **Actual code** - Not descriptions, but the real code being discussed
4. **Repository context** - AGENTS.md, GEMINI.md, or project overview
5. **Relevant source files** - Files being modified or referenced

### Context Checklist

Before running critics, verify context.md includes:
- [ ] User's exact request (copy-paste, not paraphrase)
- [ ] All sub-agent findings (Explore agents, Plan mode analysis)
- [ ] Profiling data, benchmarks, or other evidence gathered
- [ ] Actual code snippets (current implementation AND proposed changes)
- [ ] File paths for all code discussed
- [ ] Key constraints (hardware limits, API requirements, etc.)

## Pre-Flight Checks

```bash
# Verify CLI access
echo "GEMINI_API_KEY is ${GEMINI_API_KEY:+set}${GEMINI_API_KEY:-NOT SET}"
codex login status
```

## Setup

```bash
# Basic setup
bash /path/to/skill/scripts/setup_council.sh .

# Setup with topic fingerprint (recommended for auto-staleness detection)
bash /path/to/skill/scripts/setup_council.sh . --fingerprint "your_topic_identifier"
```

Creates `.llm-council/` with `critic_prompt.md`, `context.md`, `history.md`, `tmp/` directory, and optionally `topic_fingerprint.txt`.

### File Locations

| File | Location | Purpose |
|------|----------|---------|
| Critic outputs | `.llm-council/tmp/critic_{1,2}_r{round}.md` | Per-round feedback |
| Session IDs | `.llm-council/tmp/session_{gemini,codex}.txt` | Session UUID for resume |
| Verification tests | `.llm-council/tmp/*_check.md` | Dry-run validation |

### Validation Behavior

The `run_deliberation.sh` script validates each critic output:
- **File exists**: Ensures critic invocation completed
- **Not empty**: Catches API failures that produce empty responses
- **Freshness check**: Warns if file is >5 minutes old (may be stale from previous run)

If validation fails for a critic, the script continues with available critics (graceful degradation) rather than exiting.


## Context Management (Staleness Detection)

The council tracks deliberation topics to auto-clear stale context when you switch topics.

### How It Works

1. **Fingerprint**: A short identifier for your current deliberation topic (e.g., `"moe_kernel_csrc/moe"`)
2. **Auto-clear**: When fingerprints differ, old `.llm-council/` is automatically cleared
3. **Continue**: When fingerprints match, existing context is preserved for multi-round deliberation

### Usage

```bash
# Check if context is stale before setup
STATUS=$(bash scripts/check_context.sh "new_topic_fingerprint")
# Returns: fresh | continue | cleared

# Force behaviors
bash scripts/check_context.sh "any" --force-clear    # Always clear
bash scripts/check_context.sh "any" --force-continue # Keep old context
```

### Recommended Workflow

```bash
# 1. Generate fingerprint from your topic (first 100 chars + key file paths)
FINGERPRINT="moe_optimization_csrc/moe/moe_down_projection.cu"

# 2. Check and handle staleness
STATUS=$(bash scripts/check_context.sh "$FINGERPRINT")
echo "Context status: $STATUS"  # fresh, continue, or cleared

# 3. Setup if needed (fresh or cleared)
if [ "$STATUS" != "continue" ]; then
    bash scripts/setup_council.sh . --fingerprint "$FINGERPRINT"
fi

# 4. Update context.md and run deliberation
```

## CLI Quick Reference

### Gemini CLI
```bash
# Basic usage
gemini "@file @dir/ prompt"           # Include files with @ syntax

# Read-only sandbox with web search (recommended for critics)
gemini -s --allowed-tools "read_file,list_directory,search_file_content,glob,google_web_search" "prompt"

# Session management
gemini --list-sessions                 # List available sessions
gemini --resume <UUID>                 # Resume by session ID (recommended)
gemini --resume latest                 # Resume most recent session
gemini --resume 5                      # Resume by index
```

### Codex CLI
```bash
# Basic usage
codex exec "prompt"
codex exec --output-last-message out.md "prompt"

# Read-only sandbox with web search (recommended for critics)
codex exec -s read-only --search -C . "prompt"

# Session management
codex exec resume <UUID> "follow-up"   # Resume by session ID (recommended)
codex exec resume --last "follow-up"   # Resume most recent session

# Session files: ~/.codex/sessions/YYYY/MM/DD/rollout-TIMESTAMP-UUID.jsonl
```

## Building Rich Context

**See `references/context-template.md` for the complete template.**

### Example: Well-Structured Context

```markdown
# Deliberation Context

## Original User Request (Verbatim)
"Determine the next steps to continue. Use the explore or plan agent to understand 
how & why the implementation was done for llama 4 on H200, then apply relevant 
modifications that is applicable for Qwen 3 on L40S. Draft a plan to implement 
& use the llm-council skill to ensure the plan is reviewed before proposing 
the final plan."

## Session History

### Explore Agent 1: Llama4 vs Qwen3 Implementation Comparison
**Findings:**
- Qwen3 has 68% coverage of Llama4 optimizations
- Missing optimizations: bitfield expert dedup, scale folding, speculative compute
- Key difference: TOP_K=8 (Qwen3) vs TOP_K=1 (Llama4)
- Tensor Core down-projection uses mma_fp8_tf32() MMA instruction

### Explore Agent 2: H100 vs L40S Hardware Differences
**Findings:**
- L40S shared memory: 100KB (vs H100: 228KB)
- Current 6c2p warp config is optimal for L40S (31% better than Llama4's 8c4p)
- Ada Lovelace vs Hopper architecture differences

### Explore Agent 3: Down-Projection Implementation
**Findings:**
- Tensor Core GEMM enabled via MOE_MONOKERNEL_USE_TENSOR_CORES
- Only warp 0 performs accumulation (5/6 calc warps idle)
- Atomic contention with TOP_K=8

### Profiling Results
| BS | Reference | Monokernel | Down-proj % |
|----|-----------|------------|-------------|
| 1  | 3.68 ms   | 0.79 ms    | 93.8%       |
| 8  | 0.75 ms   | 1.10 ms    | 71.7%       |
| 64 | 1.35 ms   | 3.75 ms    | 81.3%       |

## Repository Context

Project: vLLM inference framework
Key directories:
- csrc/moe/moe_monokernel/ - Custom CUDA kernels
- optimization-guides/ - Reference documentation

## Current Implementation (moe_down_projection.cu:318-356)
```cpp
// Only warp 0 does reduction - other 5 calc warps idle
if (warp == 0) {
    std::uint32_t t_row = thread / 4;
    for (std::uint32_t i = 0; i < CoreDims::CALC_WARP_COUNT; ++i) {
        d0 += partial_result[w_row / 2 + i][thread + 0];
    }
    atomicAdd(&scratchpad->output_accum[token_idx * K + k_idx0], d0);
}
```

## Proposed Change

### Phase 1: Warp Shuffle Reduction
```cpp
if (warp == 0) {
    std::uint32_t t_row = thread / 4;
    std::uint32_t lane_in_group = thread % 4;
    
    // Sum partial results
    for (std::uint32_t i = 0; i < CoreDims::CALC_WARP_COUNT; ++i) {
        d0 += partial_result[w_row / 2 + i][thread + 0];
    }
    
    // Warp shuffle to reduce 4 threads' contributions
    d0 += __shfl_xor_sync(0xFFFFFFFF, d0, 1);
    d0 += __shfl_xor_sync(0xFFFFFFFF, d0, 2);
    
    // Only lane 0 of each group does atomicAdd
    if (lane_in_group == 0 && k_idx0 < Dims::HIDDEN_STATES) {
        atomicAdd(&scratchpad->output_accum[token_idx * K + k_idx0], d0);
    }
}
```

**Expected Impact:** 4x fewer atomicAdd operations per tile

## Questions for Critics
1. Is the warp shuffle reduction correct for the MMA output layout?
2. Does CALC_WARP_COUNT=6 vs T_TILE=8 create any issues?
3. Are there better approaches for reducing atomic contention?
```

## Single-Round Execution (Default)

```bash
# 1. Prepare rich context (use template from references/context-template.md)
# Claude should programmatically build context.md with ALL session history

# 2. Run critics with full file context
gemini -p \
  "@.llm-council/critic_prompt.md \
   @.llm-council/context.md \
   You are Anonymous Critic #1. Review the proposal." \
  > .llm-council/tmp/critic_1.md 2>&1 &
PID1=$!

codex exec -C . \
  "$(cat .llm-council/critic_prompt.md)

$(cat .llm-council/context.md)

Repository guidance (AGENTS.md):
$(cat AGENTS.md 2>/dev/null || echo 'No AGENTS.md found')

You are Anonymous Critic #2. Review the proposal." \
  --output-last-message .llm-council/tmp/critic_2.md &
PID2=$!

wait $PID1 $PID2

# 3. Verify results
cat .llm-council/tmp/critic_1.md
cat .llm-council/tmp/critic_2.md

# 4. ALWAYS append to history
cat >> .llm-council/history.md << 'HIST'
## Round 1
### Critic #1
$(cat .llm-council/tmp/critic_1.md)
### Critic #2
$(cat .llm-council/tmp/critic_2.md)
### Vote
[ACCEPT/REJECT tally]
HIST
```

### Processing Results

After collecting critiques:
1. Parse VOTE (ACCEPT/REJECT) from each critic
2. Review blocking issues - verify each claim before acting
3. Note non-blocking suggestions for potential improvements
4. If both ACCEPT: proceed with implementation
5. If any REJECT: address blocking issues or push back with reasoning

## Multi-Round Deliberation (Sequential)

Use for complex proposals where iterative refinement is valuable. Maximum 3 rounds.

### Key Difference from Single-Round

**Sequential execution within each round:**
1. Critic #1 (Gemini) reviews first
2. Critic #1's feedback is appended to history
3. Critic #2 (Codex) reviews with Critic #1's feedback visible
4. Claude reviews ALL feedback before revising

This allows critics to build on each other's analysis.

### Flow Diagram

```
Round 1:
  Claude proposes
  → Critic #1 reviews proposal
  → Critic #2 reviews proposal + Critic #1's feedback
  → Claude reviews ALL feedback

Round 2 (if any REJECT):
  Claude revises context.md
  → Critic #1 reviews revision + full history
  → Critic #2 resumes session + Critic #1's new feedback
  → Claude reviews

Round 3 (final, if needed):
  Same pattern, critics give final verdict
```

### Using the Deliberation Script

```bash
# Round 1 - Initial review
bash scripts/run_deliberation.sh 1 3

# If revisions needed:
# 1. Review feedback in .llm-council/history.md
# 2. Update .llm-council/context.md with revised proposal
# 3. Run Round 2

bash scripts/run_deliberation.sh 2 3

# Round 3 (final) if still not accepted
bash scripts/run_deliberation.sh 3 3
```

### What the Script Does

1. **CLI availability check**: Verifies Gemini and Codex are installed and functional
2. **Critic #1 (Gemini)**:
   - Read-only sandbox with web search enabled
   - Round 1: Fresh session, captures session ID for future rounds
   - Round 2+: Resumes via explicit session ID (`--resume <UUID>`)
3. **Append to history**: Critic #1's feedback added to `history.md` before Critic #2 runs
4. **Critic #2 (Codex)**:
   - Read-only sandbox with web search enabled
   - Round 1: Fresh session, captures session ID from `~/.codex/sessions/`
   - Round 2+: Resumes via explicit session ID (`codex exec resume <UUID>`)
5. **Graceful degradation**: If one CLI is unavailable, continues with available critic
6. **Summary**: Reports votes, critic status, and session IDs

### Session ID Management

Session IDs are stored in `.llm-council/tmp/` for robust multi-round deliberation:

| File | Purpose |
|------|---------|
| `session_gemini.txt` | Gemini session UUID |
| `session_codex.txt` | Codex session UUID |

This ensures the correct session is resumed even when running multiple deliberations.

### Session Behavior Notes

| CLI | Round 1 | Round 2+ |
|-----|---------|----------|
| **Gemini** | Fresh session, capture UUID | Resume by UUID (`--resume <UUID>`) |
| **Codex** | Fresh session, capture UUID from fs | Resume by UUID (`codex exec resume <UUID>`) |

### CLI Flags Used

| CLI | Flags | Purpose |
|-----|-------|---------|
| **Gemini** | `-s` | Sandbox mode |
| **Gemini** | `--allowed-tools` | Whitelist: read_file, list_directory, search_file_content, glob, google_web_search |
| **Codex** | `-s read-only` | Read-only sandbox |
| **Codex** | `--search` | Enable web search |
| **Codex** | `-C .` | Set working directory |

### Manual Multi-Round (Alternative)

If you need more control than the script provides:

```bash
# Round 1, Critic 1
gemini -p \
  "@.llm-council/critic_prompt.md \
   @.llm-council/context.md \
   You are Critic #1. Round 1. Review the proposal." \
  > .llm-council/tmp/critic_1_r1.md 2>&1

# Append Critic 1's feedback to history BEFORE Critic 2
cat >> .llm-council/history.md << EOF
## Round 1 - Critic #1 (Gemini)
$(cat .llm-council/tmp/critic_1_r1.md)
EOF

# Round 1, Critic 2 (sees Critic 1's feedback in history)
codex exec -C . \
  "$(cat .llm-council/critic_prompt.md)
$(cat .llm-council/context.md)
$(cat .llm-council/history.md)
You are Critic #2. Round 1. Build on Critic #1's analysis above." \
  --output-last-message .llm-council/tmp/critic_2_r1.md

# Append Critic 2's feedback
cat >> .llm-council/history.md << EOF
## Round 1 - Critic #2 (Codex)
$(cat .llm-council/tmp/critic_2_r1.md)
EOF
```

For Round 2+, Codex uses `resume --last` to maintain conversation state:

```bash
# Round 2, Critic 2 (resume session)
codex exec resume --last \
  "Round 2. Critic #1's new feedback: $(cat .llm-council/tmp/critic_1_r2.md)
   Review the updated proposal." \
  --output-last-message .llm-council/tmp/critic_2_r2.md
```

## Verifying Context is Complete

### Gemini Context Verification

The `@` syntax in Gemini CLI reads files and injects their full contents. To verify:

```bash
# Check what Gemini will see (approximate token count)
wc -c .llm-council/critic_prompt.md .llm-council/context.md .llm-council/history.md

# Verify files exist and have content
for f in .llm-council/critic_prompt.md .llm-council/context.md .llm-council/history.md; do
  if [ -s "$f" ]; then
    echo "✓ $f exists ($(wc -l < "$f") lines)"
  else
    echo "✗ $f is missing or empty!"
  fi
done

# Test that Gemini can read the files (dry run)
gemini -p \
  "@.llm-council/context.md \
   Confirm you received the context. List the section headers you see." \
  > .llm-council/tmp/gemini_context_check.md 2>&1
cat .llm-council/tmp/gemini_context_check.md
```

## Verifying Session Continuity

### Codex Session Verification

```bash
# List recent Codex sessions
ls -la ~/.codex/sessions/ | tail -10

# Verify session was created after Round 1
codex exec resume --last "What was the proposal you reviewed in Round 1? Summarize briefly." \
  --output-last-message .llm-council/tmp/codex_session_check.md
cat .llm-council/tmp/codex_session_check.md
```

## Common Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| "No critics available" | Both CLIs unavailable | Install gemini/codex CLI and authenticate |
| "PARTIAL COUNCIL" warning | One CLI unavailable | Install missing CLI or proceed with single critic |
| Session ID not captured | CLI returned before session saved | Check `--list-sessions` (Gemini) or `~/.codex/sessions/` (Codex) |
| Wrong session resumed | Session ID mismatch | Clear `.llm-council/tmp/session_*.txt` and restart Round 1 |
| Vague rejections | Sparse context | Include full session history, actual code |
| Critics miss issues | Missing files | Add source files via `@path` or `$(cat)` |
| History not updating | Forgot append | Always append to history.md after each round |
| Critics confused about project | No repo context | Include AGENTS.md/GEMINI.md content |
| Codex resume fails | No prior session or bad ID | Ensure Round 1 ran, check session_codex.txt |
| Gemini resume fails | Session not found | Check `gemini --list-sessions`, verify UUID |
| "Output not found" error | Critic invocation failed | Check API connectivity, verify credentials |
| "Output is stale" warning | Old files in tmp/ | Normal if rerunning; files auto-cleaned per round |
| Empty output error | API returned no content | Check API status, retry invocation |
| Web search fails | Network or permission | Check internet connectivity, verify CLI auth |

## Reference Files

- `references/context-template.md` - Complete context structure
- `references/critic-prompt.md` - Critic role instructions
- `references/multi-round-example.md` - Example multi-round deliberation output

## Scripts

- `scripts/setup_council.sh` - Initialize .llm-council/ (accepts `--fingerprint`)
- `scripts/check_context.sh` - Check/clear stale context
- `scripts/run_deliberation.sh` - Run sequential multi-round deliberation
