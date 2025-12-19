---
name: llm-council
description: Use Gemini CLI and Claude Code CLI as a council of critics to review proposals, plans, and approaches before risky or expensive changes. Useful when you need a second opinion on architecture/design, you are stuck debugging after 2+ failed attempts, you are about to execute a complex implementation plan, you want to validate assumptions, you are completing a major implementation phase, you are finalizing an optimization plan or CUDA kernel design, or you just produced a large change set and want an external review.
---

# LLM Council

Consult **Gemini** and **Claude Code** as external critics to review your proposal *before* you implement risky or expensive changes.

- **Critic #1:** Gemini CLI
- **Critic #2:** Claude Code CLI (defaults to the bundled `scripts/claude-aws` wrapper)

## Default behavior: parallel critics (faster)

`run_deliberation.sh` runs the two critics **in parallel by default**:

- Faster wall-clock time (roughly the slower critic’s runtime)
- Independent critiques (Critic #2 does **not** see Critic #1’s same-round output)

If you want Critic #2 to explicitly build on Critic #1’s same-round feedback, use **sequential** mode.

```bash
# Point this to wherever you installed the skill
export LLM_COUNCIL_ROOT="${LLM_COUNCIL_ROOT:-$HOME/.codex/skills/llm-council}"

# Parallel (default)
bash "$LLM_COUNCIL_ROOT/scripts/run_deliberation.sh" 1 3

# Sequential (Critic #2 sees Critic #1 output from the same round)
bash "$LLM_COUNCIL_ROOT/scripts/run_deliberation.sh" 1 3 --sequential
```

You can also set a default via environment variable:

```bash
export LLM_COUNCIL_MODE=sequential   # or parallel
```

## Runtime note

Deliberation rounds can still be long-running because critics may explore the repo deeply and run multiple commands.

- **Parallel** mode usually reduces wall-clock time versus sequential.
- If your environment has tool/command timeouts, run the scripts **outside** the agent loop and then read the artifacts from `.llm-council/`.

## Features

- **Full codebase access (YOLO mode)**:
  - Gemini runs with `-y`
  - Claude runs with `--dangerously-skip-permissions`
- **Session ID tracking**:
  - Gemini: `.llm-council/tmp/session_gemini.txt` (best-effort)
  - Claude: `.llm-council/tmp/session_claude.txt` parsed from JSON output
- **Graceful degradation**: if one critic is unavailable, the other still runs
- **Output validation**: checks that critic output exists and is non-empty
- **Round aggregate output**: writes a single per-round summary file under `.llm-council/tmp/`

## Core principle: rich context is essential

**Critics can only review what they can see.** The #1 failure mode is sparse context.

Before running critics, ensure `.llm-council/context.md` includes:

- [ ] The **original request** verbatim (copy/paste)
- [ ] The **current state** (what exists today, where)
- [ ] The **proposed plan** (step-by-step, with file paths)
- [ ] The **constraints** (hardware, perf, correctness, compatibility)
- [ ] The **actual code** involved (snippets + paths, not just descriptions)
- [ ] Any evidence gathered (profiling, benchmarks, logs)
- [ ] Open questions you want critics to answer

Use `references/context-template.md` as the canonical structure.

## Pre-flight checks

From your repo root:

```bash
# Gemini auth
echo "GEMINI_API_KEY is ${GEMINI_API_KEY:+set}${GEMINI_API_KEY:-NOT SET}"

# Claude Code availability (direct)
claude -v

# Claude Code availability (bundled Bedrock wrapper)
"$LLM_COUNCIL_ROOT/scripts/claude-aws" -v
```

> If you don’t use Bedrock, set `LLM_COUNCIL_CLAUDE_CMD=claude` to force direct `claude`.

## Installation (Codex skill)

This skill is typically installed under:

```text
~/.codex/skills/llm-council/SKILL.md
```

You can invoke it by name in chat (e.g., “Use llm-council”), but the actual critic loop is driven by scripts that you run from your **repo root** so `.llm-council/` lives next to your code.

## Directory layout

This skill uses a working directory under your repo:

```text
.llm-council/
  critic_prompt.md
  context.md
  history.md
  topic_fingerprint.txt                 # optional
  tmp/
    critic_1_r1.md
    critic_2_r1.md
    critic_2_r1.json
    round_1_aggregate.md                # NEW: single-file aggregate per round
    session_gemini.txt
    session_claude.txt
```

## Recommended workflow

### 0) Choose a topic fingerprint

Pick a short fingerprint string that represents the *topic* (helps avoid accidentally reusing stale context):

- `moe_monokernel_qwen3_l40s`
- `paged_attention_fp8_perf`
- `router_topk_grad_fix`

### 1) Check for stale context

```bash
export LLM_COUNCIL_ROOT="${LLM_COUNCIL_ROOT:-$HOME/.codex/skills/llm-council}"
bash "$LLM_COUNCIL_ROOT/scripts/check_context.sh" "YOUR_FINGERPRINT"
# outputs: fresh | continue | cleared
```

### 2) Setup (if needed)

```bash
bash "$LLM_COUNCIL_ROOT/scripts/setup_council.sh" . --fingerprint "YOUR_FINGERPRINT"
```

### 3) Fill in context.md

Edit `.llm-council/context.md` and include **real, specific context**.

### 4) Run deliberation

Round 1 (parallel default):

```bash
bash "$LLM_COUNCIL_ROOT/scripts/run_deliberation.sh" 1 3
```

If any critic votes `REJECT`, revise your plan (edit `.llm-council/context.md`), then run Round 2:

```bash
bash "$LLM_COUNCIL_ROOT/scripts/run_deliberation.sh" 2 3
```

### 5) Interpret results

- If **both** vote `ACCEPT`: proceed
- If **either** votes `REJECT`: treat as blocking until addressed (or explicitly justify why you disagree)

Artifacts:

- `.llm-council/history.md` (cumulative log)
- `.llm-council/tmp/round_<N>_aggregate.md` (single file for quick reading)

## How the critics are invoked

### Critic #1 — Gemini CLI

- Uses YOLO mode (`-y`) for deep investigation.
- Attempts to resume via `--resume <UUID>` using `.llm-council/tmp/session_gemini.txt`.

### Critic #2 — Claude Code CLI

- Uses headless mode (`-p`) with:
  - `--dangerously-skip-permissions` (YOLO)
  - `--output-format json` (so we can capture `session_id`)
  - `--tools default`
- Attempts to resume via `--resume <session_id>` using `.llm-council/tmp/session_claude.txt`.
- Command selection:
  - Defaults to the bundled `scripts/claude-aws` wrapper if present
  - Override with `LLM_COUNCIL_CLAUDE_CMD` if desired

## CLI quick reference

### Gemini

```bash
gemini --version
gemini --list-sessions
gemini --resume <UUID> -p "" < prompt.txt
```

### Claude Code

```bash
claude -v

# Headless, JSON output (what the scripts use)
claude --dangerously-skip-permissions -p --output-format json < prompt.txt

# Resume
claude --resume <SESSION_ID> --dangerously-skip-permissions -p --output-format json < prompt.txt
```
