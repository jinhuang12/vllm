# Codex Skills: MoE Monokernel Optimizer + LLM Council

This zip contains two Codex CLI skills:

- `moe-monokernel-optimizer` — a stage-gated workflow for designing/implementing fused MoE monokernels in vLLM, with explicit council checkpoints, state tracking, and a revised 4-stage Phase 3 structure.
- `llm-council` — scripts + templates to run an external critic loop (Gemini CLI + Codex CLI) before you commit to an approach.

## Install

1. Create the Codex skills directory (if it does not exist):

```bash
mkdir -p ~/.codex/skills
```

2. Copy the skill folders into `~/.codex/skills/` so you end up with:

```text
~/.codex/skills/moe-monokernel-optimizer/SKILL.md
~/.codex/skills/llm-council/SKILL.md
```

3. Enable skills in Codex CLI (if they are behind a feature flag in your version). In many builds this is:

```bash
codex --enable skills
```

## Use

From your vLLM repo root, start Codex and ask:

```text
Use moe-monokernel-optimizer for Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 on L40S TP=1
```

When you hit a council checkpoint or get stuck, invoke:

```text
Use llm-council to review my Phase 2 plan for {model}/{hardware}
```

### Running llm-council scripts

The `llm-council` scripts are intended to be run from your **repo root** so they create a working directory at `.llm-council/`.

```bash
export LLM_COUNCIL_ROOT="$HOME/.codex/skills/llm-council"
bash "$LLM_COUNCIL_ROOT/scripts/setup_council.sh" . --fingerprint "your_topic"
bash "$LLM_COUNCIL_ROOT/scripts/run_deliberation.sh" 1 3
```

Results accumulate in `.llm-council/history.md`.
