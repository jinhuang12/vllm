# vLLM MoE Monokernel Optimizer — Codex bundle

Drop this repo-local `.codex/` directory into the **vLLM repo root** to enable:

- a custom Codex slash command prompt: `moe-monokernel-optimizer`
- supporting reference docs + assets (Llama4 patch, benchmarking template)
- a non-interactive runner script that preserves prompt quality

## 1) One-time setup (recommended)

From the vLLM repo root:

```bash
export CODEX_HOME="$PWD/.codex"
```

(Optionally use `direnv` to auto-export this when you `cd` into the repo.)

## 2) Interactive usage (Codex TUI)

Start Codex from the repo root:

```bash
CODEX_HOME="$PWD/.codex" codex
```

Then run the custom prompt via the slash popup:

```text
/prompts:moe-monokernel-optimizer MODEL_ID="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8" HARDWARE="L40S" DTYPE="fp8" TP=1 TOPK=8
```

You can run a single phase by adding `PHASE=1` (or 2..5).

## 3) Non-interactive usage (codex exec)

Codex currently does **not** expand custom slash commands inside `codex exec`, so use the provided runner:

```bash
./.codex/scripts/moe-monokernel-optimizer-exec.sh \
  MODEL_ID="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8" \
  HARDWARE="L40S" DTYPE="fp8" TP=1 TOPK=8 MODE=full
```

The script:
- renders the same prompt used by the slash command
- pipes it to `codex exec -` (stdin)
- sets `CODEX_HOME` so Codex loads `.codex/AGENTS.md` and repo-local prompts

### Sandbox / approvals

For hands-off automation you’ll usually want workspace writes enabled:

- `codex exec --full-auto -` (convenience preset)
- or `codex exec --sandbox workspace-write -`

Edit the script if you want stricter defaults.

## 4) What’s in here

- `.codex/AGENTS.md` — persistent instructions loaded by Codex
- `.codex/prompts/moe-monokernel-optimizer.md` — the main workflow prompt (inlines key steps)
- `.codex/moe-monokernel-optimizer/references/**` — branching, tiling, templates, etc.
- `.codex/moe-monokernel-optimizer/assets/**` — Llama4 reference patch
- `.codex/moe-monokernel-optimizer/orchestration/**` — workflow + failure-handling + phase prompts
- `.codex/moe-monokernel-optimizer/validation/**` — validation details

## 5) Outputs

Runs write to:

`moe_monokernel_artifacts/<model>_<hardware>_<dtype>_tp<TP>/`

including `constraints.md`, `optimization_plan.md`, `validation_results.md`, and `state.json`.
