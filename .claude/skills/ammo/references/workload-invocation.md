# Workload Invocation — `new_target.py` Details

Supplementary detail for §Invocation in SKILL.md. This file holds the verbose
examples and edge-case explanations that don't belong in the orchestrator's
cold-read path.

## Parameter Table

| Parameter | Prompt format (extract from user) | `new_target.py` flag | Default (if user omits) |
|-----------|----------------------------------|---------------------|------------------------|
| Batch sizes | `--batch-sizes 1 8 32` | `--batch-sizes 1 8 32` | `[1, 8, 32]` |
| Max model length | `--max-model-len=auto` or `--max-model-len=65536` | `--max-model-len 65536` | `4096` (omit flag for default) |
| Input/output lengths | `--input-len=<N> --output-len=<N>` | `--input-len <N> --output-len <N>` | `64` / `512` (only if user doesn't specify) |
| Max num sequences | `--max-num-seqs=32` | _(not a new_target.py flag)_ — post-patch `bench.extra_args` | max(batch_sizes) |
| Multi ISL/OSL pairs | `--isl-osl=64:512,2048:256` | _(not a new_target.py flag)_ — post-patch `workload_matrix` | N/A |
| Data parallel size | `--data-parallel-size 2` | `--data-parallel-size 2` | `1` (single-process) |
| Expert parallelism | `--enable-expert-parallel` | `--enable-expert-parallel` | off |

Batch sizes define the decode buckets for all profiling and validation throughout the campaign.

## Edge Cases

**EP flags**: `--ep N` (integer) is a legacy sizing field persisted to `target.ep` in `state.json` — it does NOT enable vLLM expert parallelism and is not passed through to the benchmark command. To actually enable expert parallelism, use `--enable-expert-parallel`. Most MoE workloads should leave `--ep 1` (default) and rely on `--enable-expert-parallel` plus the TP size.

**DP cross-track contract**: `new_target.py` unconditionally injects `--distributed-executor-backend external_launcher` into `bench.extra_args` when `--data-parallel-size > 1` (vLLM requires this backend for torchrun DP). The sweep script (`run_vllm_bench_latency_sweep.py`) validates that no conflicting `--distributed-executor-backend` value was appended post-hoc (e.g., via frontend `additionalFlags`); conflicts fail fast.

**Single ISL/OSL pair** (most common): pass `--input-len` and `--output-len` directly to `new_target.py`.

**Multiple ISL/OSL pairs** (`--isl-osl=`): Do NOT pass `--input-len`/`--output-len` to `new_target.py`. Instead, after `new_target.py` creates `target.json`, patch the workload section to use `workload_matrix` — each ISL/OSL pair crossed with every batch size:

```json
"workload": {
    "workload_matrix": [
        {"input_len": 64, "output_len": 512, "batch_size": 1},
        {"input_len": 64, "output_len": 512, "batch_size": 8},
        {"input_len": 2048, "output_len": 256, "batch_size": 1},
        {"input_len": 2048, "output_len": 256, "batch_size": 8}
    ],
    "num_iters": 10
}
```

**`--max-model-len=auto`**: Omit `--max-model-len` from `new_target.py` (uses its default of 4096). Note in `target.json` that the user requested auto and the actual value depends on the model config.

**`--max-num-seqs`**: This is a vLLM serving flag, not a `new_target.py` flag. After `new_target.py` creates `target.json`, add it to `bench.extra_args`:

```json
"bench": {
    "extra_args": ["--max-num-seqs", "32"],
    ...
}
```

## Canonical Invocation

```bash
python .claude/skills/ammo/scripts/new_target.py \
  --artifact-dir kernel_opt_artifacts/{model}_{hardware}_{dtype}_tp{tp} \
  --model-id <MODEL_ID> --hardware <HW> --dtype <DTYPE> --tp <TP> \
  [--batch-sizes <BATCH_SIZES>] \
  [--input-len <INPUT_LEN> --output-len <OUTPUT_LEN>] \
  [--max-model-len <MAX_MODEL_LEN>] \
  [--data-parallel-size <DP_SIZE>] \
  [--enable-expert-parallel]
```

Substitute `<INPUT_LEN>` and `<OUTPUT_LEN>` from user's prompt. If user does not specify, omit both flags (defaults to 64/512 decode-heavy workload).
