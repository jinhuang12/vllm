# vLLM Adapter (AMMO)

Use this adapter to run AMMO stages against vLLM while keeping AMMO core generic.

## Purpose

This adapter owns vLLM-specific implementation details for:
- production-parity baseline command execution
- profiling capture command execution
- correctness + E2E speedup gating workflow

The core AMMO skill does not prescribe vLLM commands directly.

## Required artifacts

1. `{artifact_dir}/artifact_bundle.json`
2. `{artifact_dir}/target.json` for E2E sweep comparisons
3. `{artifact_dir}/constraints.md` (mandatory Stage 1 artifact)
4. `{artifact_dir}/nsys_mining.md` and `{artifact_dir}/nsys_mining.json` from `$vllm-fusion-opportunity-miner` (mandatory Stage 2 ranking evidence)

`target.json` is consumed by `scripts/run_vllm_bench_latency_sweep.py` and must include:
- target: model/dtype/tp/ep/max_model_len
- workload: input_len/output_len/batch_sizes/num_iters
- bench: baseline/opt env and optional fast-path regex evidence patterns

For promotion-grade A/B in this adapter, `target.json` must make candidate differentiation explicit by the end of Stage 4:
- `baseline_env` vs `opt_env` or baseline args vs opt args must contain at least one intentional candidate delta,
- candidate delta must map to a Stage 3/Stage 4 declared switch,
- fast-path evidence rules should be populated when path activation is not obvious from args.
- Initial scaffold runs may start with identical baseline/candidate stanzas; Stage 4 must update candidate settings when wiring the candidate path.

## vLLM constraints appendix (blocking for Stage 3 planning)

`constraints.md` must include these vLLM-specific fields before writing `optimization_plan.md`:

- vLLM runtime identity:
  - vLLM version/commit
  - CUDA/PyTorch versions used by that runtime
- parity knobs and flags:
  - CUDA graphs mode
  - compile/runtime mode
  - scheduler and serving flags that can change dispatch behavior
- baseline/profile evidence paths:
  - benchmark command used for baseline
  - nsys report path (+ ncu path if used)
  - per-bucket metrics artifact paths
- fast-path activation evidence policy:
  - expected require/forbid patterns (if optimization claim depends on a specific path)
- correctness prerequisites:
  - tolerance policy source
  - key semantic invariants that must hold for this target

If these are missing, set `artifact_bundle.json.constraints.status=blocked` and do not proceed to Stage 3 planning.

Suggested source anchors for this appendix:
- `vllm/model_executor/models/qwen3_moe.py` (model routing + topology)
- `vllm/model_executor/layers/fused_moe/layer.py` (FusedMoE dispatch)
- `vllm/model_executor/layers/fused_moe/fused_moe.py` (backend path and kernels)
- `vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py` (permute/unpermute/reduce path)

## Production-parity requirements (blocking)

Baseline, profiling, and candidate runs must keep these identical unless explicitly under study:
- model ID and weights
- dtype/quantization format
- TP/EP topology
- bucket mix (input/output lengths and batch sizes)
- CUDA graphs mode
- compile/runtime mode
- scheduler + serving knobs affecting dispatch behavior

If parity differs, results are invalid for ship decisioning.

## Adapter execution commands

Manifest: `references/adapters/vllm.manifest.json`

### Baseline capture

This adapter requires a project-local baseline command via `baseline_command`.

Example:
```bash
python scripts/run_adapter_bench.py \
  --artifact-dir <artifact_dir> \
  --manifest references/adapters/vllm.manifest.json \
  --phase baseline \
  --set baseline_command='vllm bench latency --model <model_id> --dtype <dtype> --tensor-parallel-size <tp> --max-model-len <max_model_len> --input-len <input_len> --output-len <output_len> --batch-size <bs> --num-iters <iters> --output-json <artifact_dir>/baseline.json'
```

### Profiling capture

This adapter requires a project-local profiling command via `profile_command`.

Example:
```bash
python scripts/run_adapter_bench.py \
  --artifact-dir <artifact_dir> \
  --manifest references/adapters/vllm.manifest.json \
  --phase profile \
  --set profile_command='nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true -o <artifact_dir>/profiles/baseline -- vllm bench latency <same parity args as baseline>'
```

### Stage 2 bottleneck mining (authoritative for vLLM)

After baseline `nsys` capture, run `$vllm-fusion-opportunity-miner` against the exported sqlite and save outputs under `{artifact_dir}`.

Example:
```bash
miner_dir="/home/jinhun/.codex/skills/vllm-fusion-opportunity-miner"
nsys export --type sqlite -o {artifact_dir}/profiles/baseline_bs8_sqlite {artifact_dir}/profiles/baseline_bs8.nsys-rep
python "${miner_dir}/scripts/nsys_mine.py" \
  --sqlite {artifact_dir}/profiles/baseline_bs8_sqlite.sqlite \
  --out-dir {artifact_dir}
```

Stage 2 ranking for vLLM must cite `nsys_mining.md` / `nsys_mining.json`; do not substitute ad-hoc hotspot guesses.

After mining completes, update stage artifacts in the same run:
- synthesize `constraints.md` with concrete baseline/profile/mining evidence
- synthesize `optimization_plan.md` from ranked mining opportunities
- update `artifact_bundle.json` stage/status/constraints fields to reflect actual progress

Do not leave scaffold-only docs after evidence capture.

Unattended-run rule:
- if baseline/profile/mining succeeded but stage docs are still scaffolds, the run must remain `in_progress` or be marked `blocked`; it must not be treated as complete.
- if Stage 3 produced candidates but no candidate is implemented, Stage 4 is `blocked` unless user explicitly requested planning-only output.

### E2E gating (authoritative path)

Use this as the reliable and robust E2E comparison path:
```bash
python scripts/run_adapter_bench.py \
  --artifact-dir <artifact_dir> \
  --manifest references/adapters/vllm.manifest.json \
  --phase e2e
```

Equivalent direct command:
```bash
python scripts/run_vllm_bench_latency_sweep.py --artifact-dir <artifact_dir>
```

The sweep runner records:
- per-bucket baseline vs candidate latency metrics
- command/env provenance
- fast-path evidence checks (if configured in `target.json`)
- structured markdown/json outputs for maintenance decisions

Important interpretation rule:
- same-path baseline/candidate sweeps are allowed for sanity only,
- they are not valid Stage 5 promotion evidence for kernel/e2e gates.

## Correctness + E2E speedup gates

Use default AMMO gate ordering:
1. Correctness gate pass first
2. Kernel-time gate pass second
3. E2E gate pass third

Recommended E2E decision rule for vLLM adapter usage:
- Reject candidate when correctness is not clean.
- Reject candidate when E2E uplift is statistically insignificant for primary buckets, even if local kernel wins are large.
- Require explicit envelope restriction for any justified worst-bucket regression.
- If baseline/candidate are not feature-differentiated, mark kernel/e2e gates `blocked` and return to Stage 4.

## Utility script contract

Authoritative utility script:
- `scripts/run_vllm_bench_latency_sweep.py`

Inputs:
- `--artifact-dir` (required)
- optional `--target-json`, `--execution-mode`, `--require-fastpath`

Outputs:
- `{artifact_dir}/e2e_latency*/e2e_latency_results.json`
- `{artifact_dir}/e2e_latency*/e2e_latency_results.md`
- per-bucket logs/json/status under `{artifact_dir}/e2e_latency*/`

Failure behavior:
- non-zero exit on unresolved placeholders, invalid config, command failures, timeout, or required fast-path evidence failure
- failures are blocking for promotion until investigated and re-run
