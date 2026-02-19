# Adapter Template (Project-Specific)

Use this as the default adapter reference for new GPU-heavy projects.

## Purpose

Provide AMMO with project-specific command templates and evidence signatures for:
- baseline capture
- profiling capture
- Stage 2 bottleneck mining/ranking
- E2E validation

## Setup checklist

1. Copy `adapter-template.manifest.json` and rename it for your project.
2. Fill command templates using your project's benchmark/profiling entrypoints.
3. Define fast-path evidence patterns if optimization claims depend on path activation.
4. Define `mining_evidence` (workflow summary + required Stage 2 artifacts).
5. Define a runtime-specific constraints appendix (Stage 1 required fields).
6. Define candidate-vs-baseline differentiation policy (which knob/flag/path changes and how activation is proven).
7. Document known limitations and unsupported regimes.
8. Optionally declare parallel orchestration capability (`parallel_safe`, `parallel_hypothesis_limit`, contention notes).
9. Validate with `scripts/run_adapter_bench.py --dry-run` before execution.

Template commands can use variables from `artifact_bundle.json` plus explicit overrides via:

```bash
python scripts/run_adapter_bench.py \
  --artifact-dir <artifact_dir> \
  --manifest <manifest.json> \
  --phase baseline \
  --set baseline_command='your baseline command'
```

## Command policy

- Commands should be reproducible and parity-safe.
- Do not include ambiguous shell side-effects.
- Prefer explicit argument values over hidden defaults.

## Runtime utility scripts (optional)

Adapters may reference project-specific helper scripts (for example E2E sweep runners).

If used, document in the adapter:
- script ownership and expected environment
- required inputs (files/flags/env) and produced outputs
- failure semantics (what exits non-zero, what is retriable)
- the exact phase(s) where the script is authoritative

## Constraints appendix (mandatory)

Document runtime-specific Stage 1 requirements that must exist in `constraints.md`:
- parity-critical knobs
- required baseline/profile artifacts
- fast-path evidence requirements (if applicable)
- correctness prerequisites

Also document Stage 2 mining requirements:
- mining workflow entrypoint(s)
- required ranking artifacts that Stage 3 must cite

## Evidence policy

Each phase should emit durable artifacts under `{artifact_dir}`:
- logs
- profile reports
- structured result JSON where possible

For promotion-grade validation, adapter docs must state:
- what makes candidate runs materially different from baseline,
- how to detect candidate activation from logs/profilers,
- when same-path baseline/candidate sweeps are allowed only as sanity (not promotion evidence).

For parallel top-3 execution, adapter docs should also state:
- whether concurrent candidates are safe under this runtime,
- resource partitioning rules (GPU/process isolation) for comparability,
- when to force serial fallback due to contention risk.
