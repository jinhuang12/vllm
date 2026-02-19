# Adapter Contract (AMMO)

Adapters are the project-specific extension point for AMMO.

## Required manifest fields

Each adapter manifest must provide:

- `name`: adapter identifier
- `maturity`: `stable` or `beta`
- `supported_regimes`: named workload regimes supported by this adapter
- `baseline_cmd_templates`: templates for baseline capture
- `profile_cmd_templates`: templates for profiling capture
- `e2e_cmd_templates`: templates for E2E validation
- `mining_evidence`: Stage 2 bottleneck-mining contract:
  - `workflow_summary`
  - `required_artifacts`
- `fastpath_evidence_rules`:
  - `require_patterns`
  - `forbid_patterns`
- `known_limitations`: explicit unsupported scenarios

Optional parallel-orchestration fields:
- `parallel_safe`: boolean that declares whether Stage 4 parallel top-3 execution is adapter-safe by default
- `parallel_hypothesis_limit`: integer limit for concurrent hypotheses (default policy is `3`)
- `parallel_notes`: known contention hazards and isolation requirements

## Template variables

Minimum required template variables:
- `{artifact_dir}`
- `{model_id}`
- `{dtype}`
- `{tp}`
- `{ep}`

Common optional variables:
- `{batch_size}`
- `{input_len}`
- `{output_len}`
- `{max_model_len}`
- `{warmup_iters}`
- `{num_iters}`
- `{label}`
- `{bench_cmd}`

If additional variables are required, document them in adapter docs.

## Adapter utility scripts

Adapters may define runtime-specific helper scripts. If they do, the adapter docs must specify:
- script path and ownership
- required inputs (flags/env/files) and produced artifacts
- exit-code behavior and blocker conditions
- which AMMO phase treats the script output as authoritative
- if used for Stage 2, the ranking evidence artifacts they must emit

## Adapter constraints appendix

Adapters must define which Stage 1 constraints fields are mandatory for that runtime.
At minimum, adapter docs must list:
- parity-critical runtime knobs
- required baseline/profile artifact links
- fast-path evidence requirements (if applicable)
- correctness prerequisites specific to the runtime
- Stage 2 mining workflow and required ranking artifacts

## Behavioral rules

1. Fail fast on missing required fields.
2. Fail fast on unresolved critical template variables.
3. Preserve parity knobs across baseline/candidate comparisons.
4. Emit actionable blocker messages for unsupported beta flows.
5. Never silently guess missing commands.
6. If utility scripts are referenced, they must be documented with executable examples.
7. Adapter docs must declare mandatory Stage 1 constraints fields.
8. Adapter docs must declare Stage 2 mining workflow and required evidence artifacts.
9. Adapter docs must declare how candidate runs are differentiated from baseline runs and how activation is proven.
10. If parallel fields are declared, adapter docs must define contention controls and comparability policy across concurrent runs.
11. Adapter docs must define a deterministic command-fingerprint method for comparing baseline/candidate parity in validation artifacts.

## Readiness checklist

Adapter is Stage-5-ready only if:
- manifest validates against `schemas/adapter_manifest.v1.json`
- `--dry-run` renders commands without unresolved placeholders
- baseline/profile/e2e templates exist for claimed supported regimes
- mining evidence contract is documented with required artifacts
- limitations are documented and testable
- utility scripts (if any) have documented inputs, outputs, and failure behavior
- adapter constraints appendix is present and testable
- candidate-vs-baseline differentiation and activation proof policy is explicit and testable
- parallel policy is explicit when adapter claims `parallel_safe=true`
